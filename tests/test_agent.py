"""
Agent tests. Run from project root:  python -m unittest discover -s tests -v

Real: the ML /predict pipeline (predict_core), economics.py, risk.py, the RAG pipeline code, the LangGraph graph.
TEST DOUBLES: OpenWeatherMap HTTP responses (canned), the LLM (scripted JSON), the embedder (hashing), and
generated test PDFs. None of the doubles' values are agricultural data.
"""
import contextlib
import io
import json
import os
import re
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["OPENWEATHER_API_KEY"] = "test-key-not-real"
os.environ["RAG_MIN_SCORE"] = "0.2"

with contextlib.redirect_stdout(io.StringIO()):
    import app as A  # real app: provides predict_core and crop_info
A.DEBUG = False
A.reverse_geocode = lambda lat, lon: {"country": "India", "state": "Telangana", "district": "Hyderabad"}

import agent.api as agent_api  # noqa: E402
import agent.tools as tools  # noqa: E402
from agent.facts import allowed_numbers, unsupported_numbers  # noqa: E402
from rag.ingest import ingest  # noqa: E402
from rag.pipeline import RagPipeline  # noqa: E402
from rag.store import VectorStore  # noqa: E402
from tests.test_rag import HashEmbedder, StubLLM, make_pdf  # noqa: E402

SOY_DOC = ["Soybean is sown at the onset of the monsoon when soil moisture is adequate. "
           "Well drained loamy soils are preferred for soybean sowing in the monsoon season."]


class Resp:
    def __init__(self, data):
        self._d = data

    def raise_for_status(self):
        pass

    def json(self):
        return self._d


CURRENT = {"coord": {"lat": 17.38, "lon": 78.48}, "name": "Hyderabad", "sys": {"country": "IN"},
           "main": {"temp": 27.3, "humidity": 71}, "weather": [{"main": "Clouds"}]}
FORECAST = {"list": [{"main": {"temp": 25.0 + (i % 5)}, **({"rain": {"3h": 1.0}} if i in (1, 2) else {})}
                     for i in range(40)]}   # first 8 slots: two with 1.0 mm -> 2.0 mm


class FakeHTTP:
    def __init__(self, fail_forecast=False):
        self.calls, self.fail_forecast = [], fail_forecast

    def __call__(self, url, params=None, timeout=None):
        self.calls.append(url)
        if url.endswith("/weather"):
            return Resp(CURRENT)
        if self.fail_forecast:
            import requests
            raise requests.ConnectionError("secret-url-with-appid=test-key-not-real")
        return Resp(FORECAST)


def fid(user, label):
    m = re.search(rf"\[(F\d+)\] \([a-z_]+\) {re.escape(label)}", user)
    return m.group(1) if m else "F999"


class ScriptedLLM:
    """Understand call -> canned JSON; synthesis call -> `synth(user)` JSON string."""
    def __init__(self, crop="soybean", agricultural=True, synth=None, available=True):
        self.crop, self.agricultural, self.synth, self._available, self.calls = crop, agricultural, synth, available, []

    def available(self):
        return self._available

    def generate(self, system, user):
        self.calls.append("understand" if system.startswith("You turn") else "synthesize")
        if system.startswith("You turn"):
            return json.dumps({"is_agricultural": self.agricultural, "crop": self.crop,
                               "rag_query": "soybean sowing monsoon season"})
        return self.synth(user)


def good_synth(user):
    t, r = fid(user, "Current temperature"), fid(user, "Forecast rainfall, next 24 h")
    return json.dumps({
        "verdict": "conditional",
        "summary": "The documents describe soybean sowing at the monsoon onset, but the ML model and price data do not cover soybean.",
        "points": [
            {"text": "Current temperature is 27.3 C with 2.0 mm of rain forecast in the next 24 h.", "evidence": [t, r]},
            {"text": "Documents say soybean is sown at the onset of the monsoon.", "evidence": ["D1"]},
            {"text": "You can expect about 47 quintals per hectare.", "evidence": [t]},           # invented number
            {"text": "Soil should be well drained.", "evidence": []},                            # no evidence
            {"text": "Humidity is fine.", "evidence": ["F77"]},                                 # bad evidence id
        ],
        "caveats": ["Yield and profit data are unavailable.", "Expected profit is 90000 per hectare."],  # 2nd invented
    })


def make_rag(tmp, llm_reply="Soybean is sown at the onset of the monsoon [1]."):
    docs, idx = os.path.join(tmp, "docs"), os.path.join(tmp, "idx")
    os.makedirs(docs, exist_ok=True)
    make_pdf(os.path.join(docs, "soy_test.pdf"), SOY_DOC)
    ingest(docs, idx, HashEmbedder(), log=lambda *_: None)
    pipe = RagPipeline(embedder=HashEmbedder(), llm=StubLLM(reply=llm_reply), store=VectorStore(idx))
    return lambda: pipe


class AgentBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.http = FakeHTTP()
        tools._cache.clear()
        p = mock.patch.object(tools.requests, "get", self.http)
        p.start()
        self.addCleanup(p.stop)
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(lambda: agent_api.configure_agent(A.predict_core, A.crop_info))
        self.client = A.app.test_client()

    def use(self, llm, rag=None):
        agent_api.configure_agent(A.predict_core, A.crop_info, llm=llm,
                                  region_lookup=lambda la, lo: {"state": "Telangana", "district": "Hyderabad"},
                                  rag_pipeline_getter=rag or make_rag(self.tmp.name))

    def ask(self, body):
        r = self.client.post("/api/agent/advice", json=body)
        return r.status_code, r.get_json()

    FULL = {"question": "Should I grow soybean in my farm this season?", "location": "Hyderabad",
            "soil": {"N": 90, "P": 42, "K": 43, "ph": 6.5}}


class FullFlowTests(AgentBase):
    def test_full_soybean_flow(self):
        llm = ScriptedLLM(synth=good_synth)
        self.use(llm)
        status, body = self.ask(self.FULL)
        self.assertEqual(status, 200, body)
        # graph ran each tool exactly once, in order, then synthesised (no loops)
        self.assertEqual([t["tool"] for t in body["tool_trace"]],
                         ["understand", "get_weather", "predict_crop", "estimate_yield_profit", "search_agri_documents"])
        self.assertEqual(llm.calls, ["understand", "synthesize"])
        # four labelled sections
        for k in ("model_prediction", "external_data", "retrieved_knowledge", "final_reasoning"):
            self.assertIn("kind", body[k])
        # numbers come from tools
        w = body["external_data"]["weather"]
        self.assertEqual((w["current"]["temperature_c"], w["forecast_24h"]["rainfall_mm"]), (27.3, 2.0))
        mp = body["model_prediction"]["result"]
        self.assertTrue(mp["available"])
        self.assertFalse(mp["target_crop"]["in_model_vocabulary"] or False)   # soybean is not an ML class
        eco = body["external_data"]["economics_and_risk"]
        self.assertFalse(eco["profit"]["available"])                          # no price/cost/yield files -> unavailable
        self.assertFalse(eco["risk"].get("score"))                            # soybean has no ideal ranges stored
        # retrieved knowledge has sources
        rk = body["retrieved_knowledge"]["result"]
        self.assertTrue(rk["available"])
        self.assertEqual(rk["sources"][0]["document"], "soy_test.pdf")
        # final reasoning: valid points kept, invented numbers / unsupported evidence removed
        fr = body["final_reasoning"]
        texts = [p["text"] for p in fr["points"]]
        self.assertEqual(len(texts), 2)
        self.assertFalse(any("47" in t for t in texts))
        self.assertFalse(fr["numeric_check"]["passed"])
        self.assertEqual(set(fr["numeric_check"]["unsupported_numbers"]), {"47", "90000"})
        self.assertEqual(len(fr["numeric_check"]["removed"]), 4)   # 47, no-evidence, bad id, 90000 caveat
        self.assertEqual(fr["caveats"], ["Yield and profit data are unavailable."])
        self.assertIn("D1", fr["evidence_index"])

    def test_missing_inputs_skip_tools_and_are_reported(self):
        self.use(ScriptedLLM(synth=good_synth))
        status, body = self.ask({"question": "Should I grow soybean this season?"})
        self.assertEqual(status, 200)
        self.assertEqual([t["tool"] for t in body["tool_trace"]],
                         ["understand", "estimate_yield_profit", "search_agri_documents"])
        self.assertEqual(self.http.calls, [])                      # no weather call without a location
        self.assertEqual(len(body["missing_information"]), 2)      # location + soil

    def test_out_of_scope_uses_no_tools(self):
        llm = ScriptedLLM(agricultural=False, crop=None, synth=good_synth)
        self.use(llm)
        status, body = self.ask({"question": "Who won the football match yesterday?"})
        self.assertEqual(status, 200)
        self.assertEqual(body["final_reasoning"]["status"], "out_of_scope")
        self.assertEqual([t["tool"] for t in body["tool_trace"]], ["understand"])
        self.assertEqual(llm.calls, ["understand"])

    def test_verdict_forced_to_insufficient_without_evidence(self):
        def overconfident(user):
            return json.dumps({"verdict": "recommended", "summary": "Grow it.", "points": [], "caveats": []})
        empty_rag = make_rag(self.tmp.name)  # docs exist but we ask about something else below
        self.use(ScriptedLLM(synth=overconfident), rag=lambda: __import__("rag.pipeline", fromlist=["x"]).RagPipeline(
            embedder=HashEmbedder(), llm=StubLLM(), store=VectorStore(os.path.join(self.tmp.name, "none"))))
        status, body = self.ask({"question": "Should I grow soybean this season?", "location": "Hyderabad"})
        self.assertEqual(body["final_reasoning"]["verdict"], "insufficient_evidence")

    def test_no_llm_key_returns_503_with_evidence_and_keyword_fallback(self):
        self.use(ScriptedLLM(available=False, synth=good_synth))
        status, body = self.ask(self.FULL)
        self.assertEqual(status, 503)
        self.assertFalse(body["success"])
        self.assertEqual(body["plan"]["understanding_method"], "keyword_fallback")
        self.assertEqual(body["plan"]["crop"], "soybean")                    # found by keyword
        self.assertTrue(body["evidence"]["facts"])                           # collected evidence still returned
        self.assertIsNone(body["final_reasoning"]["summary"])

    def test_api_validation(self):
        self.use(ScriptedLLM(synth=good_synth))
        bad = [{}, {"question": "hi"}, {"question": "x" * 2000}, {"question": "ok question", "lat": 10},
               {"question": "ok question", "soil": {"ph": 20}}, {"question": "ok question", "season": "Monsoon"},
               {"question": "ok question", "soil": {"N": "a"}}]
        for b in bad:
            self.assertEqual(self.ask(b)[0], 400, b)
        self.assertEqual(self.client.post("/api/agent/advice", data="nope").status_code, 400)


class WeatherToolTests(AgentBase):
    def test_parse_rain_sum_and_cache(self):
        r1 = tools.get_weather("Hyderabad")
        r2 = tools.get_weather("hyderabad ")
        self.assertEqual(r1["forecast_24h"]["rainfall_mm"], 2.0)
        self.assertEqual(r1["forecast_5d"]["total_rain_mm"], 2.0)
        self.assertTrue(r2["cached"])
        self.assertEqual(len(self.http.calls), 2)       # 1 weather + 1 forecast for two lookups

    def test_forecast_failure_is_unavailable_not_estimated_and_leaks_nothing(self):
        self.http.fail_forecast = True
        r = tools.get_weather(lat=17.38, lon=78.48)
        self.assertTrue(r["available"])
        self.assertIsNone(r["forecast_24h"])
        self.assertNotIn("test-key", json.dumps(r))
        pred = tools.make_predict_crop(A.predict_core)(r, {"N": 1, "P": 1, "K": 1, "ph": 6})
        self.assertFalse(pred["available"])
        self.assertIn("rainfall", pred["reason"])

    def test_missing_key_and_http_error_messages(self):
        with mock.patch.dict(os.environ, {"OPENWEATHER_API_KEY": ""}):
            self.assertFalse(tools.get_weather("X")["available"])
        import requests

        def boom(*a, **k):
            raise requests.ConnectionError("https://api.openweathermap.org/...appid=test-key-not-real")
        with mock.patch.object(tools.requests, "get", boom):
            r = tools.get_weather("Nowhere")
        self.assertFalse(r["available"])
        self.assertNotIn("test-key", r["reason"])


class GuardTests(unittest.TestCase):
    def test_numeric_guard(self):
        facts = [{"label": "Rain next 24 h", "value": "54.5 mm", "available": True}]
        allowed = allowed_numbers(facts, [{"text": "sow 25 days after"}], "asked about 2 acres")
        self.assertEqual(unsupported_numbers("54.5 mm [F1], 24 h, 25 days, 2 acres", allowed), [])
        self.assertEqual(unsupported_numbers("about 55 mm", allowed), ["55"])      # rounding is not allowed
        # scoped to cited evidence: 25 only exists in the passage, so a point citing only F1 may not use it
        facts2 = [dict(facts[0], id="F1"), {"id": "F2", "label": "x", "value": "7 days", "available": True}]
        scoped = allowed_numbers(facts2, [], "q", only_ids={"F2"})
        self.assertEqual(unsupported_numbers("54.5 mm", scoped), ["54.5"])
        self.assertEqual(unsupported_numbers("expect 99 mm and 1,200 kg [D1]", allowed), ["99", "1,200"])

    def test_graph_is_acyclic(self):
        """The declared graph (not just the runtime path) must have no cycle and no self-loop."""
        import agent.api as api
        edges = {(e.source, e.target) for e in api._graph.get_graph().edges}
        order = ["__start__", "understand", "get_weather", "predict_crop", "estimate_yield_profit",
                 "search_agri_documents", "synthesize", "__end__"]
        rank = {n: i for i, n in enumerate(order)}
        self.assertTrue(edges)
        for src, dst in edges:
            self.assertLess(rank[src], rank[dst], f"backward/self edge {src}->{dst}")

    def test_agent_has_exactly_four_tools(self):
        t = tools.build_tools(A.predict_core, A.crop_info, __import__("economics"), __import__("risk"), lambda: None)
        self.assertEqual(set(t), {"get_weather", "predict_crop", "estimate_yield_profit", "search_agri_documents"})


if __name__ == "__main__":
    unittest.main()
