"""
Tests for economics / risk / decision scoring. Run from project root:
    python -m unittest discover -s tests -v

All numbers in the CSV fixtures below are TEST-ONLY placeholders written to a
temp dir to exercise the arithmetic. They are not real prices, costs or yields.
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import economics as econ  # noqa: E402
import risk  # noqa: E402
import decision  # noqa: E402


def write_fixture(d, prices="", costs="", yields=""):
    for name, header, rows in (
        ("market_prices.csv", "crop,state,district,price_inr_per_quintal,price_date,source\n", prices),
        ("cultivation_costs.csv", "crop,state,cost_inr_per_hectare,reference_year,source\n", costs),
        ("historical_yields.csv", "crop,state,district,year,yield_t_per_ha,source\n", yields),
    ):
        with open(os.path.join(d, name), "w") as f:
            f.write(header + rows)
    econ._state["mtimes"].clear()  # force reload


class EconomicsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._old = econ.ECON_DIR
        econ.ECON_DIR = self.tmp.name

    def tearDown(self):
        econ.ECON_DIR = self._old
        econ._state["mtimes"].clear()
        self.tmp.cleanup()

    def test_everything_unavailable_when_files_empty(self):
        write_fixture(self.tmp.name)
        e = econ.estimate_economics("Paddy", "S", "D")
        for k in ("expected_yield", "market_price", "cultivation_cost", "revenue", "profit"):
            self.assertFalse(e[k]["available"], k)
        self.assertIn("missing", e["profit"]["reason"])

    def test_missing_files_are_graceful(self):
        econ.ECON_DIR = os.path.join(self.tmp.name, "nope")
        econ._state["mtimes"].clear()
        self.assertFalse(econ.estimate_economics("rice")["profit"]["available"])

    def test_revenue_profit_arithmetic_and_alias(self):
        yields = "".join(f"Paddy,S,D,{y},{v},t\n" for y, v in
                         [(2019, 3.0), (2020, 3.2), (2021, 3.4), (2022, 3.6), (2023, 3.8), (2018, 9.9)])
        write_fixture(self.tmp.name,
                      prices="rice,S,,2000,2099-01-01,t\n",   # 2000/quintal = 20000/tonne
                      costs="rice,,40000,2023,t\n",
                      yields=yields)
        e = econ.estimate_economics("rice", "S", "D")
        self.assertEqual(e["expected_yield"]["value"], 3.4)            # median of latest 5 years (2018 excluded)
        self.assertEqual(e["expected_yield"]["reference_best"], 9.9)    # best ever for the crop
        self.assertEqual(e["revenue"]["value"], 68000)                  # 3.4 t x 20000
        self.assertEqual(e["profit"]["value"], 28000)                   # 68000 - 40000
        self.assertEqual(e["market_price"]["level"], "state")
        self.assertEqual(e["cultivation_cost"]["level"], "national")

    def test_profit_unavailable_without_cost_but_revenue_ok(self):
        write_fixture(self.tmp.name, prices="rice,,,1000,2099-01-01,t\n", yields="rice,,,2023,2.0,t\n")
        e = econ.estimate_economics("rice")
        self.assertTrue(e["revenue"]["available"])
        self.assertFalse(e["profit"]["available"])

    def test_invalid_rows_skipped(self):
        write_fixture(self.tmp.name, prices="rice,,,abc,2099-01-01,t\nrice,,,-5,2099-01-01,t\n")
        self.assertFalse(econ.market_price("rice")["available"])
        self.assertEqual(econ.data_status()["prices"]["rows_skipped_invalid"], 2)


CROP = {"ideal_temperature": [20, 30], "ideal_humidity": [50, 80], "ideal_rainfall": [100, 200]}


class RiskTests(unittest.TestCase):
    def test_no_risk_inside_ranges(self):
        r = risk.assess_risk(CROP, {"temperature": 25, "humidity": 60, "rainfall": 150})
        self.assertEqual(r["score"], 0)
        self.assertEqual(r["level"], "Low")
        self.assertIn("drought_index", r["unavailable"])
        self.assertIn("forecast_uncertainty", r["unavailable"])
        self.assertIn("rainfall_variability", r["unavailable"])  # no pipeline data

    def test_heat_and_excess_rain(self):
        r = risk.assess_risk(CROP, {"temperature": 40, "humidity": 60, "rainfall": 300})
        self.assertEqual(r["components"]["temperature_stress"], 100)   # a full range-width above
        self.assertEqual(r["components"]["rainfall_stress"], 100)
        self.assertEqual(r["level"], "High")
        self.assertTrue(any("Heat" in d for d in r["drivers"]))
        self.assertTrue(any("Excess rainfall" in d for d in r["drivers"]))

    def test_drought_indicator_and_variability(self):
        r = risk.assess_risk(CROP, {"temperature": 25, "humidity": 60, "rainfall": 50},
                             {"rainfall_cv_pct": 30})
        self.assertEqual(r["components"]["rainfall_stress"], 50)
        self.assertEqual(r["components"]["rainfall_variability"], 60)
        self.assertNotIn("rainfall_variability", r["unavailable"])
        self.assertAlmostEqual(sum(r["weights_used"].values()), 1, places=2)

    def test_unknown_crop(self):
        self.assertIsNone(risk.assess_risk(None, {"temperature": 1, "humidity": 1, "rainfall": 1})["score"])


class DecisionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._old = econ.ECON_DIR
        econ.ECON_DIR = self.tmp.name
        write_fixture(self.tmp.name)

    def tearDown(self):
        econ.ECON_DIR = self._old
        econ._state["mtimes"].clear()
        self.tmp.cleanup()

    def recs(self):
        mk = lambda k, s: {"crop": k.title(), "crop_key": k, "overall_score": s,
                           "reasons": [{"ok": True, "text": "Common in your region"}]}
        return [mk("rice", 60), mk("maize", 58)]

    def test_no_economics_final_is_suitability_and_safety_only(self):
        crop_info = {"rice": CROP, "maize": CROP}
        out, w = decision.apply_decision_layer(self.recs(), crop_info,
                                               {"temperature": 25, "humidity": 60, "rainfall": 150})
        d = out[0]["decision"]
        self.assertEqual(d["components_unavailable"], ["yield", "profit"])
        self.assertAlmostEqual(d["components_used"]["suitability"], 0.75)
        self.assertEqual(d["final_score"], 0.75 * 60 + 0.25 * 100)
        self.assertFalse(d["profit"]["available"])
        self.assertTrue(any("Profit not estimated" in r for r in d["main_reasons"]))

    def test_risk_can_reorder(self):
        cold = dict(CROP, ideal_temperature=[10, 15])  # maize: today's 25C is far outside
        out, _ = decision.apply_decision_layer(self.recs(), {"rice": CROP, "maize": cold},
                                               {"temperature": 25, "humidity": 60, "rainfall": 150})
        self.assertEqual(out[0]["crop_key"], "rice")

    def test_profit_component_used_when_data_present(self):
        write_fixture(self.tmp.name,
                      prices="rice,,,2000,2099-01-01,t\nmaize,,,1000,2099-01-01,t\n",
                      costs="rice,,10000,2023,t\nmaize,,10000,2023,t\n",
                      yields="rice,,,2023,4.0,t\nmaize,,,2023,3.0,t\n")
        out, _ = decision.apply_decision_layer(self.recs(), {"rice": CROP, "maize": CROP},
                                               {"temperature": 25, "humidity": 60, "rainfall": 150})
        by = {r["crop_key"]: r["decision"] for r in out}
        self.assertEqual(by["rice"]["profit"]["value"], 70000)   # 4t x 20000 - 10000
        self.assertEqual(by["maize"]["profit"]["value"], 20000)  # 3t x 10000 - 10000
        self.assertEqual(by["rice"]["components"]["profit"], 100)
        self.assertAlmostEqual(by["maize"]["components"]["profit"], 100 * 20000 / 70000, places=1)
        self.assertEqual(by["rice"]["components_unavailable"], [])


if __name__ == "__main__":
    unittest.main()
