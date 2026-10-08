// UI for POST /api/agent/advice. All server text goes in via textContent (never innerHTML).
const AGENT_API = "http://127.0.0.1:5000/api/agent/advice";

const $ = id => document.getElementById(id);
function el(tag, cls, text){
    const n = document.createElement(tag);
    if(cls) n.className = cls;
    if(text !== undefined && text !== null) n.textContent = text;
    return n;
}

const VERDICT_LABEL = {
    recommended: "Recommended", conditional: "Conditionally recommended",
    not_recommended: "Not recommended", insufficient_evidence: "Not enough evidence to decide"
};

function section(title, kind, open){
    const d = el("details", "agent-section");
    if(open) d.open = true;
    d.appendChild(el("summary", null, title));
    d.appendChild(el("p", "agent-kind", kind));
    return d;
}

function factList(facts){
    const ul = el("ul", "agent-facts");
    facts.forEach(f => {
        const li = el("li");
        li.appendChild(el("span", "agent-fid", f.id));
        li.appendChild(el("strong", null, f.label + ": "));
        if(f.available){ li.appendChild(document.createTextNode(String(f.value))); }
        else { li.appendChild(el("span", "decision-unavailable", "Unavailable")); li.appendChild(el("small", null, " — " + (f.reason || ""))); }
        ul.appendChild(li);
    });
    return ul;
}

function render(data){
    const out = $("agResult");
    out.innerHTML = "";
    const fr = data.final_reasoning || {};
    const idx = fr.evidence_index || {};

    // ---- 4) final AI reasoning ------------------------------------------------
    const card = el("div", "panel agent-answer");
    if(fr.verdict){
        card.appendChild(el("span", `agent-verdict verdict-${fr.verdict}`, VERDICT_LABEL[fr.verdict] || fr.verdict));
    }
    if(fr.summary){ card.appendChild(el("p", "agent-summary", fr.summary)); }
    if(fr.points && fr.points.length){
        const ul = el("ul", "agent-points");
        fr.points.forEach(p => {
            const li = el("li");
            li.appendChild(document.createTextNode(p.text + " "));
            p.evidence.forEach(e => li.appendChild(el("span", "agent-chip", e)).setAttribute("title", idx[e] || e));
            ul.appendChild(li);
        });
        card.appendChild(ul);
    }
    (fr.caveats || []).forEach(c => card.appendChild(el("p", "agent-caveat", "⚠ " + c)));
    if(data.missing_information && data.missing_information.length){
        const box = el("div", "agent-missing");
        box.appendChild(el("strong", null, "To get a fuller answer, provide:"));
        const ul = el("ul"); data.missing_information.forEach(m => ul.appendChild(el("li", null, m)));
        box.appendChild(ul); card.appendChild(box);
    }
    const nc = fr.numeric_check;
    if(nc && !nc.passed){
        card.appendChild(el("p", "agent-caveat", `Number check: ${nc.removed.length} statement(s) were removed because they contained numbers that did not come from the tools or documents.`));
    }
    card.appendChild(el("p", "agent-note", "AI-written reasoning over the evidence below. It can be wrong; chips (F#/D#) point to the evidence each statement relies on."));
    out.appendChild(card);

    const facts = (data.evidence && data.evidence.facts) || [];

    // ---- 1) model prediction ----------------------------------------------------
    const s1 = section("1 · Model prediction", data.model_prediction.kind, true);
    const mf = facts.filter(f => f.category === "model_prediction");
    s1.appendChild(mf.length ? factList(mf) : el("p", "agent-empty", "Not run (missing inputs or earlier step unavailable)."));
    out.appendChild(s1);

    // ---- 2) external data -------------------------------------------------------
    const s2 = section("2 · External data", data.external_data.kind, true);
    const ef = facts.filter(f => f.category === "external_data");
    s2.appendChild(ef.length ? factList(ef) : el("p", "agent-empty", "Not fetched."));
    out.appendChild(s2);

    // ---- 3) retrieved knowledge -------------------------------------------------
    const s3 = section("3 · Retrieved knowledge (documents)", data.retrieved_knowledge.kind, true);
    const rk = data.retrieved_knowledge.result;
    if(rk && rk.available){
        s3.appendChild(el("p", null, rk.answer));
        const ul = el("ul", "agent-facts");
        (rk.sources || []).forEach(s => ul.appendChild(el("li", null, `[${s.passage}] ${s.title || s.document} — ${s.document}, page ${s.page}`)));
        s3.appendChild(ul);
        ((data.evidence && data.evidence.passages) || []).forEach(p => {
            const d = el("div", "chat-context-item");
            d.appendChild(el("div", "chat-context-meta", `${p.id} · ${p.document}, p.${p.page} · similarity ${p.score}`));
            d.appendChild(el("div", null, p.text));
            s3.appendChild(d);
        });
    }else{
        s3.appendChild(el("p", "agent-empty", (rk && rk.reason) || "No document knowledge was retrieved."));
    }
    out.appendChild(s3);

    // ---- tool trace ---------------------------------------------------------------
    const trace = (data.tool_trace || []).map(t => `${t.tool} ${t.ok ? "✓" : "✗"}`).join("  →  ");
    out.appendChild(el("p", "agent-trace", "Steps: " + trace + " → synthesize"));
}

$("agentForm").addEventListener("submit", async e => {
    e.preventDefault();
    const body = {question: $("agQuestion").value.trim()};
    const loc = $("agLocation").value.trim(); if(loc) body.location = loc;
    const season = $("agSeason").value; if(season) body.season = season;
    const soil = {};
    [["N","agN"],["P","agP"],["K","agK"],["ph","agPh"]].forEach(([k,id]) => {
        const v = $(id).value; if(v !== "") soil[k] = Number(v);
    });
    if(Object.keys(soil).length) body.soil = soil;

    $("agSubmit").disabled = true;
    $("agStatus").textContent = "Gathering weather, model output, data and documents…";
    $("agResult").innerHTML = "";
    try{
        const res = await fetch(AGENT_API, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
        const data = await res.json();
        if(data.evidence){ render(data); }
        $("agStatus").textContent = data.success ? "" : (data.message || "Request failed.");
    }catch(err){
        $("agStatus").textContent = "Could not reach the backend at " + AGENT_API + " — is `python app.py` running?";
    }finally{
        $("agSubmit").disabled = false;
    }
});
