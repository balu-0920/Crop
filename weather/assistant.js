// Chat UI for the agriculture RAG assistant. All server text is inserted with
// textContent (never innerHTML), so document text can't inject markup.
const RAG_API_BASE = "http://127.0.0.1:5000/api/agriculture";

const chatLog = document.getElementById("chatLog");
const chatForm = document.getElementById("chatForm");
const chatInput = document.getElementById("chatInput");
const chatSend = document.getElementById("chatSend");
const chatStatus = document.getElementById("chatStatus");

function el(tag, className, text){
    const node = document.createElement(tag);
    if(className) node.className = className;
    if(text !== undefined) node.textContent = text;
    return node;
}

function addMessage(role, text, extraClass){
    const wrap = el("div", `chat-msg chat-${role}${extraClass ? " " + extraClass : ""}`);
    wrap.appendChild(el("div", "chat-bubble", text));
    chatLog.appendChild(wrap);
    chatLog.scrollTop = chatLog.scrollHeight;
    return wrap;
}

function addSources(wrap, sources, retrieved){
    if(sources && sources.length){
        const box = el("div", "chat-sources");
        box.appendChild(el("strong", null, "Sources"));
        const ul = el("ul");
        sources.forEach(s => {
            const label = `[${s.passage}] ${s.title || s.document} — ${s.document}, page ${s.page}`;
            ul.appendChild(el("li", null, label));
        });
        box.appendChild(ul);
        wrap.appendChild(box);
    }
    if(retrieved && retrieved.length){
        const details = el("details", "chat-context");
        details.appendChild(el("summary", null, `Retrieved passages (${retrieved.length})`));
        retrieved.forEach(r => {
            const item = el("div", "chat-context-item");
            item.appendChild(el("div", "chat-context-meta",
                `${r.document}, p.${r.page} · similarity ${r.score}${r.used_in_answer ? " · used" : " · not used"}`));
            item.appendChild(el("div", null, r.excerpt));
            details.appendChild(item);
        });
        wrap.appendChild(details);
    }
}

async function loadStatus(){
    try{
        const res = await fetch(`${RAG_API_BASE}/status`);
        const s = await res.json();
        const docs = Object.keys(s.documents || {}).length;
        chatStatus.textContent = docs
            ? `Knowledge base: ${docs} document(s), ${s.chunks} passages.${s.llm_configured ? "" : " ⚠ Language model not configured (set ANTHROPIC_API_KEY in .env)."}`
            : "Knowledge base is empty — add PDFs to rag/documents/ and run: python -m rag.ingest";
    }catch(e){
        chatStatus.textContent = "Cannot reach the backend at " + RAG_API_BASE + " — is `python app.py` running?";
    }
}

chatForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const question = chatInput.value.trim();
    if(question.length < 3) return;

    addMessage("user", question);
    chatInput.value = "";
    chatSend.disabled = true;
    const pending = addMessage("bot", "Searching the documents…");

    try{
        const res = await fetch(`${RAG_API_BASE}/ask`, {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({question})
        });
        const data = await res.json();
        pending.remove();
        if(!data.success){
            const wrap = addMessage("bot", data.message || "Something went wrong.", "chat-error");
            addSources(wrap, [], data.retrieved_context);
        }else{
            const wrap = addMessage("bot", data.answer, data.answerable ? "" : "chat-unanswerable");
            addSources(wrap, data.sources, data.retrieved_context);
        }
    }catch(e){
        pending.remove();
        addMessage("bot", "Could not reach the backend. Is `python app.py` running?", "chat-error");
    }finally{
        chatSend.disabled = false;
        chatInput.focus();
    }
});

loadStatus();
