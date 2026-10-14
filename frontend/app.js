const $ = (id) => document.getElementById(id);
const selected = new Set();
let docs = [];

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text; // textContent only: document text is untrusted
  return node;
}

function setStatus(message, kind = "") {
  const box = $("upload-status");
  box.textContent = message;
  box.className = "status " + kind;
}

async function api(path, options) {
  const res = await fetch(path, options);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch (_) { /* keep statusText */ }
    throw new Error(detail);
  }
  return res.status === 204 ? null : res.json();
}

function renderDocs() {
  const list = $("doc-list");
  list.replaceChildren();
  $("empty-docs").hidden = docs.length > 0;
  for (const doc of docs) {
    const li = el("li");
    const box = el("input");
    box.type = "checkbox";
    box.checked = selected.has(doc.id);
    box.setAttribute("aria-label", "Search " + doc.name);
    box.addEventListener("change", () => (box.checked ? selected.add(doc.id) : selected.delete(doc.id)));
    const name = el("span", "name", doc.name);
    name.title = doc.name;
    const meta = el("span", "meta", doc.chunks + " chunks");
    const del = el("button", "", "×");
    del.title = "Delete " + doc.name;
    del.type = "button";
    del.addEventListener("click", async () => {
      await api("/api/documents/" + doc.id, { method: "DELETE" });
      selected.delete(doc.id);
      loadDocs();
    });
    li.append(box, name, meta, del);
    list.append(li);
  }
}

async function loadDocs() {
  docs = await api("/api/documents");
  renderDocs();
}

async function upload(files) {
  if (!files.length) return;
  const form = new FormData();
  for (const f of files) form.append("files", f);
  setStatus("Uploading and indexing...");
  try {
    const result = await api("/api/documents", { method: "POST", body: form });
    const failed = result.errors.map((e) => e.name + ": " + e.error).join(" | ");
    setStatus("Indexed " + result.documents.length + " file(s)." + (failed ? " Skipped: " + failed : ""), failed ? "error" : "ok");
  } catch (err) {
    setStatus("Upload failed: " + err.message, "error");
  }
  loadDocs();
}

function addMessage(role) {
  const wrap = el("div", "message " + role);
  const bubble = el("div", "bubble");
  wrap.append(bubble);
  $("messages").append(wrap);
  $("messages").scrollTop = $("messages").scrollHeight;
  return bubble;
}

function renderAnswer(bubble, data) {
  bubble.textContent = data.answer;
  if (data.warning) bubble.append(el("div", "warning", data.warning));
  if (data.sources.length) {
    const box = el("div", "sources");
    for (const s of data.sources) {
      const d = el("details");
      const sum = el("summary");
      sum.append(el("span", "badge", "[" + s.n + "]"), document.createTextNode(s.doc_name + ", page " + s.page));
      d.append(sum, el("p", "", s.text));
      box.append(d);
    }
    bubble.append(box);
  }
}

$("ask-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = $("question").value.trim();
  if (!question) return;
  $("question").value = "";
  addMessage("user").textContent = question;
  const bubble = addMessage("assistant");
  bubble.textContent = "Searching your documents...";
  $("ask-btn").disabled = true;
  try {
    const body = { question };
    if (selected.size) body.doc_ids = [...selected];
    const data = await api("/api/ask", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    });
    renderAnswer(bubble, data);
  } catch (err) {
    bubble.className = "bubble error";
    bubble.textContent = "Something went wrong: " + err.message;
  } finally {
    $("ask-btn").disabled = false;
    $("messages").scrollTop = $("messages").scrollHeight;
  }
});

$("select-all").addEventListener("click", () => {
  const all = docs.every((d) => selected.has(d.id));
  selected.clear();
  if (!all) docs.forEach((d) => selected.add(d.id));
  renderDocs();
});

const zone = $("dropzone");
$("file-input").addEventListener("change", (e) => { upload([...e.target.files]); e.target.value = ""; });
["dragenter", "dragover"].forEach((t) => zone.addEventListener(t, (e) => { e.preventDefault(); zone.classList.add("drag"); }));
["dragleave", "drop"].forEach((t) => zone.addEventListener(t, (e) => { e.preventDefault(); zone.classList.remove("drag"); }));
zone.addEventListener("drop", (e) => upload([...e.dataTransfer.files]));

api("/api/health").then((h) => {
  const labels = { anthropic: "Claude", openai: "OpenAI", extractive: "offline extractive mode (set an API key for LLM answers)" };
  $("mode").textContent = "Answers: " + labels[h.llm_mode];
}).catch(() => {});
loadDocs();
