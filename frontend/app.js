"use strict";

// Served from the backend, so the API lives at the same origin — relative paths.
const $ = (s) => document.querySelector(s);
const esc = (s) =>
  String(s ?? "").replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
const fmt = (n) => (n === null || n === undefined ? "—" : Number(n).toFixed(4));
const pct = (n) => (n === null || n === undefined ? "—" : (Number(n) * 100).toFixed(1) + "%");

// ── Navigation ─────────────────────────────────────────────
let metricsLoaded = false;
document.querySelectorAll(".nav").forEach((nav) => {
  nav.onclick = () => {
    document.querySelectorAll(".nav").forEach((n) => n.classList.remove("active"));
    document.querySelectorAll(".panel").forEach((p) => p.classList.remove("active"));
    nav.classList.add("active");
    $("#" + nav.dataset.tab).classList.add("active");
    $("#pageTitle").textContent = nav.dataset.title;
    $("#pageCrumb").textContent = nav.dataset.crumb;
    if (nav.dataset.tab === "metrics" && !metricsLoaded) loadMetrics();
  };
});

// ── Backend health indicator ───────────────────────────────
async function pingHealth() {
  const dot = $("#hdot"), label = $("#hstatus");
  try {
    const res = await fetch("/health");
    if (!res.ok) throw new Error();
    dot.className = "dot ok";
    label.textContent = "Backend bağlı";
  } catch {
    dot.className = "dot down";
    label.textContent = "Backend yanıt vermiyor";
  }
}
pingHealth();
setInterval(pingHealth, 15000);

// ── Shared query call ──────────────────────────────────────
async function queryApi(body) {
  const res = await fetch("/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error("HTTP " + res.status + " — " + (await res.text()));
  return res.json();
}
const loadingCard = (txt) =>
  `<div class="card"><div class="placeholder"><span class="spinner"></span> &nbsp;${esc(txt)}</div></div>`;

// ── Sohbet ─────────────────────────────────────────────────
function renderAnswer(d) {
  const badge = d.abstained
    ? `<span class="chip red">● bilgi tabanında bulunamadı</span>`
    : `<span class="chip green">● cevaplandı</span>`;
  const srcs =
    (d.sources || [])
      .map(
        (s, i) => `
      <div class="src">
        <div class="srchd">
          <span class="rank">#${i + 1}</span>
          ${s.key ? `<span class="chip key">${esc(s.key)}</span>` : ""}
          <b>${esc(s.source_doc)}</b>
          <span class="score">${fmt(s.score)}</span>
        </div>
        <details>
          <summary>metni göster</summary>
          <div class="snippet">${esc(s.text || "")}</div>
        </details>
        <div class="meta">${esc(s.section || "—")} · <span class="chip">${esc(s.modality || "?")}</span></div>
      </div>`
      )
      .join("") || `<div class="muted">kaynak yok (çekimser kapısı)</div>`;
  return `
    <div class="card">
      <div class="card-hd"><h2>Cevap</h2>${badge}</div>
      <div class="answer">${esc(d.answer)}</div>
      <div class="muted">top_score (dense kapı): <span class="score" style="margin:0">${fmt(d.top_score)}</span></div>
    </div>
    <div class="card">
      <div class="card-hd"><h2>Kaynaklar · ${(d.sources || []).length}</h2></div>
      ${srcs}
    </div>`;
}

async function runChat() {
  const question = $("#q").value.trim();
  if (!question) { $("#chatOut").innerHTML = `<div class="card err">Soru boş.</div>`; return; }
  $("#chatOut").innerHTML = loadingCard("Sorgulanıyor…");
  try {
    const d = await queryApi({
      question,
      top_k: parseInt($("#topk").value, 10) || 5,
      retrieval_mode: $("#mode").value,
    });
    $("#chatOut").innerHTML = renderAnswer(d);
  } catch (e) {
    $("#chatOut").innerHTML = `<div class="card err">Hata: ${esc(e.message)}</div>`;
  }
}
$("#ask").onclick = runChat;
$("#q").addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") runChat();
});

// ── Metrikler ──────────────────────────────────────────────
// [label, accessor(mode) -> number, higherIsBetter, formatter]
const METRIC_ROWS = [
  ["Recall@3", (m) => m.retrieval.recall_at_k["3"], true, pct],
  ["Recall@5", (m) => m.retrieval.recall_at_k["5"], true, pct],
  ["MRR", (m) => m.retrieval.mrr, true, fmt],
  ["Faithfulness", (m) => m.generation.faithfulness, true, pct],
  ["Answer relevance", (m) => m.generation.answer_relevance, true, pct],
  ["Abstention recall", (m) => m.abstention.abstention_recall, true, pct],
  ["False abstentions", (m) => m.abstention.false_abstentions, false, (n) => n],
];

async function loadMetrics() {
  metricsLoaded = true;
  const el = $("#metricsOut");
  el.innerHTML = loadingCard("Metrikler yükleniyor…");
  try {
    const res = await fetch("/metrics");
    if (!res.ok) {
      const detail = await res.json().catch(() => ({}));
      el.innerHTML = `<div class="card"><div class="placeholder"><div class="ic">📊</div>${esc(detail.detail || "Metrik verisi yok.")}</div></div>`;
      return;
    }
    el.innerHTML = renderMetrics(await res.json());
  } catch (e) {
    metricsLoaded = false;
    el.innerHTML = `<div class="card err">Hata: ${esc(e.message)}</div>`;
  }
}

function kpi(lab, mode, fmtFn, higher) {
  const dv = fmtFn ? fmtFn(mode.d) : mode.d;
  let delta = "";
  if (mode.d !== null && mode.h !== null && mode.d !== mode.h) {
    const up = mode.h > mode.d;
    const good = higher ? up : !up;
    delta = `<div class="delta ${good ? "up" : "down"}">${up ? "▲" : "▼"} hybrid: ${fmtFn ? fmtFn(mode.h) : mode.h}</div>`;
  }
  return `<div class="kpi"><div class="lab">${esc(lab)}</div><div class="val">${dv}</div>${delta}</div>`;
}

function renderMetrics(data) {
  const c = data.config || {};
  const dense = data.modes?.dense;
  const hybrid = data.modes?.hybrid;
  const pick = (acc) => ({ d: dense ? acc(dense) : null, h: hybrid ? acc(hybrid) : null });

  const kpis = `
    <div class="kpis">
      ${kpi("Recall@5 (dense)", pick((m) => m.retrieval.recall_at_k["5"]), pct, true)}
      ${kpi("MRR (dense)", pick((m) => m.retrieval.mrr), fmt, true)}
      ${kpi("Faithfulness (dense)", pick((m) => m.generation.faithfulness), pct, true)}
      ${kpi("Abstention recall", pick((m) => m.abstention.abstention_recall), pct, true)}
    </div>`;

  const rows = METRIC_ROWS.map(([label, acc, higher, format]) => {
    const dv = dense ? acc(dense) : null;
    const hv = hybrid ? acc(hybrid) : null;
    let dCls = "num", hCls = "num";
    if (dv !== null && hv !== null && dv !== hv) {
      if (higher ? dv > hv : dv < hv) dCls += " best"; else hCls += " best";
    }
    return `<tr><td>${esc(label)}</td>
      <td class="${dCls}">${format(dv)}</td>
      <td class="${hCls}">${format(hv)}</td></tr>`;
  }).join("");

  const cfg = `
    <div class="muted" style="margin-bottom:4px; line-height:1.9">
      embedding <span class="chip">${esc(c.embedding_model)}</span>
      generation <span class="chip">${esc(c.generation_model)}</span>
      judge <span class="chip">${esc(c.judge_model)}</span>
      eşik <span class="chip">${esc(c.abstention_threshold)}</span>
      top_k <span class="chip">${esc(c.top_k)}</span>
      set <span class="chip">${esc(c.n_answerable)} cevaplanabilir + ${esc(c.n_abstention)} korpus dışı</span>
    </div>`;

  return `
    ${kpis}
    <div class="card">
      <div class="card-hd"><h2>Değerlendirme · dense vs hybrid</h2><span class="chip acc">${esc(c.n_total)} soruluk golden set</span></div>
      ${cfg}
      <table>
        <thead><tr><th>Metrik</th><th class="num">Dense</th><th class="num">Hybrid</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
      <div class="muted" style="margin-top:12px">● = o metrikte daha iyi olan mod. False abstentions hariç tüm metriklerde yüksek olan daha iyidir.</div>
    </div>`;
}

// ── Prompt akışı ───────────────────────────────────────────
function stepCard(title, cls, body) {
  return `<div class="step ${cls || ""}"><h3>${esc(title)}</h3>${body}</div>`;
}

function renderTrace(d) {
  const t = d.trace || {};
  const steps = [];

  if (t.embedding) {
    steps.push(stepCard("1 · Embed · soru → vektör", "answer",
      `<div><span class="k">model:</span> ${esc(t.embedding.model)} ·
        <span class="k">boyut:</span> ${esc(t.embedding.dim)}</div>
       <div class="muted">ilk 8 değer: [${t.embedding.preview.map(esc).join(", ")}]</div>`));
  }
  if (t.sparse) {
    steps.push(stepCard("1b · Sparse · BM25 sorgu terimleri", "answer",
      `<div><span class="k">eşleşen terim sayısı:</span> ${esc(t.sparse.term_count)}</div>`));
  }
  if (t.retrieval) {
    const hits = (t.retrieval.hits || [])
      .map(
        (h) => `<div class="src">
          <div class="srchd"><span class="rank">#${h.rank}</span>
            <b>${esc(h.source_doc)}</b><span class="score">${fmt(h.score)}</span></div>
          <div class="snippet">${esc(h.text || "")}</div>
          <div class="meta">${esc(h.section || "—")} · <span class="chip">${esc(h.modality || "?")}</span></div>
        </div>`
      )
      .join("") || `<div class="muted">sonuç yok</div>`;
    steps.push(stepCard(`2 · Retrieval · ${esc(t.retrieval.mode)}`, "answer", hits));
  }
  if (t.abstention) {
    const a = t.abstention;
    const abstain = a.decision === "abstain";
    steps.push(stepCard("3 · Çekimser kararı", abstain ? "abstain" : "answer",
      `<div><span class="k">top_score:</span> ${fmt(a.top_score)} ·
        <span class="k">eşik:</span> ${esc(a.threshold)}</div>
       <div>karar: <b>${abstain ? "ÇEKİMSER · model çağrılmadı" : "CEVAPLA"}</b></div>`));
  }
  if (t.generation) {
    if (t.generation.skipped) {
      steps.push(stepCard("4 · Generation", "skip",
        `<div class="muted">atlandı — çekimser kapısı kapandı.</div>`));
    } else {
      steps.push(stepCard("4 · Generation", "answer",
        `<div><span class="k">model:</span> ${esc(t.generation.model)}</div>
         <details><summary>modele giden bağlam (context)</summary><pre>${esc(t.generation.context)}</pre></details>
         <div class="answer" style="margin-top:10px">${esc(t.generation.answer)}</div>`));
    }
  }
  return `<div class="card"><div class="card-hd"><h2>Pipeline izi</h2>
    <span class="chip ${d.abstained ? "red" : "green"}">${d.abstained ? "çekimser" : "cevaplandı"}</span></div>
    <div>${steps.join("")}</div></div>`;
}

async function runFlow() {
  const question = $("#fq").value.trim();
  if (!question) { $("#flowOut").innerHTML = `<div class="card err">Soru boş.</div>`; return; }
  $("#flowOut").innerHTML = loadingCard("Pipeline çalışıyor…");
  try {
    const d = await queryApi({
      question,
      top_k: parseInt($("#ftopk").value, 10) || 5,
      retrieval_mode: $("#fmode").value,
      debug: true,
    });
    $("#flowOut").innerHTML = renderTrace(d);
  } catch (e) {
    $("#flowOut").innerHTML = `<div class="card err">Hata: ${esc(e.message)}</div>`;
  }
}
$("#fask").onclick = runFlow;
$("#fq").addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") runFlow();
});
