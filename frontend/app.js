"use strict";

// ---------------------------------------------------------------------------
// Built-in evaluation test set (matches the quick_fit.py sample corpus)
// ---------------------------------------------------------------------------

const EVAL_QUERIES = {
  q01: "information retrieval ranking algorithms",
  q02: "neural embeddings semantic search dense vectors",
  q03: "BM25 probabilistic retrieval model parameters",
  q04: "query expansion synonyms WordNet terms",
  q05: "precision recall nDCG evaluation metrics",
  q06: "inverted index document indexing construction",
  q07: "BERT transformer language model pre-training",
  q08: "hybrid retrieval fusion BM25 embeddings",
  q09: "tokenization stemming lemmatization text preprocessing",
  q10: "approximate nearest neighbor FAISS similarity search",
};

// Graded relevance: 2 = highly relevant, 1 = partially relevant
const EVAL_QRELS = [
  // q01 — IR ranking
  {query_id:"q01", doc_id:"d001", relevance:2}, {query_id:"q01", doc_id:"d002", relevance:2},
  {query_id:"q01", doc_id:"d006", relevance:2}, {query_id:"q01", doc_id:"d007", relevance:2},
  {query_id:"q01", doc_id:"d009", relevance:2}, {query_id:"q01", doc_id:"d005", relevance:1},
  {query_id:"q01", doc_id:"d004", relevance:1}, {query_id:"q01", doc_id:"d025", relevance:1},
  {query_id:"q01", doc_id:"d045", relevance:1},
  // q02 — neural embeddings
  {query_id:"q02", doc_id:"d003", relevance:2}, {query_id:"q02", doc_id:"d010", relevance:2},
  {query_id:"q02", doc_id:"d013", relevance:2}, {query_id:"q02", doc_id:"d015", relevance:2},
  {query_id:"q02", doc_id:"d048", relevance:2}, {query_id:"q02", doc_id:"d011", relevance:1},
  {query_id:"q02", doc_id:"d012", relevance:1}, {query_id:"q02", doc_id:"d018", relevance:1},
  {query_id:"q02", doc_id:"d019", relevance:1},
  // q03 — BM25
  {query_id:"q03", doc_id:"d002", relevance:2}, {query_id:"q03", doc_id:"d001", relevance:1},
  // q04 — query expansion
  {query_id:"q04", doc_id:"d008", relevance:2}, {query_id:"q04", doc_id:"d040", relevance:2},
  {query_id:"q04", doc_id:"d060", relevance:1}, {query_id:"q04", doc_id:"d028", relevance:1},
  // q05 — evaluation metrics
  {query_id:"q05", doc_id:"d005", relevance:2}, {query_id:"q05", doc_id:"d006", relevance:2},
  {query_id:"q05", doc_id:"d007", relevance:2}, {query_id:"q05", doc_id:"d044", relevance:1},
  {query_id:"q05", doc_id:"d041", relevance:1}, {query_id:"q05", doc_id:"d055", relevance:1},
  {query_id:"q05", doc_id:"d056", relevance:1},
  // q06 — inverted index
  {query_id:"q06", doc_id:"d004", relevance:2}, {query_id:"q06", doc_id:"d030", relevance:2},
  {query_id:"q06", doc_id:"d058", relevance:1},
  // q07 — BERT / transformer
  {query_id:"q07", doc_id:"d011", relevance:2}, {query_id:"q07", doc_id:"d012", relevance:2},
  {query_id:"q07", doc_id:"d013", relevance:1}, {query_id:"q07", doc_id:"d014", relevance:1},
  {query_id:"q07", doc_id:"d020", relevance:1},
  // q08 — hybrid retrieval
  {query_id:"q08", doc_id:"d021", relevance:2}, {query_id:"q08", doc_id:"d022", relevance:2},
  {query_id:"q08", doc_id:"d023", relevance:2}, {query_id:"q08", doc_id:"d024", relevance:1},
  {query_id:"q08", doc_id:"d025", relevance:1},
  // q09 — preprocessing
  {query_id:"q09", doc_id:"d031", relevance:2}, {query_id:"q09", doc_id:"d032", relevance:2},
  {query_id:"q09", doc_id:"d033", relevance:2}, {query_id:"q09", doc_id:"d034", relevance:1},
  {query_id:"q09", doc_id:"d038", relevance:1}, {query_id:"q09", doc_id:"d039", relevance:1},
  // q10 — ANN / FAISS
  {query_id:"q10", doc_id:"d010", relevance:2}, {query_id:"q10", doc_id:"d053", relevance:2},
  {query_id:"q10", doc_id:"d003", relevance:1}, {query_id:"q10", doc_id:"d046", relevance:1},
];

// =============================================================================
// api.js — all fetch calls to the API gateway
// =============================================================================

const API_BASE = "http://localhost:8000";

async function apiSearch({ query, dataset, model, top_k, use_refinement, user_id,
                           bm25_k1, bm25_b,
                           hybrid_bm25_weight, hybrid_embedding_weight, hybrid_tfidf_weight }) {
  const body = { query, dataset, model, top_k, use_refinement, user_id };
  if (model === "bm25") {
    if (bm25_k1 !== null) body.bm25_k1 = bm25_k1;
    if (bm25_b  !== null) body.bm25_b  = bm25_b;
  }
  if (model === "hybrid_parallel") {
    if (hybrid_bm25_weight      !== null) body.hybrid_bm25_weight      = hybrid_bm25_weight;
    if (hybrid_embedding_weight !== null) body.hybrid_embedding_weight = hybrid_embedding_weight;
    if (hybrid_tfidf_weight     !== null) body.hybrid_tfidf_weight     = hybrid_tfidf_weight;
  }
  const resp = await fetch(`${API_BASE}/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: resp.statusText }));
    throw new Error(err.detail || resp.statusText);
  }
  return resp.json();
}

async function apiEvaluate({ model_name, dataset, results_per_query, qrels, k }) {
  const resp = await fetch(`${API_BASE}/evaluate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model_name, dataset, results_per_query, qrels, k }),
  });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: resp.statusText }));
    throw new Error(err.detail || resp.statusText);
  }
  return resp.json();
}

// =============================================================================
// ui.js — DOM manipulation helpers
// =============================================================================

function showError(message) {
  document.getElementById("error-text").textContent = message;
  document.getElementById("error-banner").classList.remove("hidden");
}

function dismissError() {
  document.getElementById("error-banner").classList.add("hidden");
}

function setLoading(btnId, spinnerId, textId, loading, label = "Search") {
  const btn     = document.getElementById(btnId);
  const spinner = document.getElementById(spinnerId);
  const text    = document.getElementById(textId);
  btn.disabled  = loading;
  spinner.classList.toggle("hidden", !loading);
  text.textContent = loading ? "Loading…" : label;
}

function renderResults(results) {
  const container = document.getElementById("results-container");
  if (!results || results.length === 0) {
    container.innerHTML = `<p style="color:var(--muted);margin-top:.75rem">No results found.</p>`;
    return;
  }
  container.innerHTML = results.map((r, idx) => {
    const hasText  = r.text && r.text.trim().length > 0;
    const preview  = hasText ? escHtml(r.text.substring(0, 280)) : "";
    const isTrunc  = hasText && r.text.length > 280;
    const fullId   = `doc-full-${idx}`;
    const btnId    = `doc-btn-${idx}`;
    return `
      <div class="result-card">
        <div class="result-header">
          <span class="result-rank">#${r.rank}</span>
          <span class="result-doc-id">${escHtml(r.doc_id)}</span>
          <span class="result-score">score: ${Number(r.score).toFixed(4)}</span>
        </div>
        ${hasText ? `
          <div class="result-text">
            <span id="${fullId}-short">${preview}${isTrunc ? "…" : ""}</span>
            ${isTrunc ? `<span id="${fullId}-full" class="hidden">${escHtml(r.text)}</span>` : ""}
          </div>
          ${isTrunc ? `
          <button class="expand-btn" id="${btnId}"
            onclick="toggleDoc('${fullId}','${btnId}')">Show full document</button>
          ` : ""}
        ` : `<div class="result-no-text">Full text not available — re-run pipeline to populate document store.</div>`}
      </div>`;
  }).join("");
}

function toggleDoc(fullId, btnId) {
  const shortEl = document.getElementById(fullId + "-short");
  const fullEl  = document.getElementById(fullId + "-full");
  const btn     = document.getElementById(btnId);
  if (!fullEl) return;
  const expanded = !fullEl.classList.contains("hidden");
  fullEl.classList.toggle("hidden", expanded);
  shortEl.classList.toggle("hidden", !expanded);
  btn.textContent = expanded ? "Show full document" : "Collapse";
}

function renderMeta(total, timeMs) {
  const meta = document.getElementById("results-meta");
  meta.textContent = `${total} result${total !== 1 ? "s" : ""} · ${timeMs} ms`;
  meta.classList.remove("hidden");
}

function renderRefinementBanner(refinement) {
  const banner = document.getElementById("refinement-banner");
  if (!refinement || refinement.original === refinement.final) {
    banner.classList.add("hidden");
    return;
  }
  banner.textContent = `Query refined: "${refinement.original}" → "${refinement.final}"`;
  banner.classList.remove("hidden");
}

function renderEvalTable(rows) {
  const tbody = document.getElementById("eval-table-body");
  tbody.innerHTML = rows.map(r => {
    const notFitted = r.MAP === null;
    const tip = notFitted ? ` title="${escHtml(r._reason || 'Model not ready')}"` : "";
    const cell = v => notFitted
      ? `<td style="color:var(--muted);font-style:italic"${tip}>Not fitted</td>`
      : `<td>${fmt(v)}</td>`;
    return `
    <tr>
      <td><strong>${escHtml(r.Model)}</strong></td>
      ${cell(r.MAP)}${cell(r.Recall)}${cell(r["P@10"])}${cell(r["nDCG@10"])}
    </tr>`;
  }).join("");
}

let _evalChart   = null;
let _reportChart = null;

function renderEvalChart(rows) {
  const ctx = document.getElementById("eval-chart").getContext("2d");
  if (_evalChart) _evalChart.destroy();
  _evalChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: rows.map(r => r.Model),
      datasets: [{
        label: "MAP",
        data: rows.map(r => r.MAP),
        backgroundColor: "rgba(67,97,238,.75)",
        borderColor: "#4361ee",
        borderWidth: 1.5,
        borderRadius: 4,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false }, title: { display: true, text: "MAP by Model" } },
      scales: { y: { beginAtZero: true, max: 1, ticks: { stepSize: 0.1 } } },
    },
  });
}

async function loadReport(dataset, btn) {
  document.querySelectorAll(".report-tab-btn").forEach(b => b.classList.remove("active"));
  btn.classList.add("active");

  const loading = document.getElementById("report-loading");
  const errEl   = document.getElementById("report-error");
  const content = document.getElementById("report-content");
  const meta    = document.getElementById("report-meta");

  loading.classList.remove("hidden");
  errEl.classList.add("hidden");
  content.classList.add("hidden");

  try {
    const resp = await fetch(`${API_BASE}/reports/${dataset}`);
    if (!resp.ok) {
      const e = await resp.json().catch(() => ({ detail: resp.statusText }));
      throw new Error(e.detail || resp.statusText);
    }
    const data = await resp.json();
    loading.classList.add("hidden");

    // Populate table
    const tbody = document.getElementById("report-table-body");
    const METRIC_MAP = { "MAP": "MAP", "P@10": "P@10", "nDCG@10": "nDCG@10", "Recall": "Recall" };
    tbody.innerHTML = data.rows.map(r => {
      const best = data.rows.reduce((a, b) => parseFloat(a.MAP) >= parseFloat(b.MAP) ? a : b);
      const isBest = r.Model === best.Model;
      return `<tr>
        <td><strong>${escHtml(r.Model)}</strong>${isBest ? ' <span class="best-badge">best MAP</span>' : ""}</td>
        <td>${r.MAP ?? "—"}</td>
        <td>${r.Recall ?? "—"}</td>
        <td>${r["P@10"] ?? "—"}</td>
        <td>${r["nDCG@10"] ?? "—"}</td>
      </tr>`;
    }).join("");

    meta.textContent = `Dataset: ${data.dataset_id}  ·  ${data.rows.length} models evaluated`;

    // Chart
    if (_reportChart) _reportChart.destroy();
    const ctx = document.getElementById("report-chart").getContext("2d");
    const colors = ["#4361ee","#06b6d4","#10b981","#f59e0b","#ef4444"];
    _reportChart = new Chart(ctx, {
      type: "bar",
      data: {
        labels: data.rows.map(r => r.Model),
        datasets: [
          { label: "MAP",     data: data.rows.map(r => parseFloat(r.MAP)     || 0), backgroundColor: colors[0] + "cc", borderColor: colors[0], borderWidth:1.5, borderRadius:4 },
          { label: "Recall",  data: data.rows.map(r => parseFloat(r.Recall)  || 0), backgroundColor: colors[1] + "cc", borderColor: colors[1], borderWidth:1.5, borderRadius:4 },
          { label: "nDCG@10", data: data.rows.map(r => parseFloat(r["nDCG@10"]) || 0), backgroundColor: colors[2] + "cc", borderColor: colors[2], borderWidth:1.5, borderRadius:4 },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: "top" },
          title: { display: true, text: `Evaluation — ${data.dataset_id}` },
        },
        scales: { y: { beginAtZero: true, max: 1, ticks: { stepSize: 0.1 } } },
      },
    });

    content.classList.remove("hidden");
  } catch (err) {
    loading.classList.add("hidden");
    errEl.textContent = "Could not load report: " + err.message;
    errEl.classList.remove("hidden");
  }
}

function fmt(val) {
  return typeof val === "number" ? val.toFixed(4) : (val ?? "—");
}

function escHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

// =============================================================================
// Settings — localStorage persistence
// =============================================================================

const SETTINGS_KEY = "ir_settings";

const defaultSettings = {
  stemming: false,
  lemmatization: false,
  stopwords: true,
  topk: 10,
};

function loadSettings() {
  try {
    return { ...defaultSettings, ...JSON.parse(localStorage.getItem(SETTINGS_KEY) || "{}") };
  } catch { return { ...defaultSettings }; }
}

function saveSettings(settings) {
  localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
}

function applySettingsToUI(settings) {
  document.getElementById("setting-stemming").checked   = settings.stemming;
  document.getElementById("setting-lemmatization").checked = settings.lemmatization;
  document.getElementById("setting-stopwords").checked  = settings.stopwords;
  const topkEl = document.getElementById("setting-topk");
  topkEl.value = settings.topk;
  document.getElementById("setting-topk-val").textContent = settings.topk;
}

// =============================================================================
// main — event listeners and initialisation
// =============================================================================

document.addEventListener("DOMContentLoaded", () => {

  // ── Tab switching ──────────────────────────────────────────────────────────
  let _reportLoaded = false;
  document.querySelectorAll(".tab-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const target = btn.dataset.tab;
      document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
      document.querySelectorAll(".tab-content").forEach(c => {
        c.classList.toggle("active", c.id === `tab-${target}`);
        c.classList.toggle("hidden",  c.id !== `tab-${target}`);
      });
      btn.classList.add("active");
      // Auto-load dataset1 report the first time the Evaluate tab is opened.
      if (target === "evaluate" && !_reportLoaded) {
        _reportLoaded = true;
        const firstTabBtn = document.querySelector(".report-tab-btn[data-ds='dataset1']");
        if (firstTabBtn) loadReport("dataset1", firstTabBtn);
      }
    });
  });

  // ── Mode toggle (Basic / Basic+Advanced) ──────────────────────────────────
  const modeDescs = {
    basic:    "Core IR models only — no query refinement",
    advanced: "Core IR models + query refinement + additional features",
  };
  document.querySelectorAll('input[name="search-mode"]').forEach(radio => {
    radio.addEventListener("change", () => {
      document.querySelectorAll(".mode-tab").forEach(t => t.classList.remove("mode-tab-active"));
      radio.parentElement.classList.add("mode-tab-active");
      document.getElementById("mode-desc").textContent = modeDescs[radio.value] || "";
    });
  });

  // ── Model selector → show/hide parameter panels ──────────────────────────
  const modelSelect    = document.getElementById("model-select");
  const bm25Params     = document.getElementById("bm25-params");
  const hybridParams   = document.getElementById("hybrid-params");

  function updateModelParamVisibility() {
    bm25Params.classList.toggle("hidden",   modelSelect.value !== "bm25");
    hybridParams.classList.toggle("hidden", modelSelect.value !== "hybrid_parallel");
  }
  modelSelect.addEventListener("change", updateModelParamVisibility);
  updateModelParamVisibility();

  // ── BM25 slider live values ────────────────────────────────────────────────
  ["bm25-k1", "bm25-b"].forEach(id => {
    const el  = document.getElementById(id);
    const val = document.getElementById(`${id}-val`);
    el.addEventListener("input", () => { val.textContent = el.value; });
  });

  // ── Hybrid weight sliders live values + normalised percentages ────────────
  function updateHybridWeightDisplay() {
    const wB = parseFloat(document.getElementById("hybrid-bm25-w").value);
    const wE = parseFloat(document.getElementById("hybrid-emb-w").value);
    const wT = parseFloat(document.getElementById("hybrid-tfidf-w").value);
    document.getElementById("hybrid-bm25-w-val").textContent   = wB.toFixed(2);
    document.getElementById("hybrid-emb-w-val").textContent    = wE.toFixed(2);
    document.getElementById("hybrid-tfidf-w-val").textContent  = wT.toFixed(2);
    const total = wB + wE + wT || 1;
    document.getElementById("hw-bm25-pct").textContent   = Math.round(wB / total * 100) + "%";
    document.getElementById("hw-emb-pct").textContent    = Math.round(wE / total * 100) + "%";
    document.getElementById("hw-tfidf-pct").textContent  = Math.round(wT / total * 100) + "%";
  }
  ["hybrid-bm25-w", "hybrid-emb-w", "hybrid-tfidf-w"].forEach(id => {
    document.getElementById(id).addEventListener("input", updateHybridWeightDisplay);
  });
  updateHybridWeightDisplay();

  // ── Settings slider live value ─────────────────────────────────────────────
  const topkSlider = document.getElementById("setting-topk");
  const topkVal    = document.getElementById("setting-topk-val");
  topkSlider.addEventListener("input", () => { topkVal.textContent = topkSlider.value; });

  // ── Load and apply stored settings ────────────────────────────────────────
  const settings = loadSettings();
  applySettingsToUI(settings);

  // ── Save settings ──────────────────────────────────────────────────────────
  document.getElementById("save-settings-btn").addEventListener("click", () => {
    const updated = {
      stemming:       document.getElementById("setting-stemming").checked,
      lemmatization:  document.getElementById("setting-lemmatization").checked,
      stopwords:      document.getElementById("setting-stopwords").checked,
      topk:           parseInt(topkSlider.value, 10),
    };
    saveSettings(updated);
    const status = document.getElementById("settings-status");
    status.classList.remove("hidden");
    setTimeout(() => status.classList.add("hidden"), 2000);
  });

  // ── Search ─────────────────────────────────────────────────────────────────
  async function doSearch() {
    const query = document.getElementById("query-input").value.trim();
    if (!query) { showError("Please enter a search query."); return; }

    const s = loadSettings();
    const top_k = s.topk;

    setLoading("search-btn", "search-spinner", "search-btn-text", true, "Search");
    document.getElementById("results-container").innerHTML = "";
    document.getElementById("results-meta").classList.add("hidden");
    document.getElementById("refinement-banner").classList.add("hidden");

    try {
      const mode    = document.querySelector('input[name="search-mode"]:checked').value;
      const isHybP  = modelSelect.value === "hybrid_parallel";
      const data = await apiSearch({
        query,
        dataset:                  document.getElementById("dataset-select").value,
        model:                    modelSelect.value,
        top_k,
        use_refinement:           mode === "advanced",
        user_id:                  "default",
        bm25_k1:                  modelSelect.value === "bm25" ? parseFloat(document.getElementById("bm25-k1").value) : null,
        bm25_b:                   modelSelect.value === "bm25" ? parseFloat(document.getElementById("bm25-b").value)  : null,
        hybrid_bm25_weight:       isHybP ? parseFloat(document.getElementById("hybrid-bm25-w").value)  : null,
        hybrid_embedding_weight:  isHybP ? parseFloat(document.getElementById("hybrid-emb-w").value)   : null,
        hybrid_tfidf_weight:      isHybP ? parseFloat(document.getElementById("hybrid-tfidf-w").value) : null,
      });

      renderRefinementBanner(data.refinement);
      renderMeta(data.results.length, data.time_ms);
      renderResults(data.results);
    } catch (err) {
      showError("Search failed: " + err.message);
    } finally {
      setLoading("search-btn", "search-spinner", "search-btn-text", false, "Search");
    }
  }

  document.getElementById("search-btn").addEventListener("click", doSearch);
  document.getElementById("query-input").addEventListener("keydown", e => {
    if (e.key === "Enter") doSearch();
  });

  // ── Evaluate ───────────────────────────────────────────────────────────────
  document.getElementById("eval-btn").addEventListener("click", async () => {
    const checkedModels = Array.from(
      document.querySelectorAll(".checkbox-group input[type=checkbox]:checked")
    ).map(cb => cb.value);

    if (checkedModels.length === 0) {
      showError("Select at least one model to evaluate.");
      return;
    }

    const evalDataset = document.getElementById("eval-dataset").value;
    const k           = 10;
    const progress    = document.getElementById("eval-progress");

    setLoading("eval-btn", "eval-spinner", "eval-btn-text", true, "Run Evaluation");
    document.getElementById("eval-results").classList.add("hidden");
    progress.classList.remove("hidden");
    progress.textContent = `Running evaluation against real ${evalDataset} queries… (this may take a minute)`;

    try {
      const resp = await fetch(`${API_BASE}/evaluate/run`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          dataset:     evalDataset,
          models:      checkedModels,
          max_queries: 20,
          k,
        }),
      });

      if (!resp.ok) {
        const err = await resp.json().catch(() => ({ detail: resp.statusText }));
        throw new Error(err.detail || resp.statusText);
      }

      const data = await resp.json();

      const rows = checkedModels.map(modelName => {
        const m = data.models[modelName];
        if (!m || m.error) {
          const reason = m && m.error ? m.error : "model not ready";
          return { Model: modelName, MAP: null, Recall: null, "P@10": null, "nDCG@10": null, _reason: reason };
        }
        return {
          Model:     modelName,
          MAP:       m.MAP                         ?? 0,
          Recall:    m.mean_recall                 ?? 0,
          "P@10":    m[`mean_precision_at_${k}`]   ?? 0,
          "nDCG@10": m[`mean_ndcg_at_${k}`]        ?? 0,
        };
      });

      progress.textContent = `Evaluated ${data.num_queries} queries from ${data.dataset_id}.`;
      document.getElementById("eval-results").classList.remove("hidden");
      renderEvalTable(rows);
      renderEvalChart(rows);

    } catch (err) {
      showError("Evaluation failed: " + err.message);
    } finally {
      setLoading("eval-btn", "eval-spinner", "eval-btn-text", false, "Run Evaluation");
    }
  });
});