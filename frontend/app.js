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

async function apiSearch({ query, dataset, model, top_k, use_refinement, user_id, bm25_k1, bm25_b }) {
  const body = { query, dataset, model, top_k, use_refinement, user_id };
  if (model === "bm25") {
    if (bm25_k1 !== null) body.bm25_k1 = bm25_k1;
    if (bm25_b  !== null) body.bm25_b  = bm25_b;
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
  container.innerHTML = results.map(r => {
    const snippet = r.text ? r.text.substring(0, 200) + (r.text.length > 200 ? "…" : "") : "";
    return `
      <div class="result-card">
        <div class="result-rank">${r.rank}</div>
        <div class="result-doc-id">${escHtml(r.doc_id)}</div>
        <div class="result-score">score: ${Number(r.score).toFixed(4)}</div>
        ${snippet ? `<div class="result-snippet">${escHtml(snippet)}</div>` : ""}
      </div>`;
  }).join("");
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
    const cell = v => notFitted
      ? `<td style="color:var(--muted);font-style:italic">Not fitted</td>`
      : `<td>${fmt(v)}</td>`;
    return `
    <tr>
      <td><strong>${escHtml(r.Model)}</strong></td>
      ${cell(r.MAP)}${cell(r.Recall)}${cell(r["P@10"])}${cell(r["nDCG@10"])}
    </tr>`;
  }).join("");
}

let _evalChart = null;

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
  document.querySelectorAll(".tab-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const target = btn.dataset.tab;
      document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
      document.querySelectorAll(".tab-content").forEach(c => {
        c.classList.toggle("active", c.id === `tab-${target}`);
        c.classList.toggle("hidden",  c.id !== `tab-${target}`);
      });
      btn.classList.add("active");
    });
  });

  // ── Model selector → show/hide BM25 sliders ───────────────────────────────
  const modelSelect  = document.getElementById("model-select");
  const bm25Params   = document.getElementById("bm25-params");

  function updateBm25Visibility() {
    bm25Params.classList.toggle("hidden", modelSelect.value !== "bm25");
  }
  modelSelect.addEventListener("change", updateBm25Visibility);
  updateBm25Visibility();

  // ── BM25 slider live values ────────────────────────────────────────────────
  ["bm25-k1", "bm25-b"].forEach(id => {
    const el  = document.getElementById(id);
    const val = document.getElementById(`${id}-val`);
    el.addEventListener("input", () => { val.textContent = el.value; });
  });

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
      const data = await apiSearch({
        query,
        dataset:         document.getElementById("dataset-select").value,
        model:           modelSelect.value,
        top_k,
        use_refinement:  document.getElementById("use-refinement").checked,
        user_id:         "default",
        bm25_k1:         modelSelect.value === "bm25" ? parseFloat(document.getElementById("bm25-k1").value) : null,
        bm25_b:          modelSelect.value === "bm25" ? parseFloat(document.getElementById("bm25-b").value)  : null,
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
    const queryIds    = Object.keys(EVAL_QUERIES);
    const rows        = [];

    setLoading("eval-btn", "eval-spinner", "eval-btn-text", true, "Run Evaluation");
    document.getElementById("eval-results").classList.add("hidden");
    progress.classList.remove("hidden");

    try {
      for (const modelName of checkedModels) {

        // Step 1 — run every test query through the search API
        const results_per_query = {};
        let   notFitted = false;

        for (let i = 0; i < queryIds.length; i++) {
          const qid       = queryIds[i];
          const queryText = EVAL_QUERIES[qid];
          progress.textContent =
            `[${modelName}] Searching query ${i + 1} / ${queryIds.length}: "${queryText}"`;

          try {
            const data = await apiSearch({
              query:          queryText,
              dataset:        evalDataset,
              model:          modelName,
              top_k:          k,
              use_refinement: false,
              user_id:        "eval",
              bm25_k1:        null,
              bm25_b:         null,
            });
            results_per_query[qid] = data.results || [];
          } catch (err) {
            const msg = err.message || "";
            const isNotReady =
              msg.includes("not fitted") ||
              msg.includes("not loaded")  ||
              msg.includes("FAISS index") ||
              msg.includes("503")         ||
              msg.includes("Service unreachable");
            if (isNotReady) {
              notFitted = true;
              break;   // no point continuing — model not ready
            }
            results_per_query[qid] = [];
          }
        }

        if (notFitted) {
          rows.push({ Model: modelName, MAP: null, Recall: null, "P@10": null, "nDCG@10": null });
          continue;
        }

        // Step 2 — send results + qrels to evaluation service
        progress.textContent = `[${modelName}] Computing metrics…`;
        try {
          const evalResult = await apiEvaluate({
            model_name:        modelName,
            dataset:           evalDataset,
            results_per_query,
            qrels:             EVAL_QRELS,
            k,
          });

          rows.push({
            Model:     modelName,
            MAP:       evalResult.MAP                           ?? 0,
            Recall:    evalResult.mean_recall                   ?? 0,
            "P@10":    evalResult[`mean_precision_at_${k}`]     ?? 0,
            "nDCG@10": evalResult[`mean_ndcg_at_${k}`]          ?? 0,
          });
        } catch (err) {
          rows.push({ Model: modelName, MAP: 0, Recall: 0, "P@10": 0, "nDCG@10": 0 });
          showError(`Evaluation failed for ${modelName}: ${err.message}`);
        }
      }

      if (rows.length > 0) {
        document.getElementById("eval-results").classList.remove("hidden");
        renderEvalTable(rows);
        renderEvalChart(rows);
      }
    } catch (err) {
      showError("Evaluation error: " + err.message);
    } finally {
      setLoading("eval-btn", "eval-spinner", "eval-btn-text", false, "Run Evaluation");
      progress.classList.add("hidden");
    }
  });
});