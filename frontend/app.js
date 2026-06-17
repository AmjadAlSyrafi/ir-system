"use strict";

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
  tbody.innerHTML = rows.map(r => `
    <tr>
      <td><strong>${escHtml(r.Model)}</strong></td>
      <td>${fmt(r.MAP)}</td>
      <td>${fmt(r.Recall)}</td>
      <td>${fmt(r["P@10"])}</td>
      <td>${fmt(r["nDCG@10"])}</td>
    </tr>`).join("");
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

    setLoading("eval-btn", "eval-spinner", "eval-btn-text", true, "Run Evaluation");

    try {
      // Placeholder: in a real setup the frontend would get qrels/results from
      // the gateway. Here we show the UI flow with a stub response.
      const rows = checkedModels.map(m => ({
        Model: m,
        MAP:     Math.random() * 0.5 + 0.1,
        Recall:  Math.random() * 0.5 + 0.2,
        "P@10":  Math.random() * 0.4 + 0.1,
        "nDCG@10": Math.random() * 0.5 + 0.15,
      }));

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