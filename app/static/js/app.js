/**
 * PaperMind — Three-Panel Research Workspace Script
 *
 * Controls Document Library (Left), Research Workspace (Center),
 * and Evidence Inspector (Right).
 * Communicates with POST /query and GET /health.
 * Zero external dependencies. Zero fabricated metrics.
 */

// ── State ──────────────────────────────────────────────────────────────────
const state = {
  currentPdf: /** @type {string|null} */ (null),
  isLoading: false,
  lastRequestId: /** @type {string|null} */ (null),
  lastResult: /** @type {object|null} */ (null),
  selectedCitationIdx: 0,
};

// ── DOM References ─────────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);

const dom = {
  appLayout:         $("app-layout"),

  // Panel Toggles
  toggleLeftBtn:     $("toggle-left-btn"),
  closeLeftBtn:      $("close-left-btn"),
  toggleRightBtn:    $("toggle-right-btn"),
  closeRightBtn:     $("close-right-btn"),

  // Status
  systemStatus:      $("system-status"),
  statusLabel:       $("system-status")?.querySelector(".status-label"),
  statusDot:         $("system-status")?.querySelector(".status-dot"),

  // Document Library (Left)
  pdfInput:          $("pdf-filename"),
  loadPaperBtn:      $("load-paper-btn"),
  activePaperBox:    $("active-paper-display"),
  activePaperName:   $("active-paper-name"),
  presetList:        $("preset-list"),

  // Workspace Composer (Center)
  queryInput:        $("query-input"),
  charCount:         $("query-char-count"),
  topKInput:         $("top-k-input"),
  submitBtn:         $("submit-btn"),
  shortcutHint:      $("shortcut-hint"),

  // Center States
  stateEmpty:        $("state-empty"),
  stateLoading:      $("state-loading"),
  stateError:        $("state-error"),
  stateInsufficient: $("state-insufficient"),
  stateSuccess:      $("state-success"),

  // Center Content
  loadingStage:      $("loading-stage"),
  errorMessage:      $("error-message"),
  errorDismissBtn:   $("error-dismiss-btn"),
  insufficientMsg:   $("insufficient-message"),
  answerText:        $("answer-text"),
  copyAnswerBtn:     $("copy-answer-btn"),
  contextText:       $("context-text"),
  techPanel:         $("tech-panel"),
  techGrid:          $("tech-grid"),

  // Evidence Inspector (Right)
  citationsPanel:    $("citations-panel"),
  citationCount:     $("citation-count"),
  citationList:      $("citation-list"),

  // Inspector Detail View
  inspectorEmptyTip: $("inspector-empty-tip"),
  inspectorActive:   $("inspector-active-content"),
  insLabel:          $("ins-label"),
  insType:           $("ins-type"),
  insSource:         $("ins-source"),
  insPage:           $("ins-page"),
  insId:             $("ins-id"),
  insScore:          $("ins-score"),

  // Footer
  footerRequestId:   $("footer-request-id"),
};

// ── Panel Visibility & Drawer Toggles ──────────────────────────────────────
function toggleLeftPanel() {
  const isMobile = window.innerWidth <= 900;
  if (isMobile) {
    dom.appLayout?.classList.toggle("left-open");
  } else {
    dom.appLayout?.classList.toggle("left-collapsed");
  }
}

function toggleRightPanel() {
  const isMobile = window.innerWidth <= 900;
  if (isMobile) {
    dom.appLayout?.classList.toggle("right-open");
  } else {
    dom.appLayout?.classList.toggle("right-collapsed");
  }
}

// ── Health Check ───────────────────────────────────────────────────────────
async function checkHealth() {
  try {
    const res = await fetch("/health");
    if (!res.ok) throw new Error("Status " + res.status);
    const data = await res.json();
    if (data.status === "healthy") {
      setStatus("healthy", "API Ready");
    } else {
      setStatus("error", "API Degraded");
    }
  } catch {
    setStatus("error", "API Offline");
  }
}

function setStatus(status, label) {
  if (!dom.systemStatus) return;
  dom.systemStatus.dataset.status = status;
  if (dom.statusLabel) dom.statusLabel.textContent = label;
}

// ── Document Selection ─────────────────────────────────────────────────────
function selectPaper(filename) {
  if (!filename || !filename.trim().toLowerCase().endsWith(".pdf")) {
    showValidationError(dom.pdfInput, "Filename must end with .pdf");
    return;
  }
  const cleanName = filename.trim();
  state.currentPdf = cleanName;

  if (dom.pdfInput) dom.pdfInput.value = cleanName;
  if (dom.activePaperName) dom.activePaperName.textContent = cleanName;
  if (dom.activePaperBox) dom.activePaperBox.classList.remove("hidden");

  // Highlight active preset item
  document.querySelectorAll(".doc-item-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.getAttribute("data-filename") === cleanName);
  });

  updateSubmitState();
}

// ── Validation Helpers ─────────────────────────────────────────────────────
function showValidationError(inputEl, message) {
  if (!inputEl) return;
  inputEl.classList.add("input-error");
  inputEl.setAttribute("aria-invalid", "true");
  inputEl.setCustomValidity(message);
  inputEl.reportValidity();

  inputEl.addEventListener(
    "input",
    () => {
      inputEl.classList.remove("input-error");
      inputEl.removeAttribute("aria-invalid");
      inputEl.setCustomValidity("");
    },
    { once: true }
  );
}

// ── Composer Controls ──────────────────────────────────────────────────────
function updateSubmitState() {
  const hasQuery = (dom.queryInput?.value || "").trim().length > 0;
  const hasPaper = state.currentPdf !== null;
  if (dom.submitBtn) {
    dom.submitBtn.disabled = !(hasQuery && hasPaper) || state.isLoading;
  }
}

function onQueryInput() {
  const len = (dom.queryInput?.value || "").length;
  if (dom.charCount) {
    dom.charCount.textContent = len > 0 ? `${len} chars` : "";
  }
  updateSubmitState();
}

// ── State Switching ────────────────────────────────────────────────────────
const ALL_STATES = [
  dom.stateEmpty,
  dom.stateLoading,
  dom.stateError,
  dom.stateInsufficient,
  dom.stateSuccess,
];

function showState(activeEl) {
  ALL_STATES.forEach((el) => {
    if (!el) return;
    el.classList.toggle("hidden", el !== activeEl);
  });
}

// ── Query Execution ────────────────────────────────────────────────────────
async function submitQuery() {
  if (state.isLoading) return;

  const query = (dom.queryInput?.value || "").trim();
  if (!query) {
    showValidationError(dom.queryInput, "Please enter a research question.");
    return;
  }
  if (!state.currentPdf) {
    showValidationError(dom.pdfInput, "Please select or enter a paper filename first.");
    return;
  }

  const rawTopK = parseInt(dom.topKInput?.value, 10);
  const safeTopK = Number.isFinite(rawTopK) && rawTopK > 0 ? rawTopK : 5;

  state.isLoading = true;
  updateSubmitState();

  // Reset results display
  dom.techPanel?.classList.add("hidden");
  showState(dom.stateLoading);

  // Staged loading tracker
  const stages = [
    "Ingesting & indexing multimodal document…",
    "Running hybrid dense + sparse retrieval…",
    "Applying cross-encoder candidate reranking…",
    "Synthesizing grounded citations…",
  ];
  let stageIdx = 0;
  if (dom.loadingStage) dom.loadingStage.textContent = stages[0];
  const stageTimer = setInterval(() => {
    stageIdx = (stageIdx + 1) % stages.length;
    if (dom.loadingStage) dom.loadingStage.textContent = stages[stageIdx];
  }, 2400);

  let requestId = null;

  try {
    const response = await fetch("/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        pdf_path: state.currentPdf,
        query: query,
        top_k: safeTopK,
      }),
    });

    requestId = response.headers.get("X-Request-ID") || null;

    if (!response.ok) {
      let detail = `Server responded with HTTP ${response.status}`;
      try {
        const errJson = await response.json();
        detail = errJson.detail || detail;
      } catch {
        // Fall back to HTTP status message
      }
      throw new Error(detail);
    }

    const data = await response.json();
    state.lastResult = data;
    state.lastRequestId = requestId;
    renderResult(data, requestId);

  } catch (err) {
    clearInterval(stageTimer);
    state.isLoading = false;
    updateSubmitState();
    renderError(err.message || "An unexpected error occurred during synthesis.");
    if (requestId && dom.footerRequestId) {
      dom.footerRequestId.textContent = `trace: ${requestId}`;
    }
    return;
  }

  clearInterval(stageTimer);
  state.isLoading = false;
  updateSubmitState();
}

// ── Render Results ─────────────────────────────────────────────────────────
function renderResult(data, requestId) {
  if (requestId && dom.footerRequestId) {
    dom.footerRequestId.textContent = `trace: ${requestId}`;
  }

  // Insufficient evidence case
  if (data.insufficient_evidence === true) {
    if (dom.insufficientMsg && data.answer) {
      dom.insufficientMsg.textContent = data.answer;
    }
    showState(dom.stateInsufficient);
    if (data.citations && data.citations.length > 0) {
      renderCitations(data.citations);
    }
    renderTechDetails(data, requestId);
    return;
  }

  // Success grounded synthesis
  if (dom.answerText) {
    dom.answerText.textContent = data.answer || "(No textual answer generated)";
  }
  if (dom.contextText) {
    dom.contextText.textContent = data.grounded_context || "(No raw context excerpt)";
  }

  showState(dom.stateSuccess);
  renderCitations(data.citations || []);
  renderTechDetails(data, requestId);
}

// ── Evidence Inspector Rendering ───────────────────────────────────────────
const TYPE_STYLES = {
  text:     { cls: "type-text",     label: "Text" },
  caption:  { cls: "type-caption",  label: "Caption" },
  image:    { cls: "type-image",    label: "Figure" },
  table:    { cls: "type-table",    label: "Table" },
  equation: { cls: "type-equation", label: "Formula" },
};

function renderCitations(citations) {
  if (!dom.citationList) return;

  if (!citations || citations.length === 0) {
    if (dom.citationCount) dom.citationCount.textContent = "0 cited";
    dom.citationList.innerHTML = "";
    if (dom.inspectorEmptyTip) dom.inspectorEmptyTip.classList.remove("hidden");
    if (dom.inspectorActive) dom.inspectorActive.classList.add("hidden");
    return;
  }

  if (dom.citationCount) {
    dom.citationCount.textContent = `${citations.length} cited`;
  }
  dom.citationList.innerHTML = "";

  citations.forEach((c, idx) => {
    const card = document.createElement("li");
    card.className = "citation-card" + (idx === 0 ? " selected" : "");
    card.setAttribute("data-idx", String(idx));

    // Top row
    const topRow = document.createElement("div");
    topRow.className = "citation-top-row";

    const labelBadge = document.createElement("span");
    labelBadge.className = "citation-label-badge";
    labelBadge.textContent = c.citation_label || `[${idx + 1}]`;

    const typeKey = (c.element_type || "text").toLowerCase();
    const typeMeta = TYPE_STYLES[typeKey] || { cls: "type-text", label: c.element_type || "Text" };

    const typeBadge = document.createElement("span");
    typeBadge.className = `citation-type-badge ${typeMeta.cls}`;
    typeBadge.textContent = typeMeta.label;

    topRow.appendChild(labelBadge);
    topRow.appendChild(typeBadge);

    // Source Title & Page
    const sourceTitle = document.createElement("div");
    sourceTitle.className = "citation-source";
    sourceTitle.textContent = c.source_filename || "Document Source";
    sourceTitle.title = c.source_filename || "";

    const pageBadge = document.createElement("div");
    pageBadge.className = "citation-page-badge";
    pageBadge.textContent = `Page ${c.page_number ?? "—"} · ${c.element_id || "Element"}`;

    card.appendChild(topRow);
    card.appendChild(sourceTitle);
    card.appendChild(pageBadge);

    // Score meter
    if (c.score != null) {
      const numScore = typeof c.score === "number" ? c.score : parseFloat(c.score);
      const pct = Math.min(100, Math.max(0, Math.round(numScore * 100)));

      const meter = document.createElement("div");
      meter.className = "citation-score-meter";

      const scoreMeta = document.createElement("div");
      scoreMeta.className = "score-meta";
      scoreMeta.innerHTML = `<span>Relevance Score</span><span class="score-val">${numScore.toFixed(3)}</span>`;

      const track = document.createElement("div");
      track.className = "score-bar-track";

      const fill = document.createElement("div");
      fill.className = "score-bar-fill";
      fill.style.width = `${pct}%`;

      track.appendChild(fill);
      meter.appendChild(scoreMeta);
      meter.appendChild(track);
      card.appendChild(meter);
    }

    // Click handler to select and inspect
    card.addEventListener("click", () => {
      document.querySelectorAll(".citation-card").forEach((el) => el.classList.remove("selected"));
      card.classList.add("selected");
      inspectCitation(c);
    });

    dom.citationList.appendChild(card);
  });

  // Select first citation by default
  inspectCitation(citations[0]);
}

function inspectCitation(c) {
  if (!c || !dom.inspectorActive) return;

  if (dom.inspectorEmptyTip) dom.inspectorEmptyTip.classList.add("hidden");
  dom.inspectorActive.classList.remove("hidden");

  if (dom.insLabel) dom.insLabel.textContent = c.citation_label || "[1]";

  const typeKey = (c.element_type || "text").toLowerCase();
  const typeMeta = TYPE_STYLES[typeKey] || { cls: "type-text", label: c.element_type || "Text" };
  if (dom.insType) {
    dom.insType.className = `inspector-type-pill ${typeMeta.cls}`;
    dom.insType.textContent = typeMeta.label;
  }

  if (dom.insSource) dom.insSource.textContent = c.source_filename || "Document Source";
  if (dom.insPage) dom.insPage.textContent = String(c.page_number ?? "—");
  if (dom.insId) dom.insId.textContent = c.element_id || "—";

  if (dom.insScore) {
    if (c.score != null) {
      const num = typeof c.score === "number" ? c.score : parseFloat(c.score);
      dom.insScore.textContent = `${num.toFixed(3)} (${Math.round(num * 100)}%)`;
    } else {
      dom.insScore.textContent = "—";
    }
  }
}

// ── Technical Telemetry Drawer ─────────────────────────────────────────────
function renderTechDetails(data, requestId) {
  if (!dom.techPanel || !dom.techGrid) return;

  const entries = [];
  if (requestId)              entries.push(["Trace Request ID", requestId]);
  if (data.status)            entries.push(["Execution Status", data.status]);
  if (state.currentPdf)       entries.push(["Document Context", state.currentPdf]);
  if (data.citations?.length) entries.push(["Retrieved Candidates", data.citations.length]);
  if (data.insufficient_evidence !== undefined) {
    entries.push(["Evidence Sufficiency", data.insufficient_evidence ? "Refusal (Insufficient)" : "Verified"]);
  }

  if (entries.length === 0) {
    dom.techPanel.classList.add("hidden");
    return;
  }

  dom.techGrid.innerHTML = "";
  entries.forEach(([key, val]) => {
    const dt = document.createElement("dt");
    dt.textContent = key;
    const dd = document.createElement("dd");
    dd.textContent = String(val);
    dom.techGrid.appendChild(dt);
    dom.techGrid.appendChild(dd);
  });

  dom.techPanel.classList.remove("hidden");
}

// ── Error View ─────────────────────────────────────────────────────────────
function renderError(message) {
  if (dom.errorMessage) {
    dom.errorMessage.textContent = message || "An unexpected error occurred.";
  }
  showState(dom.stateError);
  dom.techPanel?.classList.add("hidden");
}

function dismissError() {
  showState(dom.stateEmpty);
}

// ── Keyboard Shortcuts ─────────────────────────────────────────────────────
function onKeyDown(e) {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
    if (!dom.submitBtn?.disabled) {
      submitQuery();
    }
  }
}

// ── Copy Answer to Clipboard ───────────────────────────────────────────────
async function copyAnswer() {
  const text = dom.answerText?.textContent || "";
  if (!text) return;

  try {
    await navigator.clipboard.writeText(text);
    const span = dom.copyAnswerBtn?.querySelector("span");
    if (span) {
      const prev = span.textContent;
      span.textContent = "Copied!";
      setTimeout(() => {
        span.textContent = prev;
      }, 1800);
    }
  } catch {
    // Clipboard permission denied
  }
}

// ── Initialization ─────────────────────────────────────────────────────────
function init() {
  // Panel toggle event listeners
  dom.toggleLeftBtn?.addEventListener("click", toggleLeftPanel);
  dom.closeLeftBtn?.addEventListener("click", toggleLeftPanel);
  dom.toggleRightBtn?.addEventListener("click", toggleRightPanel);
  dom.closeRightBtn?.addEventListener("click", toggleRightPanel);

  // Platform-aware keyboard shortcut hint
  const isMac = navigator.platform?.toUpperCase().indexOf("MAC") >= 0 ||
                navigator.userAgent?.toUpperCase().indexOf("MAC") >= 0;
  if (dom.shortcutHint) {
    dom.shortcutHint.textContent = isMac ? "⌘ ↵" : "Ctrl ↵";
  }

  // Health polling
  checkHealth();
  setInterval(checkHealth, 30_000);

  // Load paper button & Enter key
  dom.loadPaperBtn?.addEventListener("click", () => {
    selectPaper(dom.pdfInput?.value || "");
  });

  dom.pdfInput?.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      selectPaper(dom.pdfInput?.value || "");
    }
  });

  // Preset chips / items
  dom.presetList?.addEventListener("click", (e) => {
    const btn = e.target.closest(".doc-item-btn");
    if (btn && btn.dataset.filename) {
      selectPaper(btn.dataset.filename);
    }
  });

  // Starter query prompt buttons
  document.querySelectorAll(".starter-query-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const text = btn.querySelector("span:not(.starter-icon)")?.textContent?.trim() || "";
      if (text && dom.queryInput) {
        dom.queryInput.value = text;
        onQueryInput();
        dom.queryInput.focus();
      }
    });
  });

  // Query composer inputs
  dom.queryInput?.addEventListener("input", onQueryInput);
  dom.queryInput?.addEventListener("keydown", onKeyDown);

  // Submit button
  dom.submitBtn?.addEventListener("click", submitQuery);

  // Copy answer button
  dom.copyAnswerBtn?.addEventListener("click", copyAnswer);

  // Error dismiss button
  dom.errorDismissBtn?.addEventListener("click", dismissError);

  // Default initial UI state
  showState(dom.stateEmpty);
  updateSubmitState();

  // If preset exists, pre-select default
  const defaultPreset = document.querySelector(".doc-item-btn[data-filename]");
  if (defaultPreset && defaultPreset.dataset.filename) {
    selectPaper(defaultPreset.dataset.filename);
  }
}

document.addEventListener("DOMContentLoaded", init);
