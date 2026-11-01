/**
 * PaperMind — Three-Panel Research Workspace Script
 *
 * Connects to live FastAPI backend:
 * - GET /health
 * - POST /query
 *
 * Implements:
 * 1. Live health polling & graceful offline degradation
 * 2. Schema-compliant QueryRequest & QueryResponse handling
 * 3. Grounded answer rendering with clickable inline [1], [2] citation markers
 * 4. Interactive Evidence Inspector synchronization
 * 5. Robust error handling (422, 500, network loss) with path redaction
 * 6. Duplicate submission prevention & X-Request-ID trace tracking
 *
 * Zero external dependencies. Zero fabricated metrics.
 */

// ── State ──────────────────────────────────────────────────────────────────
const state = {
  currentPdf: /** @type {string|null} */ (null),
  isLoading: false,
  lastRequestId: /** @type {string|null} */ (null),
  lastResult: /** @type {object|null} */ (null),
  citations: /** @type {Array<object>} */ ([]),
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

function toggleRightPanel(forceOpen = false) {
  const isMobile = window.innerWidth <= 900;
  if (isMobile) {
    if (forceOpen) {
      dom.appLayout?.classList.add("right-open");
    } else {
      dom.appLayout?.classList.toggle("right-open");
    }
  } else {
    if (forceOpen) {
      dom.appLayout?.classList.remove("right-collapsed");
    } else {
      dom.appLayout?.classList.toggle("right-collapsed");
    }
  }
}

// ── Health Check (GET /health) ─────────────────────────────────────────────
async function checkHealth() {
  try {
    const res = await fetch("/health", {
      method: "GET",
      cache: "no-store",
    });

    if (!res.ok) {
      setStatus("error", `API Degraded (${res.status})`);
      return;
    }

    const data = await res.json();
    if (data && data.status === "healthy") {
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
function getResolvedPdfPath(rawPath) {
  if (!rawPath || typeof rawPath !== "string") return "";
  const normalized = rawPath.trim().replace(/\\/g, "/");
  if (normalized.startsWith("data/")) {
    return normalized;
  }
  return `data/${normalized}`;
}

function selectPaper(filename) {
  if (!filename || typeof filename !== "string") return;

  const cleanName = filename.trim();
  if (!cleanName.toLowerCase().endsWith(".pdf")) {
    showValidationError(dom.pdfInput, "Filename must end with .pdf");
    return;
  }

  state.currentPdf = cleanName;

  if (dom.pdfInput) dom.pdfInput.value = cleanName;
  if (dom.activePaperName) dom.activePaperName.textContent = cleanName;
  if (dom.activePaperBox) dom.activePaperBox.classList.remove("hidden");

  // Highlight active item in document list
  document.querySelectorAll(".doc-item-btn").forEach((btn) => {
    const btnFile = btn.getAttribute("data-filename");
    const isMatch = btnFile === cleanName || btnFile === cleanName.replace(/^data[\/\\]/, "");
    btn.classList.toggle("active", isMatch);
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

function sanitizeErrorMessage(msg) {
  if (!msg || typeof msg !== "string") {
    return "An error occurred during query execution.";
  }
  // Redact Windows paths (e.g. C:\Users\... or D:\project\...)
  let clean = msg.replace(/[A-Za-z]:\\[^ \t\n\r"']+/g, "[safe_path]");
  // Redact Unix root paths
  clean = clean.replace(/\/(?:Users|home|tmp|var|etc)\/[^ \t\n\r"']+/g, "[safe_path]");
  // Redact potential API keys
  clean = clean.replace(/gsk_[A-Za-z0-9_\-]+/g, "[redacted_key]");
  clean = clean.replace(/sk-[A-Za-z0-9_\-]+/g, "[redacted_key]");
  // Remove traceback headers if present
  if (clean.includes("Traceback (most recent call last):")) {
    const parts = clean.split("Traceback (most recent call last):");
    clean = parts[0].trim() || "Internal graph execution error occurred.";
  }
  return clean;
}

// ── Composer Controls ──────────────────────────────────────────────────────
function updateSubmitState() {
  const hasQuery = (dom.queryInput?.value || "").trim().length > 0;
  const hasPaper = state.currentPdf !== null && state.currentPdf.trim().length > 0;
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

// ── Query Execution (POST /query) ──────────────────────────────────────────
async function submitQuery() {
  // Prevent duplicate submissions while request is already running
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
  }, 2200);

  let requestId = null;
  const payloadPdfPath = getResolvedPdfPath(state.currentPdf);

  try {
    const response = await fetch("/query", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        pdf_path: payloadPdfPath,
        query: query,
        top_k: safeTopK,
      }),
    });

    // Capture X-Request-ID response header
    requestId = response.headers.get("X-Request-ID") || response.headers.get("x-request-id") || null;

    if (!response.ok) {
      let detail = `Server responded with HTTP ${response.status}`;
      try {
        const errJson = await response.json();
        if (errJson.detail) {
          if (typeof errJson.detail === "string") {
            detail = errJson.detail;
          } else if (Array.isArray(errJson.detail)) {
            // FastAPI validation error array
            detail = errJson.detail.map((e) => e.msg || e.message || "Invalid input").join("; ");
          }
        }
      } catch {
        // Fall back to HTTP status message
      }
      throw new Error(sanitizeErrorMessage(detail));
    }

    const data = await response.json();
    state.lastResult = data;
    state.lastRequestId = requestId;
    state.citations = Array.isArray(data.citations) ? data.citations : [];
    renderResult(data, requestId);

  } catch (err) {
    clearInterval(stageTimer);
    state.isLoading = false;
    updateSubmitState();

    let safeMsg = err.message || "An unexpected error occurred during synthesis.";
    if (safeMsg.includes("Failed to fetch") || safeMsg.includes("NetworkError")) {
      safeMsg = "Unable to connect to the PaperMind server. Please verify the backend is running.";
    }
    renderError(safeMsg);

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

  // Case A: Insufficient evidence in paper (Grounded Refusal)
  if (data.insufficient_evidence === true) {
    if (dom.insufficientMsg) {
      // Display actual grounded refusal reason from model without fabrication
      dom.insufficientMsg.textContent =
        data.answer ||
        "The paper does not contain enough grounded evidence to answer this question with high confidence.";
    }
    showState(dom.stateInsufficient);

    // Preserve retrieved candidate evidence so user can inspect evaluated sources
    if (state.citations.length > 0) {
      renderCitations(state.citations);
    } else {
      if (dom.citationCount) dom.citationCount.textContent = "0 cited";
      if (dom.citationList) dom.citationList.innerHTML = "";
      if (dom.inspectorEmptyTip) dom.inspectorEmptyTip.classList.remove("hidden");
      if (dom.inspectorActive) dom.inspectorActive.classList.add("hidden");
    }

    renderTechDetails(data, requestId);
    return;
  }

  // Case B: Grounded synthesis answer
  showState(dom.stateSuccess);
  renderCitations(state.citations);
  renderGroundedAnswer(data.answer);

  // Initialize two-way synchronization for the default active citation
  if (state.citations.length > 0) {
    selectCitation(0, false);
  }

  if (dom.contextText) {
    dom.contextText.textContent = data.grounded_context || "(No raw context excerpt)";
  }

  renderTechDetails(data, requestId);
}

// ── Grounded Answer & Safe Markdown Parsing ───────────────────────────────
function findCitationIndex(numStr) {
  if (!state.citations || state.citations.length === 0) return -1;
  const clean = String(numStr).trim();
  const targetLabel = `[${clean}]`;
  const targetNum = parseInt(clean, 10);

  return state.citations.findIndex((c, idx) => {
    if (c.citation_label === targetLabel || c.citation_label === clean) return true;
    const labelNum = (c.citation_label || "").replace(/\D/g, "");
    if (labelNum && parseInt(labelNum, 10) === targetNum) return true;
    if (!isNaN(targetNum) && (idx + 1) === targetNum) return true;
    return false;
  });
}

function selectCitation(targetIdx, shouldScroll = true) {
  if (!state.citations || state.citations.length === 0) return;
  if (targetIdx < 0 || targetIdx >= state.citations.length) return;

  state.selectedCitationIdx = targetIdx;
  const c = state.citations[targetIdx];
  const targetNum = String(targetIdx + 1);
  const labelNum = (c.citation_label || "").replace(/\D/g, "");

  // 1. Update Right Inspector list card selection
  const cards = document.querySelectorAll(".citation-card");
  cards.forEach((card, idx) => {
    const isMatch = idx === targetIdx;
    card.classList.toggle("selected", isMatch);
    if (isMatch && shouldScroll) {
      card.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  });

  // 2. Update Inspector Detail View
  inspectCitation(c);

  // 3. Update all inline citation buttons in Answer prose (Two-Way Sync)
  document.querySelectorAll(".inline-citation:not(.inline-citation--unresolved)").forEach((el) => {
    const citeNum = el.getAttribute("data-cite-num");
    const matches = citeNum === targetNum || (labelNum && citeNum === labelNum);
    el.classList.toggle("active", Boolean(matches));
  });
}

function renderSafeInlines(text, parentEl) {
  if (!text) return;

  // Tokenize bold, code, and citations safely
  const tokenRegex = /(`[^`]+`|\*\*[^*]+\*\*|\[\d+(?:\s*,\s*\d+)*\])/g;
  let lastIdx = 0;
  let match;

  while ((match = tokenRegex.exec(text)) !== null) {
    // Append plain text preceding match
    if (match.index > lastIdx) {
      parentEl.appendChild(document.createTextNode(text.slice(lastIdx, match.index)));
    }

    const token = match[0];

    // Code token: `...`
    if (token.startsWith("`") && token.endsWith("`") && token.length >= 2) {
      const code = document.createElement("code");
      code.textContent = token.slice(1, -1);
      parentEl.appendChild(code);
    }
    // Bold token: **...**
    else if (token.startsWith("**") && token.endsWith("**") && token.length >= 4) {
      const strong = document.createElement("strong");
      renderSafeInlines(token.slice(2, -2), strong);
      parentEl.appendChild(strong);
    }
    // Citation token: [1] or [1, 2]
    else if (token.startsWith("[") && token.endsWith("]")) {
      const inner = token.slice(1, -1);
      const nums = inner.split(",").map((s) => s.trim()).filter(Boolean);

      nums.forEach((numStr) => {
        const targetIdx = findCitationIndex(numStr);

        if (targetIdx !== -1) {
          const btn = document.createElement("button");
          btn.type = "button";
          btn.className = "inline-citation";
          btn.setAttribute("data-cite-num", numStr);
          btn.setAttribute("aria-label", `View evidence citation [${numStr}]`);
          btn.textContent = `[${numStr}]`;

          if (targetIdx === state.selectedCitationIdx) {
            btn.classList.add("active");
          }

          btn.addEventListener("click", (e) => {
            e.preventDefault();
            toggleRightPanel(true);
            selectCitation(targetIdx, true);
          });

          parentEl.appendChild(btn);
        } else {
          // Unresolved citation: visibly unresolved, non-interactive marker with tooltip
          const span = document.createElement("span");
          span.className = "inline-citation inline-citation--unresolved";
          span.setAttribute("data-cite-num", numStr);
          span.setAttribute("role", "note");
          span.setAttribute("tabindex", "0");
          span.setAttribute("title", `Citation [${numStr}] not found in retrieved evidence`);
          span.setAttribute("aria-label", `Citation [${numStr}] (not found in retrieved evidence)`);
          span.textContent = `[${numStr}]`;
          parentEl.appendChild(span);
        }
      });
    }

    lastIdx = match.index + match[0].length;
  }

  // Append remaining plain text
  if (lastIdx < text.length) {
    parentEl.appendChild(document.createTextNode(text.slice(lastIdx)));
  }
}

function renderGroundedAnswer(rawAnswer) {
  if (!dom.answerText) return;

  const text = (rawAnswer || "").trim();
  if (!text) {
    dom.answerText.textContent = "(No answer generated by model)";
    return;
  }

  dom.answerText.innerHTML = "";

  const lines = text.split(/\r?\n/);
  let currentList = null;

  function flushList() {
    if (currentList) {
      dom.answerText.appendChild(currentList.element);
      currentList = null;
    }
  }

  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    const trimmed = line.trim();

    if (!trimmed) {
      flushList();
      i++;
      continue;
    }

    // 1. Heading check: e.g. ### Heading or ## Heading
    const headingMatch = trimmed.match(/^(#{1,6})\s+(.+)$/);
    if (headingMatch) {
      flushList();
      const level = Math.min(4, Math.max(2, headingMatch[1].length));
      const h = document.createElement(`h${level}`);
      renderSafeInlines(headingMatch[2], h);
      dom.answerText.appendChild(h);
      i++;
      continue;
    }

    // 2. Unordered list item: - item or * item
    const ulMatch = trimmed.match(/^[\*\-]\s+(.+)$/);
    if (ulMatch) {
      if (!currentList || currentList.type !== "ul") {
        flushList();
        currentList = { type: "ul", element: document.createElement("ul") };
      }
      const li = document.createElement("li");
      renderSafeInlines(ulMatch[1], li);
      currentList.element.appendChild(li);
      i++;
      continue;
    }

    // 3. Ordered list item: 1. item
    const olMatch = trimmed.match(/^(\d+)\.\s+(.+)$/);
    if (olMatch) {
      if (!currentList || currentList.type !== "ol") {
        flushList();
        currentList = { type: "ol", element: document.createElement("ol") };
      }
      const li = document.createElement("li");
      renderSafeInlines(olMatch[2], li);
      currentList.element.appendChild(li);
      i++;
      continue;
    }

    // 4. Standard paragraph: accumulate until empty line, heading, or list
    flushList();
    const paragraphLines = [trimmed];
    i++;
    while (i < lines.length) {
      const nextLine = lines[i].trim();
      if (!nextLine || nextLine.startsWith("#") || nextLine.match(/^[\*\-]\s+/) || nextLine.match(/^\d+\.\s+/)) {
        break;
      }
      paragraphLines.push(nextLine);
      i++;
    }

    const p = document.createElement("p");
    renderSafeInlines(paragraphLines.join(" "), p);
    dom.answerText.appendChild(p);
  }

  flushList();
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

    // Click handler to select and inspect (two-way synchronization)
    card.addEventListener("click", () => {
      selectCitation(idx, false);
    });

    dom.citationList.appendChild(card);
  });

  // Select first citation by default with two-way synchronization
  selectCitation(0, false);
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
  if (state.currentPdf)       entries.push(["Document Context", getResolvedPdfPath(state.currentPdf)]);
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
    dom.errorMessage.textContent = sanitizeErrorMessage(message);
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
    if (!dom.submitBtn?.disabled && !state.isLoading) {
      submitQuery();
    }
  }
}

// ── Copy Answer to Clipboard ───────────────────────────────────────────────
async function copyAnswer() {
  const text = dom.answerText?.innerText || dom.answerText?.textContent || "";
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
    // Clipboard permission denied or fallback
  }
}

// ── Initialization ─────────────────────────────────────────────────────────
function init() {
  // Panel toggle event listeners
  dom.toggleLeftBtn?.addEventListener("click", toggleLeftPanel);
  dom.closeLeftBtn?.addEventListener("click", toggleLeftPanel);
  dom.toggleRightBtn?.addEventListener("click", () => toggleRightPanel());
  dom.closeRightBtn?.addEventListener("click", () => toggleRightPanel());

  // Platform-aware keyboard shortcut hint
  const isMac = navigator.platform?.toUpperCase().indexOf("MAC") >= 0 ||
                navigator.userAgent?.toUpperCase().indexOf("MAC") >= 0;
  if (dom.shortcutHint) {
    dom.shortcutHint.textContent = isMac ? "⌘ ↵" : "Ctrl ↵";
  }

  // Live health polling
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

  // If preset exists, pre-select default candidate
  const defaultPreset = document.querySelector(".doc-item-btn[data-filename]");
  if (defaultPreset && defaultPreset.dataset.filename) {
    selectPaper(defaultPreset.dataset.filename);
  }
}

document.addEventListener("DOMContentLoaded", init);
