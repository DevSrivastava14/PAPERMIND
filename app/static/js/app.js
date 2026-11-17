import { formatSavedMetric } from "./evaluationDisplay.js";
import { createQuerySession, documentSelectionKey } from "./querySession.js";

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
const querySession = createQuerySession();
let activeQueryController = null;

const state = {
  currentPdf: /** @type {string|null} */ (null),
  currentDocumentId: /** @type {string|null} */ (null),
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
  activePaperBox:    $("active-paper-display"),
  activePaperName:   $("active-paper-name"),
  activePaperBadge:  $("active-paper-badge"),
  activePaperStatus: $("active-paper-status"),
  presetList:        $("preset-list"),
  libraryFeedback:   $("library-feedback"),
  uploadInput:       $("pdf-upload-input"),
  uploadStatus:      $("upload-status"),

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
  evaluationSelect:  $("evaluation-artifact-select"),
  evaluationLoadBtn: $("evaluation-load-btn"),
  evaluationStatus:  $("evaluation-status"),
  evaluationDescription: $("evaluation-description"),
  evaluationMeta:    $("evaluation-meta"),
  evaluationSummary: $("evaluation-summary"),
  evaluationQuestions:$("evaluation-questions"),

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
  insVisualBox:      $("ins-visual-box"),
  insImage:          $("ins-image"),
  insImageStatus:    $("ins-image-status"),
  insCaptionBlock:   $("ins-caption-block"),
  insCaption:        $("ins-caption"),
  insAssociatedBlock:$("ins-associated-block"),
  insAssociated:     $("ins-associated"),
  insDiagnostics:    $("ins-diagnostics"),
  insDiagnosticsGrid:$("ins-diagnostics-grid"),

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

function currentSelectionKey() {
  return documentSelectionKey(state.currentDocumentId, state.currentPdf);
}

function abandonDisplayedQuery() {
  if (activeQueryController) {
    activeQueryController.abort();
    activeQueryController = null;
  }
  state.isLoading = false;
  state.lastResult = null;
  state.citations = [];
  state.selectedCitationIdx = 0;
  dom.techPanel?.classList.add("hidden");
  if (dom.citationCount) dom.citationCount.textContent = "0 cited";
  if (dom.citationList) dom.citationList.innerHTML = "";
  if (dom.inspectorEmptyTip) dom.inspectorEmptyTip.classList.remove("hidden");
  if (dom.inspectorActive) dom.inspectorActive.classList.add("hidden");
  if (dom.answerText) dom.answerText.replaceChildren();
  if (dom.contextText) dom.contextText.textContent = "";
  showState(dom.stateEmpty);
  updateSubmitState();
}

function adoptDocumentSelection(documentInfo) {
  const changed = querySession.select(currentSelectionKey());
  updateActiveDocumentDisplay(documentInfo);
  if (dom.activePaperBox) dom.activePaperBox.classList.remove("hidden");
  updateDocumentSelectionVisuals();
  if (changed) abandonDisplayedQuery();
  else updateSubmitState();
}

function selectPaper(filename) {
  if (!filename || typeof filename !== "string") return;

  const cleanName = filename.trim();
  if (!cleanName.toLowerCase().endsWith(".pdf")) return;

  state.currentPdf = cleanName;
  state.currentDocumentId = null;
  adoptDocumentSelection({
    filename: cleanName,
    status: "ready",
  });
}

function formatDocumentStatus(status) {
  switch (status) {
    case "ready":
      return "Ready for questions";
    case "processing":
      return "Processing document…";
    case "failed":
      return "Processing failed";
    case "uploaded":
    default:
      return "Uploaded · not processed yet";
  }
}

function formatDocumentCardStatus(status) {
  switch (status) {
    case "ready":
      return "Ready";
    case "processing":
      return "Processing";
    case "failed":
      return "Failed";
    case "uploaded":
    default:
      return "Uploaded";
  }
}

function formatActiveBadge(status) {
  switch (status) {
    case "processing":
      return "Processing";
    case "failed":
      return "Error";
    case "uploaded":
      return "Uploaded";
    case "ready":
    default:
      return "Ready";
  }
}

function setLibraryFeedback(message, tone) {
  if (!dom.libraryFeedback) return;
  dom.libraryFeedback.textContent = message || "";
  if (tone) dom.libraryFeedback.dataset.tone = tone;
  else delete dom.libraryFeedback.dataset.tone;
}

function updateActiveDocumentDisplay(documentInfo) {
  const filename = documentInfo?.filename || "Uploaded PDF";
  const status = documentInfo?.status || "uploaded";
  if (dom.activePaperName) {
    dom.activePaperName.textContent = filename;
    dom.activePaperName.title = filename;
  }
  if (dom.activePaperStatus) {
    dom.activePaperStatus.textContent = formatDocumentStatus(status);
    dom.activePaperStatus.dataset.status = status;
  }
  if (dom.activePaperBox) dom.activePaperBox.dataset.status = status;
  if (dom.activePaperBadge) dom.activePaperBadge.textContent = formatActiveBadge(status);
}

function updateDocumentSelectionVisuals() {
  document.querySelectorAll(".doc-item-btn").forEach((btn) => {
    const isUploaded = Boolean(btn.dataset.documentId);
    const isSelected = isUploaded
      ? btn.dataset.documentId === state.currentDocumentId
      : !state.currentDocumentId &&
        (btn.dataset.filename === state.currentPdf ||
          btn.dataset.filename === state.currentPdf?.replace(/^data[\/\\]/, ""));
    btn.classList.toggle("active", isSelected);
    btn.setAttribute("aria-current", isSelected ? "true" : "false");
  });
}

function selectUploadedDocument(uploadedDocument) {
  state.currentDocumentId = uploadedDocument.document_id || null;
  state.currentPdf = uploadedDocument.filename || null;
  adoptDocumentSelection(uploadedDocument);
}

function removalErrorMessage(status) {
  if (status === 404) return "That document is no longer available.";
  if (status === 400) return "This document cannot be removed.";
  return "The document could not be removed.";
}

function clearResearchWorkspace() {
  querySession.invalidate();
  abandonDisplayedQuery();
}

function appendRemoveControl(item, uploadedDocument) {
  const actions = document.createElement("div");
  actions.className = "doc-item-actions";
  const removeButton = document.createElement("button");
  removeButton.type = "button";
  removeButton.className = "doc-remove-btn";
  removeButton.setAttribute(
    "aria-label",
    `Remove ${uploadedDocument.filename || "uploaded document"}`,
  );
  removeButton.title = "Remove document";
  removeButton.innerHTML = `
    <svg viewBox="0 0 16 16" fill="none" aria-hidden="true">
      <path d="M3.5 4.5h9" stroke="currentColor" stroke-width="1.3" stroke-linecap="round"/>
      <path d="M6.5 4.5V3.2a.7.7 0 0 1 .7-.7h1.6a.7.7 0 0 1 .7.7V4.5" stroke="currentColor" stroke-width="1.3" stroke-linecap="round"/>
      <path d="M4.5 4.5l.6 8.1a1 1 0 0 0 1 .9h3.8a1 1 0 0 0 1-.9l.6-8.1" stroke="currentColor" stroke-width="1.3" stroke-linejoin="round"/>
    </svg>`;
  removeButton.addEventListener("click", (event) => {
    event.stopPropagation();
    openRemovalConfirmation(item, uploadedDocument);
  });
  actions.appendChild(removeButton);
  item.appendChild(actions);
}

function openRemovalConfirmation(item, uploadedDocument) {
  dom.presetList?.querySelectorAll(".doc-confirm").forEach((confirm) => confirm.remove());
  const confirm = document.createElement("div");
  confirm.className = "doc-confirm";
  const prompt = document.createElement("p");
  prompt.textContent = "Remove this uploaded document and its indexed data?";
  const actions = document.createElement("div");
  actions.className = "doc-confirm-actions";
  const cancel = document.createElement("button");
  cancel.type = "button";
  cancel.className = "doc-confirm-cancel";
  cancel.textContent = "Cancel";
  const remove = document.createElement("button");
  remove.type = "button";
  remove.className = "doc-confirm-remove";
  remove.textContent = "Remove";
  cancel.addEventListener("click", (event) => {
    event.stopPropagation();
    confirm.remove();
  });
  remove.addEventListener("click", (event) => {
    event.stopPropagation();
    confirm.remove();
    removeUploadedDocument(item, uploadedDocument);
  });
  actions.append(cancel, remove);
  confirm.append(prompt, actions);
  confirm.addEventListener("click", (event) => event.stopPropagation());
  item.appendChild(confirm);
  cancel.focus();
}

async function removeUploadedDocument(item, uploadedDocument) {
  const removeButton = item.querySelector(".doc-remove-btn");
  if (removeButton) removeButton.disabled = true;
  const filename = uploadedDocument.filename || "Uploaded document";
  setLibraryFeedback(`Removing ${filename}…`);
  try {
    const response = await fetch(
      `/documents/${encodeURIComponent(uploadedDocument.document_id)}`,
      { method: "DELETE" },
    );
    if (!response.ok) throw new Error(removalErrorMessage(response.status));
    const wasActive = state.currentDocumentId === uploadedDocument.document_id;
    const refreshed = await loadUploadedDocuments();
    if (wasActive) {
      clearResearchWorkspace();
      selectPaper("sample_paper.pdf");
    }
    setLibraryFeedback(
      refreshed
        ? `Removed ${filename}.`
        : `Removed ${filename}, but the document list could not be refreshed.`,
      refreshed ? "success" : "error",
    );
  } catch (error) {
    if (removeButton) removeButton.disabled = false;
    setLibraryFeedback(error.message || "The document could not be removed.", "error");
    if (error.message === "That document is no longer available.") {
      const wasActive = state.currentDocumentId === uploadedDocument.document_id;
      await loadUploadedDocuments();
      if (wasActive) {
        clearResearchWorkspace();
        selectPaper("sample_paper.pdf");
      }
    }
  }
}

function renderUploadedDocuments(documents) {
  if (!dom.presetList) return;
  dom.presetList.querySelectorAll(".uploaded-doc-item").forEach((item) => item.remove());
  documents.forEach((uploadedDocument) => {
    const filename = uploadedDocument.filename || "Uploaded PDF";
    const status = uploadedDocument.status || "uploaded";
    const item = document.createElement("div");
    item.className = "doc-item-btn uploaded-doc-item";
    item.dataset.documentId = uploadedDocument.document_id || "";
    item.dataset.filename = filename;
    item.dataset.status = status;
    item.dataset.origin = "upload";
    item.setAttribute("role", "button");
    item.setAttribute("tabindex", "0");

    const icon = document.querySelector(".doc-item-icon")?.cloneNode(true);
    if (icon) item.appendChild(icon);

    const info = document.createElement("div");
    info.className = "doc-item-info";
    const name = document.createElement("span");
    name.className = "doc-item-name";
    name.textContent = filename;
    name.title = filename;
    const statusLine = document.createElement("span");
    statusLine.className = "doc-status-line";
    const dot = document.createElement("span");
    dot.className = "doc-status-dot";
    dot.setAttribute("aria-hidden", "true");
    const statusLabel = document.createElement("span");
    statusLabel.className = "doc-item-type";
    statusLabel.textContent = formatDocumentCardStatus(status);
    statusLine.append(dot, statusLabel);
    info.append(name, statusLine);
    item.appendChild(info);
    appendRemoveControl(item, uploadedDocument);
    item.addEventListener("click", (event) => {
      if (event.target.closest(".doc-remove-btn, .doc-confirm")) return;
      selectUploadedDocument(uploadedDocument);
    });
    item.addEventListener("keydown", (event) => {
      if (event.target !== item) return;
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        selectUploadedDocument(uploadedDocument);
      }
    });
    dom.presetList.appendChild(item);
  });
  const selectedDocument = documents.find(
    (documentInfo) => documentInfo.document_id === state.currentDocumentId,
  );
  if (selectedDocument) updateActiveDocumentDisplay(selectedDocument);
  else if (
    state.currentDocumentId &&
    !documents.some((documentInfo) => documentInfo.document_id === state.currentDocumentId)
  ) {
    clearResearchWorkspace();
    selectPaper("sample_paper.pdf");
  }
  updateDocumentSelectionVisuals();
}

async function loadUploadedDocuments() {
  try {
    const response = await fetch("/documents", { cache: "no-store" });
    if (!response.ok) throw new Error("Unable to load uploaded documents");
    const payload = await response.json();
    renderUploadedDocuments(Array.isArray(payload.documents) ? payload.documents : []);
    return true;
  } catch {
    if (dom.uploadStatus) dom.uploadStatus.textContent = "Uploaded document list is unavailable.";
    return false;
  }
}

async function uploadDocument(file) {
  if (!file) return;
  if (!file.name.toLowerCase().endsWith(".pdf")) {
    if (dom.uploadStatus) dom.uploadStatus.textContent = "Choose a PDF file.";
    return;
  }

  const formData = new FormData();
  formData.append("file", file);
  if (dom.uploadStatus) dom.uploadStatus.textContent = "Uploading and validating PDF…";
  if (dom.uploadInput) dom.uploadInput.disabled = true;
  try {
    const response = await fetch("/documents", { method: "POST", body: formData });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.detail || "Upload failed.");
    if (dom.uploadStatus) {
      dom.uploadStatus.textContent = `${payload.filename} uploaded. Processing is not started yet.`;
    }
    await loadUploadedDocuments();
  } catch (error) {
    if (dom.uploadStatus) dom.uploadStatus.textContent = error.message || "Upload failed.";
  } finally {
    if (dom.uploadInput) {
      dom.uploadInput.disabled = false;
      dom.uploadInput.value = "";
    }
  }
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
async function consumeSSE(response, onEvent) {
  if (!response.body) throw new Error("Streaming response body is unavailable.");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    const frames = buffer.split("\n\n");
    buffer = frames.pop() || "";
    for (const frame of frames) {
      const eventMatch = frame.match(/^event:\s*(.+)$/m);
      const dataMatch = frame.match(/^data:\s*(.+)$/m);
      if (!dataMatch) continue;
      await onEvent(
        eventMatch ? eventMatch[1].trim() : "message",
        JSON.parse(dataMatch[1]),
      );
    }
    if (done) break;
  }

  if (buffer.trim()) {
    const eventMatch = buffer.match(/^event:\s*(.+)$/m);
    const dataMatch = buffer.match(/^data:\s*(.+)$/m);
    if (dataMatch) {
      await onEvent(
        eventMatch ? eventMatch[1].trim() : "message",
        JSON.parse(dataMatch[1]),
      );
    }
  }
}

function waitForNextPaint() {
  return new Promise((resolve) => {
    requestAnimationFrame(resolve);
  });
}

async function submitQuery() {
  // Prevent duplicate submissions while request is already running
  if (state.isLoading) return;

  const query = (dom.queryInput?.value || "").trim();
  if (!query) {
    showValidationError(dom.queryInput, "Please enter a research question.");
    return;
  }
  if (!state.currentPdf) {
    setLibraryFeedback("Select a document before asking a question.", "error");
    return;
  }

  const rawTopK = parseInt(dom.topKInput?.value, 10);
  const safeTopK = Number.isFinite(rawTopK) && rawTopK > 0 ? rawTopK : 5;

  if (activeQueryController) activeQueryController.abort();
  const ticket = querySession.begin();
  const controller = new AbortController();
  activeQueryController = controller;
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
    if (!querySession.isCurrent(ticket)) return;
    stageIdx = (stageIdx + 1) % stages.length;
    if (dom.loadingStage) dom.loadingStage.textContent = stages[stageIdx];
  }, 2200);

  let requestId = null;
  const payload = {
    query: query,
    top_k: safeTopK,
  };
  if (state.currentDocumentId) {
    payload.document_id = state.currentDocumentId;
  } else {
    payload.pdf_path = getResolvedPdfPath(state.currentPdf);
  }

  try {
    const response = await fetch("/query/stream", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
    if (!querySession.isCurrent(ticket)) return;

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

    let completed = null;
    let streamedAnswer = "";
    await consumeSSE(response, async (event, eventPayload) => {
      if (!querySession.isCurrent(ticket)) return;
      if (event === "retrieval") {
        state.citations = Array.isArray(eventPayload.citations) ? eventPayload.citations : [];
        if (dom.contextText) dom.contextText.textContent = eventPayload.grounded_context || "";
        renderCitations(state.citations);
        renderTechDetails(eventPayload, requestId);
      } else if (event === "token") {
        streamedAnswer += typeof eventPayload.text === "string" ? eventPayload.text : "";
        if (dom.answerText) {
          if (dom.stateSuccess?.classList.contains("hidden")) {
            showState(dom.stateSuccess);
          }
          dom.answerText.textContent = streamedAnswer;
          await waitForNextPaint();
        }
      } else if (event === "error") {
        throw new Error(eventPayload.error || "Streaming generation failed.");
      } else if (event === "complete") {
        completed = {
          ...eventPayload,
          answer: eventPayload.answer || streamedAnswer,
          citations: Array.isArray(eventPayload.citations) ? eventPayload.citations : state.citations,
        };
      }
    });
    if (!querySession.isCurrent(ticket)) return;
    if (!completed) throw new Error("The answer stream ended before completion.");
    state.lastResult = completed;
    state.lastRequestId = requestId;
    state.citations = Array.isArray(completed.citations) ? completed.citations : [];
    renderResult(completed, requestId);
    if (querySession.isCurrent(ticket) && state.currentDocumentId) await loadUploadedDocuments();

  } catch (err) {
    if (!querySession.isCurrent(ticket) || err?.name === "AbortError") return;

    let safeMsg = err.message || "An unexpected error occurred during synthesis.";
    if (safeMsg.includes("Failed to fetch") || safeMsg.includes("NetworkError")) {
      safeMsg = "Unable to connect to the PaperMind server. Please verify the backend is running.";
    }
    renderError(safeMsg);

    if (requestId && dom.footerRequestId) {
      dom.footerRequestId.textContent = `trace: ${requestId}`;
    }
  } finally {
    clearInterval(stageTimer);
    if (querySession.isCurrent(ticket)) {
      state.isLoading = false;
      if (activeQueryController === controller) activeQueryController = null;
      updateSubmitState();
    }
  }
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

  const isImage = typeKey === "image";
  dom.insVisualBox?.classList.toggle("hidden", !isImage);
  dom.insCaptionBlock?.classList.toggle("hidden", !isImage || !c.caption);
  dom.insAssociatedBlock?.classList.toggle("hidden", !isImage || !c.associated_text);
  if (dom.insCaption) dom.insCaption.textContent = c.caption || "";
  if (dom.insAssociated) dom.insAssociated.textContent = c.associated_text || "";

  const diagnostics = c.diagnostics;
  if (dom.insDiagnostics && dom.insDiagnosticsGrid) {
    dom.insDiagnosticsGrid.innerHTML = "";
    const diagnosticEntries = diagnostics ? [
      ["Dense", diagnostics.dense_score != null
        ? `${Number(diagnostics.dense_score).toFixed(3)} · rank ${diagnostics.dense_rank ?? "—"}`
        : "Not retrieved"],
      ["BM25", diagnostics.bm25_score != null
        ? `${Number(diagnostics.bm25_score).toFixed(3)} · rank ${diagnostics.bm25_rank ?? "—"}`
        : "Not retrieved"],
      ["RRF", diagnostics.rrf_score != null
        ? `${Number(diagnostics.rrf_score).toFixed(5)} · rank ${diagnostics.rrf_rank ?? "—"}`
        : "Unavailable"],
      ["RRF contribution", diagnostics.rrf_dense_contribution != null || diagnostics.rrf_bm25_contribution != null
        ? `dense ${diagnostics.rrf_dense_contribution != null ? Number(diagnostics.rrf_dense_contribution).toFixed(5) : "—"} · BM25 ${diagnostics.rrf_bm25_contribution != null ? Number(diagnostics.rrf_bm25_contribution).toFixed(5) : "—"}`
        : "Unavailable"],
      ["CrossEncoder", diagnostics.cross_encoder_score != null ? Number(diagnostics.cross_encoder_score).toFixed(3) : "Unavailable"],
      ["CLIP", diagnostics.clip_score != null ? Number(diagnostics.clip_score).toFixed(3) : "Unavailable"],
      ["Final", diagnostics.final_score != null
        ? `${Number(diagnostics.final_score).toFixed(3)} · rank ${diagnostics.final_rank ?? "—"}`
        : `rank ${diagnostics.final_rank ?? "—"}`],
    ] : [];
    diagnosticEntries.forEach(([label, value]) => {
      const dt = document.createElement("dt");
      dt.textContent = label;
      const dd = document.createElement("dd");
      dd.textContent = value;
      dom.insDiagnosticsGrid.append(dt, dd);
    });
    dom.insDiagnostics.classList.toggle("hidden", diagnosticEntries.length === 0);
  }

  if (!isImage || !dom.insImage || !dom.insImageStatus) return;
  dom.insImage.removeAttribute("src");
  dom.insImage.alt = `Retrieved figure from ${c.source_filename || "source document"}`;
  dom.insImageStatus.textContent = "Image unavailable for this evidence item.";
  if (typeof c.image_data_url === "string" && c.image_data_url.startsWith("data:image/")) {
    dom.insImageStatus.textContent = "";
    dom.insImage.src = c.image_data_url;
    dom.insImage.onerror = () => {
      dom.insImage.removeAttribute("src");
      dom.insImageStatus.textContent = "Image unavailable for this evidence item.";
    };
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
  const retrieval = data.retrieval_diagnostics;
  if (retrieval?.candidate_count != null) entries.push(["Rerank Candidates", retrieval.candidate_count]);
  if (retrieval?.result_count != null) entries.push(["Final Results", retrieval.result_count]);
  if (retrieval?.reranker) entries.push(["Active Reranker", retrieval.reranker]);
  if (retrieval?.retrieval_latency_ms != null) {
    entries.push(["Retrieval Latency", `${Number(retrieval.retrieval_latency_ms).toFixed(1)} ms`]);
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

function addEvaluationText(parent, tag, text, className = "") {
  const node = document.createElement(tag);
  node.textContent = text;
  if (className) node.className = className;
  parent.appendChild(node);
  return node;
}

const EVALUATION_LABELS = {
  dense: "Dense retrieval",
  bm25: "BM25",
  hybrid_rrf: "Hybrid (RRF)",
  hybrid_reranked: "Hybrid reranked",
  faithfulness: "Faithfulness",
  answer_relevancy: "Answer relevancy",
  contextual_relevancy: "Contextual relevancy",
  contextual_precision: "Contextual precision",
  contextual_recall: "Contextual recall",
  partial_recall: "Partial recall",
  complete_recall: "Complete recall",
  mrr: "MRR",
  timestamp: "Recorded",
  date: "Recorded",
  total_questions: "Questions",
  questions: "Questions",
  top_k: "Results per question",
  configurations: "Configurations",
  evaluation_type: "Evaluation type",
  average_latency_ms: "Average latency",
  provider: "Judge provider",
  duration_seconds: "Duration (seconds)",
  all_configurations_completed: "All configurations completed",
  generation_model: "Answer model",
  evaluation_model: "Judge model",
  api_failures: "API failures",
  evaluation_rate_limit_events: "Judge rate-limit events",
  evaluation_rate_limit_waits: "Judge rate-limit waits",
  evaluation_metric_attempts: "Judge attempts",
  retrieval_only: "Retrieval only",
};

function evaluationLabel(value) {
  const key = String(value);
  if (EVALUATION_LABELS[key]) return EVALUATION_LABELS[key];
  return key.replaceAll("_", " ");
}

function appendSavedMetric(parent, label, metric) {
  const formatted = formatSavedMetric(metric);
  addEvaluationText(parent, "p", `${label}: ${formatted.label}`);
  if (!formatted.detail) return;
  const technical = document.createElement("details");
  technical.className = "evaluation-technical";
  const summary = document.createElement("summary");
  summary.textContent = "Technical detail";
  const detail = document.createElement("p");
  detail.textContent = formatted.detail;
  technical.append(summary, detail);
  parent.appendChild(technical);
}

function formatEvaluationMetadata(metadata) {
  const entries = [];
  Object.entries(metadata || {}).forEach(([key, value]) => {
    if (value === null || value === undefined || value === "") return;
    if (key === "name" || key === "reproducible_with") return;
    if (key === "pdf_path") {
      const filename = String(value).split(/[/\\]/).pop();
      if (filename) entries.push(["Document", filename]);
      return;
    }
    if (key === "groq_status" && value && typeof value === "object") {
      if (value.run === false) entries.push(["Answer-quality judge", "Not run"]);
      return;
    }
    if (key === "notes" && Array.isArray(value)) {
      const notes = value.filter((note) => typeof note === "string" && !/day\s*\d+/i.test(note));
      if (notes.length) entries.push(["Notes", notes.join(" ")]);
      return;
    }
    if (Array.isArray(value)) {
      if (!value.length) return;
      const display = value.every((item) => typeof item !== "object")
        ? value.map((item) => evaluationLabel(item)).join(", ")
        : value.map((item) => JSON.stringify(item)).join(", ");
      entries.push([evaluationLabel(key), display]);
      return;
    }
    if (key === "average_latency_ms" && value && typeof value === "object") {
      const display = Object.entries(value)
        .map(([config, latency]) => `${evaluationLabel(config)}: ${latency} ms`)
        .join(" · ");
      entries.push(["Average latency", display]);
      return;
    }
    if (typeof value === "object") return;
    const display = typeof value === "boolean"
      ? (value ? "Yes" : "No")
      : (EVALUATION_LABELS[String(value)] || String(value));
    entries.push([evaluationLabel(key), display]);
  });
  return entries;
}

function renderEvaluationArtifact(artifact) {
  const metadata = artifact.metadata || {};
  dom.evaluationMeta.replaceChildren();
  dom.evaluationSummary.replaceChildren();
  dom.evaluationQuestions.replaceChildren();

  addEvaluationText(dom.evaluationMeta, "span", artifact.title || "Saved evaluation");
  formatEvaluationMetadata(metadata).forEach(([label, value]) => {
    addEvaluationText(dom.evaluationMeta, "span", `${label}: ${value}`);
  });
  if (dom.evaluationDescription) {
    dom.evaluationDescription.textContent = artifact.description || "";
    dom.evaluationDescription.classList.toggle("hidden", !artifact.description);
  }

  Object.entries(artifact.summary || {}).forEach(([config, metrics]) => {
    Object.entries(metrics).forEach(([name, metric]) => {
      const formatted = formatSavedMetric(metric);
      addEvaluationText(
        dom.evaluationSummary,
        "dt",
        `${evaluationLabel(config)} · ${evaluationLabel(name)}`,
      );
      const value = addEvaluationText(dom.evaluationSummary, "dd", formatted.label);
      if (formatted.detail) value.title = formatted.detail;
    });
  });

  (artifact.questions || []).forEach((question) => {
    const details = document.createElement("details");
    details.className = "evaluation-question";
    const summary = document.createElement("summary");
    summary.textContent = `${question.id || "Question"}${question.query ? ` · ${question.query}` : ""}`;
    details.appendChild(summary);
    if (question.status) addEvaluationText(details, "p", `Status: ${question.status}`);
    Object.entries(question.configurations || {}).forEach(([config, value]) => {
      if (value && typeof value === "object" && "status" in value) {
        appendSavedMetric(details, evaluationLabel(config), value);
        return;
      }
      const metricText = Array.isArray(value)
        ? `${value.length} retrieved items`
        : typeof value === "object"
          ? Object.entries(value).map(([key, item]) => {
            if (key === "retrieved_evidence" && Array.isArray(item)) return `retrieved items: ${item.length}`;
            if (item === null || item === undefined) return `${evaluationLabel(key)}: Metric unavailable`;
            return `${evaluationLabel(key)}: ${typeof item === "object" ? JSON.stringify(item) : item}`;
          }).join(" · ")
          : String(value);
      addEvaluationText(details, "p", `${evaluationLabel(config)}: ${metricText}`);
    });
    dom.evaluationQuestions.appendChild(details);
  });

  dom.evaluationMeta.classList.remove("hidden");
  dom.evaluationSummary.classList.toggle("hidden", dom.evaluationSummary.childElementCount === 0);
  dom.evaluationQuestions.classList.toggle("hidden", dom.evaluationQuestions.childElementCount === 0);
  dom.evaluationStatus.textContent = artifact.incomplete
    ? "Saved evaluation loaded. Some saved results are incomplete."
    : "Saved evaluation loaded.";
}

async function loadEvaluationArtifacts() {
  if (!dom.evaluationSelect) return;
  try {
    const response = await fetch("/evaluation/results");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const payload = await response.json();
    payload.artifacts?.forEach((artifact) => {
      if (!artifact.available) return;
      const option = document.createElement("option");
      option.value = artifact.id;
      option.textContent = artifact.title;
      option.title = artifact.description || artifact.title;
      dom.evaluationSelect.appendChild(option);
    });
  } catch {
    dom.evaluationStatus.textContent = "Saved evaluations are unavailable.";
  }
}

async function loadSelectedEvaluation() {
  const artifactId = dom.evaluationSelect?.value;
  if (!artifactId) {
    dom.evaluationStatus.textContent = "Select a saved evaluation to review.";
    return;
  }
  dom.evaluationStatus.textContent = "Loading saved evaluation…";
  try {
    const response = await fetch(`/evaluation/results?artifact_id=${encodeURIComponent(artifactId)}`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const payload = await response.json();
    renderEvaluationArtifact(payload.artifact);
  } catch {
    dom.evaluationStatus.textContent = "Unable to load the selected evaluation.";
  }
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

  dom.presetList?.addEventListener("click", (e) => {
    if (e.target.closest(".doc-remove-btn, .doc-confirm, .uploaded-doc-item")) return;
    const btn = e.target.closest(".doc-item-btn");
    if (btn?.dataset.origin === "sample" && btn.dataset.filename) {
      selectPaper(btn.dataset.filename);
    }
  });

  dom.uploadInput?.addEventListener("change", () => {
    uploadDocument(dom.uploadInput.files?.[0]);
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
  dom.evaluationLoadBtn?.addEventListener("click", loadSelectedEvaluation);
  loadEvaluationArtifacts();
  loadUploadedDocuments();

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
