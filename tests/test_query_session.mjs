import assert from "node:assert/strict";
import test from "node:test";

import { createQuerySession, documentSelectionKey } from "../app/static/js/querySession.js";
import { formatSavedMetric } from "../app/static/js/evaluationDisplay.js";

test("a normal query completion is accepted", () => {
  const session = createQuerySession();
  assert.equal(session.select(documentSelectionKey(null, "sample_paper.pdf")), true);
  const token = session.begin();
  assert.equal(session.isCurrent(token), true);
});

test("document removal invalidates an active query and its late response", () => {
  const session = createQuerySession();
  session.select(documentSelectionKey("doc-upload", "paper.pdf"));
  const active = session.begin();

  assert.equal(session.select(documentSelectionKey(null, "sample_paper.pdf")), true);
  assert.equal(session.isCurrent(active), false);
  session.invalidate();
  assert.equal(session.isCurrent(active), false);
});

test("switching documents invalidates the query started for the previous document", () => {
  const session = createQuerySession();
  session.select(documentSelectionKey("doc-one", "one.pdf"));
  const first = session.begin();

  assert.equal(session.select(documentSelectionKey("doc-two", "two.pdf")), true);
  assert.equal(session.isCurrent(first), false);

  const second = session.begin();
  assert.equal(session.isCurrent(second), true);
  assert.equal(session.isCurrent(first), false);
});

test("an older request cannot overwrite a newer request", () => {
  const session = createQuerySession();
  session.select(documentSelectionKey(null, "sample_paper.pdf"));
  const older = session.begin();
  const newer = session.begin();

  assert.equal(session.isCurrent(older), false);
  assert.equal(session.isCurrent(newer), true);
});

test("selecting the same document does not invalidate its query", () => {
  const session = createQuerySession();
  const key = documentSelectionKey(null, "sample_paper.pdf");
  session.select(key);
  const token = session.begin();
  assert.equal(session.select(key), false);
  assert.equal(session.isCurrent(token), true);
});

test("saved metric values stay visible and judge failures stay descriptive", () => {
  assert.deepEqual(formatSavedMetric({ value: 1, status: "available" }), {
    label: "1.0000",
    detail: "",
  });
  assert.deepEqual(formatSavedMetric({ value: 0, status: "available" }), {
    label: "0.0000",
    detail: "",
  });
  assert.deepEqual(
    formatSavedMetric({
      value: null,
      status: "failed",
      error: "RetryError: RetryError[<Future at 0x1 state=finished raised RateLimitError>]",
    }),
    {
      label: "Evaluation incomplete",
      detail: "RetryError: RetryError[<Future at 0x1 state=finished raised RateLimitError>]",
    },
  );
  assert.deepEqual(formatSavedMetric({ value: null, status: "unavailable" }), {
    label: "Metric unavailable",
    detail: "",
  });
  assert.deepEqual(formatSavedMetric({ value: 0.3529, status: "failed", error: "below threshold" }), {
    label: "0.3529",
    detail: "below threshold",
  });
});
