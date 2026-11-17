/** Tracks which query may update the workspace. */

export function documentSelectionKey(documentId, pdfName) {
  if (documentId) return `id:${documentId}`;
  if (pdfName) return `pdf:${pdfName}`;
  return "";
}

export function createQuerySession() {
  let generation = 0;
  let selectionKey = "";

  return {
    selectionKey() {
      return selectionKey;
    },
    select(key) {
      const next = key || "";
      if (next === selectionKey) return false;
      selectionKey = next;
      generation += 1;
      return true;
    },
    begin() {
      generation += 1;
      return { generation, selectionKey };
    },
    invalidate() {
      generation += 1;
    },
    isCurrent(token) {
      return Boolean(token)
        && token.generation === generation
        && token.selectionKey === selectionKey;
    },
  };
}
