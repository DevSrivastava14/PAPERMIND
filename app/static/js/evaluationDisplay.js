/** User-facing labels for saved evaluation metrics. Raw records stay unchanged. */

export function formatSavedMetric(metric) {
  if (metric && typeof metric.value === "number" && Number.isFinite(metric.value)) {
    return {
      label: metric.value.toFixed(4),
      detail: metric.status === "failed" && metric.error ? String(metric.error) : "",
    };
  }
  if (metric && metric.value !== null && metric.value !== undefined && metric.status !== "failed" && metric.status !== "unavailable") {
    return { label: String(metric.value), detail: "" };
  }
  if (metric && metric.status === "failed") {
    return {
      label: "Evaluation incomplete",
      detail: metric.error ? String(metric.error) : "",
    };
  }
  return { label: "Metric unavailable", detail: "" };
}
