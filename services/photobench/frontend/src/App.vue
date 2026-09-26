<script setup>
import { computed, onMounted, onUnmounted, reactive, ref, watch } from "vue";

const graphQualityCache = new Map();
const graphQualityRequests = new Map();
const graphQualityRetryAt = new Map();

const EXECUTION_PHASES = [
  { key: "model_deploy", label: "模型部署" },
  { key: "scope_setup", label: "创建相册" },
  { key: "identity_seed", label: "预置身份" },
  { key: "photo_import", label: "照片/视频导入" },
  { key: "pipeline_processing", label: "流水线处理" },
  { key: "qa_eval", label: "QA 评测" },
];

const config = ref(null);
const manifests = ref([]);
const profiles = ref([]);
const vllmTargets = ref({});
const runs = ref([]);
const activeRunId = ref(null);
const activeRun = ref(null);
const graphQuality = ref(null);
const graphQualityLoading = ref(false);
const qaPage = ref({ items: [], page: 1, page_size: 20, total: 0, pages: 1 });
const qaDetails = reactive({});
const openQaItems = reactive(new Set());
const loadingQaItems = reactive(new Set());
const qaPageSize = ref(20);
const qaFilters = reactive({ search: "", score: "", task_type: "", tag: "", angle: "", difficulty: "", answerability: "", agent_status: "", primary: "" });
const reviewDrafts = reactive({});
const selectedAlbum = ref("album3-max-video10");
const albumCountLabel = (manifest) => {
  const videos = Number(manifest?.video_count || 0);
  const base = `${manifest.face_count}人 / ${manifest.photo_count}图`;
  return videos ? `${base} / ${videos}视频` : base;
};
const selectedQa = ref("mixed-image-video-487q");
const selectedModels = reactive(new Set());
const sentrixUrl = ref("");
const judgeUrl = ref("");
const judgeModel = ref("");
const judgeApiKey = ref("");
const judgeApiKeyDirty = ref(false);
const vllmTargetId = ref("");
const vllmManagerUrl = ref("");
const modelEndpoint = ref("");
const modelEndpointUserEdited = ref(false);
const endpointModels = ref([]);
const selectedEndpointModel = ref("");
const currentModelInfo = ref(null);
const currentModelLoading = ref(false);
const currentModelError = ref("");
const currentModelPopoverOpen = ref(false);
const modelTestState = ref("idle");
const modelTestMessage = ref("");
const connectionConfigState = ref("idle");
const connectionConfigMessage = ref("");
const rejudgePrompt = ref("");
const judgePromptKinds = ref([]);
const activePromptKind = ref("answer_quality");
const promptDrafts = reactive({ answer_quality: "", task_decision: "", evidence: "" });
const judgeProviderId = ref("");
watch(judgeProviderId, (newId) => {
  const provider = (config.value?.judge_providers || []).find((p) => p.id === newId);
  if (provider?.url) {
    judgeUrl.value = provider.url;
    judgeModel.value = provider.model || judgeModel.value;
    markConnectionConfigDirty();
  }
});

const suiteRunning = ref(false);
const rejudgeSubmitting = ref(false);
const reviewSaving = ref(false);
const loading = ref(true);
const activeView = ref("runs");
const qaBrowserAlbum = ref("album3-max-video10");
const qaBrowserSet = ref("mixed-image-video-487q");
const qaBrowserItems = ref([]);
const qaBrowserSearch = ref("");
const qaBrowserTag = ref("");
const qaBrowserLoading = ref(false);
const qaBrowserError = ref("");
const qaBrowserMediaResolution = ref(null);
const error = ref("");
const lightbox = ref(null);
const judgeModal = ref(null);
let pollTimer = null;
let destroyed = false;

const api = async (path, options = {}) => {
  const response = await fetch(path, { headers: { "content-type": "application/json", ...(options.headers || {}) }, ...options });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
};
const post = (path, body) => api(path, { method: "POST", body: JSON.stringify(body) });
const esc = (value) => String(value ?? "");
// Display the endpoint's exact served ID.  The value is also sent unchanged
// in API requests, so aliases such as qwen3-vl-4b and qwen3-vl-4b-instruct
// remain distinguishable.
const modelDisplayName = (value) => String(value || "");
const modelName = (run) => {
  // 新记录保存了实际发送到 OpenAI-compatible endpoint 的 served alias。
  // 优先展示它，不把 qwen3-vl-4b 和 qwen3-vl-4b-instruct 混成一个名称。
  const actualName = run?.served_model_name
    || run?.current_model_snapshot?.served_model_name;
  if (actualName) return String(actualName);
  return run?.model_profile || run?.model_name || run?.profile || "unknown";
};
const albumName = (run) => run?.scope_name || run?.album_id || run?.qa_name || "album";
const qaName = (run) => run?.qa_set || run?.qa_name || "qa";
const fmtDate = (value) => value ? new Date(value).toLocaleString("zh-CN", { month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit" }) : "-";
const duration = (run) => {
  if (!run?.started_at) return "-";
  const end = run.finished_at ? new Date(run.finished_at) : new Date();
  const seconds = Math.max(0, Math.round((end - new Date(run.started_at)) / 1000));
  return seconds < 60 ? `${seconds}s` : `${Math.floor(seconds / 60)}m${seconds % 60}s`;
};
const fmtMs = (value) => value == null ? "-" : value >= 1000 ? `${(value / 1000).toFixed(1)}s` : `${Number(value).toFixed(0)}ms`;
const fmtPct = (value) => value == null ? "-" : `${(Number(value) * 100).toFixed(1)}%`;
const fmtSignedPct = (value) => {
  if (value == null) return "-";
  const number = Number(value);
  return `${number > 0 ? "+" : ""}${(number * 100).toFixed(1)}%`;
};
const fmtTokens = (value) => value == null || !Number.isFinite(Number(value)) ? "-" : `${Math.round(Number(value)).toLocaleString("en-US")} token`;
const scoreClass = (score) => score === 2 ? "score-2" : score === 1 ? "score-1" : score === 0 ? "score-0" : "score-none";
const statusLabel = (status) => ({ done: "完成", running: "进行中", pending: "等待", cancelling: "停止中", failed: "失败", completed: "完成", completed_with_errors: "完成但有错误", interrupted: "中断", cancelled: "已取消", partial: "部分完成", stalled: "已停滞", not_run: "未执行", skipped: "不适用" }[status] || status || "等待");
function setModelSelected(modelId, checked) {
  if (checked && modelId === "__current__") {
    selectedModels.clear();
    selectedModels.add(modelId);
  } else if (checked) {
    selectedModels.delete("__current__");
    selectedModels.add(modelId);
  } else {
    selectedModels.delete(modelId);
  }
}

const qaOptions = computed(() => manifests.value.find((m) => m.album_id === selectedAlbum.value)?.qa_sets || ["compact-10q"]);
const hasRunning = computed(() => runs.value.some((run) => ["running", "pending", "cancelling"].includes(run.status) || run.rejudge?.status === "running"));
const activeRejudge = computed(() => activeRun.value?.rejudge || null);
const visibleQaItems = computed(() => {
  const items = qaPage.value?.items || [];
  const task = activeRejudge.value;
  if (!task || task.status === "completed") return items;
  if (!["running", "failed", "interrupted"].includes(task.status)) return items;
  return items.filter((item) => {
    const judge = item.judge || {};
    return judge.rejudge_id === task.rejudge_id && ["completed", "failed"].includes(judge.status);
  });
});
const rejudgePercent = computed(() => {
  const task = activeRejudge.value;
  return task?.total ? Math.round(((task.completed || 0) / task.total) * 100) : 0;
});
const canRejudge = computed(() => Boolean(
  (activeRun.value?.item_count || activeRun.value?.summary?.completed)
  && !["running", "pending"].includes(activeRun.value.status)
  && activeRejudge.value?.status !== "running"
  && rejudgePrompt.value.trim()
));
function averageMetric(values) {
  const valid = values.map(Number).filter(Number.isFinite);
  return valid.length ? valid.reduce((sum, value) => sum + value, 0) / valid.length : null;
}
function nearestRankPercentile(values, percentile) {
  const valid = values.map(Number).filter(Number.isFinite).sort((a, b) => a - b);
  if (!valid.length) return null;
  return valid[Math.max(0, Math.min(valid.length - 1, Math.ceil(valid.length * percentile) - 1))];
}
function inferMediaType(value, explicit = "") {
  if (["image", "video"].includes(String(explicit || "").toLowerCase())) return String(explicit).toLowerCase();
  const text = String(value || "").split(/[?#]/)[0];
  return /\.(?:mp4|mov|m4v|webm|mkv|avi)$/i.test(text) || /^video-\d+$/i.test(text.split("/").pop() || "") ? "video" : "image";
}
function mediaKey(ref) {
  const mediaType = inferMediaType(ref?.media_id || ref?.file_name || ref?.image_id, ref?.media_type);
  const name = String(ref?.media_id || ref?.file_name || ref?.image_id || "").split(/[/?#]/).filter(Boolean).pop() || "";
  return `${mediaType}:${mediaType === "video" ? name.replace(/\.[^.]+$/, "") : name}`.toLocaleLowerCase();
}
function mediaRefs(item, prefix = "retrieval") {
  const typed = item?.[`${prefix}_media_refs`];
  if (Array.isArray(typed)) return typed.map((ref) => ({ ...ref, media_type: inferMediaType(ref?.media_id, ref?.media_type), media_id: String(ref?.media_id || "") })).filter((ref) => ref.media_id);
  const refs = [
    ...((item?.[`${prefix}_image_ids`] || []).map((media_id) => ({ media_type: inferMediaType(media_id), media_id }))),
    ...((item?.[`${prefix}_video_ids`] || []).map((media_id) => ({ media_type: "video", media_id }))),
  ];
  return [...new Map(refs.map((ref) => [mediaKey(ref), ref])).values()];
}
function microMetrics(items, prefix) {
  const rows = items.map((item) => item?.[`${prefix}_retrieval_counts`]).filter((row) => row && Number(row.gt) > 0);
  if (!rows.length) return { precision: null, recall: null, f1: null, metricCount: 0 };
  const gt = rows.reduce((sum, row) => sum + Number(row.gt || 0), 0);
  const predicted = rows.reduce((sum, row) => sum + Number(row.predicted || 0), 0);
  const matched = rows.reduce((sum, row) => sum + Number(row.matched || 0), 0);
  const precision = predicted ? matched / predicted : (gt ? 0 : null);
  const recall = gt ? matched / gt : null;
  return { precision, recall, f1: precision != null && recall != null && precision + recall ? 2 * precision * recall / (precision + recall) : (gt ? 0 : null), metricCount: rows.length };
}
function macroMetrics(items, prefix) {
  const rows = items.map((item) => item?.[`${prefix}_retrieval_counts`]).filter((row) => row && Number(row.gt) > 0);
  const values = rows.map((row) => {
    const gt = Number(row.gt || 0);
    const predicted = Number(row.predicted || 0);
    const matched = Number(row.matched || 0);
    const precision = predicted ? matched / predicted : 0;
    const recall = matched / gt;
    const f1 = precision + recall ? 2 * precision * recall / (precision + recall) : 0;
    return { precision, recall, f1 };
  });
  return { precision: averageMetric(values.map((row) => row.precision)), recall: averageMetric(values.map((row) => row.recall)), f1: averageMetric(values.map((row) => row.f1)), metricCount: values.length };
}
function effectiveRunSummary(run) {
  const saved = run?.summary || {};
  const items = run?.items || [];
  const metricItems = items.filter((item) => String(item?.answerability || "").toLowerCase() !== "unanswerable");
  const judged = items.filter((item) => item.judge?.score != null && item.judge?.consistency_status !== "inconsistent");
  const recalls = metricItems.map((item) => item.retrieval_recall).filter((value) => Number.isFinite(Number(value)));
  const scores = judged.map((item) => Number(item.judge.score)).filter(Number.isFinite);
  const evidenceScores = items.map((item) => item.evidence_judge?.score).filter((score) => [0, 1, 2].includes(score));
  const typedRetrieval = items.some((item) => Object.prototype.hasOwnProperty.call(item, "retrieval_media_refs"));
  const retrievalItems = metricItems.filter((item) => mediaRefs(item).length);
  const retrievalTp = retrievalItems.reduce((sum, item) => sum + (item.matched_file_names || []).length, 0);
  const retrievalPredicted = retrievalItems.reduce((sum, item) => sum + (item.retrieved_file_names || []).length, 0);
  const retrievalGt = retrievalItems.reduce((sum, item) => sum + (item.retrieval_image_ids || []).length, 0);
  const retrievalPrecision = retrievalPredicted ? retrievalTp / retrievalPredicted : (retrievalGt ? 0 : null);
  const retrievalRecall = retrievalGt ? retrievalTp / retrievalGt : null;
  const retrievalF1 = retrievalPrecision != null && retrievalRecall != null && retrievalPrecision + retrievalRecall
    ? 2 * retrievalPrecision * retrievalRecall / (retrievalPrecision + retrievalRecall) : (retrievalGt ? 0 : null);
  const mediaMicro = microMetrics(metricItems, "media");
  const imageMicro = microMetrics(metricItems, "image");
  const videoMicro = microMetrics(metricItems, "video");
  const mediaMacro = macroMetrics(metricItems, "media");
  const imageMacro = macroMetrics(metricItems, "image");
  const videoMacro = macroMetrics(metricItems, "video");
  const actionJudges = items.flatMap((item) => item.task_judges?.length ? item.task_judges : [item.task_judge])
    .filter((judge) => [true, false].includes(judge?.correct));
  const parseTotals = items.map((item) => item.agent_stability?.json_parse_total).filter(Number.isFinite);
  const parseSuccesses = items.map((item) => item.agent_stability?.json_parse_success).filter(Number.isFinite);
  const completion = items.map((item) => item.agent_stability?.completed_within_steps).filter((value) => typeof value === "boolean");
  const agentTaskLatencies = items.map((item) => Number(item.timing_breakdown?.agent_wall_ms)).filter(Number.isFinite);
  const agentLoopCounts = items.map((item) => {
    const calls = itemCallMetrics(item);
    return calls.some((call) => call.call_type)
      ? calls.filter((call) => ["agent", "recovery"].includes(call.call_type)).length
      : null;
  }).filter(Number.isFinite);
  const dist = { ...(saved.judge_distribution || {}) };
  if (!Object.keys(saved.judge_distribution || {}).length) scores.forEach((score) => { dist[String(score)] = (dist[String(score)] || 0) + 1; });
  const llm = items.map(itemLlmSummary).filter(Boolean);
  const allCalls = items.flatMap(itemCallMetrics);
  const callTokenCounts = allCalls.map((call) => {
    const prompt = Number(call.preflight_prompt_tokens ?? call.prompt_tokens);
    const completion = Number(call.completion_tokens);
    return Number.isFinite(prompt) && Number.isFinite(completion) ? { prompt, completion, context: prompt + completion } : null;
  }).filter(Boolean);
  const promptTokens = allCalls.map((call) => Number(call.preflight_prompt_tokens ?? call.prompt_tokens)).filter(Number.isFinite);
  const completionTokens = callTokenCounts.map((call) => call.completion);
  const contextTokens = callTokenCounts.map((call) => call.context);
  return {
    ...saved,
    completed: saved.completed ?? items.length,
    total: saved.total ?? run?.qa_count ?? items.length,
    judge_valid_count: saved.judge_valid_count ?? judged.length,
    judge_distribution: dist,
    retrieval_recall_mean: saved.retrieval_recall_macro ?? (typedRetrieval ? mediaMacro.recall : averageMetric(recalls)),
    retrieval_precision_macro: saved.retrieval_precision_macro ?? (typedRetrieval ? mediaMacro.precision : averageMetric(retrievalItems.map((item) => item.retrieval_precision).filter(Number.isFinite))),
    retrieval_recall_macro: saved.retrieval_recall_macro ?? (typedRetrieval ? mediaMacro.recall : averageMetric(recalls)),
    retrieval_f1_macro: saved.retrieval_f1_macro ?? (typedRetrieval ? mediaMacro.f1 : averageMetric(retrievalItems.map((item) => item.retrieval_f1).filter(Number.isFinite))),
    retrieval_excluded_unanswerable_count: saved.retrieval_excluded_unanswerable_count ?? (items.length - metricItems.length),
    answer_quality_mean: saved.answer_quality_mean ?? (scores.length ? averageMetric(scores) : null),
    exact_accuracy: saved.exact_accuracy ?? (scores.length ? scores.filter((score) => score === 2).length / scores.length : null),
    core_accuracy: saved.core_accuracy ?? (scores.length ? scores.filter((score) => score >= 1).length / scores.length : null),
    retrieval_precision_micro: saved.retrieval_precision_micro ?? (typedRetrieval ? mediaMicro.precision : retrievalPrecision),
    retrieval_recall_micro: saved.retrieval_recall_micro ?? (typedRetrieval ? mediaMicro.recall : retrievalRecall),
    retrieval_f1_micro: saved.retrieval_f1_micro ?? (typedRetrieval ? mediaMicro.f1 : retrievalF1),
    retrieval_metric_count: saved.retrieval_metric_count ?? retrievalItems.length,
    retrieval_metric_scope: saved.retrieval_metric_scope ?? (typedRetrieval ? "all_media" : "legacy_image_only"),
    media_retrieval_precision_micro: saved.media_retrieval_precision_micro ?? (typedRetrieval ? mediaMicro.precision : null),
    media_retrieval_recall_micro: saved.media_retrieval_recall_micro ?? (typedRetrieval ? mediaMicro.recall : null),
    media_retrieval_f1_micro: saved.media_retrieval_f1_micro ?? (typedRetrieval ? mediaMicro.f1 : null),
    media_retrieval_precision_macro: saved.media_retrieval_precision_macro ?? (typedRetrieval ? mediaMacro.precision : null),
    media_retrieval_recall_macro: saved.media_retrieval_recall_macro ?? (typedRetrieval ? mediaMacro.recall : null),
    media_retrieval_f1_macro: saved.media_retrieval_f1_macro ?? (typedRetrieval ? mediaMacro.f1 : null),
    media_retrieval_metric_count: saved.media_retrieval_metric_count ?? (typedRetrieval ? mediaMicro.metricCount : null),
    image_retrieval_precision_micro: saved.image_retrieval_precision_micro ?? (typedRetrieval ? imageMicro.precision : retrievalPrecision),
    image_retrieval_recall_micro: saved.image_retrieval_recall_micro ?? (typedRetrieval ? imageMicro.recall : retrievalRecall),
    image_retrieval_f1_micro: saved.image_retrieval_f1_micro ?? (typedRetrieval ? imageMicro.f1 : retrievalF1),
    image_retrieval_precision_macro: saved.image_retrieval_precision_macro ?? (typedRetrieval ? imageMacro.precision : averageMetric(retrievalItems.map((item) => item.retrieval_precision).filter(Number.isFinite))),
    image_retrieval_recall_macro: saved.image_retrieval_recall_macro ?? (typedRetrieval ? imageMacro.recall : averageMetric(recalls)),
    image_retrieval_f1_macro: saved.image_retrieval_f1_macro ?? (typedRetrieval ? imageMacro.f1 : averageMetric(retrievalItems.map((item) => item.retrieval_f1).filter(Number.isFinite))),
    image_retrieval_metric_count: saved.image_retrieval_metric_count ?? (typedRetrieval ? imageMicro.metricCount : retrievalItems.length),
    video_retrieval_precision_micro: saved.video_retrieval_precision_micro ?? (typedRetrieval ? videoMicro.precision : null),
    video_retrieval_recall_micro: saved.video_retrieval_recall_micro ?? (typedRetrieval ? videoMicro.recall : null),
    video_retrieval_f1_micro: saved.video_retrieval_f1_micro ?? (typedRetrieval ? videoMicro.f1 : null),
    video_retrieval_precision_macro: saved.video_retrieval_precision_macro ?? (typedRetrieval ? videoMacro.precision : null),
    video_retrieval_recall_macro: saved.video_retrieval_recall_macro ?? (typedRetrieval ? videoMacro.recall : null),
    video_retrieval_f1_macro: saved.video_retrieval_f1_macro ?? (typedRetrieval ? videoMacro.f1 : null),
    video_retrieval_metric_count: saved.video_retrieval_metric_count ?? (typedRetrieval ? videoMicro.metricCount : null),
    evidence_distribution: saved.evidence_distribution ?? { 0: evidenceScores.filter((score) => score === 0).length, 1: evidenceScores.filter((score) => score === 1).length, 2: evidenceScores.filter((score) => score === 2).length },
    evidence_valid_count: saved.evidence_valid_count ?? evidenceScores.length,
    evidence_mean: saved.evidence_mean ?? (evidenceScores.length ? averageMetric(evidenceScores) : null),
    evidence_fully_supported_rate: saved.evidence_fully_supported_rate ?? (evidenceScores.length ? evidenceScores.filter((score) => score === 2).length / evidenceScores.length : null),
    evidence_basically_supported_rate: saved.evidence_basically_supported_rate ?? (evidenceScores.length ? evidenceScores.filter((score) => score >= 1).length / evidenceScores.length : null),
    task_decision_labeled_count: saved.task_decision_labeled_count ?? actionJudges.length,
    task_decision_valid_count: saved.task_decision_valid_count ?? actionJudges.length,
    task_decision_accuracy: saved.task_decision_accuracy ?? (actionJudges.length ? actionJudges.filter((judge) => judge.correct).length / actionJudges.length : null),
    json_parse_total: saved.json_parse_total ?? (parseTotals.length ? parseTotals.reduce((sum, value) => sum + value, 0) : null),
    json_parse_success: saved.json_parse_success ?? (parseSuccesses.length ? parseSuccesses.reduce((sum, value) => sum + value, 0) : null),
    json_parse_success_rate: saved.json_parse_success_rate ?? (parseTotals.length ? parseSuccesses.reduce((sum, value) => sum + value, 0) / parseTotals.reduce((sum, value) => sum + value, 0) : null),
    qa_completion_valid_count: saved.qa_completion_valid_count ?? completion.length,
    qa_completion_within_steps_rate: saved.qa_completion_within_steps_rate ?? (completion.length ? completion.filter(Boolean).length / completion.length : null),
    agent_task_latency_mean_ms: saved.agent_task_latency_mean_ms ?? averageMetric(agentTaskLatencies),
    agent_loop_calls_mean: saved.agent_loop_calls_mean ?? averageMetric(agentLoopCounts),
    llm_ttft_ms_mean: saved.llm_ttft_ms_mean ?? averageMetric(llm.map((item) => item.ttft_ms_avg)),
    llm_tokens_per_second_mean: saved.llm_tokens_per_second_mean ?? averageMetric(llm.map((item) => item.tokens_per_second_avg)),
    prompt_tokens_total: saved.prompt_tokens_total ?? (allCalls.length ? llm.reduce((sum, item) => sum + (Number(item.prompt_tokens_total) || 0), 0) : null),
    completion_tokens_total: saved.completion_tokens_total ?? (allCalls.length ? llm.reduce((sum, item) => sum + (Number(item.completion_tokens_total) || 0), 0) : null),
    llm_prompt_tokens_max: saved.llm_prompt_tokens_max ?? (promptTokens.length ? Math.max(...promptTokens) : null),
    llm_prompt_tokens_p95: saved.llm_prompt_tokens_p95 ?? nearestRankPercentile(promptTokens, 0.95),
    llm_completion_tokens_max: saved.llm_completion_tokens_max ?? (completionTokens.length ? Math.max(...completionTokens) : null),
    llm_completion_tokens_p95: saved.llm_completion_tokens_p95 ?? nearestRankPercentile(completionTokens, 0.95),
    llm_context_tokens_max: saved.llm_context_tokens_max ?? (contextTokens.length ? Math.max(...contextTokens) : null),
    llm_context_tokens_p95: saved.llm_context_tokens_p95 ?? nearestRankPercentile(contextTokens, 0.95),
    llm_context_samples_count: saved.llm_context_samples_count ?? contextTokens.length,
    agent_throughput_latency_mode: saved.agent_throughput_latency_mode
      ?? (run?.phases?.qa_eval?.agent_phase_wall_ms != null ? "measured_agent_phase" : "historical_interval_estimate"),
    agent_phase_wall_ms: saved.agent_phase_wall_ms ?? run?.phases?.qa_eval?.agent_phase_wall_ms ?? null,
    agent_phase_completed_count: saved.agent_phase_completed_count ?? run?.phases?.qa_eval?.agent_completed ?? null,
    agent_throughput_qa_per_s: saved.agent_throughput_qa_per_s ?? null,
    judge_phase_wall_ms: saved.judge_phase_wall_ms ?? run?.phases?.qa_eval?.judge_phase_wall_ms ?? null,
    judge_concurrency: saved.judge_concurrency ?? run?.phases?.qa_eval?.judge_concurrency ?? run?.judge_concurrency ?? null,
  };
}
function resultPhaseStatus(phase) {
  if (phase?.status) return phase.status;
  return ["cancelled", "interrupted", "failed"].includes(activeRun.value?.status) ? "not_run" : "pending";
}

function imageUrl(image) {
  return image?.media_url || (image?.asset_id ? `${sentrixUrl.value.replace(/\/$/, "")}/api/assets/${image.asset_id}/file` : "");
}
function actionLabel(value) {
  return ({ answer: "回答", refuse: "拒答", clarify: "澄清", none: "无有效行为" })[value] || "未记录";
}
function evidenceScoreLabel(judge) {
  const score = judge?.score;
  if (score === 2) return "2 分：媒体证据支持回答";
  if (score === 1) return "1 分：媒体证据部分支持";
  if (score === 0) return "0 分：媒体证据无法支持";
  return judge?.reason === "not_applicable" ? "不适用" : judge?.reason === "no_answer" ? "无回答，未评分" : "未记录";
}
function itemRetrievalMetrics(item) {
  if (!mediaRefs(item).length) return "本题无标准媒体，不计入检索指标";
  const total = `总媒体 P ${fmtPct(item?.media_retrieval_precision ?? item?.retrieval_precision)} · R ${fmtPct(item?.media_retrieval_recall ?? item?.retrieval_recall)} · F1 ${fmtPct(item?.media_retrieval_f1 ?? item?.retrieval_f1)}`;
  if (!Object.prototype.hasOwnProperty.call(item || {}, "retrieval_media_refs")) return `${total} · 历史图片口径`;
  const parts = [total];
  if (mediaRefs(item).some((ref) => ref.media_type === "image")) parts.push(`图片 R ${fmtPct(item?.image_retrieval_recall)}`);
  if (mediaRefs(item).some((ref) => ref.media_type === "video")) parts.push(`视频 R ${fmtPct(item?.video_retrieval_recall)}`);
  return parts.join(" · ");
}
function itemParseRate(item) {
  const stability = item?.agent_stability || {};
  if (stability.json_parse_total == null) return "未记录";
  return `${stability.json_parse_success ?? 0}/${stability.json_parse_total} 次模型输出解析为合法动作 (${fmtPct(stability.json_parse_rate)})`;
}
function executionState(item) {
  const stability = item?.agent_stability || {};
  const status = String(item?.agent_status || item?.guard_debug?.status || "").toLowerCase();
  const termination = String(item?.termination_reason || item?.guard_debug?.termination_reason || "").toLowerCase();
  const turns = Array.isArray(item?.runtime_turns) ? item.runtime_turns : [];
  const outcome = String(item?.turn_outcome || turns[turns.length - 1]?.turn_outcome || "").toLowerCase();
  const failure = [status, termination, outcome].some((value) => /error|failed|failure|timeout|cancel|blocked|parse_failure|model_error/.test(value));
  const partial = [status, termination, outcome].some((value) => /partial|limit|budget|incomplete|tool_call_limit|step_limit/.test(value));
  if (failure) return { key: "failed", label: "执行失败" };
  if (partial) return { key: "partial", label: "部分完成" };
  if (outcome === "final_answer" || ["complete", "completed", "done", "success"].includes(status)
      || ["complete", "completed", "done", "success"].includes(termination)
      || stability.completed_within_steps === true) {
    return { key: "complete", label: "已完成" };
  }
  if (status || termination || outcome || stability.completed_within_steps === false) return { key: "unknown", label: "状态待确认" };
  return { key: "unknown", label: "未记录" };
}
function executionStateClass(item) { return `status-${executionState(item).key}`; }
function completionLabel(item) {
  return executionState(item).label;
}
function taskDecisionLabel(item) {
  const judge = item?.task_judge || {};
  if (!judge.expected_action) return "未标注";
  return `期望${actionLabel(judge.expected_action)}，实际${actionLabel(judge.actual_action)}`;
}
function judgeReason(judge) {
  if (!judge?.reason || ["not_applicable", "no_answer"].includes(judge.reason)) return "";
  return judge.reason;
}
function conversationTurns(item) {
  return Array.isArray(item?.conversation) ? item.conversation : [];
}
function conversationIdLabel(item) {
  return item?.conversation_id || conversationTurns(item).find((turn) => turn?.conversation_id)?.conversation_id || "历史结果未记录会话 ID";
}
function conversationContextLabel(turn, index) {
  const count = Number.isFinite(Number(turn?.context_turn_count)) ? Number(turn.context_turn_count) : index;
  return count > 0 ? `本轮携带前 ${count} 轮对话上下文` : "首轮，无历史上下文";
}
function turnScore(score) {
  return [0, 1, 2].includes(score) ? `${score} 分` : "不适用";
}
function albumLocalUrl(fileName, mediaType = "") {
  const album = activeRun.value?.album_id || "";
  const video = inferMediaType(fileName, mediaType) === "video";
  const collection = video ? "videos" : /^faceid_[^/]+\.(?:jpe?g|png|webp)$/i.test(String(fileName || "")) ? "faces" : "photos";
  const mediaFile = video && !/\.(?:mp4|mov|m4v|webm|mkv|avi)$/i.test(String(fileName || "")) ? `${fileName}.mp4` : fileName;
  return (album && mediaFile) ? `/api/albums/${encodeURIComponent(album)}/${collection}/${encodeURIComponent(mediaFile)}` : "";
}
function isVideoMedia(image) {
  if (image?.media_type === "video") return true;
  return [image?.media_id, image?.video_id, image?.image_id, image?.file_name, image?.media_url].some((value) => {
    const text = String(value || "");
    const fileName = text.split(/[/?#]/).filter(Boolean).pop() || "";
    return /^video-\d+(?:\.mp4)?$/i.test(fileName) || /\.mp4(?:$|[?#])/i.test(text);
  });
}
function decorateMedia(list) {
  return (list || []).map((img) => {
    if (img?.media_url) return img;
    const sourceId = img?.media_id || img?.image_id || img?.video_id || "";
    const parts = sourceId.split("/");
    const file = parts.length >= 2 ? parts[parts.length - 1] : (img?.file_name || sourceId);
    const album = activeRun.value?.album_id || (parts.length === 2 ? parts[0] : "");
    const video = isVideoMedia(img);
    const collection = video ? "videos" : /^faceid_[^/]+\.(?:jpe?g|png|webp)$/i.test(file) ? "faces" : "photos";
    const mediaFile = video && !/\.mp4$/i.test(file) ? `${file}.mp4` : file;
    const local = (album && mediaFile) ? `/api/albums/${encodeURIComponent(album)}/${collection}/${encodeURIComponent(mediaFile)}` : "";
    return local ? { ...img, media_type: video ? "video" : (img.media_type || "image"), media_url: local } : img;
  });
}
function itemMedia(item, gt = false) {
  if (gt) {
    if (item.gt_media?.length) return decorateMedia(item.gt_media);
    if (item.gt_images?.length && !Object.prototype.hasOwnProperty.call(item || {}, "retrieval_media_refs")) return decorateMedia(item.gt_images);
    return decorateMedia(mediaRefs(item).map((ref) => {
      const fileName = ref.media_type === "video" && !/\.[^.]+$/.test(ref.media_id) ? `${ref.media_id}.mp4` : ref.media_id.split("/").pop();
      return { ...ref, file_name: fileName, matched: (item.matched_file_names || []).includes(fileName) };
    }));
  }
  // Model recall is the upstream evidence projection, not only explicit delivery.
  if (item.evidence_source_media?.length) return decorateMedia(item.evidence_source_media);
  if (item.evidence_source_images?.length) return decorateMedia(item.evidence_source_images);
  if (item.evidence_source_file_names?.length) {
    return decorateMedia(item.evidence_source_file_names.map((file_name) => ({ file_name, media_type: inferMediaType(file_name), media_url: albumLocalUrl(file_name) })));
  }
  if (item.predicted_media?.length) return decorateMedia(item.predicted_media);
  if (item.predicted_images?.length) return decorateMedia(item.predicted_images);
  if (item.predicted_file_names?.length) {
    return item.predicted_file_names.map((file_name) => ({ file_name, media_type: inferMediaType(file_name), media_url: albumLocalUrl(file_name) }));
  }
  // A validator may leave all candidates as candidate_only.  They are not
  // answer evidence, but hiding them makes a healthy retrieval look empty
  // in the evaluation UI.  Show a bounded representative window here; the
  // full candidate set remains in retrieval metrics and the trace.
  if (item.retrieved_candidate_media?.length) {
    return decorateMedia(item.retrieved_candidate_media.slice(0, 6));
  }
  if (item.retrieved_candidate_images?.length) {
    return decorateMedia(item.retrieved_candidate_images.slice(0, 6));
  }
  return (item.retrieved_file_names || []).slice(0, 6)
    .map((file_name) => ({ file_name, media_type: inferMediaType(file_name), media_url: albumLocalUrl(file_name) }));
}
function itemEvidenceMedia(item) {
  const media = item?.evidence_source_media || item?.evidence_source_images || [];
  if (media.length) return decorateMedia(media);
  const names = item?.evidence_source_file_names || [];
  return decorateMedia(names.map((file_name) => ({ file_name, media_type: inferMediaType(file_name), media_url: albumLocalUrl(file_name) })));
}
function isDirectEvidence(item, media) {
  const ref = typeof media === "string" ? { media_id: media, media_type: inferMediaType(media) } : media;
  const key = mediaKey(ref);
  const answerRefs = mediaRefs(item, "answer_evidence");
  const claimRefs = (item?.answer_claims || []).flatMap((claim) => mediaRefs(claim, "evidence"));
  return [...answerRefs, ...claimRefs].some((candidate) => mediaKey(candidate) === key);
}
function judgeInput(item) {
  const input = item.judge?.input || {};
  return {
    complete: Array.isArray(input.messages),
    rawJson: Array.isArray(input.messages) ? JSON.stringify(input, null, 2) : "",
  };
}
function openJudgeInput(item) { judgeModal.value = { qaId: item?.qa_id || "", ...judgeInput(item) }; }
function closeJudgeInput() { judgeModal.value = null; }
function toolBindingLabel(trace) {
  if (trace?.round_binding_source === "step_id") return "按步骤 ID 精确绑定";
  return trace?.round_binding_source === "inferred_single_model_call" ? "单轮数据推断归属" : "按执行轨迹绑定";
}
function retrievalBackendLabel(trace) {
  const channels = trace?.retrieval_timing?.channels || {};
  const backends = [...new Set(Object.values(channels).map((channel) => channel && channel.backend).filter(Boolean))];
  return backends.length ? backends.join("/") : "";
}
function retrievalBackendDegraded(trace) {
  const label = retrievalBackendLabel(trace);
  return Boolean(label) && label.split("/").some((backend) => backend !== "qdrant");
}
function judgeRoundState(item) {
  const task = activeRejudge.value;
  const judge = item.judge || {};
  if (!task) return "normal";
  if (judge.rejudge_id !== task.rejudge_id) return task.status === "running" ? "pending" : "normal";
  if (task.status !== "running" && (judge.status === "pending" || judge.status === "running")) return "interrupted";
  if (judge.status === "pending" || judge.status === "running" || judge.status === "failed") return judge.status;
  return "updated";
}
function judgeScoreLabel(item) {
  if (item.judge?.consistency_status === "inconsistent") return "评分异常";
  const state = judgeRoundState(item);
  if (state === "pending") return "待重新评分";
  if (state === "running") return "评分中";
  if (state === "failed") return "评分失败";
  if (state === "interrupted") return "本轮未完成";
  return item.judge?.score == null ? "未评分" : `${item.judge.score}分`;
}
function phaseSeconds(phase, preferredKey = "total_seconds") {
  if (!phase) return null;
  const preferred = Number(phase[preferredKey]);
  if (Number.isFinite(preferred) && preferred > 0) return preferred;
  if (phase.started_at && phase.finished_at) {
    const elapsed = (new Date(phase.finished_at) - new Date(phase.started_at)) / 1000;
    if (Number.isFinite(elapsed) && elapsed >= 0) return elapsed;
  }
  return Number.isFinite(preferred) ? preferred : null;
}
function fmtSeconds(value) {
  if (value == null || !Number.isFinite(Number(value))) return "-";
  const seconds = Number(value);
  if (seconds < 0.001) return "<1ms";
  if (seconds < 0.1) return `${Math.round(seconds * 1000)}ms`;
  if (seconds < 10) return `${seconds.toFixed(2).replace(/0+$/, "").replace(/\.$/, "")}s`;
  return `${seconds.toFixed(1).replace(/\.0$/, "")}s`;
}
function fmtNumber(value, suffix = "", digits = 1) {
  if (value == null || !Number.isFinite(Number(value))) return "-";
  return `${Number(value).toFixed(digits).replace(/\.0$/, "")}${suffix}`;
}
function fmtMemory(value) {
  if (value == null || !Number.isFinite(Number(value))) return "-";
  return `${(Number(value) / 1024).toFixed(2)} GiB`;
}
function gpuMetricRows(phase = {}) {
  if (phase.memory_pressure) {
    const mp = phase.memory_pressure || {};
    const used = phase.memory_used_gib || {};
    const comp = phase.compressed_gib || {};
    const swap = phase.swap_used_gib || {};
    const thermal = phase.thermal_state || {};
    const cpu = phase.cpu_percent || {};
    const modelMem = phase.model_process_memory_used_mib || {};
    const arb = phase.arbiter_summary || {};
    const arbDist = arb.state_distribution || {};
    const arbLabel = Object.keys(arbDist).length ? Object.entries(arbDist).map(([k, v]) => `${k}=${v}`).join(" ") : "-";
    const thermalLabel = (v) => v == null ? "-" : (["nominal", "fair", "serious", "critical"][Math.round(v)] ?? `${v}`);
    return [
      ["内存压力", fmtNumber(mp.mean), `峰值 ${fmtNumber(mp.peak)} · P95 ${fmtNumber(mp.p95)}`, true],
      ["整机内存占用", used.mean == null ? "-" : `${Number(used.mean).toFixed(2)} GiB`, `峰值 ${used.peak == null ? "-" : `${Number(used.peak).toFixed(2)} GiB`} · P95 ${used.p95 == null ? "-" : `${Number(used.p95).toFixed(2)} GiB`}`],
      ["压缩内存", comp.mean == null ? "-" : `${Number(comp.mean).toFixed(2)} GiB`, `峰值 ${comp.peak == null ? "-" : `${Number(comp.peak).toFixed(2)} GiB`} · macOS 内存压缩器占用`],
      ["Swap 用量", swap.mean == null ? "-" : `${Number(swap.mean).toFixed(2)} GiB`, `峰值 ${swap.peak == null ? "-" : `${Number(swap.peak).toFixed(2)} GiB`} · 换页开始即压力信号`],
      ["散热状态", thermal.mean == null ? "-" : thermalLabel(thermal.mean), `峰值 ${thermal.peak == null ? "-" : thermalLabel(thermal.peak)} · NSProcessInfo.thermalState`, true],
      ["CPU 占用", cpu.mean == null ? "-" : fmtNumber(cpu.mean, "%"), `峰值 ${cpu.peak == null ? "-" : fmtNumber(cpu.peak, "%")} · 全核采样`],
      ["模型进程内存", modelMem.mean == null ? "-" : fmtMemory(modelMem.mean), `峰值 ${modelMem.peak == null ? "-" : fmtMemory(modelMem.peak)} · mlx 进程 RSS（Metal 分配不在其中）`],
      ["调度状态", arbLabel, `worker_scale 均值 ${arb.worker_scale_mean == null ? "-" : arb.worker_scale_mean} · 预占峰值 ${arb.preempt_count_max ?? 0}`, true],
      ["Import/Agent 活跃峰值", `import ${arb.import_active_peak ?? 0} · agent ${arb.agent_vlm_active_peak ?? 0}`, "采样期内 VLM 令牌持有峰值"],
      ["采样数量", phase.samples_count == null ? "-" : `${phase.samples_count} 次`, "macOS 系统采样点"],
    ];
  }
  const temp = phase.temperature_c || {};
  const util = phase.gpu_utilization_pct || {};
  const memory = phase.memory_used_mib || {};
  const modelMemory = phase.model_process_memory_used_mib || {};
  const kvCache = phase.kv_cache_usage_pct || {};
  const power = phase.power_draw_w || {};
  const clock = phase.sm_clock_mhz || {};
  const processLimit = phase.model_process_memory_limit_mib;
  const processLimitLabel = processLimit == null ? "模型进程显存上限" : `${fmtMemory(processLimit)} 上限告警`;
  return [
    ["模型进程显存", fmtMemory(modelMemory.mean), `峰值 ${fmtMemory(modelMemory.peak)} · P95 ${fmtMemory(modelMemory.p95)}`, true],
    ["采样数量", phase.samples_count == null ? "-" : `${phase.samples_count} 次`, "GPU 原始采样点"],
    ["GPU 利用率", fmtNumber(util.mean, "%"), `峰值 ${fmtNumber(util.peak, "%")} · P95 ${fmtNumber(util.p95, "%")}`],
    ["整卡显存", fmtMemory(memory.mean), `峰值 ${fmtMemory(memory.peak)} · P95 ${fmtMemory(memory.p95)}`],
    [processLimitLabel, phase.model_process_over_limit_samples == null ? "-" : `${phase.model_process_over_limit_samples} 次`, processLimit == null ? "Manager 未返回告警阈值" : `模型进程 NVML 占用超过 ${fmtMemory(processLimit)} 的采样次数`],
    ["KV Cache 使用率", fmtNumber(kvCache.mean, "%"), `峰值 ${fmtNumber(kvCache.peak, "%")} · P95 ${fmtNumber(kvCache.p95, "%")}`],
    ["GPU 温度", fmtNumber(temp.mean, "°C"), `峰值 ${fmtNumber(temp.peak, "°C")} · P95 ${fmtNumber(temp.p95, "°C")}`],
    ["GPU 功耗", fmtNumber(power.mean, "W"), `峰值 ${fmtNumber(power.peak, "W")} · P95 ${fmtNumber(power.p95, "W")}`],
    ["SM 时钟", fmtNumber(clock.mean, "MHz"), `峰值 ${fmtNumber(clock.peak, "MHz")} · P95 ${fmtNumber(clock.p95, "MHz")}`],
  ];
}
function comparableMemoryProfile(run) {
  if (!run) return null;
  if (run?.memory_profile) return { ...run.memory_profile, source: "replay" };
  const gpu = run?.phases?.gpu_metrics;
  if (!gpu) return { source: "pending", status: "pending", memory_profile: {}, questions_completed: run.summary?.completed, questions_total: run.summary?.total };
  if (!gpu?.memory_profile) return { source: "pending", status: gpu.status, memory_profile: {}, questions_completed: run.summary?.completed, questions_total: run.summary?.total };
  return {
    status: gpu.status,
    source: "gpu_metrics",
    memory_profile: gpu.memory_profile,
    model_process_memory_used_mib: gpu.model_process_memory_used_mib,
    questions_completed: run.summary?.completed,
    questions_total: run.summary?.total,
  };
}
function memoryProfileRows(profile = {}) {
  const memory = profile.memory_profile || {};
  const isBenchmarkGpuProfile = profile.source === "gpu_metrics";
  if (memory.method === "macos_unified_memory_v1") {
    return [
      ["内存占用峰值", memory.memory_used_peak_gib == null ? "-" : `${Number(memory.memory_used_peak_gib).toFixed(2)} GiB`, "16GB 统一内存整机峰值", true],
      ["模型进程空闲占用", memory.idle_model_process_memory_gib == null ? "-" : `${Number(memory.idle_model_process_memory_gib).toFixed(2)} GiB`, "mlx 进程 RSS 采样最小值"],
      ["Swap 峰值", memory.swap_used_peak_gib == null ? "-" : `${Number(memory.swap_used_peak_gib).toFixed(2)} GiB`, "换页压力信号"],
      ["压缩内存峰值", memory.compressed_peak_gib == null ? "-" : `${Number(memory.compressed_peak_gib).toFixed(2)} GiB`, "macOS 内存压缩器峰值"],
      ["复测进度", `${profile.questions_completed ?? 0}/${profile.questions_total ?? 0} 题`, `请求失败 ${profile.failed_requests ?? 0} · 答案不保存`],
      ["原测评数据一致性", profile.items_integrity_ok === true ? "通过" : profile.status === "completed" ? "未通过" : "待完成", profile.answers_persisted === false ? "原答案未写入" : "记录状态异常"],
    ];
  }
  const processMemory = profile.model_process_memory_used_mib || {};
  return [
    ["可比较工作负载显存", memory.comparable_workload_memory_gib == null ? "-" : `${Number(memory.comparable_workload_memory_gib).toFixed(2)} GiB`, "固定基础占用 + 本次 KV Cache 实际峰值", true],
    ["固定基础占用", memory.fixed_base_memory_gib == null ? "-" : `${Number(memory.fixed_base_memory_gib).toFixed(2)} GiB`, "空载模型进程显存 - 预分配 KV Cache 容量", true],
    ["KV Cache 容量", memory.kv_cache_capacity_gib == null ? "-" : `${Number(memory.kv_cache_capacity_gib).toFixed(2)} GiB`, memory.kv_cache_capacity_tokens == null ? "未记录 token 容量" : `${Number(memory.kv_cache_capacity_tokens).toLocaleString("en-US")} token`],
    ["KV Cache 实际峰值", memory.kv_cache_used_peak_gib == null ? "-" : `${Number(memory.kv_cache_used_peak_gib).toFixed(3)} GiB`, `使用率峰值 ${fmtNumber(memory.kv_cache_usage_peak_pct, "%")}`],
    ["模型权重", memory.weight_gib == null ? "-" : `${Number(memory.weight_gib).toFixed(2)} GiB`, `激活峰值 ${memory.peak_activation_gib == null ? "-" : `${memory.peak_activation_gib} GiB`} · CUDA Graph ${memory.cuda_graph_gib == null ? "-" : `${memory.cuda_graph_gib} GiB`}`],
    ["vLLM 进程预留显存", fmtMemory(processMemory.peak), `空载 ${memory.idle_process_memory_gib == null ? "-" : `${Number(memory.idle_process_memory_gib).toFixed(2)} GiB`} · 不用于跨模型需求比较`],
    isBenchmarkGpuProfile
      ? ["评测采样覆盖", `${profile.questions_completed ?? "-"}/${profile.questions_total ?? "-"} 题`, "来自本次正式评测 GPU 采样"]
      : ["复测进度", `${profile.questions_completed ?? 0}/${profile.questions_total ?? 0} 题`, `请求失败 ${profile.failed_requests ?? 0} · 答案不保存`],
    isBenchmarkGpuProfile
      ? ["数据来源", "正式评测采样", "与本次 run 的 QA/GPU 采样同时记录"]
      : ["原测评数据一致性", profile.items_integrity_ok === true ? "通过" : profile.status === "completed" ? "未通过" : "待完成", profile.answers_persisted === false ? "原答案未写入" : "记录状态异常"],
  ];
}
function aggregateMetricRows(phase = {}) {
  // The aggregate phase is a historical snapshot and may predate rejudge or
  // consistency filtering. Use the run-level effective summary as the single
  // source of truth for the detail view.
  const summary = effectiveRunSummary(activeRun.value);
  const dist = summary.judge_distribution || {};
  const evidenceDist = summary.evidence_distribution || {};
  const throughputSamples = Number(summary.agent_throughput_latency_sample_count);
  const throughputTotal = Number(summary.agent_throughput_latency_total_count ?? summary.total);
  const throughputNote = summary.agent_throughput_latency_mode === "measured_agent_phase"
    ? `Agent 阶段实际墙钟 ÷ ${summary.agent_phase_completed_count ?? throughputTotal} 题 · 并发 ${activeRun.value?.qa_concurrency ?? "-"} · 不含 Judge，Judge 可与 Agent 并行`
    : Number.isFinite(throughputSamples) && Number.isFinite(throughputTotal)
      ? `历史记录按 Agent/Judge 时间线回退估算 · ${throughputSamples}/${throughputTotal} 题，不能视为实测`
      : "历史记录未保存 Agent 独立阶段墙钟";
  const typedMediaMetrics = summary.retrieval_metric_scope === "all_media";
  const rows = [
    [typedMediaMetrics ? "媒体检索 Precision" : "历史图片检索 Precision", fmtPct(summary.retrieval_precision_macro), `逐 QA 求值后平均 · ${summary.retrieval_metric_count ?? 0} 题有 GT · 排除 ${summary.retrieval_excluded_unanswerable_count ?? 0} 道不可回答题`, true],
    [typedMediaMetrics ? "媒体检索 Recall" : "历史图片检索 Recall", fmtPct(summary.retrieval_recall_macro), typedMediaMetrics ? "每道 QA 的 Recall 等权平均；图视频按类型与稳定标识匹配" : "每道 QA 的 Recall 等权平均；历史 run 无法补算视频指标", true],
    ["回答质量均分", summary.answer_quality_mean == null ? "-" : `${summary.answer_quality_mean} / 2`, `Valid ${summary.judge_valid_count ?? 0}/${summary.total ?? 0} · Invalid ${(summary.total ?? 0) - (summary.judge_valid_count ?? 0)} · 0:${dist["0"] || 0} · 1:${dist["1"] || 0} · 2:${dist["2"] || 0}`, true],
    ["步数内 QA 完成率", fmtPct(summary.qa_completion_within_steps_rate), `有效记录 ${summary.qa_completion_valid_count ?? 0} 题`, true],
    ["JSON 解析成功率", fmtPct(summary.json_parse_success_rate), summary.json_parse_total == null ? "历史记录未保存解析轨迹" : `${summary.json_parse_success ?? 0}/${summary.json_parse_total} 个需解析模型输出`, true],
    ["Agent 并发吞吐折算时延", fmtMs(summary.agent_throughput_latency_ms), throughputNote, true],
    ["平均调用轮数", summary.agent_loop_calls_mean == null ? "未记录" : `${Number(summary.agent_loop_calls_mean).toFixed(2)} 轮`, "仅 Agent/Recovery，不含 L2 Judge、Final Writer 和工具内部模型", true],
    ["累计输入 token", fmtTokens(summary.prompt_tokens_total), "所有主 Agent 模型调用输入 token 累计", true],
    ["累计输出 token", fmtTokens(summary.completion_tokens_total), "所有主 Agent 模型调用输出 token 累计", true],
    ["平均任务完成时间", fmtMs(summary.agent_task_latency_mean_ms), activeRun.value?.qa_concurrency > 1
      ? `每道 QA 各自计时的平均值（输入→最终回答，不含 Judge）；并发 ${activeRun.value.qa_concurrency} 负载下含排队与批内干扰，勿与串行 run 直接对比`
      : "每道 QA 各自计时的平均值（输入→最终回答，不含 Judge）", true],
    [typedMediaMetrics ? "媒体检索 F1" : "历史图片检索 F1", fmtPct(summary.retrieval_f1_macro), "逐 QA 计算 F1 后等权平均"],
    ["图片检索 P / R / F1", `${fmtPct(summary.image_retrieval_precision_macro)} / ${fmtPct(summary.image_retrieval_recall_macro)} / ${fmtPct(summary.image_retrieval_f1_macro)}`, typedMediaMetrics ? `${summary.image_retrieval_metric_count ?? 0} 题含图片 GT · 逐 QA 平均` : "历史图片口径 · 逐 QA 平均"],
    ["视频检索 P / R / F1", typedMediaMetrics ? `${fmtPct(summary.video_retrieval_precision_macro)} / ${fmtPct(summary.video_retrieval_recall_macro)} / ${fmtPct(summary.video_retrieval_f1_macro)}` : "未记录", typedMediaMetrics ? `${summary.video_retrieval_metric_count ?? 0} 题含视频 GT · 逐 QA 平均` : "历史 run 无 typed media，禁止推测"],
    ["Judge LLM 平均时延", fmtMs(summary.judge_llm_latency_mean_ms), `每题 Judge 评分调用平均耗时 · Judge 阶段墙钟 ${fmtMs(summary.judge_phase_wall_ms)}`],
    ["任务判断准确率", fmtPct(summary.task_decision_accuracy), `标注 ${summary.task_decision_labeled_count ?? 0} 题 · Judge 有效 ${summary.task_decision_valid_count ?? 0} 题`],
    ["证据对应均分", summary.evidence_mean == null ? "未记录" : `${summary.evidence_mean} / 2`, `0:${evidenceDist["0"] || 0} · 1:${evidenceDist["1"] || 0} · 2:${evidenceDist["2"] || 0}`],
    ["证据完全支持率", fmtPct(summary.evidence_fully_supported_rate), "证据 Judge = 2"],
    ["端到端测评总时延（不含 Judge）", fmtMs(summary.benchmark_e2e_latency_excluding_judge_ms), "身份/关系及图片导入开始至全部 QA 完成，已扣除 Judge 时延"],
    ["完全准确率", fmtPct(summary.exact_accuracy), "Judge 评分为 2 的比例"],
    ["核心准确率", fmtPct(summary.core_accuracy), "Judge 评分为 1 或 2 的比例"],
    ["LLM TTFT 均值", fmtMs(summary.llm_ttft_ms_mean), "首 token 响应时间"],
    ["LLM 生成速度", summary.llm_tokens_per_second_mean == null ? "-" : `${Number(summary.llm_tokens_per_second_mean).toFixed(1)} token/s`, "主 Agent 平均生成速度"],
  ];
  const face = summary.face_clustering;
  if (face?.available) {
    const gallery = face.identity_gallery_available === true;
    const inclusive = gallery || face.image_identity_link_available === true;
    const f1 = gallery ? face.identity_gallery_f1 : face.image_identity_link_f1;
    const precision = gallery ? face.identity_gallery_precision : face.image_identity_link_precision;
    const recall = gallery ? face.identity_gallery_recall : face.image_identity_link_recall;
    rows.push([gallery ? "人脸聚类 F1（身份库对齐）" : (inclusive ? "人脸聚类 F1（图像级身份关联）" : "人脸聚类 F1"), fmtPct(inclusive ? f1 : face.same_person_pair_f1), gallery
      ? `身份库命中 ${face.identity_gallery_true_positive_count ?? 0} · 可评估图像 ${face.identity_gallery_evaluable_image_count ?? 0}`
      : inclusive
        ? `同身份图像对 ${face.image_identity_link_true_positive_count ?? 0} · 可评估图像 ${face.image_identity_link_evaluable_image_count ?? 0}`
      : `同人脸对 ${face.true_positive_pair_count ?? 0} · 可评估人脸 ${face.evaluable_face_count ?? 0}`, true]);
    rows.push([gallery ? "人脸聚类 P / R（身份库对齐）" : (inclusive ? "人脸聚类 P / R（图像级）" : "人脸聚类 P / R"), `${fmtPct(inclusive ? precision : face.same_person_pair_precision)} / ${fmtPct(inclusive ? recall : face.same_person_pair_recall)}`, gallery
      ? "每张多脸图逐人检测，检测框通过 seeded confirmed cluster 对齐身份库后与 GT 身份集合比较"
      : inclusive
        ? "多脸图按 GT 身份集合与跨图共享 cluster 的一致性计算；不猜测框与身份的对应关系"
      : "仅纳入身份标注与检测脸可无歧义对应的样本", true]);
    if (inclusive && face.same_person_pair_f1 != null) {
      rows.push(["严格单脸 pairwise F1", fmtPct(face.same_person_pair_f1), `严格可评估人脸 ${face.evaluable_face_count ?? 0}；多脸图不纳入此项`]);
    }
  } else if (face?.reason) {
    rows.push(["人脸聚类质量", "未评分", face.reason]);
  } else if (activeRun.value?.status === "running") {
    // Face-clustering quality is computed during aggregate after QA finishes.
    // Keep a visible placeholder while the run is active instead of silently
    // hiding the metric and making the dashboard look incomplete.
    rows.push(["人脸聚类质量", "待汇总", "QA 测评完成后计算同人脸对 P / R / F1；当前批次仍在运行"]);
  }
  const gpu = summary.gpu_metrics;
  if (gpu?.samples_count) {
    rows.push(["GPU 采样", `${gpu.samples_count} 次`, `来源 ${gpu.source || "local"} · 显存峰值 ${gpu.memory_used_mib?.peak ?? "-"} MiB`]);
  }
  return rows;
}
function tokenDistributionRows() {
  const summary = effectiveRunSummary(activeRun.value);
  return [
    ["最大输入 token", fmtTokens(summary.llm_prompt_tokens_max), "单次调用 prompt_tokens 最大值"],
    ["P95 输入 token", fmtTokens(summary.llm_prompt_tokens_p95), "95% 的调用输入不超过此值"],
    ["最大输出 token", fmtTokens(summary.llm_completion_tokens_max), "用于评估 max_tokens / max_new_tokens"],
    ["P95 输出 token", fmtTokens(summary.llm_completion_tokens_p95), "95% 的调用输出不超过此值"],
    ["最大总上下文", fmtTokens(summary.llm_context_tokens_max), "单次调用输入 token + 输出 token"],
    ["P95 总上下文", fmtTokens(summary.llm_context_tokens_p95), "用于评估 max_model_len"],
  ];
}
function tokenDistributionCount() {
  return effectiveRunSummary(activeRun.value).llm_context_samples_count ?? 0;
}
function graphQualityPrimaryRows() {
  const quality = graphQuality.value;
  if (!quality?.available) return [];
  const hasGroundTruth = quality.reference_edge_ground_truth_available === true;
  const matched = quality.reference_edge_true_positive_count ?? 0;
  const predicted = quality.evaluable_predicted_edge_count ?? 0;
  const reference = quality.reference_edge_count ?? 0;
  return [
    [hasGroundTruth ? "可验证边真实准确率（P）" : "可验证边规则准确率（P）", fmtPct(quality.reference_edge_precision), hasGroundTruth ? `${matched}/${predicted} 条构建边匹配独立 GT` : `${matched}/${predicted} 条可验证构建边与确定性参考规则一致`, true],
    [hasGroundTruth ? "可验证边真实召回率（R）" : "可验证边规则召回率（R）", fmtPct(quality.reference_edge_recall), hasGroundTruth ? `${matched}/${reference} 条独立 GT 关系被构建` : `${matched}/${reference} 条确定性参考关系被构建`, true],
    [hasGroundTruth ? "可验证边真实 F1" : "可验证边规则 F1", fmtPct(quality.reference_edge_f1), hasGroundTruth ? "独立 GT 下准确率和召回率的综合指标" : "基于源字段和建图规则的工程回归指标，不代表独立人工真值", true],
    ["节点来源可追溯率", fmtPct(quality.node_source_traceability_rate), `${quality.total_nodes ?? 0} 个节点都有来源；不代表内容或关系一定正确`, true],
  ];
}
function graphQualityDiagnosticRows() {
  const quality = graphQuality.value;
  if (!quality?.available) return [];
  return [
    ["源字段规则一致率（内部诊断）", quality.source_consistency_precision == null ? "-" : `${fmtPct(quality.source_consistency_precision)} / ${fmtPct(quality.source_consistency_recall)}`, `${quality.source_consistency_true_positive_count ?? 0}/${quality.source_consistency_reference_edge_count ?? 0}，与上方规则 P/R 同源`],
    ["节点连接率", fmtPct(quality.node_connected_rate), `${quality.connected_nodes ?? 0}/${quality.total_nodes ?? 0} 个节点至少连接一条边`],
    ["有效边率", fmtPct(quality.valid_edge_rate), `${quality.valid_edges ?? 0}/${quality.total_edges ?? 0} 条边通过端点、时序和字段校验`],
    ["边证据支持率", fmtPct(quality.edge_evidence_support_rate), `${quality.supported_edges ?? 0}/${quality.total_edges ?? 0} 条边的证据字段能找到支持；不是独立真值准确率`],
    ["边一致性率", fmtPct(quality.edge_consistency_rate), `${quality.consistent_edges ?? 0}/${quality.total_edges ?? 0} 条边通过端点、时间顺序和关系语义约束`],
    ["重复边率", fmtPct(quality.duplicate_edge_rate), `${quality.duplicate_edges ?? 0} 条重复边 / ${quality.total_edges ?? 0} 条边`],
    ["不可验证边占比", quality.total_edges ? fmtPct((quality.unverifiable_edge_count ?? 0) / quality.total_edges) : "-", "语义相似、因果等暂无独立参考真值的边，不混入真实性分数"],
  ];
}
function graphIdentityRows() {
  const identity = graphQuality.value?.face_clustering;
  if (!identity?.available) return [];
  const gallery = identity.identity_gallery_available === true;
  const inclusive = identity.image_identity_link_available === true;
  const diagnostics = Number(identity.metric_definition_version) >= 2
    ? [
      [gallery ? "身份库 GT 图像覆盖率" : (inclusive ? "GT 标注图像资产覆盖率" : "单身份 GT 图片资产覆盖率"), fmtPct(gallery ? (identity.identity_gallery_gt_asset_count / Math.max(1, identity.gt_single_image_count + identity.gt_multi_identity_image_count)) : (inclusive ? identity.image_identity_link_asset_coverage_rate : identity.asset_name_coverage_rate)),
        gallery
          ? `${identity.identity_gallery_gt_asset_count ?? 0}/${(identity.gt_single_image_count ?? 0) + (identity.gt_multi_identity_image_count ?? 0)} 张 GT 图像进入身份库评估`
          : inclusive
            ? `${identity.image_identity_link_gt_asset_count ?? 0}/${(identity.gt_single_image_count ?? 0) + (identity.gt_multi_identity_image_count ?? 0)} 张 GT 按文件名存在于当前 scope`
          : `${identity.asset_name_coverage_count ?? 0}/${identity.gt_single_image_count ?? 0} 张标注图片按文件名存在于当前 scope；缺失 ${identity.gt_assets_missing_from_scope_count ?? 0} 张`],
      ["scope 内 GT 图片恰好单脸率", fmtPct(identity.single_face_detection_of_present_assets_rate),
        `${identity.detected_single_face_count ?? 0}/${identity.asset_name_coverage_count ?? 0} 张；多脸 ${identity.detected_multiple_faces_count ?? 0} 张、无脸记录 ${identity.detected_zero_face_count ?? 0} 张`],
      ["单脸样本入簇率", fmtPct(identity.single_face_clustered_rate),
        `${identity.evaluable_face_count ?? 0}/${identity.detected_single_face_count ?? 0} 张恰好单脸 GT 已有 cluster_id`],
      ...(gallery ? [["身份库识别图像率", fmtPct(identity.identity_gallery_identified_image_rate),
        `${identity.identity_gallery_identified_image_count ?? 0}/${identity.identity_gallery_evaluable_image_count ?? 0} 张有至少一个已知身份命中`]] : []),
      ...(inclusive ? [["多脸图像入簇覆盖率", fmtPct(identity.image_identity_link_clustered_image_rate),
        `${identity.image_identity_link_clustered_image_count ?? 0}/${identity.image_identity_link_detected_image_count ?? 0} 张有检测的 GT 图像包含 cluster_id`]] : []),
    ]
    : [["覆盖率诊断", "待刷新", "当前是旧版快照，尚未区分相册缺图、零脸检测和多脸图；重启 PhotoBench 后会自动重算"]];
  return [
    [gallery ? "同一人物聚类 F1（身份库对齐）" : (inclusive ? "同一人物聚类 F1（图像级身份关联）" : "同一人物聚类 F1"), fmtPct(gallery ? identity.identity_gallery_f1 : (inclusive ? identity.image_identity_link_f1 : identity.same_person_pair_f1)), gallery
      ? "多脸图逐框经过 seeded confirmed identity cluster 对齐后，与 GT 身份集合计算 P/R/F1"
      : inclusive
        ? "多脸图通过 GT 身份集合与跨图共享 cluster 评估，不进行未经标注支持的逐框配对"
      : "准确率和召回率的综合指标", true],
    [gallery ? "同一人物聚类准确率（身份库）" : (inclusive ? "同一人物聚类准确率（图像级）" : "同一人物聚类准确率"), fmtPct(gallery ? identity.identity_gallery_precision : (inclusive ? identity.image_identity_link_precision : identity.same_person_pair_precision)),
      gallery ? "身份库命中的身份中，属于该 GT 图片真实人物的比例" : (inclusive ? "预测共享 cluster 的图像对中，GT 确实共享人物的比例" : "聚到一起的脸对中，真实同人的比例"), true],
    [gallery ? "不同人物识别错误率" : (inclusive ? "不同人物图像错误合并率" : "不同人物错误合并率"), fmtPct(gallery ? (identity.identity_gallery_false_positive_count / Math.max(1, identity.identity_gallery_true_positive_count + identity.identity_gallery_false_positive_count)) : (inclusive ? (identity.image_identity_link_false_positive_count / Math.max(1, identity.image_identity_link_true_positive_count + identity.image_identity_link_false_positive_count)) : identity.false_merge_rate)),
      gallery ? "身份库命中的身份不在该 GT 图片身份集合中的比例" : (inclusive ? "GT 不共享人物但被同一 cluster 连接的图像对比例" : "聚到一起的脸对中，实际不同人的比例"), true],
    [gallery ? "同一人物聚类召回率（身份库）" : (inclusive ? "同一人物聚类召回率（图像级）" : "同一人物聚类召回率"), fmtPct(gallery ? identity.identity_gallery_recall : (inclusive ? identity.image_identity_link_recall : identity.same_person_pair_recall)),
      gallery ? "GT 身份集合中的人物被检测并正确落入身份库 cluster 的比例" : (inclusive ? "GT 共享人物的图像对中，成功通过 cluster 连接的比例" : "真实同人的脸对中，成功聚到一起的比例"), true],
    ...(inclusive && identity.same_person_pair_f1 != null ? [["严格单脸 pairwise F1", fmtPct(identity.same_person_pair_f1), `仅 ${identity.evaluable_face_count ?? 0} 张无歧义单脸样本`]] : []),
    ...diagnostics,
  ];
}
function graphQuestionTypeRows() {
  const rows = effectiveRunSummary(activeRun.value).graph_qa_by_type;
  return Array.isArray(rows) ? rows : [];
}
function graphEdgeTypeSummary() {
  const counts = graphQuality.value?.edge_type_counts;
  if (!counts || typeof counts !== "object") return "未记录";
  const entries = Object.entries(counts).filter(([, count]) => Number(count) > 0);
  return entries.length ? entries.map(([type, count]) => `${type} ${count}`).join(" · ") : "未记录";
}
function graphEvaluableEdgeSummary() {
  const counts = graphQuality.value?.evaluable_edge_subtype_counts;
  if (!counts || typeof counts !== "object") return "未记录";
  const labels = { SHARES_RELATION: "共享关系", MENTIONS_OBJECT: "对象提及", TIME_PRECEDES: "时间先后",
    CLIP_CONTAINS: "片段包含", VIDEO_CONTAINS: "视频包含", MENTIONS_PERSON: "人物提及",
    OCCURRED_AT: "地点", CAPTURED_ON: "日期" };
  const entries = Object.entries(counts).filter(([, count]) => Number(count) > 0);
  return entries.length ? entries.map(([type, count]) => `${labels[type] || type} ${count}`).join(" · ") : "未记录";
}
function graphQualityUnavailableText() {
  const quality = graphQuality.value;
  if (graphQualityLoading.value || quality?.status === "computing" || quality?.reason === "snapshot_in_progress") {
    return "图结构快照正在生成，完成后会自动显示统计。";
  }
  return quality?.reason ? `暂无可用图结构快照：${quality.reason}` : "该运行暂无可用图结构快照。";
}
async function loadGraphQuality() {
  const runId = activeRunId.value;
  if (!runId) { graphQuality.value = null; return; }
  if (graphQualityCache.has(runId)) {
    graphQuality.value = graphQualityCache.get(runId);
    return;
  }
  if (graphQualityRequests.has(runId)) return graphQualityRequests.get(runId);
  if ((graphQualityRetryAt.get(runId) || 0) > Date.now()) return;
  graphQualityLoading.value = true;
  const request = (async () => {
    try {
      // Graph memory is a mutable shared index. Ask PhotoBench for the
      // run-bound snapshot instead of auditing whichever scope was built last.
      const payload = await api(`/api/runs/${encodeURIComponent(runId)}/graph-quality`);
      if (payload?.status === "computing") {
        // The backend single-flights this expensive snapshot. Avoid hammering
        // it on every progress poll while the one calculation is in flight.
        // A completed run does not keep the normal progress poll alive, so
        // schedule one bounded retry here as well. Otherwise a first request
        // that sees `snapshot_in_progress` leaves the graph QA and face
        // quality panels permanently empty until the user reselects the run.
        graphQualityRetryAt.set(runId, Date.now() + 5000);
        window.setTimeout(() => {
          if (!destroyed && activeRunId.value === runId) loadGraphQuality();
        }, 5100);
      } else {
        graphQualityCache.set(runId, payload);
        graphQualityRetryAt.delete(runId);
      }
      if (activeRunId.value === runId) graphQuality.value = payload;
    } catch (error) {
      graphQualityRetryAt.set(runId, Date.now() + 5000);
      if (activeRunId.value === runId) graphQuality.value = { available: false, reason: error.message };
    } finally {
      graphQualityRequests.delete(runId);
      if (activeRunId.value === runId) graphQualityLoading.value = false;
    }
  })();
  graphQualityRequests.set(runId, request);
  return request;
}
function itemCallMetrics(item) {
  return Array.isArray(item?.model_call_metrics)
    ? item.model_call_metrics.filter((metric) => metric && typeof metric === "object")
    : [];
}
function itemExecutionTrace(item) {
  return Array.isArray(item?.execution_trace)
    ? item.execution_trace.filter((step) => step && typeof step === "object")
    : [];
}
function conversationTurnNumber(value, fallback = 0) {
  const turn = Number(value);
  return Number.isInteger(turn) && turn >= 0 ? turn : fallback;
}
function agentLoopGroups(item) {
  const savedCalls = itemCallMetrics(item)
    .map((call, globalIndex) => ({ ...call, _globalCallIndex: globalIndex }))
    .filter((call) => callType(call) !== "tool_internal");
  const traceModels = itemExecutionTrace(item).filter((step) => ["model", "writer", "judge"].includes(String(step.stage || step.type || "")));
  const turns = conversationTurns(item);
  const knownTurns = new Set();
  savedCalls.forEach((call) => knownTurns.add(conversationTurnNumber(call.conversation_turn)));
  traceModels.forEach((step) => knownTurns.add(conversationTurnNumber(step.conversation_turn)));
  turns.forEach((_, index) => knownTurns.add(index));
  if (!knownTurns.size) knownTurns.add(0);
  const metricKeys = new Set(savedCalls.map((call) => `${conversationTurnNumber(call.conversation_turn)}:${call.step_id || ""}`));
  const traceOnlyCalls = traceModels.filter((step) => {
    if (String(step.call_type || "") === "tool_internal") return false;
    const key = `${conversationTurnNumber(step.conversation_turn)}:${step.step_id || ""}`;
    return (step.status === "error" || step.status === "failed" || step.parse_status === "failed") && !metricKeys.has(key);
  }).map((step, index) => {
    const turnIndex = conversationTurnNumber(step.conversation_turn);
    const turn = turns[turnIndex] || {};
    return { ...step, conversation_turn: turnIndex, call_type: step.call_type || "agent",
      status: step.status || "error", turn_outcome: step.turn_outcome || turn.turn_outcome || "model_error",
      next_step: step.next_step || turn.next_step, call_observation: step.call_observation || {
        kind: step.call_type || "agent", purpose: "模型调用在生成性能指标前失败",
        trigger: turn.message || "当前对话轮", outcome: turn.termination_reason || step.error || step.detail || "调用失败",
        source: "execution_trace_only",
      }, _traceOnly: true, _globalCallIndex: `trace-${index}` };
  });
  return [...knownTurns].sort((a, b) => a - b).map((turnIndex) => ({
    turnIndex, turn: turns[turnIndex] || {},
    calls: [...savedCalls, ...traceOnlyCalls].filter((call) => conversationTurnNumber(call.conversation_turn) === turnIndex),
  }));
}
function showAgentLoopGroupHeaders(item) {
  return conversationTurns(item).length > 1 || agentLoopGroups(item).length > 1;
}
function turnTerminationLabel(turn) {
  const reason = String(turn?.termination_reason || "");
  if (/token budget preflight failed|tokenize-current|502 Bad Gateway|tokenize.*502/i.test(reason)) return "上下文 token 预检失败（tokenize 接口 502）";
  return ({
    complete: "正常完成",
    parse_failure: "JSON 解析失败",
    model_error: "模型请求失败",
    context_blocked: "上下文或 token 预检拦截",
    tool_call_limit: "达到最大工具调用步数",
    step_limit: "达到最大执行步数",
  })[reason] || reason || "未记录";
}
function turnCompletionLabel(turn) {
  return executionState(turn).label;
}
function turnRecoveryCount(group) {
  return group?.calls?.filter((call) => callType(call) === "recovery").length ?? 0;
}
function average(values) {
  const valid = values.map(Number).filter(Number.isFinite);
  return valid.length ? valid.reduce((sum, value) => sum + value, 0) / valid.length : null;
}
function itemLlmSummary(item) {
  const calls = itemCallMetrics(item);
  if (!calls.length) return null;
  const saved = item?.llm_summary || {};
  const numeric = (key, fallback) => Number.isFinite(Number(saved[key])) ? Number(saved[key]) : fallback;
  return {
    call_count: numeric("call_count", calls.length),
    streamed_count: numeric("streamed_count", calls.filter((call) => call.streamed === true).length),
    ttft_ms_avg: numeric("ttft_ms_avg", average(calls.map((call) => call.ttft_ms))),
    total_ms_sum: numeric("total_ms_sum", calls.reduce((sum, call) => sum + (Number(call.total_ms) || 0), 0)),
    prompt_tokens_total: numeric("prompt_tokens_total", calls.reduce((sum, call) => sum + (Number(call.prompt_tokens) || 0), 0)),
    completion_tokens_total: numeric("completion_tokens_total", calls.reduce((sum, call) => sum + (Number(call.completion_tokens) || 0), 0)),
    tokens_per_second_avg: numeric("tokens_per_second_avg", average(calls.map((call) => call.tokens_per_second))),
  };
}
function fmtTokenRate(value) {
  return value == null || !Number.isFinite(Number(value)) ? "-" : `${Number(value).toFixed(1)} token/s`;
}
function callStatus(call) {
  if (call?.status === "context_budget_exceeded") return "调用前拦截";
  if (call?.status === "error") return "调用失败";
  if (call?.status && call.status !== "complete") return call.status;
  return call?.streamed === true ? "成功 · 流式" : call?.streamed === false ? "成功 · 非流式" : "未记录";
}
function callOutcome(call) {
  const outcome = call?.turn_outcome;
  if (outcome === "tool_call") return `本轮结果：调用工具 ${call?.next_step || callObservation(call).relatedTool || ""}`.trim();
  if (outcome === "final_answer") return "本轮结果：正常回答结束";
  if (outcome === "parse_failure") return "本轮结果：JSON 解析失败";
  if (outcome === "model_error") return "本轮结果：模型请求失败";
  if (outcome === "context_blocked") return "本轮结果：上下文或 token 预检拦截";
  if (outcome === "step_limit") return "本轮结果：达到最大执行步数";
  return "本轮结果：历史记录未保存";
}
function callOutcomeClass(call) {
  return ["parse_failure", "model_error", "context_blocked", "step_limit"].includes(call?.turn_outcome)
    ? "outcome-failed" : call?.turn_outcome ? "outcome-ok" : "";
}
function callType(call) {
  return call?.call_type || call?.call_observation?.kind || "legacy";
}
function callTypeLabel(call) {
  if (call?.call_observation?.label) return call.call_observation.label;
  return ({
    planner: "Agent 2.0 目标分解与规划",
    agent: "Agent 决策 / 回答",
    recovery: "Agent 恢复调用",
    writer: "最终回答重写",
    faithfulness_judge: "L2 事实一致性检查",
    tool_internal: "工具内部模型调用",
    legacy: "历史模型调用",
  })[callType(call)] || call.call_type;
}
function callTypeDescription(call) {
  if (call?.call_observation?.purpose) return call.call_observation.purpose;
  return ({
    planner: "解析用户目标并声明最小充分证据需求（TaskState/EvidenceLedger）",
    agent: "模型选择工具或直接生成回答",
    recovery: "由解析失败、重复工具或 Guard 纠正触发",
    writer: "仅按受控事实重写最终回答，不调用工具",
    faithfulness_judge: "检查回答与工具事实是否一致，不调用工具",
    tool_internal: "工具执行过程中调用模型完成视觉识别或 OCR",
    legacy: "旧记录未保存调用类型",
  })[callType(call)] || "后端记录的模型调用类型";
}
function showToolBranch(call) {
  return !["planner", "writer", "faithfulness_judge", "tool_internal"].includes(callType(call));
}
function noToolLabel(call) {
  if (callType(call) === "agent") return "该调用直接生成回答，未触发工具。";
  if (callType(call) === "recovery") return "该恢复调用未触发工具。";
  return "该历史调用没有可绑定的工具记录。";
}
function callBudget(call) {
  const prompt = Number(call?.preflight_prompt_tokens ?? call?.prompt_tokens);
  const output = Number(call?.effective_max_tokens);
  const limit = Number(call?.max_model_len);
  if (![prompt, output, limit].every(Number.isFinite)) return "-";
  return `${prompt} + ${output} / ${limit}`;
}
function callObservation(call) {
  const observation = call?.call_observation || {};
  const source = ({
    backend_recorded: "后端直接记录",
    historical_trace_aligned: "历史执行轨迹确定性对齐",
    historical_unresolved: "历史记录信息不足",
    execution_trace_only: "失败执行轨迹",
  })[observation.source] || observation.source || "未记录";
  return {
    purpose: observation.purpose || "未记录",
    trigger: observation.trigger || "未记录",
    outcome: observation.outcome || "未记录",
    source,
    relatedTool: observation.related_tool || "-",
    parentStep: observation.parent_step_id || call?.parent_step_id || "-",
  };
}
function itemToolTrace(item) {
  return Array.isArray(item?.tool_trace)
    ? item.tool_trace.filter((trace) => trace && typeof trace === "object")
    : [];
}
function itemDetail(summary) { return qaDetails[summary?.index] || null; }

function toolExecutionSteps(item) {
  return itemExecutionTrace(item).filter((step) => String(step.stage || step.type || "") === "tool");
}

function callIndexForStep(item, stepId, conversationTurn) {
  if (!stepId) return null;
  const calls = itemCallMetrics(item);
  const exact = calls.findIndex((call) => String(call.step_id || "") === String(stepId)
    && (conversationTurn == null || conversationTurnNumber(call.conversation_turn) === conversationTurnNumber(conversationTurn)));
  if (exact >= 0) return exact;
  const fallback = calls.findIndex((call) => String(call.step_id || "") === String(stepId));
  return fallback >= 0 ? fallback : null;
}

function normalizedToolCallIndex(item, trace, traceIndex) {
  const parentStepId = trace?.parent_step_id;
  const parentIndex = callIndexForStep(item, parentStepId, trace?.conversation_turn);
  if (parentIndex != null) return parentIndex;
  const sameTurnTraces = itemToolTrace(item).filter((candidate) => trace?.conversation_turn == null
    || conversationTurnNumber(candidate.conversation_turn) === conversationTurnNumber(trace.conversation_turn));
  const localTraceIndex = sameTurnTraces.indexOf(trace);
  const executionStep = toolExecutionSteps(item).filter((step) => trace?.conversation_turn == null
    || conversationTurnNumber(step.conversation_turn) === conversationTurnNumber(trace.conversation_turn))[localTraceIndex >= 0 ? localTraceIndex : traceIndex];
  const executionIndex = callIndexForStep(item, executionStep?.parent_step_id, trace?.conversation_turn);
  if (executionIndex != null) return executionIndex;
  const savedIndex = Number(trace?.model_call_index);
  return Number.isInteger(savedIndex) ? savedIndex : null;
}

function toolsForGroupedCall(item, call) {
  const turnIndex = conversationTurnNumber(call?.conversation_turn);
  return itemToolTrace(item).filter((trace, traceIndex) => {
    const sameTurn = conversationTurnNumber(trace.conversation_turn) === turnIndex;
    return sameTurn && normalizedToolCallIndex(item, trace, traceIndex) === Number(call?._globalCallIndex);
  });
}

function toolDurationMs(trace) {
  const duration = Number(trace?.duration_ms);
  if (Number.isFinite(duration) && duration >= 0) return duration;
  const latency = Number(trace?.latency_s);
  return Number.isFinite(latency) && latency >= 0 ? latency * 1000 : null;
}

function toolLatencySegments(trace) {
  const timing = trace?.retrieval_timing || {};
  const segments = [];
  const add = (kind, label, value) => {
    const ms = Number(value);
    if (Number.isFinite(ms) && ms > 0) segments.push({ kind, label, ms });
  };
  add("nested-tool", "查询构建", timing.query_build_ms);
  Object.entries(timing.channels || {}).forEach(([channel, value]) => {
    add("nested-tool", channel, value?.latency_ms);
  });
  add("nested-tool", "融合", timing.fusion_ms);
  add("nested-tool", "后处理", timing.postprocess_ms);
  const total = Number(timing.total_ms);
  const accounted = segments.reduce((sum, segment) => sum + segment.ms, 0);
  if (Number.isFinite(total) && total > accounted + 0.5) segments.push({ kind: "other", label: "工具其他", ms: total - accounted });
  return segments;
}

function callAgentLoopTiming(item, call) {
  const modelMs = Number(call?.total_ms);
  const validModelMs = Number.isFinite(modelMs) && modelMs >= 0 ? modelMs : null;
  const saved = call?.agent_loop_timing || {};
  const tools = toolsForGroupedCall(item, call);
  const toolMs = Number.isFinite(Number(saved.tool_ms))
    ? Number(saved.tool_ms)
    : tools.reduce((sum, trace) => sum + (toolDurationMs(trace) || 0), 0);
  const ttftMs = Number(call?.ttft_ms);
  const validTtftMs = Number.isFinite(ttftMs) && ttftMs >= 0 ? ttftMs : null;
  const generationMs = Number.isFinite(Number(saved.model_generation_ms))
    ? Number(saved.model_generation_ms)
    : validModelMs == null ? null : Math.max(0, validModelMs - (validTtftMs || 0));
  const totalMs = Number.isFinite(Number(call?.agent_loop_total_ms))
    ? Number(call.agent_loop_total_ms)
    : validModelMs == null ? null : validModelMs + toolMs;
  return { modelMs: validModelMs, ttftMs: validTtftMs, generationMs, toolMs, totalMs, tools };
}

function callLatencySegments(item, call) {
  const timing = callAgentLoopTiming(item, call);
  const segments = [];
  if (timing.ttftMs != null && timing.ttftMs > 0) segments.push({ kind: "ttft", label: "TTFT", ms: timing.ttftMs });
  if (timing.generationMs != null && timing.generationMs > 0) segments.push({ kind: "generation", label: "模型生成 / 回答", ms: timing.generationMs });
  timing.tools.forEach((trace, index) => {
    const ms = toolDurationMs(trace);
    if (ms != null && ms > 0) segments.push({ kind: "tool", label: trace.tool || `工具 ${index + 1}`, ms, trace });
  });
  const accounted = segments.reduce((sum, segment) => sum + segment.ms, 0);
  if (timing.totalMs != null && timing.totalMs > accounted + 0.5) {
    segments.push({ kind: "other", label: "Agent 编排", ms: timing.totalMs - accounted });
  }
  return segments;
}

function latencySegmentTitle(segment) {
  const detail = segment.trace?.retrieval_timing?.total_ms != null
    ? `\n检索内部耗时 ${fmtMs(segment.trace.retrieval_timing.total_ms)}` : "";
  return `${segment.label} · ${fmtMs(segment.ms)}${detail}`;
}

function qaLatencySegments(item) {
  const loops = agentLoopGroups(item).flatMap((group) => group.calls).map((call, index) => {
    const timing = callAgentLoopTiming(item, call);
    return timing.totalMs == null ? null : {
      kind: "agent-loop",
      label: `Agent Loop ${index + 1} · ${callTypeLabel(call)}`,
      shortLabel: `Loop ${index + 1}`,
      ms: timing.totalMs,
      call,
    };
  }).filter(Boolean);
  const breakdown = itemTimingBreakdown(item);
  const loopMs = loops.reduce((sum, segment) => sum + segment.ms, 0);
  const judgeMs = Number(breakdown.judge_ms);
  if (Number.isFinite(judgeMs) && judgeMs > 0) loops.push({ kind: "judge", label: "Judge", shortLabel: "Judge", ms: judgeMs });
  const judgeQueueMs = Number(breakdown.judge_queue_wait_ms);
  if (Number.isFinite(judgeQueueMs) && judgeQueueMs > 0) {
    loops.push({ kind: "judge-queue", label: "Judge 排队 / 编排", shortLabel: "Judge 排队", ms: judgeQueueMs });
  }
  const wallMs = Number(breakdown.wall_clock_ms);
  const otherMs = Number.isFinite(wallMs)
    ? Math.max(0, wallMs - loopMs - (Number.isFinite(judgeMs) ? judgeMs : 0) - (Number.isFinite(judgeQueueMs) ? judgeQueueMs : 0))
    : 0;
  if (otherMs > 0.5) loops.push({ kind: "other", label: "其他未归因时延", shortLabel: "其他", ms: otherMs });
  return loops;
}

function latencySegmentStyle(segment, segments) {
  const total = segments.reduce((sum, value) => sum + value.ms, 0) || 1;
  return { flex: `${Math.max(segment.ms, 0.1)} 1 0%`, "--segment-share": `${(segment.ms / total) * 100}%` };
}

function runtimeDebugTurns(item) {
  const turns = item?.runtime_turns || [];
  return Array.isArray(turns) ? turns.filter((turn) => turn && Array.isArray(turn.debug_trace)) : [];
}

function debugTraceForTurn(item, turnIndex) {
  const turn = runtimeDebugTurns(item).find((t) => Number(t?.index) === Number(turnIndex));
  return turn?.debug_trace || [];
}

function debugStepForCall(item, call, callIndexInTurn) {
  const turnIndex = conversationTurnNumber(call?.conversation_turn);
  const steps = debugTraceForTurn(item, turnIndex);
  const ctype = callType(call);
  const wantType = ctype === "faithfulness_judge" ? "judge" : "model";
  const candidates = steps.filter((s) => {
    if (wantType === "judge") return s?.type === "judge";
    if (wantType === "model") return s?.type === "model" && (s?.call_type || "agent") !== "faithfulness_judge";
    return false;
  });
  return candidates[callIndexInTurn] || null;
}

function debugStepsForCall(item, call) {
  const turnIndex = conversationTurnNumber(call?.conversation_turn);
  const steps = debugTraceForTurn(item, turnIndex);
  const ctype = callType(call);
  if (ctype === "faithfulness_judge") return steps.filter((s) => s?.type === "judge");
  if (ctype === "agent" || ctype === "recovery") {
    return steps.filter((s) => s?.type === "model" && (s?.call_type || "agent") !== "faithfulness_judge");
  }
  return [];
}

function debugStepForCallInGroup(item, group, call) {
  const turnIndex = conversationTurnNumber(call?.conversation_turn);
  const steps = debugTraceForTurn(item, turnIndex);
  if (call?.step_id) {
    const matched = steps.find((s) => s?.step_id === call.step_id);
    if (matched) return matched;
  }
  const ctype = callType(call);
  if (ctype === "planner") {
    return steps.find((s) => s?.type === "planner" || s?.call_type === "planner") || null;
  }
  if (ctype === "faithfulness_judge") {
    const judges = steps.filter((s) => s?.type === "judge" || s?.call_type === "faithfulness_judge");
    const index = group.calls.filter((c) => callType(c) === "faithfulness_judge").indexOf(call);
    return judges[index] || null;
  }
  if (ctype === "agent" || ctype === "recovery") {
    const models = steps.filter((s) => s?.type === "model" && s?.call_type !== "faithfulness_judge" && s?.call_type !== "planner");
    const index = group.calls.filter((c) => ["agent", "recovery"].includes(callType(c))).indexOf(call);
    return models[index] || null;
  }
  // Fallback match by index if legacy
  const index = group.calls.indexOf(call);
  const models = steps.filter((s) => s?.type === "model" || s?.type === "planner" || s?.type === "judge");
  return models[index] || null;
}

function debugToolsForCall(item, group, call) {
  const modelStep = debugStepForCallInGroup(item, group, call);
  if (!modelStep) return [];
  const turnIndex = conversationTurnNumber(call?.conversation_turn);
  const steps = debugTraceForTurn(item, turnIndex);
  return steps.filter((s) => s?.type === "tool"
    && String(s?.parent_step_id) === String(modelStep.step_id));
}
function debugPromptAnnotations(item, group, call) {
  return debugStepForCallInGroup(item, group, call)?.prompt_annotations || [];
}
function getAgent2Trace(item) {
  if (!item) return null;
  if (item.agent2_trace && (item.agent2_trace.task_declaration || item.agent2_trace.task_state || item.agent2_trace.evidence_ledger)) {
    return item.agent2_trace;
  }
  const turns = item.runtime_turns || item.conversation || [];
  for (const t of turns) {
    if (t?.agent2_trace && (t.agent2_trace.task_declaration || t.agent2_trace.task_state || t.agent2_trace.evidence_ledger)) {
      return t.agent2_trace;
    }
  }
  return item.agent2_trace || null;
}
function getAgent2Requirements(item) {
  const trace = getAgent2Trace(item);
  if (!trace) return [];
  return trace.task_state?.requirements || trace.task_declaration?.requirements || trace.requirements || [];
}
function getAgent2LedgerEntries(item) {
  const trace = getAgent2Trace(item);
  if (!trace) return [];
  return trace.evidence_ledger?.entries || [];
}

function formatAgent2EvidenceValue(value) {
  if (value == null) return "已记录证据，但没有提取出的文本值";
  if (typeof value === "string") {
    const text = value.trim();
    if (!text) return "已记录证据，但没有提取出的文本值";
    try {
      return JSON.stringify(JSON.parse(text), null, 2);
    } catch {
      return value;
    }
  }
  const formatted = JSON.stringify(value, null, 2);
  return formatted ?? String(value);
}

function agent2EvidenceRows(value) {
  return Math.max(3, formatAgent2EvidenceValue(value).split("\n").length);
}
function agent2EvidenceHeight(value) {
  return String(agent2EvidenceRows(value) * 15.5 + 16) + "px";
}
function agent2RequirementStatusLabel(status) {
  return ({ satisfied: "已满足", partially_supported: "部分满足", open: "待满足", failed: "未满足" })[status] || status || "未记录";
}
function agent2RequirementStatusClass(status) {
  return status === "satisfied" ? "status-complete" : status === "partially_supported" ? "status-partial" : "status-failed";
}
function agent2EvidenceTypeLabel(type) {
  return ({
    // Legacy aliases kept for historical runs.
    visual: "视觉", text: "文字 / OCR", temporal: "时间", geographic: "地点",
    identity: "身份", semantic: "语义",
    // Canonical Agent2 evidence types emitted by the backend.
    memory_asset: "照片 / 视频", memory_reference: "记忆引用",
    visual_observation: "视觉观察", visible_text: "可见文字 / OCR",
    structured_fact: "结构化事实", temporal_metadata: "时间元数据",
    location_metadata: "地点元数据", confirmed_identity: "已确认身份",
    user_statement: "用户陈述", transcript: "转写文本",
  })[type] || type || "证据";
}
function agent2RequirementSummary(item) {
  const requirements = getAgent2Requirements(item);
  const satisfied = requirements.filter((req) => req.status === "satisfied").length;
  const partial = requirements.filter((req) => req.status === "partially_supported").length;
  return `${satisfied}/${requirements.length} 项满足${partial ? `，${partial} 项部分满足` : ""}`;
}
function agent2DecisionSummary(item) {
  const decisions = getAgent2Trace(item)?.planner_decisions || [];
  const decision = decisions[decisions.length - 1] || {};
  return decision.status === "accepted" ? "规划已接受" : decision.status === "fallback" ? "规划失败，已回退主流程" : decision.status || "已记录";
}

function attributionSummary(item) {
  const attribution = item?.attribution || {};
  const layers = attribution.layers || {};
  return {
    primary: attribution.primary || "未归因",
    failed: Object.entries(layers).filter(([, status]) => status === "fail").map(([key]) => key).join(" / ") || "无",
  };
}
function attributionLabel(key) {
  return ({ R: "检索", V: "视觉", O: "OCR", T: "工具", S: "综合", G: "Guard", J: "Judge", PASS: "通过" })[key] || key;
}
function attributionClass(status) { return status === "fail" ? "score-0" : status === "pass" ? "score-2" : "score-none"; }
function toolsForCall(item, callIndex) {
  return itemToolTrace(item).filter((trace) => Number(trace.model_call_index) === callIndex);
}
function unboundTools(item) {
  return itemToolTrace(item).filter((trace, traceIndex) => normalizedToolCallIndex(item, trace, traceIndex) == null);
}
function toolStatusLabel(trace) {
  const status = trace?.status || "未知";
  return trace?.reason ? `${status} · ${trace.reason}` : status;
}
function toolPerformanceRows() {
  const performance = effectiveRunSummary(activeRun.value).tool_performance || {};
  return Object.entries(performance).map(([name, metrics]) => ({ name, ...metrics }));
}
function deliveryBreakdown() {
  const bd = effectiveRunSummary(activeRun.value).delivery_breakdown;
  if (!bd || (!bd.deterministic_delivery_count && !bd.ocr_partial_count)) return null;
  return {
    detCount: bd.deterministic_delivery_count || 0,
    detKinds: Object.entries(bd.deterministic_delivery_kinds || {}),
    ocrPartial: bd.ocr_partial_count || 0,
    ocrReasons: Object.entries(bd.ocr_partial_reasons || {}),
  };
}
function shortHash(value) { return value ? String(value).slice(0, 12) : "-"; }
function snapshotSummary(snapshot) {
  const gpu = snapshot?.gpu?.[0] || {};
  const process = snapshot?.process_memory || {};
  const state = snapshot?.manager || {};
  return {
    time: snapshot?.captured_at ? fmtDate(snapshot.captured_at) : "未记录",
    model: state.served_model_name || state.profile || "-",
    gpu: gpu.name || "-",
    memory: fmtMemory(gpu.memory_used_mib),
    processMemory: fmtMemory(process.process_memory_used_mib),
  };
}
function guardSummary(item) {
  const guard = item?.guard_debug || {};
  const codes = Array.isArray(guard.l1_codes) ? guard.l1_codes : [];
  const det = guard.deterministic_delivery || {};
  const delivery = item?.delivery_status || {};
  return {
    recorded: Boolean(Object.keys(guard).length || item?.termination_reason || item?.agent_status),
    status: guard.status || item?.agent_status || "-",
    termination: normalizedTerminationReason(item),
    recoveries: guard.recovery_attempts ?? "-",
    codes: codes.length ? codes.join("、") : "无",
    deterministic: det.rendered ? (det.kind || "未知") : "",
    ocrPartial: delivery.ocr_partial ? (delivery.ocr_partial_reason || "unknown") : "",
  };
}
function normalizedTerminationReason(item) {
  const reason = String(item?.agent_reason || "");
  if (/token budget preflight failed|tokenize-current|502 Bad Gateway/i.test(reason)) return "上下文 token 预检失败（tokenize 接口 502）";
  return item?.guard_debug?.termination_reason || item?.termination_reason || "-";
}
function terminationDisplayLabel(item) {
  const raw = normalizedTerminationReason(item);
  if (raw === "complete" || (raw === "-" && executionState(item).key === "complete")) return "正常完成";
  if (raw === "-") return "未记录";
  return turnTerminationLabel({ ...item, termination_reason: raw });
}
function hasToolTrace(item) {
  return Object.prototype.hasOwnProperty.call(item || {}, "tool_trace")
    || item?.timing_breakdown?.tool_trace_recorded === true;
}
function itemTimingBreakdown(item) {
  const saved = item?.timing_breakdown || {};
  const modelMs = saved.model_ms ?? item?.llm_summary?.total_ms_sum ?? null;
  return {
    wall_clock_ms: saved.wall_clock_ms ?? item?.wall_clock_ms ?? null,
    agent_wall_ms: saved.agent_wall_ms ?? null,
    model_ms: modelMs,
    tool_ms: saved.tool_ms ?? null,
    judge_ms: saved.judge_ms ?? item?.judge_ms ?? null,
    other_ms: saved.other_ms ?? null,
  };
}
function retrievalChannelRows(item) {
  const rows = [];
  itemToolTrace(item).forEach((trace, toolIndex) => {
    const timing = trace.retrieval_timing || {};
    Object.entries(timing.channels || {}).forEach(([channel, value]) => {
      rows.push({
        key: `${toolIndex}-${channel}`,
        tool_round: toolIndex + 1,
        tool_name: trace.tool || "未知工具",
        channel,
        latency_ms: value?.latency_ms,
        embedding_ms: value?.embedding_ms,
        candidate_count: value?.candidate_count,
        status: value?.status,
      });
    });
  });
  return rows;
}
function phaseSummary(key, phase) {
  if (!phase) return "";
  if (key === "model_deploy" && phase.unload_seconds != null) return `卸载 ${fmtSeconds(phase.unload_seconds)} · 加载 ${fmtSeconds(phase.load_seconds)} · 健康检查 ${fmtSeconds(phase.health_check_seconds)}`;
  if (key === "scope_setup") return `创建 ${fmtSeconds(phaseSeconds(phase, "create_seconds"))}`;
  if (key === "identity_seed") {
    const relationshipImport = phase.family_relationship_import || {};
    const relationshipText = relationshipImport.requested
      ? ` · 关系 ${relationshipImport.imported ?? 0}/${relationshipImport.requested}`
      : "";
    return `预置 ${phase.seeded_count ?? 0} 人${relationshipText} · ${fmtSeconds(phaseSeconds(phase, "upload_seconds"))}`;
  }
  if (key === "photo_import") return `导入 ${phase.accepted_count ?? 0}/${phase.total_media ?? phase.total_photos ?? "?"} 个媒体 · 失败 ${phase.failed_count ?? 0} · ${fmtSeconds(phaseSeconds(phase, "upload_seconds"))}`;
  if (key === "pipeline_processing" && phase.progress) return `${phase.progress.processed || 0}/${phase.progress.total || 0} 资产完成 · ${phase.progress.failed || 0} 失败 · ${phase.progress.skipped || 0} 跳过 · 阶段墙钟 ${fmtSeconds(phaseSeconds(phase))}`;
  if (key === "qa_eval") {
    const p = phase.progress;
    if (p && (p.completed != null || p.in_flight != null)) {
      const agentText = `Agent ${p.agent_completed ?? p.completed ?? 0}/${p.agent_total ?? p.total ?? "?"}`;
      const judgeText = `Judge ${p.judge_completed ?? 0}/${p.judge_total ?? p.total ?? "?"}`;
      const concurrencyText = p.qa_concurrency > 1 || p.judge_concurrency > 1
        ? ` · 并发 Agent ${p.qa_concurrency ?? "-"} / Judge ${p.judge_concurrency ?? "-"}` : "";
      const failedText = phase.failed_count ? ` · 失败 ${phase.failed_count}` : "";
      return `${agentText} · ${judgeText}${failedText}${concurrencyText} · ${fmtSeconds(phaseSeconds(phase))}`;
    }
    return `${activeRun.value?.item_count || 0} 题 · ${fmtSeconds(phaseSeconds(phase))}`;
  }
  const elapsed = phaseSeconds(phase);
  if (elapsed != null) return `耗时 ${fmtSeconds(elapsed)}`;
  return "";
}
function phaseErrorDetails(phase) {
  if (!phase) return [];
  const details = Array.isArray(phase.error_details) ? phase.error_details : [];
  if (details.length) return details;
  return phase.error ? [{ sample_id: "阶段错误", status: phase.status || "failed", reason: phase.error }] : [];
}
function phaseErrorSummary(phase) {
  const count = phaseErrorDetails(phase).length;
  if (!count) return "";
  const skipped = Number(phase?.skipped_asset_count || phase?.progress?.skipped || 0);
  return skipped ? `${count} 条错误记录，${skipped} 个样本已跳过` : `${count} 条错误记录`;
}
function openImage(image) { const url = imageUrl(image); if (url) lightbox.value = { url, name: image.file_name || image.image_id || "图片" }; }
function phasePercent(phase) {
  const p = phase?.progress;
  return p?.total ? Math.min(100, Math.round(((Number(p.processed) || 0) + (Number(p.failed) || 0) + (Number(p.skipped) || 0)) / p.total * 100)) : 0;
}
function pipelineMetricRows(phase = {}) {
  const metrics = phase.pipeline_metrics || {};
  const timings = metrics.stage_timings || {};
  const imageCount = Number(metrics.image_count);
  const wallSeconds = phaseSeconds(phase);
  const hasImageCount = Number.isFinite(imageCount) && imageCount > 0;
  const averageWallSeconds = hasImageCount && Number.isFinite(wallSeconds)
    ? wallSeconds / imageCount
    : phase.average_seconds_per_photo;
  const row = (key, label) => {
    const value = timings[key];
    return value ? [label, fmtSeconds(value.mean_seconds), `总计 ${fmtSeconds(value.sum_seconds)} · P95 ${fmtSeconds(value.p95_seconds)}`] : null;
  };
  const cumulativeCallRow = (key, label) => {
    const value = timings[key];
    return value ? [
      label,
      `单次均值 ${fmtSeconds(value.mean_seconds)}`,
      `${value.count ?? "-"} 次调用累计 ${fmtSeconds(value.sum_seconds)} · P95 ${fmtSeconds(value.p95_seconds)}；并发调用可使累计耗时大于阶段墙钟`,
    ] : null;
  };
  return [
    ["评测图片墙钟均值", averageWallSeconds == null ? "未记录" : fmtSeconds(averageWallSeconds), hasImageCount ? `阶段墙钟 ${fmtSeconds(wallSeconds)} ÷ ${imageCount} 张评测图片；不含身份参考图` : "历史运行未记录评测图片数，按旧口径展示"],
    ["实际图片并发", metrics.effective_workers == null ? "-" : `${metrics.effective_workers} 路`, `配置 ${metrics.configured_workers ?? "-"} · vLLM 上限 ${metrics.vllm_max_num_seqs ?? "-"}`],
    ["事件总结", metrics.event_count == null ? "-" : `${metrics.event_summary_call_count ?? 0}/${metrics.event_count} 次`, `批次耗时 ${fmtSeconds(metrics.event_summary_wall_seconds)}`],
    cumulativeCallRow("vlm_image_description_seconds", "VLM 图片描述调用"),
    row("face_detection_seconds", "人脸检测"),
    row("image_clip_seconds", "图片 CLIP"),
    row("face_clustering_seconds", "人脸归类"),
    row("event_clustering_seconds", "事件聚类"),
    row("text_embedding_seconds", "文本 embedding"),
  ].filter(Boolean);
}

async function loadRuns() { runs.value = (await api("/api/runs")).runs || []; }
function runProgressLabel(run) {
  if (run?.mode === "build") return "—";
  const progress = run?.phases?.qa_eval?.progress;
  if (progress && progress.judge_total != null) {
    return `${progress.judge_completed ?? 0}/${progress.judge_total}`;
  }
  return `${run?.summary?.completed || 0}/${run?.summary?.total || run?.qa_count || 0}`;
}
async function loadQaPage(page = qaPage.value.page || 1) {
  if (!activeRunId.value) return;
  const runId = activeRunId.value;
  const params = new URLSearchParams({ page: String(page), page_size: String(qaPageSize.value) });
  Object.entries(qaFilters).forEach(([key, value]) => { if (value) params.set(key, value); });
  const payload = await api(`/api/runs/${encodeURIComponent(runId)}/items?${params}`);
  if (activeRunId.value !== runId) return;
  qaPage.value = payload;
  const refreshDetails = [];
  for (const summary of qaPage.value.items || []) {
    const detail = qaDetails[summary.index];
    if (detail) {
      const oldJudge = detail.judge || {};
      const newJudge = summary.judge || {};
      if (newJudge.rejudge_id && (newJudge.rejudge_id !== oldJudge.rejudge_id
        || (newJudge.status === "completed" && oldJudge.status !== "completed"))) {
        refreshDetails.push(summary.index);
      } else {
        Object.assign(oldJudge, newJudge);
      }
    }
  }
  await Promise.all(refreshDetails.map((index) => loadQaDetail(index, { force: true })));
}
async function applyQaFilters() { await loadQaPage(1); }
async function resetQaFilters() {
  Object.assign(qaFilters, { search: "", score: "", task_type: "", tag: "", angle: "", difficulty: "", answerability: "", agent_status: "", primary: "" });
  await loadQaPage(1);
}
async function loadActiveRun({ resetPage = false } = {}) {
  if (!activeRunId.value) return;
  const runId = activeRunId.value;
  const payload = await api(`/api/runs/${encodeURIComponent(runId)}`);
  if (activeRunId.value !== runId) return;
  activeRun.value = payload;
  const fallbackSummary = effectiveRunSummary(activeRun.value);
  runs.value = runs.value.map((run) => run.run_id === activeRunId.value
    ? { ...run, summary: { ...(run.summary || {}), ...fallbackSummary } }
    : run);
  if (resetPage) {
    qaPage.value = { items: [], page: 1, page_size: qaPageSize.value, total: 0, pages: 1 };
    Object.keys(qaDetails).forEach((key) => delete qaDetails[key]);
    openQaItems.clear();
    Object.keys(reviewDrafts).forEach((key) => delete reviewDrafts[key]);
    const reviewPayload = await api(`/api/runs/${encodeURIComponent(runId)}/reviews`);
    Object.assign(reviewDrafts, reviewPayload.reviews || {});
  }
  await loadQaPage(resetPage ? 1 : qaPage.value.page);
  loadGraphQuality();
}
function reviewFor(summary) {
  const qaId = String(summary?.qa_id || "");
  if (!reviewDrafts[qaId]) reviewDrafts[qaId] = { verdict: "", note: "" };
  return reviewDrafts[qaId];
}
async function saveReviews() {
  if (!activeRunId.value || reviewSaving.value) return;
  reviewSaving.value = true;
  try {
    const reviews = Object.fromEntries(Object.entries(reviewDrafts).filter(([, value]) => value?.verdict));
    await post(`/api/runs/${encodeURIComponent(activeRunId.value)}/reviews`, { reviews });
    await loadQaPage(qaPage.value.page);
  } finally { reviewSaving.value = false; }
}
async function loadQaDetail(index, { force = false } = {}) {
  if (!activeRunId.value || loadingQaItems.has(index) || (!force && qaDetails[index])) return;
  const runId = activeRunId.value;
  loadingQaItems.add(index);
  try {
    const payload = await api(`/api/runs/${encodeURIComponent(runId)}/items/${index}`);
    if (activeRunId.value === runId) qaDetails[index] = payload.item;
  } finally { loadingQaItems.delete(index); }
}
async function toggleQa(summary) {
  const index = summary.index;
  if (openQaItems.has(index)) { openQaItems.delete(index); return; }
  openQaItems.add(index);
  await loadQaDetail(index);
}
async function changeQaPage(page) {
  const target = Math.max(1, Math.min(Number(page), qaPage.value.pages || 1));
  if (target === qaPage.value.page) return;
  await loadQaPage(target);
  document.querySelector("#qa-results")?.scrollIntoView({ behavior: "smooth", block: "start" });
}
async function changeQaPageSize() { await loadQaPage(1); }
async function selectRun(run) { activeRunId.value = run.run_id; await loadActiveRun({ resetPage: true }); document.querySelector("#detail-region")?.scrollIntoView({ behavior: "smooth", block: "start" }); }
async function loadProfiles() {
  if (!vllmManagerUrl.value.trim()) {
    profiles.value = [];
    selectedModels.forEach((modelId) => {
      if (modelId !== "__current__") selectedModels.delete(modelId);
    });
    return;
  }
  profiles.value = (await post("/api/profiles", {
    vllm_target_id: vllmTargetId.value,
    vllm_manager_url: vllmManagerUrl.value.trim(),
    model_base_url: modelEndpointUserEdited.value ? modelEndpoint.value.trim() : "",
  })).profiles || [];
}
function onModelManagerInput() {
  markConnectionConfigDirty();
  profiles.value = [];
  currentModelInfo.value = null;
  currentModelPopoverOpen.value = false;
  currentModelError.value = "";
  modelTestState.value = "idle";
  modelTestMessage.value = "";
  selectedModels.delete("__current__");
  selectedModels.forEach((modelId) => {
    if (modelId !== "__current__") selectedModels.delete(modelId);
  });
}
function markConnectionConfigDirty() {
  if (loading.value) return;
  connectionConfigState.value = "dirty";
  connectionConfigMessage.value = "有未保存修改";
}
async function saveConnectionConfig() {
  if (connectionConfigState.value === "saving") return;
  connectionConfigState.value = "saving";
  connectionConfigMessage.value = "正在保存…";
  try {
    const result = await api("/api/config", {
      method: "POST",
      body: JSON.stringify({
        sentrix_url: sentrixUrl.value.trim(),
        judge_url: judgeUrl.value.trim(),
        judge_model: judgeModel.value.trim(),
        vllm_manager_url: vllmManagerUrl.value.trim(),
        model_base_url: modelEndpoint.value.trim(),
        endpoint_model: selectedEndpointModel.value,
        judge_provider_id: judgeProviderId.value,
        ...(judgeApiKeyDirty.value ? { judge_api_key: judgeApiKey.value } : {}),
      }),
    });
    const saved = result.runtime_config || {};
    config.value = { ...config.value, runtime_config: saved,
      default_sentrix_url: saved.sentrix_url,
      default_judge_url: saved.judge_url,
      default_vllm_api_url: saved.vllm_manager_url,
      default_vllm_model_base_url: saved.model_base_url };
    sentrixUrl.value = saved.sentrix_url || sentrixUrl.value;
    judgeUrl.value = saved.judge_url || judgeUrl.value;
    judgeModel.value = saved.judge_model || judgeModel.value;
    judgeApiKey.value = "";
    judgeApiKeyDirty.value = false;
    vllmManagerUrl.value = saved.vllm_manager_url || "";
    modelEndpoint.value = saved.model_base_url || "";
    modelEndpointUserEdited.value = Boolean(saved.model_base_url);
    selectedEndpointModel.value = saved.endpoint_model || "";
    endpointModels.value = [];
    currentModelInfo.value = null;
    currentModelError.value = "";
    modelTestState.value = "idle";
    modelTestMessage.value = "";
    connectionConfigState.value = "saved";
    connectionConfigMessage.value = `已保存 · ${new Date().toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" })}`;
    if (modelEndpoint.value.trim()) await loadCurrentModel({ openPopover: false });
  } catch (e) {
    connectionConfigState.value = "error";
    connectionConfigMessage.value = `保存失败：${e.message}`;
  }
}
const PROMPT_KIND_LABELS = { answer_quality: "回答质量", task_decision: "任务判断", evidence: "证据核验" };
function promptKindMeta(kind) { return judgePromptKinds.value.find((k) => k.kind === kind) || null; }
async function loadJudgePrompts() {
  try {
    const data = await fetch("/api/judge-prompts").then((r) => r.json());
    judgePromptKinds.value = data.kinds || [];
    for (const meta of judgePromptKinds.value) {
      promptDrafts[meta.kind] = meta.custom || meta.default || "";
    }
    rejudgePrompt.value = promptDrafts.answer_quality || "";
  } catch { /* keep defaults from config */ }
}
function switchPromptKind(kind) {
  activePromptKind.value = kind;
  rejudgePrompt.value = promptDrafts[kind] || "";
}
function resetJudgePrompt() {
  const meta = promptKindMeta(activePromptKind.value);
  rejudgePrompt.value = meta?.default || config.value?.judge_prompt || "";
  promptDrafts[activePromptKind.value] = rejudgePrompt.value;
}

const exportScores = ref(["0", "1", "2"]);
const deleteScopeAfterRun = ref(false);

// ---- 工作模式：全链路 / 复用相册测评 / 构建相册 ----
const RUN_MODES_UI = [
  { id: "full", label: "全链路测试", hint: "新建相册 → 身份预置 → 导入图片和视频 → 流水线处理 → QA 测评", button: "启动评测" },
  { id: "reuse", label: "复用相册测评", hint: "选择后端已有相册，跳过导入与处理，直接 QA 测评（相册不会被删除）", button: "启动复用测评" },
  { id: "build", label: "构建相册", hint: "新建相册并导入图片/视频完成数据处理；产物相册保留供复用或在线测试，不做 QA 测评", button: "启动相册构建" },
];
const runMode = ref("full");
const runModeMeta = computed(() => RUN_MODES_UI.find((m) => m.id === runMode.value) || RUN_MODES_UI[0]);
const memorySpaces = ref([]);
const reuseBases = ref([]);
const memorySpacesLoading = ref(false);
const existingScopeId = ref("");
const selectedReuseBaseId = ref("");
const scopeAlbumHint = ref("");
async function loadMemorySpaces() {
  memorySpacesLoading.value = true;
  try {
    const data = await fetch(`/api/memory-spaces?sentrix_url=${encodeURIComponent(sentrixUrl.value)}`).then((r) => r.json());
    memorySpaces.value = data.spaces || [];
    reuseBases.value = data.reuse_bases || [];
  } catch { memorySpaces.value = []; reuseBases.value = []; }
  finally { memorySpacesLoading.value = false; }
}
const selectedSpace = computed(() => memorySpaces.value.find((s) => s.id === existingScopeId.value) || null);
const selectedReuseBase = computed(() => reuseBases.value.find((item) => item.base_id === selectedReuseBaseId.value) || null);
// 隐藏"照片"下拉后，QA 题目与 GT 对照的数据集依据从所选现有相册自动推断：
// 优先查 run 历史（scope_id → album_id），再按相册名子串匹配（长 id 优先），最后保持当前选择。
function inferAlbumForScope(space) {
  if (!space) return null;
  const run = runs.value.find((r) => r.scope_id === space.id && (r.mode === "build" || r.mode === "full" || !r.mode));
  if (run?.album_id && manifests.value.some((m) => m.album_id === run.album_id)) return run.album_id;
  const name = String(space.name || "");
  const candidates = manifests.value.map((m) => m.album_id).filter((id) => name.includes(id));
  if (candidates.length) return candidates.sort((a, b) => b.length - a.length)[0];
  return null;
}
watch(existingScopeId, () => {
  if (runMode.value !== "reuse") return;
  const album = inferAlbumForScope(selectedSpace.value);
  if (album) {
    selectedAlbum.value = album;
    scopeAlbumHint.value = `题目对照数据集：${album}`;
  } else {
    scopeAlbumHint.value = "未能自动匹配数据集，题目对照沿用当前 QA 数据集所属数据";
  }
});
watch(selectedReuseBaseId, () => {
  if (runMode.value !== "reuse") return;
  const base = selectedReuseBase.value;
  if (!base) return;
  existingScopeId.value = base.scope_id || "";
  if (base.album_id) {
    selectedAlbum.value = base.album_id;
    const manifest = manifests.value.find((item) => item.album_id === base.album_id);
    const options = Object.keys(manifest?.qa_sets || {});
    if (options.length && !options.includes(selectedQa.value)) selectedQa.value = options[0];
  }
  scopeAlbumHint.value = `复用基座：${base.album_id} · ${base.model_profile}；QA 数据集：${base.album_id}`;
});
function onModeChange() {
  existingScopeId.value = "";
  selectedReuseBaseId.value = "";
  scopeAlbumHint.value = "";
  if (runMode.value === "reuse" && !memorySpaces.value.length) loadMemorySpaces();
}
const startDisabledReason = computed(() => {
  if (hasRunning.value || suiteRunning.value) return "已有任务运行中";
  if (!modelEndpoint.value.trim()) return "请先填写模型服务地址";
  if (!selectedModels.size) return "请先选择模型";
  if ([...selectedModels].some((modelId) => modelId !== "__current__") && !vllmManagerUrl.value.trim()) return "选择注册表模型需要模型管理器地址";
  if (selectedModels.has("__current__") && !selectedEndpointModel.value) return "请先从模型服务中选择要复用的模型";
  if (runMode.value === "reuse" && !existingScopeId.value) return "请先选择要复用的相册";
  return "";
});
const modeLabel = (mode) => (({ full: "全链路", reuse: "复用测评", build: "构建相册" })[mode || "full"] || mode);
const modeBadgeClass = (mode) => (({ full: "mode-full", reuse: "mode-reuse", build: "mode-build" })[mode || "full"] || "mode-full");
function exportSftTraces() {
  if (!activeRunId.value) return;
  const scores = exportScores.value;
  if (!scores.length) { window.alert("请至少勾选一个评分再导出"); return; }
  window.open(`/api/runs/${encodeURIComponent(activeRunId.value)}/export-sft?scores=${scores.join(",")}`, "_blank");
}
async function saveJudgePrompt() {
  const prompt = rejudgePrompt.value.trim();
  if (!prompt) { window.alert("提示词不能为空"); return; }
  try {
    await post("/api/judge-prompts", { kind: activePromptKind.value, system_prompt: prompt });
    promptDrafts[activePromptKind.value] = prompt;
    window.alert(`已保存「${PROMPT_KIND_LABELS[activePromptKind.value] || activePromptKind.value}」提示词，重新评分与后续评测将使用它`);
  } catch (e) { window.alert("保存失败：" + e.message); }
}
async function startRejudge() {
  if (!canRejudge.value || rejudgeSubmitting.value) return;
  if (!window.confirm(`仅使用现有 ${activeRun.value.item_count || 0} 条 Agent 回答重新调用 Judge。旧评分将保留在历史记录中，确定开始？`)) return;
  rejudgeSubmitting.value = true;
  error.value = "";
  try {
   await post(`/api/runs/${encodeURIComponent(activeRunId.value)}/rejudge`, {
    judge_url: judgeUrl.value.trim(),
    judge_model: judgeModel.value.trim(),
    judge_provider_id: judgeProviderId.value,
    ...(judgeApiKeyDirty.value ? { judge_api_key: judgeApiKey.value } : {}),
    system_prompt: promptDrafts.answer_quality || rejudgePrompt.value,
    task_system_prompt: promptDrafts.task_decision || undefined,
    evidence_system_prompt: promptDrafts.evidence || undefined,
    });
    await loadRuns(); await loadActiveRun(); startPolling();
  } catch (e) { error.value = e.message; }
  finally { rejudgeSubmitting.value = false; }
}
async function loadCurrentModel({ openPopover = true } = {}) {
  const endpoint = modelEndpoint.value.trim();
  if (!endpoint) {
    currentModelError.value = "请先填写模型服务地址，例如 192.168.0.153:8100";
    return;
  }
  currentModelLoading.value = true;
  currentModelError.value = "";
  try {
    const requestedModel = selectedEndpointModel.value;
    const requestBody = {
      model_base_url: endpoint,
      vllm_target_id: vllmTargetId.value,
    };
    let result;
    try {
      result = await post("/api/current-model", { ...requestBody, model: requestedModel });
    } catch (selectionError) {
      if (!requestedModel) throw selectionError;
      selectedEndpointModel.value = "";
      result = await post("/api/current-model", requestBody);
      currentModelError.value = `已清除不可用的已保存模型：${selectionError.message}`;
    }
    endpointModels.value = result.served_models || [];
    if (result.served_model_name) selectedEndpointModel.value = result.served_model_name;
    else if (!endpointModels.value.includes(requestedModel)) selectedEndpointModel.value = "";
    currentModelInfo.value = result;
    currentModelPopoverOpen.value = openPopover;
  } catch (e) {
    currentModelInfo.value = null;
    endpointModels.value = [];
    currentModelPopoverOpen.value = false;
    currentModelError.value = e.message;
  } finally {
    currentModelLoading.value = false;
  }
}
async function onEndpointModelChange() {
  selectedModels.delete("__current__");
  modelTestState.value = "idle";
  modelTestMessage.value = "";
  markConnectionConfigDirty();
  await loadCurrentModel({ openPopover: false });
}
async function testEndpointModel() {
  if (!selectedEndpointModel.value || modelTestState.value === "testing") return;
  modelTestState.value = "testing";
  modelTestMessage.value = "正在发送最小 POST 请求…";
  try {
    const result = await post("/api/test-model", {
      model_base_url: modelEndpoint.value.trim(),
      model: selectedEndpointModel.value,
    });
    modelTestState.value = "success";
    modelTestMessage.value = `POST 可用 · ${result.latency_ms} ms`;
  } catch (e) {
    modelTestState.value = "error";
    modelTestMessage.value = `POST 测试失败：${e.message}`;
  }
}
function onModelEndpointInput() {
  modelEndpointUserEdited.value = true;
  endpointModels.value = [];
  selectedEndpointModel.value = "";
  currentModelInfo.value = null;
  currentModelPopoverOpen.value = false;
  currentModelError.value = "";
  modelTestState.value = "idle";
  modelTestMessage.value = "";
  selectedModels.delete("__current__");
  markConnectionConfigDirty();
}
async function startSuite() {
  const blocked = startDisabledReason.value;
  if (blocked && !blocked.includes("运行中")) { window.alert(blocked); return; }
  if (blocked) return;
  if (runMode.value === "build" && !window.confirm("构建相册模式只导入并处理数据，不做 QA 测评；产物相册会保留。确定开始？")) return;
  suiteRunning.value = true;
  try {
   const result = await post("/api/runs", { album_id: selectedAlbum.value, qa_set: runMode.value === "build" ? undefined : selectedQa.value, mode: runMode.value, existing_scope_id: runMode.value === "reuse" ? existingScopeId.value : undefined, models: [...selectedModels], sentrix_url: sentrixUrl.value.trim(), judge_url: judgeUrl.value.trim(), judge_model: judgeModel.value.trim(), judge_provider_id: judgeProviderId.value, ...(judgeApiKeyDirty.value ? { judge_api_key: judgeApiKey.value } : {}), vllm_target_id: vllmTargetId.value, vllm_manager_url: vllmManagerUrl.value.trim(), model_base_url: selectedModels.has("__current__") || modelEndpointUserEdited.value ? modelEndpoint.value.trim() : "", endpoint_model: selectedModels.has("__current__") ? selectedEndpointModel.value : "", delete_scope_after_run: runMode.value === "full" ? deleteScopeAfterRun.value : false });
    activeRunId.value = result.run_ids[0];
    await loadRuns(); await loadActiveRun({ resetPage: true }); startPolling();
  } catch (e) { error.value = e.message; } finally { suiteRunning.value = false; }
}
async function stopSuite() {
  if (!window.confirm("确定停止当前所有评测任务？")) return;
  await api("/api/cancel-active", { method: "POST", body: "{}" });
  await loadRuns(); await loadActiveRun();
}
async function deleteRun(run) {
  if (!window.confirm("删除此评测？")) return;
  await api(`/api/runs/${encodeURIComponent(run.run_id)}`, { method: "DELETE" });
  if (activeRunId.value === run.run_id) { activeRunId.value = null; activeRun.value = null; }
  await loadRuns();
}
function startPolling() {
  if (pollTimer || destroyed) return;
  const poll = async () => {
    pollTimer = null;
    try {
      await loadRuns();
      if (activeRunId.value) await loadActiveRun();
      if (hasRunning.value && !destroyed) pollTimer = window.setTimeout(poll, 2000);
    } catch { if (!destroyed) pollTimer = window.setTimeout(poll, 5000); }
  };
  poll();
}
const arbiterStatus = ref(null);
let arbiterTimer = null;
async function loadArbiterStatus() {
  if (!sentrixUrl.value) return;
  try {
    const resp = await fetch(`${sentrixUrl.value.replace(/\/$/, "")}/api/arbiter/status`, { signal: AbortSignal.timeout(3000) });
    if (resp.ok) arbiterStatus.value = await resp.json();
  } catch (_) { /* backend 暂不可达时保持上次值 */ }
}
function arbStateLabel(s) {
  return ({idle: "空闲", import_running: "导入运行中", agent_active: "Agent 活跃", preempting: "抢占中"})[s] || s || "-";
}
function thermalStateLabel(v) {
  return ({0: "nominal", 1: "fair", 2: "serious", 3: "critical"})[v] ?? "-";
}
async function init() {
  try {
    config.value = await api("/api/config");
    vllmTargets.value = config.value.vllm_targets || {};
    vllmTargetId.value = config.value.default_vllm_target_id || Object.keys(vllmTargets.value)[0] || "";
    const runtimeConfig = config.value.runtime_config || {};
    vllmManagerUrl.value = runtimeConfig.vllm_manager_url || vllmTargets.value[vllmTargetId.value]?.manager_url || config.value.default_vllm_api_url || "";
    modelEndpoint.value = runtimeConfig.model_base_url || config.value.default_vllm_model_base_url || vllmTargets.value[vllmTargetId.value]?.model_base_url || "";
    modelEndpointUserEdited.value = Boolean(runtimeConfig.model_base_url);
    selectedEndpointModel.value = runtimeConfig.endpoint_model || "";
    sentrixUrl.value = runtimeConfig.sentrix_url || config.value.default_sentrix_url;
    judgeUrl.value = runtimeConfig.judge_url || config.value.default_judge_url;
    judgeModel.value = runtimeConfig.judge_model || config.value.judge_model || "";
    judgeApiKey.value = "";
    judgeApiKeyDirty.value = false;
    rejudgePrompt.value = config.value.custom_judge_prompt || config.value.judge_prompt || "";
    await loadJudgePrompts();
    judgeProviderId.value = runtimeConfig.judge_provider_id || config.value.default_judge_provider_id || (config.value.judge_providers?.[0]?.id || "");
    connectionConfigState.value = "saved";
    connectionConfigMessage.value = "已读取配置文件";
    manifests.value = (await api("/api/manifests")).manifests || [];
    await loadRuns(); if (vllmManagerUrl.value.trim()) await loadProfiles();
    if (modelEndpoint.value.trim()) await loadCurrentModel({ openPopover: false });
    // Open the newest completed run when nothing is active. Leaving the
    // detail panel unselected after a finished evaluation makes all aggregate
    // metric sections—including graph-by-QA-type and face clustering—appear
    // absent even though the run already contains them.
    const current = runs.value.find((run) => ["running", "pending"].includes(run.status))
      || runs.value.find((run) => run.status === "completed");
    if (current) {
      activeRunId.value = current.run_id;
      await loadActiveRun({ resetPage: true });
      if (["running", "pending"].includes(current.status)) startPolling();
    }
  } catch (e) { error.value = e.message; } finally { loading.value = false; }
  loadArbiterStatus();
  arbiterTimer = setInterval(loadArbiterStatus, 3000);
}
const qaBrowserOptions = computed(() => manifests.value.find((m) => m.album_id === qaBrowserAlbum.value)?.qa_sets || []);
const qaBrowserTags = computed(() => [...new Set(qaBrowserItems.value.flatMap(item => Array.isArray(item.tags) ? item.tags : []))].sort());
const visibleQaBrowserItems = computed(() => {
  const query = qaBrowserSearch.value.trim().toLocaleLowerCase();
  return qaBrowserItems.value.filter(item => {
    if (qaBrowserTag.value && !(item.tags || []).includes(qaBrowserTag.value)) return false;
    if (!query) return true;
    return [item.qa_id, item.question, item.answer, ...(item.tags || [])]
      .some(value => String(value || "").toLocaleLowerCase().includes(query));
  });
});
async function loadQaBrowser() {
  qaBrowserLoading.value = true;
  qaBrowserError.value = "";
  try {
    const data = await api(`/api/qa-dataset?album_id=${encodeURIComponent(qaBrowserAlbum.value)}&qa_set=${encodeURIComponent(qaBrowserSet.value)}&sentrix_url=${encodeURIComponent(sentrixUrl.value.trim())}`);
    qaBrowserItems.value = data.items || [];
    qaBrowserMediaResolution.value = data.media_resolution || null;
  } catch (e) { qaBrowserError.value = e.message; qaBrowserItems.value = []; qaBrowserMediaResolution.value = null; }
  finally { qaBrowserLoading.value = false; }
}
function qaTypeLabel(t) {
  return ({event_memory_qa:"事件记忆",single_evidence_memory_qa:"单图证据",relationship_qa:"关系问答",multi_hop:"多跳/因果",multi_turn_clarify:"多轮澄清",multi_turn_disambiguation:"多轮消歧",ambiguous_retrieval:"模糊检索",evidence_insufficient:"证据不足",unsupported_retrieval:"无依据检索",instruction_injection:"指令注入",prompt_injection:"提示注入",data_exfiltration:"数据泄露",authority_impersonation:"权限伪造",mixed_injection:"混合注入",indirect_injection:"间接注入",jailbreak_attempt:"越狱尝试"}[t]) || t || "未分类";
}
function qaActionBadge(a) {
  return ({answer:"回答",refuse:"拒答",clarify:"澄清"}[a]) || a || "-";
}
function qaAnswerabilityLabel(v) {
  return ({answerable:"可回答",unanswerable:"不可回答",ambiguous:"有歧义",unsafe_request:"不安全请求",answerable_after_clarification:"澄清后可回答",mixed:"混合"}[v]) || v || "-";
}
function qaEvidenceRefs(item) {
  return mediaRefs(item, "retrieval");
}
function qaIsVideoEvidence(item, media) {
  const mediaId = typeof media === "string" ? media : media?.media_id;
  return item.video_id === mediaId || inferMediaType(mediaId, media?.media_type) === "video";
}
function qaHasVideoEvidence(item) {
  return qaEvidenceRefs(item).some((ref) => ref.media_type === "video");
}
function qaMediaUrl(albumId, item, media) {
  if (typeof media !== "string" && media?.media_url) return media.media_url;
  const assetId = typeof media === "string" ? "" : media?.asset_id;
  return assetId && sentrixUrl.value
    ? `${sentrixUrl.value.replace(/\/$/, "")}/api/assets/${encodeURIComponent(assetId)}/file`
    : "";
}
function qaClaimMediaRefs(claim) {
  return mediaRefs(claim, "evidence");
}
function qaConversationTurns(item) {
  const conv = item.conversation;
  if (conv && Array.isArray(conv)) return conv;
  return [{ message: item.question, expected_action: item.expected_action, reference_answer: item.answer }];
}
function qaReferenceLabel(turn) {
  return turn?.expected_action === "clarify" ? "参考澄清示例" : "参考回答";
}
onMounted(init);
onUnmounted(() => { destroyed = true; if (pollTimer) clearTimeout(pollTimer); if (arbiterTimer) clearInterval(arbiterTimer); });
</script>

<template>
  <main v-if="!loading" class="app-shell">
    <nav class="view-tabs">
      <button :class="['view-tab', { active: activeView === 'runs' }]" @click="activeView = 'runs'">评测运行</button>
      <button :class="['view-tab', { active: activeView === 'qa-browser' }]" @click="activeView = 'qa-browser'; loadQaBrowser()">QA 数据集浏览</button>
    </nav>
    <template v-if="activeView === 'runs'">
    <section v-if="arbiterStatus" class="section arbiter-section">
      <div class="section-head">
        <h2>实时调度状态（VLMArbiter）</h2>
        <span class="live-badge">每 3s 刷新</span>
      </div>
      <div class="arbiter-grid">
        <div class="arbiter-cell"><span>调度器状态</span><strong>{{ arbStateLabel(arbiterStatus.state) }}</strong></div>
        <div class="arbiter-cell"><span>Worker 系数</span><strong>{{ arbiterStatus.worker_scale }}</strong></div>
        <div class="arbiter-cell"><span>内存压力</span><strong>{{ fmtNumber(arbiterStatus.memory_pressure) }}</strong></div>
        <div class="arbiter-cell"><span>门控 soft / hard</span><strong>{{ arbiterStatus.memory_gate_threshold }} / {{ arbiterStatus.memory_critical_threshold }}</strong></div>
        <div class="arbiter-cell"><span>散热状态</span><strong>{{ thermalStateLabel(arbiterStatus.thermal_state) }}</strong></div>
        <div class="arbiter-cell"><span>Import 活跃</span><strong>{{ arbiterStatus.import_active }}</strong></div>
        <div class="arbiter-cell"><span>Agent VLM 活跃</span><strong>{{ arbiterStatus.agent_vlm_active }}</strong></div>
        <div class="arbiter-cell"><span>预占次数</span><strong>{{ arbiterStatus.preempt_count }}</strong></div>
      </div>
    </section>
    <section class="section config-section">
      <div class="section-head">
<h2>评测配置</h2>
<div class="section-head-actions">
<span v-if="hasRunning" class="live-badge">实时更新中</span>
<button class="btn ghost compact" type="button" :disabled="connectionConfigState === 'saving'" @click="saveConnectionConfig">{{ connectionConfigState === 'saving' ? '保存中…' : '保存测评配置' }}</button>
<span class="config-save-status" :class="`state-${connectionConfigState}`">{{ connectionConfigMessage }}</span>
</div>
</div>
      <div class="config-groups">
        <div class="config-group">
          <div class="config-group-head"><div><strong>任务范围</strong><span>选择本次要构建或测评的数据范围</span></div></div>
          <div class="config-grid config-grid-scope">
        <label>工作模式<select v-model="runMode" :disabled="suiteRunning || hasRunning" @change="onModeChange">
<option v-for="mode in RUN_MODES_UI" :key="mode.id" :value="mode.id">{{ mode.label }}</option>
</select>
<span class="config-help mode-hint">{{ runModeMeta.hint }}</span>
</label>
        <label v-if="runMode !== 'reuse'">照片<select v-model="selectedAlbum">
<option v-for="manifest in manifests" :key="manifest.album_id" :value="manifest.album_id">{{ manifest.album_name }} ({{ albumCountLabel(manifest) }})</option>
</select>
<span v-if="runMode === 'build'" class="config-help">图片、视频与身份来源，处理后相册保留</span>
</label>
        <label v-if="runMode === 'reuse'">复用基座<select v-model="selectedReuseBaseId" :disabled="memorySpacesLoading">
<option value="" disabled>{{ memorySpacesLoading ? '加载中…' : (reuseBases.length ? '请选择相册基座（相册 + 模型）' : '无可复用基座（检查 Sentrix 后端地址）') }}</option>
<option v-for="base in reuseBases" :key="base.base_id" :value="base.base_id">{{ base.album_id }} · {{ base.model_profile }} · {{ base.scope_name }}</option>
</select>
<span class="config-help">{{ scopeAlbumHint || '选择后直接复用已生成相册，不重新处理照片' }}</span>
</label>
        <label v-if="runMode !== 'build'">QA 数据集<select v-model="selectedQa">
<option v-for="qa in qaOptions" :key="qa" :value="qa">{{ qa }}</option>
</select>
<span v-if="runMode === 'reuse'" class="config-help">题目与对照随所选现有相册自动匹配</span>
</label>
          </div>
        </div>
        <div class="config-group">
          <div class="config-group-head"><div><strong>评测服务</strong><span>配置 Sentrix 后端、Judge 评分服务及认证信息</span></div></div>
          <div class="config-grid config-grid-judge">
        <label>Sentrix 后端<input v-model="sentrixUrl" type="text" @input="markConnectionConfigDirty" placeholder="例如 192.168.0.153:8091" />
</label>
       <label>Judge 服务<input v-model="judgeUrl" type="text" @input="markConnectionConfigDirty" placeholder="例如 192.168.1.65:1234/v1" />
</label>
        <label>Judge 模型<input v-model="judgeModel" type="text" @input="markConnectionConfigDirty" placeholder="例如 qwen3.5-4b-mlx" />
          <span class="config-help">Judge 请求使用此 model 字段。</span>
</label>
        <label>Judge API key
          <input v-model="judgeApiKey" type="password" autocomplete="new-password" @input="judgeApiKeyDirty = true; markConnectionConfigDirty()" placeholder="留空表示不使用或保留已保存值" />
          <span class="config-help">仅保存到评测编排器本机环境变量；{{ config?.runtime_config?.judge_api_key_set ? `已配置（${config.runtime_config.judge_api_key_hint}）` : '当前未配置' }}。</span>
</label>
          </div>
        </div>
        <div class="config-group config-group-model">
          <div class="config-group-head"><div><strong>模型服务</strong><span>模型服务地址必填；模型管理器仅用于扫描注册表和切换模型</span></div></div>
          <div class="config-model-endpoint">
        <label>模型服务地址
          <div class="endpoint-line">
          <input v-model="modelEndpoint" type="text" @input="onModelEndpointInput" placeholder="例如 192.168.0.153:8100 或 http://192.168.0.153:8100/v1" />
          <div class="current-model-control">
            <button class="current-model-trigger" type="button" :class="{ active: currentModelPopoverOpen }" :disabled="currentModelLoading" @click="currentModelInfo ? currentModelPopoverOpen = !currentModelPopoverOpen : loadCurrentModel()">
              <span class="current-model-icon">{{ currentModelLoading ? '…' : '↗' }}</span>
              <span>{{ currentModelLoading ? '正在读取' : currentModelInfo ? '可用模型' : '获取模型列表' }}</span>
              <span class="current-model-chevron">{{ currentModelPopoverOpen ? '⌃' : '⌄' }}</span>
            </button>
            <div v-if="currentModelInfo && currentModelPopoverOpen" class="current-model-popover" role="status" aria-live="polite">
              <span class="popover-arrow"></span>
              <div class="current-model-popover-head"><span>MODEL ENDPOINT</span><button type="button" aria-label="关闭" @click="currentModelPopoverOpen = false">×</button></div>
              <strong>{{ currentModelInfo.served_model_name ? modelDisplayName(currentModelInfo.served_model_name) : `${currentModelInfo.served_models.length} 个模型待选择` }}</strong>
              <div class="current-model-status"><i></i>{{ currentModelInfo.manager_available ? '已读取 Manager 当前运行状态' : '已连接 OpenAI-compatible 端点' }}</div>
              <dl>
                <div><dt>模型服务</dt><dd>{{ currentModelInfo.model_base_url }}</dd></div>
                <div v-if="currentModelInfo.state?.profile"><dt>Profile</dt><dd>{{ currentModelInfo.state.profile }}</dd></div>
                <div v-if="currentModelInfo.state?.max_model_len"><dt>上下文 / 并发</dt><dd>{{ currentModelInfo.state.max_model_len }} / {{ currentModelInfo.state.max_num_seqs || '-' }}</dd></div>
                <div v-if="currentModelInfo.state?.dtype"><dt>精度</dt><dd>{{ currentModelInfo.state.dtype }}</dd></div>
              </dl>
              <button class="popover-refresh" type="button" @click="loadCurrentModel">重新获取</button>
            </div>
          </div>
          </div>
          <span class="config-help">填写模型服务的 IP:端口；可带或不带 /v1，实际请求会自动补齐。</span>
          <span v-if="currentModelError" class="config-help error">{{ currentModelError }}</span>
</label>
      </div>
          <div class="config-model-manager">
            <label>模型管理器地址（可选）
              <input v-model="vllmManagerUrl" type="text" @input="onModelManagerInput" @change="loadProfiles" placeholder="例如 192.168.0.153:8500；无管理器可留空" />
              <span class="config-help">填写后从 Manager 的模型注册表自动扫描；留空时不显示普通模型选择。</span>
            </label>
            <button class="btn ghost compact model-registry-refresh" type="button" :disabled="!vllmManagerUrl.trim()" @click="loadProfiles">刷新模型注册表</button>
          </div>
          <div class="model-current-choice">
            <label class="endpoint-model-select">可用模型
              <select v-model="selectedEndpointModel" :disabled="!endpointModels.length" @change="onEndpointModelChange">
                <option value="">{{ endpointModels.length ? '请选择模型' : '先获取模型列表' }}</option>
                <option v-for="model in endpointModels" :key="model" :value="model">{{ modelDisplayName(model) }}</option>
              </select>
            </label>
            <button class="btn ghost compact endpoint-test-button" type="button" :disabled="!selectedEndpointModel || modelTestState === 'testing'" @click="testEndpointModel">{{ modelTestState === 'testing' ? '测试中…' : '测试 POST' }}</button>
            <span v-if="modelTestMessage" class="model-test-feedback" :class="`state-${modelTestState}`">{{ modelTestMessage }}</span>
            <label class="check endpoint-reuse-check" :class="{ active: selectedModels.has('__current__') }">
              <input type="checkbox" :checked="selectedModels.has('__current__')" :disabled="!selectedEndpointModel || !currentModelInfo?.served_model_name" @change="setModelSelected('__current__', $event.target.checked)" />复用所选模型<span v-if="selectedEndpointModel">（{{ modelDisplayName(selectedEndpointModel) }}，不启停）</span>
            </label>
          </div>
          <div v-if="vllmManagerUrl.trim()" class="model-picker">
<span class="field-label">模型注册表（可多选，串行测试）</span>
<label v-for="profile in profiles" :key="profile.id" class="check" :class="{ active: selectedModels.has(profile.id) }">
<input type="checkbox" :checked="selectedModels.has(profile.id)" :disabled="!profile.available" @change="setModelSelected(profile.id, $event.target.checked)" />{{ profile.id }}<span v-if="profile.source === 'cloud_api'">（云端 API）</span><span v-if="!profile.available">（不可用）</span>
</label>
<span v-if="!profiles.length" class="config-help">尚未从模型管理器注册表加载模型，请点击“刷新模型注册表”。</span>
          </div>
          <div v-else class="model-manager-empty">未配置模型管理器。Ollama、llama.cpp 等端点只会按请求使用上方选中的模型；测评服务不会自动启停或切换模型。</div>
        </div>
      </div>
      <div class="actions">
<label v-if="runMode === 'full'" class="check"><input type="checkbox" v-model="deleteScopeAfterRun" :disabled="suiteRunning || hasRunning" />完成后删除相册</label>
<button class="btn" :disabled="Boolean(startDisabledReason)" @click="startSuite">{{ hasRunning ? '已有任务运行中' : runModeMeta.button }}</button>
<button class="btn warn" :disabled="!hasRunning" @click="stopSuite">停止全部</button>
</div>
      <p v-if="error" class="error">{{ error }}</p>
    </section>

    <section id="runs-region" class="section">
      <div class="section-head">
<h2>评测记录</h2>
<span class="muted">{{ runs.length }} 条</span>
</div>
      <div class="runs-list">
<table>
<thead>
<tr>
<th>模型</th>
<th>相册</th>
<th>开始时间</th>
<th>耗时</th>
<th>状态</th>
<th>进度</th>
<th>媒体召回率</th>
<th>质量均分</th>
<th>
</th>
</tr>
</thead>
<tbody>
<tr v-for="run in runs" :key="run.run_id" class="run-row" :class="{ selected: activeRunId === run.run_id }" @click="selectRun(run)">
<td>
<b>{{ modelName(run) }}</b>
<span class="mode-badge" :class="modeBadgeClass(run.mode)">{{ modeLabel(run.mode) }}</span>
</td>
<td class="muted small">{{ albumName(run) }}</td>
<td class="muted small">{{ fmtDate(run.started_at) }}</td>
<td class="muted small">{{ duration(run) }}</td>
<td>
<span class="phase-status" :class="run.status">{{ statusLabel(run.status) }}</span>
</td>
<td>{{ runProgressLabel(run) }}</td>
<td :title="run.status === 'running' && run.summary?.live_preview ? '阶段性指标：仅基于目前已完成的题目，任务结束后以最终汇总为准' : ''">{{ fmtPct(run.summary?.media_retrieval_recall_micro ?? run.summary?.retrieval_recall_micro) }}</td>
<td :title="run.status === 'running' && run.summary?.live_preview ? '阶段性指标：仅统计目前已有 Judge 评分的题目，任务结束后以最终汇总为准' : ''">{{ run.summary?.answer_quality_mean ?? "-" }}</td>
<td>
<button class="btn danger compact" @click.stop="deleteRun(run)">删除</button>
</td>
</tr>
</tbody>
</table>
</div>
    </section>

    <section id="detail-region">
<div v-if="!activeRun" class="section muted">点击上方列表中的某条记录查看详情</div>
<section v-else class="section detail-section">
      <div class="section-head detail-section-head">
        <h2>{{ modelName(activeRun) }} · {{ albumName(activeRun) }} · {{ qaName(activeRun) }}</h2>
        <div class="detail-head-tools">
          <span class="phase-status" :class="activeRun.status">{{ statusLabel(activeRun.status) }}</span>
          <div class="export-control">
            <span class="field-label">导出轨迹</span>
            <span class="export-score-filter">
              <label class="checkbox-inline"><input type="checkbox" value="0" v-model="exportScores">0 分</label>
              <label class="checkbox-inline"><input type="checkbox" value="1" v-model="exportScores">1 分</label>
              <label class="checkbox-inline"><input type="checkbox" value="2" v-model="exportScores">2 分</label>
            </span>
            <button class="btn compact" @click="exportSftTraces">导出 SFT JSON</button>
          </div>
        </div>
      </div>
      <p class="run-meta">开始 {{ fmtDate(activeRun.started_at) }} · 总耗时 {{ duration(activeRun) }}<template v-if="activeRun.mode === 'reuse'"> · 复用相册 {{ activeRun.scope_name || activeRun.scope_id || activeRun.existing_scope_id }}<span v-if="(activeRun.scope_reused_from_runs || []).length">（源自 run {{ activeRun.scope_reused_from_runs.join('、') }}）</span><span v-else>（外部创建，非编排器产物）</span></template><template v-else-if="activeRun.mode === 'build'"> · 产出相册 {{ activeRun.scope_id || '-' }}（已保留，可在复用测评中使用）</template></p>
      <div v-if="activeRun.fatal_error" class="run-error-banner"><b>任务终止原因</b><span>{{ activeRun.fatal_error }}</span><small v-if="activeRun.failed_phase">失败阶段：{{ EXECUTION_PHASES.find((item) => item.key === activeRun.failed_phase)?.label || activeRun.failed_phase }}</small></div>
      <section v-if="activeRun.mode !== 'build'" class="rejudge-card">
        <div class="rejudge-head">
<div>
<h3>重新 Judge 评分</h3>
<p>只复用本次运行已有的题目、标准答案和 Agent 回答，不重新执行相册处理、模型切换或 Agent 问答。</p>
</div>
<span v-if="activeRejudge" class="phase-status" :class="activeRejudge.status">{{ statusLabel(activeRejudge.status) }}</span>
</div>
        <div class="prompt-kind-tabs">
          <button v-for="meta in judgePromptKinds" :key="meta.kind" type="button"
                  class="btn ghost compact prompt-kind-tab" :class="{ active: activePromptKind === meta.kind }"
                  @click="switchPromptKind(meta.kind)">
            {{ PROMPT_KIND_LABELS[meta.kind] || meta.kind }}<span v-if="meta.custom" class="custom-badge">已自定义</span>
          </button>
        </div>
        <label class="rejudge-prompt">Judge System Prompt（{{ PROMPT_KIND_LABELS[activePromptKind] || activePromptKind }}）<textarea v-model="rejudgePrompt" :disabled="activeRejudge?.status === 'running'" rows="8" spellcheck="false">
</textarea>
</label>
        <div class="rejudge-toolbar">
<span class="muted small">{{ rejudgePrompt.length }} 字符 · Judge {{ judgeModel || '-' }}</span>
<div class="rejudge-actions">
<button class="btn ghost compact" :disabled="activeRejudge?.status === 'running'" @click="saveJudgePrompt">保存提示词</button>
<button class="btn ghost compact" :disabled="activeRejudge?.status === 'running'" @click="resetJudgePrompt">恢复默认</button>
<button class="btn compact" :disabled="!canRejudge || rejudgeSubmitting" @click="startRejudge">{{ activeRejudge?.status === 'running' ? '重新评分中…' : '重新评分全部 QA' }}</button>
</div>
</div>
        <div v-if="activeRejudge" class="rejudge-progress">
<div class="rejudge-progress-meta">
<span>{{ activeRejudge.completed || 0 }}/{{ activeRejudge.total || 0 }} 题</span>
<span>失败 {{ activeRejudge.failed || 0 }} · {{ rejudgePercent }}%</span>
</div>
<div class="phase-bar">
<div class="phase-bar-fill" :style="{ width: rejudgePercent + '%' }">
</div>
</div>
<p v-if="activeRejudge.error" class="error">{{ activeRejudge.error }}</p>
</div>
      </section>
      <h3>Pipeline 执行阶段</h3>
<div class="phase-list">
<article v-for="(phaseDef, index) in EXECUTION_PHASES" :key="phaseDef.key" class="phase-card">
<div class="phase-title">
<div class="phase-name">
<span class="phase-step">{{ index + 1 }}</span>
<b>{{ phaseDef.label }}</b>
</div>
<span class="phase-status" :class="activeRun.phases?.[phaseDef.key]?.status || 'pending'">{{ statusLabel(activeRun.phases?.[phaseDef.key]?.status || 'pending') }}</span>
</div>
<p class="phase-summary">{{ phaseSummary(phaseDef.key, activeRun.phases?.[phaseDef.key]) }}</p>
<details v-if="phaseErrorDetails(activeRun.phases?.[phaseDef.key]).length" class="phase-errors">
<summary>{{ phaseErrorSummary(activeRun.phases?.[phaseDef.key]) }}</summary>
<ul>
<li v-for="(detail, detailIndex) in phaseErrorDetails(activeRun.phases?.[phaseDef.key]).slice(0, 20)" :key="`${detail.sample_id || 'error'}-${detailIndex}`">
<b>{{ detail.sample_id || '未知样本' }}</b><span>{{ detail.reason || detail.error || '未提供原因' }}</span><small>{{ detail.error_type || detail.status || '错误' }}<template v-if="detail.asset_id"> · {{ detail.asset_id }}</template></small>
</li>
</ul>
<p v-if="phaseErrorDetails(activeRun.phases?.[phaseDef.key]).length > 20">仅展示前 20 条，完整记录已保存在 run.json。</p>
</details>
<div v-if="phaseDef.key === 'pipeline_processing' && activeRun.phases?.[phaseDef.key]?.progress" class="phase-bar">
<div class="phase-bar-fill" :style="{ width: phasePercent(activeRun.phases[phaseDef.key]) + '%' }">
</div>
</div>
<div v-if="phaseDef.key === 'pipeline_processing' && pipelineMetricRows(activeRun.phases?.[phaseDef.key]).length" class="phase-metrics pipeline-metrics">
<div v-for="row in pipelineMetricRows(activeRun.phases?.[phaseDef.key])" :key="row[0]" class="phase-metric">
<span>{{ row[0] }}</span>
<strong>{{ row[1] }}</strong>
<small>{{ row[2] }}</small>
</div>
</div>
</article>
</div>
      <h3 class="result-heading">结果指标</h3>
<div class="result-phase-list">
        <article class="phase-card result-phase-card gpu-result-card">
<div class="phase-title">
<b>GPU 指标</b>
<span class="phase-status" :class="resultPhaseStatus(activeRun.phases?.gpu_metrics)">{{ statusLabel(resultPhaseStatus(activeRun.phases?.gpu_metrics)) }}</span>
</div>
<p class="metric-calc-time">指标计算耗时 {{ fmtSeconds(phaseSeconds(activeRun.phases?.gpu_metrics)) }} · {{ activeRun.phases?.gpu_metrics?.memory_pressure ? "macOS 统一内存系统级采样（含模型 Metal 分配）" : "模型进程显存为 NVML 按 PID 汇总的实际占用，KV Cache 为 vLLM 逻辑使用率" }}{{ activeRun.qa_concurrency > 1 ? ` · QA 并发 ${activeRun.qa_concurrency}（时延含排队，勿与串行 run 直接对比）` : "" }}</p>
<div class="phase-metrics">
<div v-for="row in gpuMetricRows(activeRun.phases?.gpu_metrics)" :key="row[0]" :class="['phase-metric', { 'priority-metric': row[3] }]">
<span>{{ row[0] }}</span>
<strong>{{ row[1] }}</strong>
<small>{{ row[2] }}</small>
</div>
</div>
</article>
        <article class="phase-card result-phase-card gpu-result-card">
<div class="phase-title">
<b>{{ comparableMemoryProfile(activeRun)?.source === 'replay' ? '可比较显存复测' : '可比较显存' }}</b>
<span class="phase-status" :class="comparableMemoryProfile(activeRun)?.status || 'pending'">{{ statusLabel(comparableMemoryProfile(activeRun)?.status || 'pending') }}</span>
</div>
<p class="metric-calc-time">{{ comparableMemoryProfile(activeRun)?.source === 'gpu_metrics' ? '来自本次正式评测 GPU 采样；' : comparableMemoryProfile(activeRun)?.source === 'replay' ? '复用现有相册与问题，不运行 Benchmark/Judge，不保存本次回答；' : '本次 run 的 GPU 采样结束后生成；' }}可比较显存 = 固定基础占用 + KV Cache 实际峰值。</p>
<p v-if="comparableMemoryProfile(activeRun).error" class="error">{{ comparableMemoryProfile(activeRun).error }}</p>
<div class="phase-metrics">
<div v-for="row in memoryProfileRows(comparableMemoryProfile(activeRun) || {})" :key="row[0]" :class="['phase-metric', { 'priority-metric': row[3] }]">
<span>{{ row[0] }}</span>
<strong>{{ row[1] }}</strong>
<small>{{ row[2] }}</small>
</div>
</div>
</article>
        <article class="phase-card result-phase-card aggregate-result-card">
<div class="phase-title">
<b>指标汇总</b>
<span class="phase-status" :class="resultPhaseStatus(activeRun.phases?.aggregate)">{{ statusLabel(resultPhaseStatus(activeRun.phases?.aggregate)) }}</span>
</div>
<p class="metric-calc-time">指标计算耗时 {{ fmtSeconds(phaseSeconds(activeRun.phases?.aggregate)) }}</p>
<div class="phase-metrics">
<div v-for="row in aggregateMetricRows(activeRun.phases?.aggregate)" :key="row[0]" :class="['phase-metric', { 'priority-metric': row[3] }]">
<span>{{ row[0] }}</span>
<strong>{{ row[1] }}</strong>
<small>{{ row[2] }}</small>
</div>
</div>
<div class="token-distribution-section">
<div class="phase-title">
<b>主 Agent 单次调用 Token 分布</b>
<span class="muted small">共 {{ tokenDistributionCount() }} 次模型调用</span>
</div>
<p class="metric-calc-time">输入、输出和总上下文均按每次主 Agent 模型调用独立统计</p>
<div class="token-distribution-grid">
<div v-for="row in tokenDistributionRows()" :key="row[0]" class="phase-metric">
<span>{{ row[0] }}</span>
<strong>{{ row[1] }}</strong>
<small>{{ row[2] }}</small>
</div>
</div>
</div>
<div class="token-distribution-section graph-quality-section">
<div class="phase-title">
<b>可验证边指标（规则参考 / 独立 GT）</b>
<span class="muted small" v-if="graphQuality?.available">{{ graphQuality.total_nodes }} 节点 · {{ graphQuality.total_edges }} 条边</span>
<span class="muted small" v-else>{{ graphQualityLoading || graphQuality?.status === 'computing' ? '图结构快照生成中…' : '暂无可用快照' }}</span>
</div>
<p class="metric-calc-time">可验证边按确定性规则筛选；有独立人工/外部 GT 时显示真实 P/R/F1，否则显示源字段规则参考 P/R/F1，用于建图回归检查。</p>
<div v-if="graphQuality?.available" class="graph-quality-group">
<div class="graph-quality-group-title"><b>可验证边 P/R/F1</b><span>{{ graphQuality.reference_edge_ground_truth_available === true ? '独立关系 GT' : '确定性规则参考' }}</span></div>
<div class="token-distribution-grid graph-quality-primary-grid">
<div v-for="row in graphQualityPrimaryRows()" :key="row[0]" :class="['phase-metric', { 'priority-metric': row[3] }]">
<span>{{ row[0] }}</span>
<strong>{{ row[1] }}</strong>
<small>{{ row[2] }}</small>
</div>
</div>
</div>
<div v-if="graphQuality?.available" class="graph-quality-group graph-quality-diagnostic-group">
<div class="graph-quality-group-title"><b>结构健康（用于定位图的缺陷，不等同真实性）</b><span>全量规则检查</span></div>
<div class="token-distribution-grid">
<div v-for="row in graphQualityDiagnosticRows()" :key="row[0]" class="phase-metric">
<span>{{ row[0] }}</span>
<strong>{{ row[1] }}</strong>
<small>{{ row[2] }}</small>
</div>
</div>
</div>
<div v-if="graphQuality?.available && graphIdentityRows().length" class="graph-quality-group graph-quality-diagnostic-group">
<div class="graph-quality-group-title"><b>人脸身份聚类质量</b><span>{{ graphQuality.face_clustering.identity_gallery_available ? '身份库对齐 · 支持多脸' : (graphQuality.face_clustering.image_identity_link_available ? '多脸图像级身份关联 · pairwise' : (Number(graphQuality.face_clustering.metric_definition_version) >= 2 ? '严格单脸子集 · pairwise' : '旧版口径 · 等待刷新')) }}</span></div>
<div class="token-distribution-grid graph-quality-primary-grid">
<div v-for="row in graphIdentityRows()" :key="row[0]" :class="['phase-metric', { 'priority-metric': row[3] }]">
<span>{{ row[0] }}</span>
<strong>{{ row[1] }}</strong>
<small>{{ row[2] }}</small>
</div>
</div>
<p class="metric-calc-time">{{ graphQuality.face_clustering.identity_gallery_available ? '当前主指标使用身份库对齐：检测框通过评测开始前 seeded 的 confirmed identity cluster 映射到身份 ID，再与每张图的 GT 身份集合计算多脸 P/R/F1；未知脸不计为已知身份，GT 不反向参与聚类。' : (graphQuality.face_clustering.image_identity_link_available ? '当前主指标使用图像级身份关联：GT 只提供每张图的身份集合，因此多脸图通过“GT 身份集合是否相交”与“检测 cluster 集合是否相交”计算跨图同人链接；不会伪造逐框对应。严格单脸 pairwise 作为辅助诊断，资产缺失、零脸、多脸和未入簇分别统计。' : (Number(graphQuality.face_clustering.metric_definition_version) >= 2 ? '当前是严格单脸口径：只纳入单身份标注、恰好一张检测脸且已入簇的可对齐样本；多脸图不猜测框与身份的对应关系。' : '当前是旧版缓存指标：请重启 PhotoBench 后自动重算。')) }}</p>
</div>
<p v-if="graphQuality?.available" class="metric-calc-time">参考关系 {{ graphQuality.reference_edge_count ?? 0 }} 条 · 已构建可验证边 {{ graphQuality.evaluable_predicted_edge_count ?? 0 }} 条（{{ graphEvaluableEdgeSummary() }}） · 未纳入 P/R/F1 {{ graphQuality.unverifiable_edge_count ?? 0 }} 条。</p>
<div v-if="graphQuality?.available && graphQuestionTypeRows().length" class="graph-quality-type-wrap">
<div class="phase-title graph-quality-subtitle">
<b>按问题类型的图记忆 QA 表现</b>
<span class="muted small">全量题目分桶 · {{ effectiveRunSummary(activeRun).graph_qa_metric_scope === 'end_to_end_retrieval_chain' ? '整条检索链路' : '历史记录' }}</span>
</div>
<p class="metric-calc-time">检索 P/R/F1 与回答质量按评测元数据分组统计；“图效果”只显示同一题加入图重排后，对媒体 GT 召回的实际变化。</p>
<div class="call-table-wrap">
<table class="call-table graph-quality-type-table">
<thead><tr><th>问题类型</th><th>题数</th><th>图路由</th><th>图效果</th><th>检索 Recall</th><th>检索 Precision</th><th>检索 F1</th><th>回答质量</th><th>证据支持率</th></tr></thead>
<tbody>
<tr v-for="row in graphQuestionTypeRows()" :key="row.type">
<td>{{ row.label }}</td>
<td>{{ row.sample_count }}<span class="muted small">（{{ row.retrieval_metric_count }} 题有媒体 GT）</span></td>
<td>{{ row.graph_routed_count ? `${row.graph_enabled_count}/${row.graph_routed_count}` : '-' }}<span class="muted small" v-if="row.graph_intents">{{ Object.keys(row.graph_intents).join('、') }}</span></td>
<td v-if="row.graph_effect_count"><span :class="row.graph_gt_improved_count > row.graph_gt_worsened_count ? 'metric-good' : 'muted'">召回 {{ fmtSignedPct(row.graph_recall_delta) }}</span><span class="muted small"> · 提升 {{ row.graph_gt_improved_count }} 题<template v-if="row.graph_gt_worsened_count">，降低 {{ row.graph_gt_worsened_count }} 题</template> · 重排 {{ row.graph_rank_changed_count ?? 0 }} 题<template v-if="row.graph_candidate_set_changed_count"> · 候选集变化 {{ row.graph_candidate_set_changed_count }} 题</template></span></td>
<td v-else>-</td>
<td>{{ fmtPct(row.retrieval_recall) }}</td>
<td>{{ fmtPct(row.retrieval_precision) }}</td>
<td>{{ fmtPct(row.retrieval_f1) }}</td>
<td>{{ row.answer_quality_mean == null ? '-' : `${row.answer_quality_mean} / 2` }}<span class="muted small">（{{ row.judge_valid_count }}）</span></td>
<td>{{ fmtPct(row.evidence_supported_rate) }}<span class="muted small" v-if="row.evidence_valid_count">（{{ row.evidence_valid_count }}）</span></td>
</tr>
</tbody>
</table>
</div>
</div>
<p v-if="!graphQuality?.available" class="qa-performance-empty">{{ graphQualityUnavailableText() }}</p>
</div>
</article>
        <article v-if="deliveryBreakdown()" class="phase-card result-phase-card">
          <div class="phase-title">
            <b>确定性交付与 OCR partial</b>
            <span class="muted small">结构层诊断</span>
          </div>
          <div class="tool-performance-grid">
            <div class="tool-performance-row" v-if="deliveryBreakdown().detCount">
              <strong>确定性渲染</strong>
              <span>{{ deliveryBreakdown().detCount }} 题直接渲染，未走模型生成</span>
              <span v-if="deliveryBreakdown().detKinds.length">类型 {{ deliveryBreakdown().detKinds.map(([k, v]) => `${k}×${v}`).join("、") }}</span>
            </div>
            <div class="tool-performance-row" v-if="deliveryBreakdown().ocrPartial">
              <strong>OCR partial</strong>
              <span>{{ deliveryBreakdown().ocrPartial }} 题因 OCR 失败以 partial 语义收尾</span>
              <span v-if="deliveryBreakdown().ocrReasons.length">原因 {{ deliveryBreakdown().ocrReasons.map(([k, v]) => `${k}×${v}`).join("、") }}</span>
            </div>
          </div>
        </article>
        <article class="phase-card result-phase-card tool-result-card">
          <div class="phase-title">
            <b>主 Agent 工具性能</b>
            <span class="muted small">{{ toolPerformanceRows().length }} 类工具</span>
          </div>
          <p class="metric-calc-time">按工具调用次数、成功率和耗时汇总；不展示具体后端实现与内部推理细节</p>
          <div v-if="toolPerformanceRows().length" class="tool-performance-grid">
            <div v-for="tool in toolPerformanceRows()" :key="tool.name" class="tool-performance-row">
              <strong>{{ tool.name }}</strong>
              <span>调用 {{ tool.calls }} · 成功 {{ tool.ok_rate == null ? "-" : fmtPct(tool.ok_rate) }}</span>
              <span>P50 {{ fmtMs(tool.p50_ms) }} · P95 {{ fmtMs(tool.p95_ms) }} · max {{ fmtMs(tool.max_ms) }}</span>
            </div>
          </div>
          <p v-else class="qa-performance-empty">该运行没有记录工具调用。</p>
        </article>
        <article class="phase-card result-phase-card traceability-card">
          <details>
            <summary><strong>运行可追溯信息</strong><span>数据集完整性与模型运行时起止快照</span></summary>
            <div class="traceability-grid">
              <div><b>输入数据校验</b><span>数据集 {{ shortHash(activeRun.input_integrity?.dataset_sha256) }}</span><span>Manifest {{ shortHash(activeRun.input_integrity?.manifest_sha256) }} · QA {{ shortHash(activeRun.input_integrity?.qa_sha256) }}</span><span>{{ activeRun.input_integrity ? `${activeRun.input_integrity.files_checked} 个文件 · 缺失 ${activeRun.input_integrity.missing_files?.length || 0}` : "该历史运行未记录" }}</span></div>
              <div><b>运行开始</b><span>{{ snapshotSummary(activeRun.hardware_snapshots?.start).time }} · {{ snapshotSummary(activeRun.hardware_snapshots?.start).model }}</span><span>{{ snapshotSummary(activeRun.hardware_snapshots?.start).gpu }}</span><span>整卡 {{ snapshotSummary(activeRun.hardware_snapshots?.start).memory }} · 模型进程 {{ snapshotSummary(activeRun.hardware_snapshots?.start).processMemory }}</span></div>
              <div><b>运行结束</b><span>{{ snapshotSummary(activeRun.hardware_snapshots?.end).time }} · {{ snapshotSummary(activeRun.hardware_snapshots?.end).model }}</span><span>{{ snapshotSummary(activeRun.hardware_snapshots?.end).gpu }}</span><span>整卡 {{ snapshotSummary(activeRun.hardware_snapshots?.end).memory }} · 模型进程 {{ snapshotSummary(activeRun.hardware_snapshots?.end).processMemory }}</span></div>
            </div>
          </details>
        </article>
      </div>
      <section id="qa-results" class="qa-results-section">
        <div class="qa-results-heading">
          <div><h3>QA 逐题结果</h3><span class="muted small">筛选结果 {{ qaPage.total }} / 全部 {{ qaPage.unfiltered_total ?? qaPage.total }} 条</span></div>
          <div class="pager" v-if="qaPage.pages > 1">
            <button class="btn ghost compact" :disabled="!qaPage.has_previous" @click="changeQaPage(qaPage.page - 1)">上一页</button>
            <span>{{ qaPage.page }} / {{ qaPage.pages }}</span>
            <button class="btn ghost compact" :disabled="!qaPage.has_next" @click="changeQaPage(qaPage.page + 1)">下一页</button>
          </div>
          <label class="page-size-control">每页
            <select v-model.number="qaPageSize" @change="changeQaPageSize"><option :value="20">20</option><option :value="50">50</option><option :value="100">100</option></select>
          </label>
        </div>
        <form class="qa-filters" @submit.prevent="applyQaFilters">
          <input v-model="qaFilters.search" type="search" placeholder="搜索题号或问题" />
          <select v-model="qaFilters.score"><option value="">全部 Judge 分数</option><option value="2">2 分</option><option value="1">1 分</option><option value="0">0 分</option></select>
          <select v-model="qaFilters.task_type"><option value="">全部任务类型</option><option v-for="value in qaPage.facets?.task_types || []" :key="value" :value="value">{{ value }}</option></select>
          <select v-model="qaFilters.tag"><option value="">全部标签</option><option v-for="value in qaPage.facets?.tags || []" :key="value" :value="value">{{ value }}</option></select>
          <select v-model="qaFilters.angle"><option value="">全部问题角度</option><option v-for="value in qaPage.facets?.angles || []" :key="value" :value="value">{{ value }}</option></select>
          <select v-model="qaFilters.difficulty"><option value="">全部难度</option><option v-for="value in qaPage.facets?.difficulties || []" :key="value" :value="value">{{ value }}</option></select>
          <select v-model="qaFilters.answerability"><option value="">全部可回答性</option><option v-for="value in qaPage.facets?.answerabilities || []" :key="value" :value="value">{{ value }}</option></select>
          <select v-model="qaFilters.agent_status"><option value="">全部 Agent 状态</option><option v-for="value in qaPage.facets?.agent_statuses || []" :key="value" :value="value">{{ value }}</option></select>
          <select v-model="qaFilters.primary"><option value="">全部归因层</option><option v-for="value in qaPage.facets?.attribution_layers || []" :key="value" :value="value">{{ attributionLabel(value) }}</option></select>
          <button class="btn compact" type="submit">筛选</button><button class="btn ghost compact" type="button" @click="resetQaFilters">重置</button>
          <button class="btn ghost compact" type="button" :disabled="reviewSaving || ['running','pending'].includes(activeRun.status)" @click="saveReviews">{{ reviewSaving ? '保存中…' : '保存人工复核' }}</button>
        </form>
        <div v-if="activeRejudge?.status === 'running' && !visibleQaItems.length" class="qa-results-empty">正在等待本轮第一条 Judge 评分结果…</div>
        <article v-for="summary in visibleQaItems" :key="summary.index" class="qa-item" :class="{ open: openQaItems.has(summary.index) }">
          <button class="item-head qa-toggle" type="button" @click="toggleQa(summary)">
            <span class="item-idx">{{ String(summary.index + 1).padStart(2, "0") }}</span>
            <strong>{{ summary.question }}</strong>
            <span class="score" :class="[scoreClass(summary.judge?.score), 'judge-round-' + judgeRoundState(summary)]">{{ judgeScoreLabel(summary) }}</span>
            <span v-if="summary.ground_truth_count > 0" class="muted small">精确率 {{ fmtPct(summary.retrieval_precision) }} · 召回率 {{ fmtPct(summary.retrieval_recall) }} · F1 {{ fmtPct(summary.retrieval_f1) }} · 命中 {{ summary.matched_count }}/{{ summary.ground_truth_count }}</span>
            <span v-if="summary.evidence_judge?.score != null" class="score" :class="scoreClass(summary.evidence_judge.score)">证据 {{ summary.evidence_judge.score }}</span>
            <span class="muted small">模型 {{ summary.model_call_count }} · 工具 {{ summary.tool_call_count }}</span>
            <span class="qa-chevron" aria-hidden="true">⌄</span>
          </button>
          <div v-if="openQaItems.has(summary.index)" class="qa-expanded">
            <div v-if="loadingQaItems.has(summary.index)" class="qa-results-empty">正在加载该题完整记录…</div>
            <template v-else-if="itemDetail(summary)">
              <div class="qa-meta-tags">
                <span v-if="itemDetail(summary).task_type">{{ itemDetail(summary).task_type }}</span>
                <span v-if="itemDetail(summary).question_type">{{ itemDetail(summary).question_type }}</span>
                <span v-if="itemDetail(summary).angle">{{ itemDetail(summary).angle }}</span>
                <span v-if="itemDetail(summary).difficulty">{{ itemDetail(summary).difficulty }}</span>
                <span v-if="itemDetail(summary).answerability">{{ itemDetail(summary).answerability }}</span>
                <span v-for="tag in itemDetail(summary).tags || []" :key="tag" class="qa-run-tag">{{ tag }}</span>
              </div>
              <section v-if="conversationTurns(itemDetail(summary)).length > 1" class="result-conversation-card">
                <header class="result-conversation-head">
                  <div><small>MULTI-TURN CONVERSATION</small><strong>同一个多轮对话样本</strong><span>{{ conversationTurns(itemDetail(summary)).length }} 轮按顺序执行，后续轮次复用同一会话上下文</span></div>
                  <div class="conversation-identity"><b>{{ conversationIdLabel(itemDetail(summary)) }}</b><span>{{ itemDetail(summary).conversation_context_mode === 'shared_conversation_id' ? '共享 conversation_id' : '历史结果按已保存轮序展示' }}</span></div>
                </header>
                <div class="result-conversation-flow">
                  <article v-for="(turn, turnIndex) in conversationTurns(itemDetail(summary))" :key="turn.index ?? turnIndex" class="result-conversation-turn">
                    <div class="result-turn-marker"><b>{{ turnIndex + 1 }}</b><span>{{ conversationContextLabel(turn, turnIndex) }}</span></div>
                    <div class="result-message result-user-message"><small>用户 · 第 {{ turnIndex + 1 }} 轮</small><p>{{ turn.message }}</p></div>
                    <div class="result-message result-assistant-message"><small>模型回答</small><p>{{ turn.final_answer || turn.answer || "未完成" }}</p></div>
                    <div class="result-turn-scores">
                      <span>任务行为 <b>期望{{ actionLabel(turn.expected_action) }} / 实际{{ actionLabel(turn.task_judge?.actual_action) }}</b><em :class="turn.task_judge?.correct === true ? 'pass' : turn.task_judge?.correct === false ? 'fail' : ''">{{ turn.task_judge?.correct === true ? '一致' : turn.task_judge?.correct === false ? '不一致' : '未记录' }}</em></span>
                      <span>回答质量 <b>{{ turnScore(turn.judge?.score) }}</b></span>
                      <span>媒体证据 <b>{{ evidenceScoreLabel(turn.evidence_judge) }}</b></span>
                      <span>轮次结果 <b>{{ turn.turn_outcome || turn.termination_reason || "未记录" }}</b></span>
                    </div>
                    <div class="result-turn-reasons" v-if="judgeReason(turn.judge) || judgeReason(turn.task_judge) || judgeReason(turn.evidence_judge)">
                      <p v-if="judgeReason(turn.judge)"><b>质量：</b>{{ judgeReason(turn.judge) }}</p>
                      <p v-if="judgeReason(turn.task_judge)"><b>行为：</b>{{ judgeReason(turn.task_judge) }}</p>
                      <p v-if="judgeReason(turn.evidence_judge)"><b>证据：</b>{{ judgeReason(turn.evidence_judge) }}</p>
                    </div>
                  </article>
                </div>
              </section>
              <div class="item-body">
                <div>
                  <h4>{{ conversationTurns(itemDetail(summary)).length > 1 ? "最终一轮回答" : "模型回答" }}</h4><p>{{ itemDetail(summary).final_answer || itemDetail(summary).answer || itemDetail(summary).error || "未完成" }}</p>
                  <div class="capability-grid">
                    <span><small>任务判断</small><b>{{ taskDecisionLabel(itemDetail(summary)) }}</b></span>
                    <span><small>任务判断结果</small><b>{{ itemDetail(summary).task_judge?.correct === true ? "一致" : itemDetail(summary).task_judge?.correct === false ? "不一致" : "未记录" }}</b></span>
                    <span v-if="itemDetail(summary).task_judges?.length > 1"><small>多轮评分口径</small><b>每轮独立评分，并携带截至该轮的完整对话</b></span>
                    <span><small>证据对应</small><b>{{ evidenceScoreLabel(itemDetail(summary).evidence_judge) }}</b></span>
                    <span><small>媒体检索</small><b>{{ itemRetrievalMetrics(itemDetail(summary)) }}</b></span>
                    <span><small>JSON 解析</small><b>{{ itemParseRate(itemDetail(summary)) }}</b></span>
                    <span><small>步数内完成</small><b>{{ completionLabel(itemDetail(summary)) }}</b></span>
                  </div>
                  <h4>模型召回媒体（{{ itemMedia(itemDetail(summary)).length }}）</h4>
                  <div class="image-grid"><div v-for="media in itemMedia(itemDetail(summary))" :key="media.asset_id || media.file_name" class="image-tile"><video v-if="isVideoMedia(media) && imageUrl(media)" :src="imageUrl(media)" controls playsinline preload="metadata"></video><img v-else-if="imageUrl(media)" :src="imageUrl(media)" :alt="media.file_name" loading="lazy" @click="openImage(media)" /><span v-else class="image-empty">无媒体</span><span class="image-label">{{ media.file_name || media.media_id || media.image_id }}</span></div><span v-if="!itemMedia(itemDetail(summary)).length" class="muted small">模型没有返回可识别的媒体</span></div>
                  <h4>回答来源媒体（{{ itemEvidenceMedia(itemDetail(summary)).length }}）</h4>
                  <div class="image-grid"><div v-for="media in itemEvidenceMedia(itemDetail(summary)).slice(0, 3)" :key="`evidence-${media.asset_id || media.file_name}`" class="image-tile"><video v-if="isVideoMedia(media) && imageUrl(media)" :src="imageUrl(media)" controls playsinline preload="metadata"></video><img v-else-if="imageUrl(media)" :src="imageUrl(media)" :alt="media.file_name" loading="lazy" @click="openImage(media)" /><span v-else class="image-empty">无媒体</span><span class="image-label">{{ media.file_name || media.media_id || media.image_id }}</span></div><span v-if="!itemEvidenceMedia(itemDetail(summary)).length" class="muted small">没有记录可展示的证据来源</span></div>
                  <details v-if="itemEvidenceMedia(itemDetail(summary)).length > 3" class="qa-detail-block"><summary>查看更多来源（{{ itemEvidenceMedia(itemDetail(summary)).length - 3 }}）</summary><div class="image-grid"><div v-for="media in itemEvidenceMedia(itemDetail(summary)).slice(3)" :key="`evidence-more-${media.asset_id || media.file_name}`" class="image-tile"><video v-if="isVideoMedia(media) && imageUrl(media)" :src="imageUrl(media)" controls playsinline preload="metadata"></video><img v-else-if="imageUrl(media)" :src="imageUrl(media)" :alt="media.file_name" loading="lazy" @click="openImage(media)" /><span v-else class="image-empty">无媒体</span><span class="image-label">{{ media.file_name || media.media_id || media.image_id }}</span></div></div></details>
                </div>
                <div>
                  <h4>正确答案</h4><p>{{ itemDetail(summary).reference_answer }}</p>
                  <h4>检索 GT 媒体（{{ itemMedia(itemDetail(summary), true).length }}）</h4>
                  <div class="image-grid"><div v-for="media in itemMedia(itemDetail(summary), true)" :key="media.asset_id || `${media.media_type}-${media.media_id}`" class="image-tile"><video v-if="isVideoMedia(media) && imageUrl(media)" :src="imageUrl(media)" controls playsinline preload="metadata"></video><img v-else-if="imageUrl(media)" :src="imageUrl(media)" :alt="media.file_name" loading="lazy" @click="openImage(media)" /><span v-else class="image-empty">无媒体</span><span class="image-label">{{ media.file_name || media.media_id || media.image_id }}<em v-if="media.matched === false"> · 未召回</em></span></div></div>
                  <h4 v-if="judgeReason(itemDetail(summary).judge)">回答质量评分说明</h4><p v-if="judgeReason(itemDetail(summary).judge)" class="muted">{{ judgeReason(itemDetail(summary).judge) }}</p>
                  <h4 v-if="judgeReason(itemDetail(summary).task_judge)">任务判断说明</h4><p v-if="judgeReason(itemDetail(summary).task_judge)" class="muted">{{ judgeReason(itemDetail(summary).task_judge) }}</p>
                  <h4 v-if="judgeReason(itemDetail(summary).evidence_judge)">媒体证据评分说明</h4><p v-if="judgeReason(itemDetail(summary).evidence_judge)" class="muted">{{ judgeReason(itemDetail(summary).evidence_judge) }}</p>
                </div>
              </div>
              <section class="qa-performance">
                <div class="timing-breakdown">
                  <span>端到端 <b>{{ fmtMs(itemTimingBreakdown(itemDetail(summary)).wall_clock_ms) }}</b></span><span>Agent 总耗时 <b>{{ fmtMs(itemTimingBreakdown(itemDetail(summary)).agent_wall_ms) }}</b></span><span>模型 <b>{{ fmtMs(itemTimingBreakdown(itemDetail(summary)).model_ms) }}</b></span><span>工具 <b>{{ hasToolTrace(itemDetail(summary)) ? fmtMs(itemTimingBreakdown(itemDetail(summary)).tool_ms) : "未记录" }}</b></span><span>Judge <b>{{ fmtMs(itemTimingBreakdown(itemDetail(summary)).judge_ms) }}</b></span><span>其他 <b>{{ hasToolTrace(itemDetail(summary)) ? fmtMs(itemTimingBreakdown(itemDetail(summary)).other_ms) : "未记录" }}</b></span>
                </div>
                <div v-if="qaLatencySegments(itemDetail(summary)).length" class="latency-composition qa-latency-composition">
                  <div class="latency-composition-head"><strong>QA 样本时延组成</strong><span>{{ fmtMs(itemTimingBreakdown(itemDetail(summary)).wall_clock_ms) }}</span></div>
                  <div class="latency-bar" role="img" aria-label="QA 样本时延组成">
                    <span v-for="(segment, segmentIndex) in qaLatencySegments(itemDetail(summary))" :key="`${segment.kind}-${segmentIndex}`" class="latency-segment" :class="`latency-segment-${segment.kind}`" :style="latencySegmentStyle(segment, qaLatencySegments(itemDetail(summary)))" :data-tooltip="`${segment.label} · ${fmtMs(segment.ms)}`"><b>{{ fmtMs(segment.ms) }}</b></span>
                  </div>
                  <div class="latency-segment-legend"><span v-for="(segment, segmentIndex) in qaLatencySegments(itemDetail(summary))" :key="`${segment.kind}-legend-${segmentIndex}`"><i :class="`latency-dot latency-dot-${segment.kind}`"></i>{{ segment.shortLabel }} {{ fmtMs(segment.ms) }}</span></div>
                </div>
                <div v-if="agentLoopGroups(itemDetail(summary)).some((group) => group.calls.length)" class="agent-loop-groups">
                  <section v-for="group in agentLoopGroups(itemDetail(summary))" :key="group.turnIndex" class="agent-loop-group">
                    <header v-if="showAgentLoopGroupHeaders(itemDetail(summary))" class="agent-loop-turn-head">
                      <div><strong>第 {{ group.turnIndex + 1 }} 轮 Agent Loop</strong><span>{{ group.turn.message || itemDetail(summary).question }}</span></div>
                      <span>{{ group.calls.length }} 次模型调用 · {{ group.turn.answer ? "已回答" : group.turn.termination_reason || "未完成" }}</span>
                    </header>
                    <div v-if="group.calls.length" class="call-tree">
                      <details v-for="(call, callIndex) in group.calls" :key="`${group.turnIndex}-${call.step_id || call._globalCallIndex || callIndex}`" :class="['call-node', `call-type-${callType(call)}`, { 'trace-only-call': call._traceOnly }]">
                        <summary>
                          <span class="call-round">{{ callIndex + 1 }}</span><strong>{{ callTypeLabel(call) }}</strong><span v-if="['agent','recovery'].includes(callType(call))" :class="['call-outcome', callOutcomeClass(call)]">{{ callOutcome(call) }}</span><span>{{ call.role || "-" }} · {{ call.model || modelName(activeRun) }}</span><span>TTFT {{ fmtMs(call.ttft_ms) }}</span><span>Agent Loop 总时延 {{ fmtMs(callAgentLoopTiming(itemDetail(summary), call).totalMs) }}</span><span>模型 {{ fmtMs(call.total_ms) }}</span><span>Token {{ call.preflight_prompt_tokens ?? call.prompt_tokens ?? "-" }} / {{ call.completion_tokens ?? "-" }}</span><span>{{ fmtTokenRate(call.tokens_per_second) }}</span><span class="stream-state" :class="{ streamed: call.streamed === true }">{{ callStatus(call) }}</span>
                        </summary>
                          <div class="call-node-body">
                            <div v-if="callLatencySegments(itemDetail(summary), call).length" class="latency-composition call-latency-composition">
                              <div class="latency-composition-head"><strong>Agent Loop 时延组成</strong><span>{{ fmtMs(callAgentLoopTiming(itemDetail(summary), call).totalMs) }}</span></div>
                              <div class="latency-bar" role="img" aria-label="Agent Loop 时延组成">
                                <span v-for="(segment, segmentIndex) in callLatencySegments(itemDetail(summary), call)" :key="`${segment.kind}-${segmentIndex}`" class="latency-segment" :class="`latency-segment-${segment.kind}`" :style="latencySegmentStyle(segment, callLatencySegments(itemDetail(summary), call))" :data-tooltip="latencySegmentTitle(segment)"><b>{{ fmtMs(segment.ms) }}</b></span>
                              </div>
                              <div class="latency-segment-legend"><span v-for="(segment, segmentIndex) in callLatencySegments(itemDetail(summary), call)" :key="`${segment.kind}-legend-${segmentIndex}`"><i :class="`latency-dot latency-dot-${segment.kind}`"></i>{{ segment.label }} {{ fmtMs(segment.ms) }}</span></div>
                            </div>
                            <div class="call-purpose-grid"><span><small>用途</small><b>{{ callTypeDescription(call) }}</b></span><span><small>触发</small><b>{{ callObservation(call).trigger }}</b></span><span><small>结果</small><b>{{ callObservation(call).outcome }}</b></span><span><small>记录来源</small><b>{{ callObservation(call).source }}</b></span></div>
                          <div class="call-budget-line"><span>请求预算</span><b>{{ callBudget(call) }}</b><span v-if="call.step_id">步骤 {{ call.step_id }}</span><span v-if="callObservation(call).relatedTool !== '-'">关联工具 {{ callObservation(call).relatedTool }}</span><span v-if="callObservation(call).parentStep !== '-'">父步骤 {{ callObservation(call).parentStep }}</span></div>
                          <details v-if="debugStepForCallInGroup(itemDetail(summary), group, call)" class="debug-inline">
                            <summary>完整输入 / 输出</summary>
                            <div v-if="debugStepForCallInGroup(itemDetail(summary), group, call).type === 'judge' || debugStepForCallInGroup(itemDetail(summary), group, call).call_type === 'faithfulness_judge'">
                              <p class="muted small">评判结论</p><pre>{{ JSON.stringify({ faithful: debugStepForCallInGroup(itemDetail(summary), group, call).faithful, problems: debugStepForCallInGroup(itemDetail(summary), group, call).problems }, null, 2) }}</pre>
                              <p v-if="debugStepForCallInGroup(itemDetail(summary), group, call).debug?.prompt" class="muted small">评判提示词</p><pre v-if="debugStepForCallInGroup(itemDetail(summary), group, call).debug?.prompt">{{ JSON.stringify(debugStepForCallInGroup(itemDetail(summary), group, call).debug?.prompt, null, 2) }}</pre>
                              <p v-if="debugStepForCallInGroup(itemDetail(summary), group, call).debug?.raw" class="muted small">评判原始回答</p><pre v-if="debugStepForCallInGroup(itemDetail(summary), group, call).debug?.raw">{{ debugStepForCallInGroup(itemDetail(summary), group, call).debug?.raw }}</pre>
                            </div>
                            <div v-else>
                              <p v-if="debugPromptAnnotations(itemDetail(summary), group, call).length" class="muted small">内部控制消息（不是用户原话）</p><pre v-if="debugPromptAnnotations(itemDetail(summary), group, call).length">{{ JSON.stringify(debugPromptAnnotations(itemDetail(summary), group, call), null, 2) }}</pre>
                              <p class="muted small">完整提示词</p><pre>{{ JSON.stringify(debugStepForCallInGroup(itemDetail(summary), group, call).prompt, null, 2) }}</pre>
                              <p class="muted small">模型原始回答</p><pre>{{ debugStepForCallInGroup(itemDetail(summary), group, call).raw_full || debugStepForCallInGroup(itemDetail(summary), group, call).raw }}</pre>
                            </div>
                          </details>
                          <div v-if="showToolBranch(call) && toolsForGroupedCall(itemDetail(summary), call).length" class="tool-tree">
                            <details v-for="(trace, toolIndex) in toolsForGroupedCall(itemDetail(summary), call)" :key="toolIndex" class="tool-node">
                              <summary><strong>{{ trace.tool || "未知工具" }}</strong><span>{{ toolStatusLabel(trace) }}</span><span>总耗时 {{ trace.latency_s == null ? "-" : fmtMs(Number(trace.latency_s) * 1000) }}</span><span class="binding-source">{{ toolBindingLabel(trace) }}</span><span v-if="retrievalBackendLabel(trace)" class="retrieval-backend" :class="{ degraded: retrievalBackendDegraded(trace) }">检索后端 {{ retrievalBackendLabel(trace) }}</span></summary>
                              <div v-if="toolLatencySegments(trace).length" class="latency-composition nested-tool-latency">
                                <div class="latency-composition-head"><strong>工具内部时延组成</strong><span>{{ fmtMs(toolDurationMs(trace)) }}</span></div>
                                <div class="latency-bar" role="img" aria-label="工具内部时延组成">
                                  <span v-for="(segment, segmentIndex) in toolLatencySegments(trace)" :key="`${segment.kind}-${segmentIndex}`" class="latency-segment" :class="`latency-segment-${segment.kind}`" :style="latencySegmentStyle(segment, toolLatencySegments(trace))" :data-tooltip="`${segment.label} · ${fmtMs(segment.ms)}`"><b>{{ fmtMs(segment.ms) }}</b></span>
                                </div>
                                <div class="latency-segment-legend"><span v-for="(segment, segmentIndex) in toolLatencySegments(trace)" :key="`${segment.kind}-legend-${segmentIndex}`"><i :class="`latency-dot latency-dot-${segment.kind}`"></i>{{ segment.label }} {{ fmtMs(segment.ms) }}</span></div>
                              </div>
                              <details v-if="debugToolsForCall(itemDetail(summary), group, call)[toolIndex]" class="debug-inline">
                                <summary>完整工具输入 / 输出</summary>
                                <p class="muted small">工具输入</p><pre>{{ JSON.stringify(debugToolsForCall(itemDetail(summary), group, call)[toolIndex].arguments, null, 2) }}</pre>
                                <p class="muted small">工具输出</p><pre>{{ JSON.stringify(debugToolsForCall(itemDetail(summary), group, call)[toolIndex].observation, null, 2) }}</pre>
                              </details>
                            </details>
                          </div>
                          <p v-else-if="showToolBranch(call)" class="qa-performance-empty">{{ noToolLabel(call) }}</p>
                        </div>
                      </details>
                    </div>
                    <p v-else class="qa-performance-empty">本轮没有保存模型调用性能或失败轨迹。</p>
                    <div v-if="group.turn.agent_status || group.turn.termination_reason || group.turn.turn_outcome" :class="['turn-status-strip', executionStateClass(group.turn)]">
                      <strong>第 {{ group.turnIndex + 1 }} 轮状态</strong><span class="status-pill">{{ turnCompletionLabel(group.turn) }}</span><span>运行状态 <b>{{ group.turn.agent_status || "未记录" }}</b></span><span>终止原因 <b>{{ turnTerminationLabel(group.turn) }}</b></span><span>JSON 解析 <b>{{ group.turn.parse_status || "未记录" }}</b></span><span>恢复 <b>{{ turnRecoveryCount(group) }} 次</b></span>
                    </div>
                  </section>
                </div>
                <p v-else class="qa-performance-empty">该历史结果未记录主模型调用性能或失败轨迹。</p>
                <details v-if="unboundTools(itemDetail(summary)).length" class="call-node unbound-tools">
                  <summary><strong>未绑定模型轮次的工具序列</strong><span>{{ unboundTools(itemDetail(summary)).length }} 次 · 历史数据未保存精确轮次关系</span></summary>
                  <div class="call-node-body tool-tree"><details v-for="(trace, toolIndex) in unboundTools(itemDetail(summary))" :key="toolIndex" class="tool-node"><summary><strong>#{{ toolIndex + 1 }} {{ trace.tool || "未知工具" }}</strong><span>{{ toolStatusLabel(trace) }}</span><span>总耗时 {{ trace.latency_s == null ? "-" : fmtMs(Number(trace.latency_s) * 1000) }}</span><span v-if="retrievalBackendLabel(trace)" class="retrieval-backend" :class="{ degraded: retrievalBackendDegraded(trace) }">检索后端 {{ retrievalBackendLabel(trace) }}</span></summary></details></div>
                </details>
                <section v-if="guardSummary(itemDetail(summary)).recorded" :class="['execution-status-panel', executionStateClass(itemDetail(summary))]">
                  <div class="execution-status-head"><div><small>执行结果</small><strong>{{ conversationTurns(itemDetail(summary)).length > 1 ? "最终一轮状态汇总" : "Agent 结束状态" }}</strong></div><span class="status-pill">{{ completionLabel(itemDetail(summary)) }}</span></div>
                  <div class="execution-status-grid"><span><small>运行状态</small><b>{{ guardSummary(itemDetail(summary)).status }}</b></span><span><small>终止原因</small><b>{{ terminationDisplayLabel(itemDetail(summary)) }}</b></span><span><small>恢复次数</small><b>{{ guardSummary(itemDetail(summary)).recoveries }} 次</b></span><span><small>JSON 解析</small><b>{{ itemParseRate(itemDetail(summary)) }}</b></span><span class="execution-status-wide"><small>运行说明</small><b>{{ itemDetail(summary).agent_reason || "本轮 Agent 已按正常流程结束" }}</b></span></div>
                </section>
              </section>
              <!-- Agent 2.0 的首轮规划与证据账本是第一步调用的审计结果，直接展示摘要。 -->
              <details v-if="getAgent2Trace(itemDetail(summary))" class="agent2-summary-panel">
                <summary class="agent2-summary-toggle">
                  <span class="agent2-toggle-copy"><small>首轮规划调用 · Agent 2.0 Shadow</small><strong>目标分解与证据账本详情</strong><span>回答前先把用户目标拆成可验证的证据需求，再记录工具带回的证据</span></span>
                  <span class="agent2-counters"><span>{{ agent2RequirementSummary(itemDetail(summary)) }}</span><span>{{ getAgent2LedgerEntries(itemDetail(summary)).length }} 条证据</span><span>{{ agent2DecisionSummary(itemDetail(summary)) }}</span></span>
                </summary>
                <div class="agent2-summary-body">
                <div class="agent2-overview-grid"><span><small>目标</small><b>{{ getAgent2Trace(itemDetail(summary))?.task_declaration?.goal || "未记录" }}</b></span><span><small>作用域</small><b>{{ getAgent2Trace(itemDetail(summary))?.task_declaration?.scope_id || "未记录" }}</b></span><span><small>需求数</small><b>{{ getAgent2Requirements(itemDetail(summary)).length }} 项证据需求</b></span><span><small>证据数</small><b>{{ getAgent2LedgerEntries(itemDetail(summary)).length }} 条账本记录</b></span></div>
                <div v-if="getAgent2Requirements(itemDetail(summary)).length" class="agent2-detail-section"><div class="agent2-section-label">证据需求</div><div class="agent2-requirement-list"><div v-for="(req, rIdx) in getAgent2Requirements(itemDetail(summary))" :key="rIdx" class="agent2-requirement"><div class="agent2-requirement-head"><b>{{ req.id }}</b><span>{{ agent2EvidenceTypeLabel(req.evidence_type) }}</span><em :class="agent2RequirementStatusClass(req.status)">{{ agent2RequirementStatusLabel(req.status) }}</em></div><p>{{ req.description || "未记录需求描述" }}</p><small>证据引用：{{ (req.evidence_refs || []).join("、") || "暂无" }}<span v-if="req.unmet_reason"> · 未满足原因：{{ req.unmet_reason }}</span></small></div></div></div>
                <div v-if="getAgent2LedgerEntries(itemDetail(summary)).length" class="agent2-detail-section"><div class="agent2-section-label">证据账本</div><div class="agent2-evidence-list"><div v-for="(entry, eIdx) in getAgent2LedgerEntries(itemDetail(summary))" :key="eIdx" class="agent2-evidence"><div class="agent2-evidence-head"><b>{{ entry.capability || agent2EvidenceTypeLabel(entry.evidence_type) }}</b><span>{{ agent2EvidenceTypeLabel(entry.evidence_type) }}</span><em>{{ entry.certainty || "未定" }}</em></div><textarea class="agent2-json-view" readonly wrap="off" :rows="agent2EvidenceRows(entry.extracted_value ?? entry.value ?? null)" :style="{ height: agent2EvidenceHeight(entry.extracted_value ?? entry.value ?? null) }" :value="formatAgent2EvidenceValue(entry.extracted_value ?? entry.value ?? null)" aria-label="证据账本 JSON 内容"></textarea><small>来源：{{ (entry.provenance_refs || entry.input_refs || []).join("、") || entry.tool_call_id || "未记录" }}<span v-if="entry.asset_id"> · 图片：{{ entry.asset_id }}</span><span v-if="entry.unmatched_reason"> · {{ entry.unmatched_reason === "evidence_incompatible" ? "非当前需求证据" : entry.unmatched_reason }}</span></small></div></div></div>
                </div>
              </details>

              <details v-if="runtimeDebugTurns(itemDetail(summary)).length" class="call-node debug-trace-node">
                <summary><strong>调试详情：完整运行时轨迹</strong><span>{{ runtimeDebugTurns(itemDetail(summary)).length }} 轮用户问答 · 提示词 / 回答 / 工具输入输出 / 评判</span></summary>
                <div class="debug-trace-body">
                  <details v-for="(turn, turnIndex) in runtimeDebugTurns(itemDetail(summary))" :key="turnIndex" class="debug-turn">
                    <summary><span class="debug-turn-index">第 {{ turn.index + 1 }} 轮</span><strong>{{ turn.message || "问答" }}</strong><span class="debug-turn-meta">{{ turn.debug_trace.length }} 个步骤</span></summary>
                    <div class="debug-steps">
                      <details v-for="(step, stepIndex) in turn.debug_trace" :key="stepIndex" class="debug-step" :class="'debug-step-' + (step.type || 'step')">
                        <summary><span class="debug-step-index">{{ stepIndex + 1 }}</span><strong>{{ step.type === 'model' ? '模型步骤' : step.type === 'tool' ? '工具步骤' : step.type === 'judge' ? '评判步骤' : step.type }}</strong><span class="debug-step-status">{{ step.status || '已记录' }}</span></summary>
                        <div v-if="step.type === 'model'">
                          <p class="muted small">提示词</p><pre>{{ JSON.stringify(step.prompt, null, 2) }}</pre>
                          <p class="muted small">模型回答</p><pre>{{ step.raw_full || step.raw }}</pre>
                        </div>
                        <div v-else-if="step.type === 'tool'">
                          <p class="muted small">工具输入</p><pre>{{ JSON.stringify(step.arguments, null, 2) }}</pre>
                          <p class="muted small">工具输出</p><pre>{{ JSON.stringify(step.observation, null, 2) }}</pre>
                        </div>
                        <div v-else-if="step.type === 'judge'">
                          <p class="muted small">评判结论</p><pre>{{ JSON.stringify({ faithful: step.faithful, problems: step.problems }, null, 2) }}</pre>
                          <div v-if="step.debug"><p class="muted small">评判提示词</p><pre>{{ JSON.stringify(step.debug.prompt, null, 2) }}</pre><p class="muted small">评判回答</p><pre>{{ step.debug.raw }}</pre></div>
                        </div>
                        <div v-else><pre>{{ JSON.stringify(step, null, 2) }}</pre></div>
                      </details>
                    </div>
                  </details>
                </div>
              </details>
              <div class="item-footer"><button class="judge-details-trigger" type="button" @click="openJudgeInput(itemDetail(summary))">JUDGE 模型输入 <span>↗</span></button></div>
              <div class="review-editor"><label>人工复核<select v-model="reviewFor(summary).verdict"><option value="">未复核</option><option value="correct">正确</option><option value="partial">部分正确</option><option value="wrong">错误</option></select></label><input v-model="reviewFor(summary).note" placeholder="复核备注（可选）" /></div>
            </template>
          </div>
        </article>
        <div class="pager pager-bottom" v-if="qaPage.pages > 1"><button class="btn ghost compact" :disabled="!qaPage.has_previous" @click="changeQaPage(qaPage.page - 1)">上一页</button><span>第 {{ qaPage.page }} / {{ qaPage.pages }} 页</span><button class="btn ghost compact" :disabled="!qaPage.has_next" @click="changeQaPage(qaPage.page + 1)">下一页</button></div>
      </section>
    </section>
</section>
    </template>

    <template v-if="activeView === 'qa-browser'">
      <section class="qa-browser-page">
        <header class="qa-browser-hero">
          <div>
            <p class="qa-browser-kicker">DATASET REVIEW</p>
            <h1>QA 数据集审阅</h1>
            <p>按题检查对话设计、参考回答、图视频 GT 和证据元数据。</p>
          </div>
          <div class="qa-browser-count"><strong>{{ visibleQaBrowserItems.length }}</strong><span>{{ visibleQaBrowserItems.length === qaBrowserItems.length ? '道题目' : `/ ${qaBrowserItems.length} 道` }}</span></div>
        </header>
        <div class="qa-browser-toolbar">
          <label><span>相册</span><select v-model="qaBrowserAlbum" @change="qaBrowserSet = (qaBrowserOptions[0] || 'compact-10q'); loadQaBrowser()">
            <option v-for="manifest in manifests" :key="manifest.album_id" :value="manifest.album_id">{{ manifest.album_name }} · {{ albumCountLabel(manifest) }}</option>
          </select></label>
          <label><span>QA 数据集</span><select v-model="qaBrowserSet" @change="loadQaBrowser">
            <option v-for="qa in qaBrowserOptions" :key="qa" :value="qa">{{ qa }}</option>
          </select></label>
          <label><span>搜索</span><input v-model="qaBrowserSearch" type="search" placeholder="QA ID、问题或答案"></label>
          <label><span>标签</span><select v-model="qaBrowserTag"><option value="">全部标签</option><option v-for="tag in qaBrowserTags" :key="tag" :value="tag">{{ tag }}</option></select></label>
          <button class="btn ghost" @click="loadQaBrowser">刷新数据</button>
        </div>
        <p v-if="qaBrowserError" class="error">{{ qaBrowserError }}</p>
        <p v-else-if="qaBrowserMediaResolution && qaBrowserMediaResolution.status !== 'no_media'" class="muted small qa-media-source-status">
          媒体来源：Sentrix 后端 · 已解析 {{ qaBrowserMediaResolution.resolved_count || 0 }} · 缺失 {{ qaBrowserMediaResolution.missing_count || 0 }} · 不唯一 {{ qaBrowserMediaResolution.ambiguous_count || 0 }}
        </p>
        <div v-if="qaBrowserLoading" class="qa-browser-empty">正在加载数据集…</div>
        <div v-else class="qa-browser-list">
          <article v-for="(item, idx) in visibleQaBrowserItems" :key="item.qa_id || idx" class="qa-review-card">
            <header class="qa-review-header">
              <div class="qa-review-index">{{ String(idx + 1).padStart(2, '0') }}</div>
              <div class="qa-review-title">
                <div class="qa-browser-meta">
                  <span class="qa-badge" :class="'badge-' + (item.expected_action || 'answer')">{{ qaActionBadge(item.expected_action) }}</span>
                  <span class="qa-type-tag">{{ qaTypeLabel(item.question_type) }}</span>
                  <span class="qa-answerability-tag">{{ qaAnswerabilityLabel(item.answerability) }}</span>
                  <span v-if="item.difficulty" class="qa-difficulty-tag">{{ item.difficulty }}</span>
                  <span v-for="tag in item.tags || []" :key="tag" class="qa-data-tag">{{ tag }}</span>
                  <span v-if="qaConversationTurns(item).length > 1" class="multi-turn-badge">{{ qaConversationTurns(item).length }} 轮</span>
                </div>
                <span class="qa-review-id">{{ item.qa_id || `qa-${idx + 1}` }}</span>
              </div>
            </header>
            <div class="qa-review-layout" :class="{ 'no-evidence': !qaEvidenceRefs(item).length }">
              <div class="qa-dialogue-panel">
                <div v-for="(turn, ti) in qaConversationTurns(item)" :key="ti" class="qa-turn-row">
                  <div class="qa-turn-side user-side">
                    <div class="qa-speaker"><span class="qa-avatar user-avatar">问</span><b>用户</b><small>第 {{ ti + 1 }} 轮</small></div>
                    <div class="qa-bubble user-bubble">
                      <p>{{ turn.message }}</p>
                      <span v-if="turn.expected_action" class="bubble-action-hint" :class="'hint-' + turn.expected_action">期望行为 · {{ qaActionBadge(turn.expected_action) }}</span>
                    </div>
                  </div>
                  <div class="qa-turn-divider"><span>{{ ti + 1 }}</span></div>
                  <div class="qa-turn-side answer-side">
                    <div class="qa-speaker answer-speaker"><small>GT</small><b>{{ qaReferenceLabel(turn) }}</b><span class="qa-avatar answer-avatar">答</span></div>
                    <div class="qa-bubble answer-bubble"><p>{{ turn.reference_answer || item.answer || '（无参考答案）' }}</p></div>
                  </div>
                </div>
              </div>
              <aside v-if="qaEvidenceRefs(item).length" class="qa-evidence-panel">
                <div class="qa-evidence-head"><div><span>RETRIEVAL GROUND TRUTH</span><strong>检索 GT 媒体</strong><small>问题对应的直接证据</small></div><b>{{ qaEvidenceRefs(item).length }}</b></div>
                <div class="gt-gallery" :class="{ 'video-gallery': qaHasVideoEvidence(item) }">
                  <div v-for="media in qaEvidenceRefs(item)" :key="mediaKey(media)" class="gt-thumb-card" :class="{ 'direct-evidence': isDirectEvidence(item, media), 'video-evidence': qaIsVideoEvidence(item, media) }">
                    <video v-if="qaIsVideoEvidence(item, media) && qaMediaUrl(qaBrowserAlbum, item, media)" :src="qaMediaUrl(qaBrowserAlbum, item, media)" controls playsinline preload="none"></video>
                    <button v-else-if="qaMediaUrl(qaBrowserAlbum, item, media)" class="gt-image-button" type="button" @click="lightbox = { url: qaMediaUrl(qaBrowserAlbum, item, media), name: media.media_id.split('/').pop() }"><img :src="qaMediaUrl(qaBrowserAlbum, item, media)" :alt="media.media_id" loading="lazy" /></button>
                    <span v-else class="image-empty">{{ media.mapping_status === 'ambiguous' ? '媒体标识不唯一' : 'Sentrix 后端未找到' }}</span>
                    <span>{{ media.media_id.split('/').pop() }}<em>{{ qaIsVideoEvidence(item, media) ? '视频证据' : isDirectEvidence(item, media) ? '直接证据' : '事件相关' }}</em></span>
                  </div>
                </div>
              </aside>
            </div>
            <div class="qa-review-foot">
              <details v-if="item.answer_claims && item.answer_claims.length" class="qa-detail-block">
                <summary>证据声明 <b>{{ item.answer_claims.length }}</b></summary>
                <div v-for="(claim, ci) in item.answer_claims" :key="ci" class="qa-claim"><span class="claim-type">{{ claim.support_type || claim.claim_id }}</span><span class="claim-text">{{ claim.text }}</span><span v-if="qaClaimMediaRefs(claim).length" class="claim-evidence">{{ qaClaimMediaRefs(claim).map(ref => `${ref.media_type === 'video' ? '视频' : '图片'}:${ref.media_id.split('/').pop()}`).join(' · ') }}</span></div>
              </details>
              <details v-if="item.person_references && item.person_references.length" class="qa-detail-block">
                <summary>人物引用 <b>{{ item.person_references.length }}</b></summary>
                <div class="qa-person-grid"><div v-for="(person, pi) in item.person_references" :key="pi" class="qa-person-ref"><span class="person-name">{{ person.name }}</span><small v-if="person.aliases?.length">{{ person.aliases.join(' / ') }}</small><small v-if="person.face_id">Face {{ person.face_id }}</small></div></div>
              </details>
              <details v-if="item.query_anchors || item.scope_anchor || item.required_evidence_sources || item.event_id || item.angle" class="qa-detail-block">
                <summary>题目元数据</summary>
                <div class="qa-meta-grid"><div v-if="item.query_anchors"><b>查询锚点</b><span>{{ item.query_anchors }}</span></div><div v-if="item.scope_anchor"><b>范围锚点</b><span>{{ item.scope_anchor }}</span></div><div v-if="item.required_evidence_sources"><b>证据源</b><span>{{ Array.isArray(item.required_evidence_sources) ? item.required_evidence_sources.join(', ') : item.required_evidence_sources }}</span></div><div v-if="item.event_id"><b>事件 ID</b><span>{{ item.event_id }}</span></div><div v-if="item.angle"><b>考察角度</b><span>{{ item.angle }}</span></div></div>
              </details>
            </div>
          </article>
        </div>
      </section>
    </template>
  </main>
  <div v-else class="loading">加载评测数据…</div>
  <div v-if="lightbox" class="lightbox" @click="lightbox = null">
<div>
<img :src="lightbox.url" :alt="lightbox.name" />
<p>{{ lightbox.name }}</p>
</div>
  </div>
  <Teleport to="body"><div v-if="judgeModal" class="judge-modal-backdrop" @click.self="closeJudgeInput"><section class="judge-modal" role="dialog" aria-modal="true" aria-label="Judge 模型原始输入"><header><div><h3>JUDGE 模型原始输入</h3><span class="muted small">{{ judgeModal.qaId }}</span></div><button class="judge-modal-close" type="button" aria-label="关闭" @click="closeJudgeInput">×</button></header><pre v-if="judgeModal.complete">{{ judgeModal.rawJson }}</pre><p v-else class="judge-input-note">该历史结果在运行时未保存 Judge 原始请求 JSON，无法恢复。</p></section></div></Teleport>
</template>
