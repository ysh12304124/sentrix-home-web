<script setup>
import { computed, onMounted, onUnmounted, reactive, ref } from "vue";

const EXECUTION_PHASES = [
  { key: "model_deploy", label: "模型部署" },
  { key: "scope_setup", label: "创建相册" },
  { key: "identity_seed", label: "预置身份" },
  { key: "photo_import", label: "照片导入" },
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
const qaPage = ref({ items: [], page: 1, page_size: 20, total: 0, pages: 1 });
const qaDetails = reactive({});
const openQaItems = reactive(new Set());
const loadingQaItems = reactive(new Set());
const qaPageSize = ref(20);
const qaFilters = reactive({ search: "", score: "", task_type: "", angle: "", difficulty: "", answerability: "", agent_status: "", primary: "" });
const reviewDrafts = reactive({});
const selectedAlbum = ref("album3-14");
const selectedQa = ref("compact-10q");
const selectedModels = reactive(new Set());
const sentrixUrl = ref("");
const judgeUrl = ref("");
const vllmTargetId = ref("");
const rejudgePrompt = ref("");
const judgeProviderId = ref("");
const suiteRunning = ref(false);
const rejudgeSubmitting = ref(false);
const reviewSaving = ref(false);
const loading = ref(true);
const activeView = ref("runs");
const qaBrowserAlbum = ref("album3");
const qaBrowserSet = ref("full-album3-38q");
const qaBrowserItems = ref([]);
const qaBrowserLoading = ref(false);
const qaBrowserError = ref("");
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
const modelName = (run) => run?.model_profile || run?.model_name || run?.profile || "unknown";
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
const fmtTokens = (value) => value == null || !Number.isFinite(Number(value)) ? "-" : `${Math.round(Number(value)).toLocaleString("en-US")} token`;
const scoreClass = (score) => score === 2 ? "score-2" : score === 1 ? "score-1" : score === 0 ? "score-0" : "score-none";
const statusLabel = (status) => ({ done: "完成", running: "进行中", pending: "等待", cancelling: "停止中", failed: "失败", completed: "完成", interrupted: "中断", cancelled: "已取消", partial: "部分采样", not_run: "未执行" }[status] || status || "等待");
function setModelSelected(modelId, checked) {
  if (checked) selectedModels.add(modelId);
  else selectedModels.delete(modelId);
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
function effectiveRunSummary(run) {
  const saved = run?.summary || {};
  const items = run?.items || [];
  const judged = items.filter((item) => item.judge?.score != null && item.judge?.consistency_status !== "inconsistent");
  const recalls = items.map((item) => item.retrieval_recall).filter((value) => Number.isFinite(Number(value)));
  const scores = judged.map((item) => Number(item.judge.score)).filter(Number.isFinite);
  const evidenceScores = items.map((item) => item.evidence_judge?.score).filter((score) => [0, 1, 2].includes(score));
  const retrievalItems = items.filter((item) => Array.isArray(item.retrieval_image_ids) && item.retrieval_image_ids.length);
  const retrievalTp = retrievalItems.reduce((sum, item) => sum + (item.matched_file_names || []).length, 0);
  const retrievalPredicted = retrievalItems.reduce((sum, item) => sum + (item.predicted_file_names || []).length, 0);
  const retrievalGt = retrievalItems.reduce((sum, item) => sum + (item.retrieval_image_ids || []).length, 0);
  const retrievalPrecision = retrievalPredicted ? retrievalTp / retrievalPredicted : (retrievalGt ? 0 : null);
  const retrievalRecall = retrievalGt ? retrievalTp / retrievalGt : null;
  const retrievalF1 = retrievalPrecision != null && retrievalRecall != null && retrievalPrecision + retrievalRecall
    ? 2 * retrievalPrecision * retrievalRecall / (retrievalPrecision + retrievalRecall) : (retrievalGt ? 0 : null);
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
    retrieval_recall_mean: saved.retrieval_recall_mean ?? averageMetric(recalls),
    answer_quality_mean: saved.answer_quality_mean ?? (scores.length ? averageMetric(scores) : null),
    exact_accuracy: saved.exact_accuracy ?? (scores.length ? scores.filter((score) => score === 2).length / scores.length : null),
    core_accuracy: saved.core_accuracy ?? (scores.length ? scores.filter((score) => score >= 1).length / scores.length : null),
    retrieval_precision_micro: saved.retrieval_precision_micro ?? retrievalPrecision,
    retrieval_recall_micro: saved.retrieval_recall_micro ?? retrievalRecall,
    retrieval_f1_micro: saved.retrieval_f1_micro ?? retrievalF1,
    retrieval_metric_count: saved.retrieval_metric_count ?? retrievalItems.length,
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
  if (score === 2) return "2 分：图片支持回答";
  if (score === 1) return "1 分：图片部分支持";
  if (score === 0) return "0 分：图片无法支持";
  return judge?.reason === "not_applicable" ? "不适用" : judge?.reason === "no_answer" ? "无回答，未评分" : "未记录";
}
function itemRetrievalMetrics(item) {
  if (!(item?.retrieval_image_ids || []).length) return "本题无标准图片，不计入检索指标";
  return `精确率 ${fmtPct(item?.retrieval_precision)} · 召回率 ${fmtPct(item?.retrieval_recall)} · F1 ${fmtPct(item?.retrieval_f1)}`;
}
function itemParseRate(item) {
  const stability = item?.agent_stability || {};
  if (stability.json_parse_total == null) return "未记录";
  return `${stability.json_parse_success ?? 0}/${stability.json_parse_total} 次模型输出解析为合法动作 (${fmtPct(stability.json_parse_rate)})`;
}
function completionLabel(item) {
  const completed = item?.agent_stability?.completed_within_steps;
  if (completed === true) return "完成";
  if (completed === false) {
    const reason = normalizedTerminationReason(item) || item?.agent_status;
    return reason ? `未完成（${reason}）` : "未完成";
  }
  return "未记录";
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
function albumLocalUrl(fileName) {
  const album = activeRun.value?.album_id || "";
  return (album && fileName) ? `/api/albums/${encodeURIComponent(album)}/photos/${encodeURIComponent(fileName)}` : "";
}
function decorateImages(list) {
  return (list || []).map((img) => {
    if (img?.media_url) return img;
    const parts = (img?.image_id || "").split("/");
    const album = parts.length === 2 ? parts[0] : (activeRun.value?.album_id || "");
    const file = parts.length === 2 ? parts[1] : (img?.file_name || "");
    const local = (album && file) ? `/api/albums/${encodeURIComponent(album)}/photos/${encodeURIComponent(file)}` : "";
    return local ? { ...img, media_url: local } : img;
  });
}
function itemImages(item, gt = false) {
  if (gt) {
    if (item.gt_images?.length) return decorateImages(item.gt_images);
    return (item.retrieval_image_ids || []).map((id) => ({ image_id: id, file_name: id.split("/").pop(), matched: (item.matched_file_names || []).includes(id.split("/").pop()) }));
  }
  if (item.predicted_images?.length) return decorateImages(item.predicted_images);
  return (item.predicted_file_names || []).map((file_name) => ({ file_name, media_url: albumLocalUrl(file_name) }));
}
function isDirectEvidence(item, imageId) {
  return (item?.answer_evidence_image_ids || []).includes(imageId)
    || (item?.answer_claims || []).some((claim) => (claim.evidence_image_ids || []).includes(imageId));
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
function memoryProfileRows(profile = {}) {
  const memory = profile.memory_profile || {};
  const processMemory = profile.model_process_memory_used_mib || {};
  return [
    ["可比较工作负载显存", memory.comparable_workload_memory_gib == null ? "-" : `${Number(memory.comparable_workload_memory_gib).toFixed(2)} GiB`, "固定基础占用 + 本次 KV Cache 实际峰值", true],
    ["固定基础占用", memory.fixed_base_memory_gib == null ? "-" : `${Number(memory.fixed_base_memory_gib).toFixed(2)} GiB`, "空载模型进程显存 - 预分配 KV Cache 容量", true],
    ["KV Cache 容量", memory.kv_cache_capacity_gib == null ? "-" : `${Number(memory.kv_cache_capacity_gib).toFixed(2)} GiB`, memory.kv_cache_capacity_tokens == null ? "未记录 token 容量" : `${Number(memory.kv_cache_capacity_tokens).toLocaleString("en-US")} token`],
    ["KV Cache 实际峰值", memory.kv_cache_used_peak_gib == null ? "-" : `${Number(memory.kv_cache_used_peak_gib).toFixed(3)} GiB`, `使用率峰值 ${fmtNumber(memory.kv_cache_usage_peak_pct, "%")}`],
    ["模型权重", memory.weight_gib == null ? "-" : `${Number(memory.weight_gib).toFixed(2)} GiB`, `激活峰值 ${memory.peak_activation_gib == null ? "-" : `${memory.peak_activation_gib} GiB`} · CUDA Graph ${memory.cuda_graph_gib == null ? "-" : `${memory.cuda_graph_gib} GiB`}`],
    ["vLLM 进程预留显存", fmtMemory(processMemory.peak), `空载 ${memory.idle_process_memory_gib == null ? "-" : `${Number(memory.idle_process_memory_gib).toFixed(2)} GiB`} · 不用于跨模型需求比较`],
    ["复测进度", `${profile.questions_completed ?? 0}/${profile.questions_total ?? 0} 题`, `请求失败 ${profile.failed_requests ?? 0} · 答案不保存`],
    ["原测评数据一致性", profile.items_integrity_ok === true ? "通过" : profile.status === "completed" ? "未通过" : "待完成", profile.answers_persisted === false ? "原答案未写入" : "记录状态异常"],
  ];
}
function aggregateMetricRows(phase = {}) {
  // The aggregate phase is a historical snapshot and may predate rejudge or
  // consistency filtering. Use the run-level effective summary as the single
  // source of truth for the detail view.
  const summary = effectiveRunSummary(activeRun.value);
  const dist = summary.judge_distribution || {};
  const evidenceDist = summary.evidence_distribution || {};
  return [
    ["图片检索 Precision", fmtPct(summary.retrieval_precision_micro), `图片级微平均 · ${summary.retrieval_metric_count ?? 0} 题有 GT 图`, true],
    ["图片检索 Recall", fmtPct(summary.retrieval_recall_micro), "图片级微平均；列表为题均宏平均", true],
    ["回答质量均分", summary.answer_quality_mean == null ? "-" : `${summary.answer_quality_mean} / 2`, `Judge 0:${dist["0"] || 0} · 1:${dist["1"] || 0} · 2:${dist["2"] || 0}`, true],
    ["步数内 QA 完成率", fmtPct(summary.qa_completion_within_steps_rate), `有效记录 ${summary.qa_completion_valid_count ?? 0} 题`, true],
    ["Token 用量", summary.prompt_tokens_total == null && summary.completion_tokens_total == null ? "未记录" : `${summary.prompt_tokens_total ?? "-"} / ${summary.completion_tokens_total ?? "-"}`, "输入 / 输出 token", true],
    ["平均任务完成时间", fmtMs(summary.agent_task_latency_mean_ms), "每道 QA 从输入到最终回答的平均 Agent 总耗时，不含 Judge", true],
    ["平均任务调用轮数", summary.agent_loop_calls_mean == null ? "未记录" : `${Number(summary.agent_loop_calls_mean).toFixed(2)} 轮`, "仅 Agent/Recovery，不含 L2 Judge、Final Writer 和工具内部模型", true],
    ["图片检索 F1", fmtPct(summary.retrieval_f1_micro), "微平均，平衡噪声与漏召回"],
    ["任务判断准确率", fmtPct(summary.task_decision_accuracy), `标注 ${summary.task_decision_labeled_count ?? 0} 题 · Judge 有效 ${summary.task_decision_valid_count ?? 0} 题`],
    ["证据对应均分", summary.evidence_mean == null ? "未记录" : `${summary.evidence_mean} / 2`, `0:${evidenceDist["0"] || 0} · 1:${evidenceDist["1"] || 0} · 2:${evidenceDist["2"] || 0}`],
    ["证据完全支持率", fmtPct(summary.evidence_fully_supported_rate), "证据 Judge = 2"],
    ["JSON 解析成功率", fmtPct(summary.json_parse_success_rate), summary.json_parse_total == null ? "历史记录未保存解析轨迹" : `${summary.json_parse_success ?? 0}/${summary.json_parse_total} 个需解析模型输出`],
    ["端到端测评总时延（不含 Judge）", fmtMs(summary.benchmark_e2e_latency_excluding_judge_ms), "身份/关系及图片导入开始至全部 QA 完成，已扣除 Judge 时延"],
    ["完全准确率", fmtPct(summary.exact_accuracy), "Judge 评分为 2 的比例"],
    ["核心准确率", fmtPct(summary.core_accuracy), "Judge 评分为 1 或 2 的比例"],
    ["LLM TTFT 均值", fmtMs(summary.llm_ttft_ms_mean), "首 token 响应时间"],
    ["LLM 生成速度", summary.llm_tokens_per_second_mean == null ? "-" : `${Number(summary.llm_tokens_per_second_mean).toFixed(1)} token/s`, "主 Agent 平均生成速度"],
  ];
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
  const savedCalls = itemCallMetrics(item).map((call, globalIndex) => ({ ...call, _globalCallIndex: globalIndex }));
  const traceModels = itemExecutionTrace(item).filter((step) => ["model", "writer", "judge"].includes(String(step.stage || step.type || "")));
  const turns = conversationTurns(item);
  const knownTurns = new Set();
  savedCalls.forEach((call) => knownTurns.add(conversationTurnNumber(call.conversation_turn)));
  traceModels.forEach((step) => knownTurns.add(conversationTurnNumber(step.conversation_turn)));
  turns.forEach((_, index) => knownTurns.add(index));
  if (!knownTurns.size) knownTurns.add(0);
  const metricKeys = new Set(savedCalls.map((call) => `${conversationTurnNumber(call.conversation_turn)}:${call.step_id || ""}`));
  const traceOnlyCalls = traceModels.filter((step) => {
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
  if (turn?.turn_outcome === "final_answer" && turn?.agent_status !== "error") return "完成";
  if (turn?.answer && ["complete", "completed", "done", "success"].includes(String(turn?.agent_status || "").toLowerCase())) return "完成";
  if (turn?.agent_status || turn?.termination_reason || turn?.turn_outcome) return `未完成（${turnTerminationLabel(turn)}）`;
  return "未记录";
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
function toolsForGroupedCall(item, call) {
  if (call?._traceOnly) return [];
  const turnIndex = conversationTurnNumber(call?.conversation_turn);
  return itemToolTrace(item).filter((trace) => {
    const sameTurn = conversationTurnNumber(trace.conversation_turn) === turnIndex;
    if (trace.parent_step_id && call?.step_id) return sameTurn && String(trace.parent_step_id) === String(call.step_id);
    return Number(trace.model_call_index) === Number(call?._globalCallIndex);
  });
}
function unboundTools(item) {
  return itemToolTrace(item).filter((trace) => !Number.isInteger(Number(trace.model_call_index)));
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
  if (key === "photo_import") return `导入 ${phase.accepted_count ?? 0}/${phase.total_photos ?? "?"} 张 · ${fmtSeconds(phaseSeconds(phase, "upload_seconds"))}`;
  if (key === "pipeline_processing" && phase.progress) return `${phase.progress.processed || 0}/${phase.progress.total || 0} 资产完成 · ${phase.progress.failed || 0} 失败 · 阶段墙钟 ${fmtSeconds(phaseSeconds(phase))}`;
  if (key === "qa_eval") return `${activeRun.value?.item_count || 0} 题 · ${fmtSeconds(phaseSeconds(phase))}`;
  const elapsed = phaseSeconds(phase);
  if (elapsed != null) return `耗时 ${fmtSeconds(elapsed)}`;
  return "";
}
function openImage(image) { const url = imageUrl(image); if (url) lightbox.value = { url, name: image.file_name || image.image_id || "图片" }; }
function phasePercent(phase) { const p = phase?.progress; return p?.total ? Math.round((p.processed / p.total) * 100) : 0; }
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
  Object.assign(qaFilters, { search: "", score: "", task_type: "", angle: "", difficulty: "", answerability: "", agent_status: "", primary: "" });
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
async function loadProfiles() { profiles.value = (await post("/api/profiles", { vllm_target_id: vllmTargetId.value })).profiles || []; }
function resetJudgePrompt() { rejudgePrompt.value = config.value?.judge_prompt || ""; }

const exportScoreFilter = ref("all");
const deleteScopeAfterRun = ref(false);
function exportSftTraces() {
  if (!activeRunId.value) return;
  const filter = exportScoreFilter.value === "all" ? "" : `?min_score=${exportScoreFilter.value}`;
  window.open(`/api/runs/${encodeURIComponent(activeRunId.value)}/export-sft${filter}`, "_blank");
}
async function saveJudgePrompt() {
  const prompt = rejudgePrompt.value.trim();
  if (!prompt) { window.alert("提示词不能为空"); return; }
  try {
    await post("/api/judge-prompt", { system_prompt: prompt });
    window.alert("已保存，后续评测将使用此提示词");
  } catch (e) { window.alert("保存失败：" + e.message); }
}
async function startRejudge() {
  if (!canRejudge.value || rejudgeSubmitting.value) return;
  if (!window.confirm(`仅使用现有 ${activeRun.value.item_count || 0} 条 Agent 回答重新调用 Judge。旧评分将保留在历史记录中，确定开始？`)) return;
  rejudgeSubmitting.value = true;
  error.value = "";
  try {
   await post(`/api/runs/${encodeURIComponent(activeRunId.value)}/rejudge`, {
    judge_url: judgeUrl.value,
    system_prompt: rejudgePrompt.value,
    });
    await loadRuns(); await loadActiveRun(); startPolling();
  } catch (e) { error.value = e.message; }
  finally { rejudgeSubmitting.value = false; }
}
async function startSuite() {
  if (hasRunning.value || suiteRunning.value) return;
  if (!selectedModels.size) { window.alert("请至少选择一个模型"); return; }
  suiteRunning.value = true;
  try {
   const result = await post("/api/runs", { album_id: selectedAlbum.value, qa_set: selectedQa.value, models: [...selectedModels], sentrix_url: sentrixUrl.value, judge_url: judgeUrl.value, vllm_target_id: vllmTargetId.value, delete_scope_after_run: deleteScopeAfterRun.value });
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
async function init() {
  try {
    config.value = await api("/api/config");
    vllmTargets.value = config.value.vllm_targets || {};
    vllmTargetId.value = config.value.default_vllm_target_id || Object.keys(vllmTargets.value)[0] || "";
    sentrixUrl.value = config.value.default_sentrix_url; judgeUrl.value = config.value.default_judge_url; rejudgePrompt.value = config.value.custom_judge_prompt || config.value.judge_prompt || "";
    judgeProviderId.value = config.value.default_judge_provider_id || (config.value.judge_providers?.[0]?.id || "");
    manifests.value = (await api("/api/manifests")).manifests || [];
    await loadRuns(); await loadProfiles();
    const current = runs.value.find((run) => ["running", "pending"].includes(run.status));
    if (current) { activeRunId.value = current.run_id; await loadActiveRun({ resetPage: true }); startPolling(); }
  } catch (e) { error.value = e.message; } finally { loading.value = false; }
}
const qaBrowserOptions = computed(() => manifests.value.find((m) => m.album_id === qaBrowserAlbum.value)?.qa_sets || []);
async function loadQaBrowser() {
  qaBrowserLoading.value = true;
  qaBrowserError.value = "";
  try {
    const data = await api(`/api/qa-dataset?album_id=${encodeURIComponent(qaBrowserAlbum.value)}&qa_set=${encodeURIComponent(qaBrowserSet.value)}`);
    qaBrowserItems.value = data.items || [];
  } catch (e) { qaBrowserError.value = e.message; qaBrowserItems.value = []; }
  finally { qaBrowserLoading.value = false; }
}
function qaTypeLabel(t) {
  return ({event_memory_qa:"事件记忆",single_evidence_memory_qa:"单图证据",relationship_qa:"关系问答",multi_turn_clarify:"多轮澄清",multi_turn_disambiguation:"多轮消歧",ambiguous_retrieval:"模糊检索",evidence_insufficient:"证据不足",unsupported_retrieval:"无依据检索",instruction_injection:"指令注入",prompt_injection:"提示注入",data_exfiltration:"数据泄露",authority_impersonation:"权限伪造",mixed_injection:"混合注入",indirect_injection:"间接注入",jailbreak_attempt:"越狱尝试"}[t]) || t || "未分类";
}
function qaActionBadge(a) {
  return ({answer:"回答",refuse:"拒答",clarify:"澄清"}[a]) || a || "-";
}
function qaAnswerabilityLabel(v) {
  return ({answerable:"可回答",unanswerable:"不可回答",ambiguous:"有歧义",unsafe_request:"不安全请求",answerable_after_clarification:"澄清后可回答",mixed:"混合"}[v]) || v || "-";
}
function qaPhotoUrl(albumId, relPath) {
  const fileName = relPath.split("/").pop();
  return `/api/albums/${encodeURIComponent(albumId)}/photos/${encodeURIComponent(fileName)}`;
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
onUnmounted(() => { destroyed = true; if (pollTimer) clearTimeout(pollTimer); });
</script>

<template>
  <main v-if="!loading" class="app-shell">
    <nav class="view-tabs">
      <button :class="['view-tab', { active: activeView === 'runs' }]" @click="activeView = 'runs'">评测运行</button>
      <button :class="['view-tab', { active: activeView === 'qa-browser' }]" @click="activeView = 'qa-browser'; loadQaBrowser()">QA 数据集浏览</button>
    </nav>
    <template v-if="activeView === 'runs'">
    <section class="section config-section">
      <div class="section-head">
<h2>评测配置</h2>
<span v-if="hasRunning" class="live-badge">实时更新中</span>
</div>
      <div class="config-grid">
        <label>相册<select v-model="selectedAlbum">
<option v-for="manifest in manifests" :key="manifest.album_id" :value="manifest.album_id">{{ manifest.album_name }} ({{ manifest.face_count }}人 / {{ manifest.photo_count }}图)</option>
</select>
</label>
        <label>QA 数据集<select v-model="selectedQa">
<option v-for="qa in qaOptions" :key="qa" :value="qa">{{ qa }}</option>
</select>
</label>
        <label>Sentrix 后端<input v-model="sentrixUrl" type="text" />
</label>
       <label>Judge 服务<input v-model="judgeUrl" type="text" />
</label>
        <label>Judge 模型<select v-model="judgeProviderId">
<option v-for="provider in (config?.judge_providers || [])" :key="provider.id" :value="provider.id">{{ provider.label }}</option>
</select>
<span class="config-help">{{ (config?.judge_providers || []).find((p) => p.id === judgeProviderId)?.model || '-' }}</span>
</label>
        <label>vLLM 服务目标<select v-model="vllmTargetId" @change="loadProfiles">
<option v-for="(target, id) in vllmTargets" :key="id" :value="id">{{ target.label }}</option>
</select>
<span class="config-help">{{ vllmTargets[vllmTargetId]?.manager_url }} → {{ vllmTargets[vllmTargetId]?.model_base_url }}</span>
</label>
      </div>
      <div class="model-picker">
<span class="field-label">选择模型（可多选，串行测试）</span>
<label v-for="profile in profiles" :key="profile.id" class="check" :class="{ active: selectedModels.has(profile.id) }">
<input type="checkbox" :checked="selectedModels.has(profile.id)" :disabled="!profile.available" @change="setModelSelected(profile.id, $event.target.checked)" />{{ profile.id }}<span v-if="!profile.available">（不可用）</span>
</label>
</div>
      <div class="actions">
<label class="check"><input type="checkbox" v-model="deleteScopeAfterRun" :disabled="suiteRunning || hasRunning" />完成后删除相册</label>
<button class="btn" :disabled="suiteRunning || hasRunning" @click="startSuite">{{ hasRunning ? '已有任务运行中' : '启动评测' }}</button>
<button class="btn warn" :disabled="!hasRunning" @click="stopSuite">停止全部</button>
<button class="btn ghost" @click="loadProfiles">刷新模型列表</button>
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
<th>题均召回率</th>
<th>质量均分</th>
<th>
</th>
</tr>
</thead>
<tbody>
<tr v-for="run in runs" :key="run.run_id" class="run-row" :class="{ selected: activeRunId === run.run_id }" @click="selectRun(run)">
<td>
<b>{{ modelName(run) }}</b>
</td>
<td class="muted small">{{ albumName(run) }}</td>
<td class="muted small">{{ fmtDate(run.started_at) }}</td>
<td class="muted small">{{ duration(run) }}</td>
<td>
<span class="phase-status" :class="run.status">{{ statusLabel(run.status) }}</span>
</td>
<td>{{ run.summary?.completed || 0 }}/{{ run.summary?.total || run.qa_count || 0 }}</td>
<td>{{ fmtPct(run.summary?.retrieval_recall_mean) }}</td>
<td>{{ run.summary?.answer_quality_mean ?? "-" }}</td>
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
      <div class="section-head">
<h2>{{ modelName(activeRun) }} · {{ albumName(activeRun) }} · {{ qaName(activeRun) }}</h2>
<span class="phase-status" :class="activeRun.status">{{ statusLabel(activeRun.status) }}</span>
<span class="field-label">导出轨迹</span>
<select v-model="exportScoreFilter" class="input compact">
  <option value="all">全部</option>
  <option value="1">1 分及以上</option>
  <option value="2">2 分</option>
</select>
<button class="btn compact" @click="exportSftTraces">导出 SFT json</button>
</div>
      <p class="run-meta">开始 {{ fmtDate(activeRun.started_at) }} · 总耗时 {{ duration(activeRun) }}</p>
      <section class="rejudge-card">
        <div class="rejudge-head">
<div>
<h3>重新 Judge 评分</h3>
<p>只复用本次运行已有的题目、标准答案和 Agent 回答，不重新执行相册处理、模型切换或 Agent 问答。</p>
</div>
<span v-if="activeRejudge" class="phase-status" :class="activeRejudge.status">{{ statusLabel(activeRejudge.status) }}</span>
</div>
        <label class="rejudge-prompt">Judge System Prompt<textarea v-model="rejudgePrompt" :disabled="activeRejudge?.status === 'running'" rows="8" spellcheck="false">
</textarea>
</label>
        <div class="rejudge-toolbar">
<span class="muted small">{{ rejudgePrompt.length }} 字符 · Judge {{ config?.judge_model || '-' }}</span>
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
<p class="metric-calc-time">指标计算耗时 {{ fmtSeconds(phaseSeconds(activeRun.phases?.gpu_metrics)) }} · 模型进程显存为 NVML 按 PID 汇总的实际占用，KV Cache 为 vLLM 逻辑使用率</p>
<div class="phase-metrics">
<div v-for="row in gpuMetricRows(activeRun.phases?.gpu_metrics)" :key="row[0]" :class="['phase-metric', { 'priority-metric': row[3] }]">
<span>{{ row[0] }}</span>
<strong>{{ row[1] }}</strong>
<small>{{ row[2] }}</small>
</div>
</div>
</article>
        <article v-if="activeRun.memory_profile" class="phase-card result-phase-card gpu-result-card">
<div class="phase-title">
<b>可比较显存复测</b>
<span class="phase-status" :class="activeRun.memory_profile.status">{{ statusLabel(activeRun.memory_profile.status) }}</span>
</div>
<p class="metric-calc-time">复用现有相册与问题，不运行 Benchmark/Judge，不保存本次回答。可比较显存 = 固定基础占用 + KV Cache 实际峰值。</p>
<p v-if="activeRun.memory_profile.error" class="error">{{ activeRun.memory_profile.error }}</p>
<div class="phase-metrics">
<div v-for="row in memoryProfileRows(activeRun.memory_profile)" :key="row[0]" :class="['phase-metric', { 'priority-metric': row[3] }]">
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
                    <div class="result-message result-assistant-message"><small>模型回答</small><p>{{ turn.answer || "未完成" }}</p></div>
                    <div class="result-turn-scores">
                      <span>任务行为 <b>期望{{ actionLabel(turn.expected_action) }} / 实际{{ actionLabel(turn.task_judge?.actual_action) }}</b><em :class="turn.task_judge?.correct === true ? 'pass' : turn.task_judge?.correct === false ? 'fail' : ''">{{ turn.task_judge?.correct === true ? '一致' : turn.task_judge?.correct === false ? '不一致' : '未记录' }}</em></span>
                      <span>回答质量 <b>{{ turnScore(turn.judge?.score) }}</b></span>
                      <span>图片证据 <b>{{ evidenceScoreLabel(turn.evidence_judge) }}</b></span>
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
                  <h4>{{ conversationTurns(itemDetail(summary)).length > 1 ? "最终一轮回答" : "模型回答" }}</h4><p>{{ itemDetail(summary).answer || itemDetail(summary).error || "未完成" }}</p>
                  <div class="capability-grid">
                    <span><small>任务判断</small><b>{{ taskDecisionLabel(itemDetail(summary)) }}</b></span>
                    <span><small>任务判断结果</small><b>{{ itemDetail(summary).task_judge?.correct === true ? "一致" : itemDetail(summary).task_judge?.correct === false ? "不一致" : "未记录" }}</b></span>
                    <span v-if="itemDetail(summary).task_judges?.length > 1"><small>多轮评分口径</small><b>每轮独立评分，并携带截至该轮的完整对话</b></span>
                    <span><small>证据对应</small><b>{{ evidenceScoreLabel(itemDetail(summary).evidence_judge) }}</b></span>
                    <span><small>图片检索</small><b>{{ itemRetrievalMetrics(itemDetail(summary)) }}</b></span>
                    <span><small>JSON 解析</small><b>{{ itemParseRate(itemDetail(summary)) }}</b></span>
                    <span><small>步数内完成</small><b>{{ completionLabel(itemDetail(summary)) }}</b></span>
                  </div>
                  <h4>模型召回图片（{{ itemImages(itemDetail(summary)).length }}）</h4>
                  <div class="image-grid"><div v-for="image in itemImages(itemDetail(summary))" :key="image.asset_id || image.file_name" class="image-tile"><img v-if="imageUrl(image)" :src="imageUrl(image)" :alt="image.file_name" @click="openImage(image)" /><span v-else class="image-empty">无图</span><span class="image-label">{{ image.file_name || image.image_id }}</span></div><span v-if="!itemImages(itemDetail(summary)).length" class="muted small">模型没有返回可识别的图片</span></div>
                </div>
                <div>
                  <h4>正确答案</h4><p>{{ itemDetail(summary).reference_answer }}</p>
                  <h4>检索 GT 图片（{{ itemImages(itemDetail(summary), true).length }}）</h4>
                  <div class="image-grid"><div v-for="image in itemImages(itemDetail(summary), true)" :key="image.asset_id || image.file_name" class="image-tile"><img v-if="imageUrl(image)" :src="imageUrl(image)" :alt="image.file_name" @click="openImage(image)" /><span v-else class="image-empty">无图</span><span class="image-label">{{ image.file_name || image.image_id }}<em v-if="image.matched === false"> · 未召回</em></span></div></div>
                  <h4 v-if="judgeReason(itemDetail(summary).judge)">回答质量评分说明</h4><p v-if="judgeReason(itemDetail(summary).judge)" class="muted">{{ judgeReason(itemDetail(summary).judge) }}</p>
                  <h4 v-if="judgeReason(itemDetail(summary).task_judge)">任务判断说明</h4><p v-if="judgeReason(itemDetail(summary).task_judge)" class="muted">{{ judgeReason(itemDetail(summary).task_judge) }}</p>
                  <h4 v-if="judgeReason(itemDetail(summary).evidence_judge)">图片证据评分说明</h4><p v-if="judgeReason(itemDetail(summary).evidence_judge)" class="muted">{{ judgeReason(itemDetail(summary).evidence_judge) }}</p>
                </div>
              </div>
              <section class="qa-performance">
                <div class="timing-breakdown">
                  <span>端到端 <b>{{ fmtMs(itemTimingBreakdown(itemDetail(summary)).wall_clock_ms) }}</b></span><span>Agent 总耗时 <b>{{ fmtMs(itemTimingBreakdown(itemDetail(summary)).agent_wall_ms) }}</b></span><span>模型 <b>{{ fmtMs(itemTimingBreakdown(itemDetail(summary)).model_ms) }}</b></span><span>工具 <b>{{ hasToolTrace(itemDetail(summary)) ? fmtMs(itemTimingBreakdown(itemDetail(summary)).tool_ms) : "未记录" }}</b></span><span>Judge <b>{{ fmtMs(itemTimingBreakdown(itemDetail(summary)).judge_ms) }}</b></span><span>其他 <b>{{ hasToolTrace(itemDetail(summary)) ? fmtMs(itemTimingBreakdown(itemDetail(summary)).other_ms) : "未记录" }}</b></span>
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
                          <span class="call-round">{{ callIndex + 1 }}</span><strong>{{ callTypeLabel(call) }}</strong><span v-if="['agent','recovery'].includes(callType(call))" :class="['call-outcome', callOutcomeClass(call)]">{{ callOutcome(call) }}</span><span>{{ call.role || "-" }} · {{ call.model || modelName(activeRun) }}</span><span>TTFT {{ fmtMs(call.ttft_ms) }}</span><span>总时延 {{ fmtMs(call.total_ms) }}</span><span>Token {{ call.preflight_prompt_tokens ?? call.prompt_tokens ?? "-" }} / {{ call.completion_tokens ?? "-" }}</span><span>{{ fmtTokenRate(call.tokens_per_second) }}</span><span class="stream-state" :class="{ streamed: call.streamed === true }">{{ callStatus(call) }}</span>
                        </summary>
                        <div class="call-node-body">
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
                              <p class="muted small">完整提示词</p><pre>{{ JSON.stringify(debugStepForCallInGroup(itemDetail(summary), group, call).prompt, null, 2) }}</pre>
                              <p class="muted small">模型原始回答</p><pre>{{ debugStepForCallInGroup(itemDetail(summary), group, call).raw_full || debugStepForCallInGroup(itemDetail(summary), group, call).raw }}</pre>
                            </div>
                          </details>
                          <div v-if="showToolBranch(call) && toolsForGroupedCall(itemDetail(summary), call).length" class="tool-tree">
                            <details v-for="(trace, toolIndex) in toolsForGroupedCall(itemDetail(summary), call)" :key="toolIndex" class="tool-node">
                              <summary><strong>{{ trace.tool || "未知工具" }}</strong><span>{{ toolStatusLabel(trace) }}</span><span>总耗时 {{ trace.latency_s == null ? "-" : fmtMs(Number(trace.latency_s) * 1000) }}</span><span class="binding-source">{{ toolBindingLabel(trace) }}</span></summary>
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
                    <details v-if="group.turn.agent_status || group.turn.termination_reason || group.turn.turn_outcome" class="call-node guard-node turn-guard-node">
                      <summary><strong>第 {{ group.turnIndex + 1 }} 轮结束状态</strong><span>{{ turnCompletionLabel(group.turn) }}</span></summary>
                      <div class="guard-grid"><span>运行状态 <b>{{ group.turn.agent_status || "未记录" }}</b></span><span>终止原因 <b>{{ turnTerminationLabel(group.turn) }}</b></span><span>轮次结果 <b>{{ group.turn.turn_outcome || "未记录" }}</b></span><span>JSON 解析 <b>{{ group.turn.parse_status || "未记录" }}</b></span><span>恢复次数 <b>{{ turnRecoveryCount(group) }}</b></span><span>下一步 <b>{{ group.turn.next_step || "未记录" }}</b></span></div>
                    </details>
                  </section>
                </div>
                <p v-else class="qa-performance-empty">该历史结果未记录主模型调用性能或失败轨迹。</p>
                <details v-if="unboundTools(itemDetail(summary)).length" class="call-node unbound-tools">
                  <summary><strong>未绑定模型轮次的工具序列</strong><span>{{ unboundTools(itemDetail(summary)).length }} 次 · 历史数据未保存精确轮次关系</span></summary>
                  <div class="call-node-body tool-tree"><details v-for="(trace, toolIndex) in unboundTools(itemDetail(summary))" :key="toolIndex" class="tool-node"><summary><strong>#{{ toolIndex + 1 }} {{ trace.tool || "未知工具" }}</strong><span>{{ toolStatusLabel(trace) }}</span><span>总耗时 {{ trace.latency_s == null ? "-" : fmtMs(Number(trace.latency_s) * 1000) }}</span></summary></details></div>
                </details>
                <details v-if="guardSummary(itemDetail(summary)).recorded" class="call-node guard-node final-agent-status"><summary><strong>{{ conversationTurns(itemDetail(summary)).length > 1 ? "最终一轮状态汇总" : "Agent 结束状态" }}</strong><span>{{ completionLabel(itemDetail(summary)) }}</span></summary><div class="guard-grid"><span>运行状态 <b>{{ guardSummary(itemDetail(summary)).status }}</b></span><span>终止原因 <b>{{ guardSummary(itemDetail(summary)).termination || "正常完成" }}</b></span><span>恢复次数 <b>{{ guardSummary(itemDetail(summary)).recoveries }}</b></span><span>运行详情 <b>{{ itemDetail(summary).agent_reason || "-" }}</b></span></div></details>
              </section>
                            <!-- Agent 2.0 目标与证据账本轨迹 -->
              <details v-if="getAgent2Trace(itemDetail(summary))" class="call-node agent2-trace-node">
                <summary><strong>Agent 2.0 目标驱动与证据账本（TaskState / EvidenceLedger）</strong><span>{{ getAgent2LedgerEntries(itemDetail(summary)).length }} 条证据 · {{ getAgent2Requirements(itemDetail(summary)).length }} 项需求</span></summary>
                <div class="debug-trace-body">
                  <div v-if="getAgent2Trace(itemDetail(summary))?.task_declaration" class="call-node-body">
                    <p class="muted small"><strong>规划目标 (Goal)：</strong> {{ getAgent2Trace(itemDetail(summary)).task_declaration.goal }}</p>
                    <p class="muted small"><strong>空间作用域 (Scope)：</strong> {{ getAgent2Trace(itemDetail(summary)).task_declaration.scope_id }}</p>
                  </div>
                  <div v-if="getAgent2Requirements(itemDetail(summary)).length" class="call-node-body">
                    <p class="muted small"><strong>证据需求分解 (Requirements)：</strong></p>
                    <table class="history-table" style="margin-top: 6px;">
                      <thead><tr><th>ID</th><th>证据类型</th><th>状态</th><th>描述</th><th>证据引用</th><th>未满足原因</th></tr></thead>
                      <tbody>
                        <tr v-for="(req, rIdx) in getAgent2Requirements(itemDetail(summary))" :key="rIdx">
                          <td>{{ req.id }}</td>
                          <td><code>{{ req.evidence_type }}</code></td>
                          <td><span :class="req.status === 'satisfied' ? 'tag-green' : req.status === 'partially_supported' ? 'tag-yellow' : 'tag-muted'">{{ req.status }}</span></td>
                          <td>{{ req.description || '-' }}</td>
                          <td>{{ (req.evidence_refs || []).join(', ') || '-' }}</td>
                          <td><span v-if="req.unmet_reason" class="tag-red">{{ req.unmet_reason }}</span><span v-else>-</span></td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                  <div v-if="getAgent2LedgerEntries(itemDetail(summary)).length" class="call-node-body">
                    <p class="muted small"><strong>证据账本记录 (Evidence Ledger Entries)：</strong></p>
                    <pre>{{ JSON.stringify(getAgent2LedgerEntries(itemDetail(summary)), null, 2) }}</pre>
                  </div>
                  <div v-if="getAgent2Trace(itemDetail(summary))?.planner_decisions?.length" class="call-node-body">
                    <p class="muted small"><strong>规划决策 (Planner Decisions)：</strong></p>
                    <pre>{{ JSON.stringify(getAgent2Trace(itemDetail(summary)).planner_decisions, null, 2) }}</pre>
                  </div>
                </div>
              </details>

              <details v-if="runtimeDebugTurns(itemDetail(summary)).length" class="call-node debug-trace-node">
                <summary><strong>完整运行时轨迹（提示词 / 回答 / 工具输入输出 / 评判）</strong><span>{{ runtimeDebugTurns(itemDetail(summary)).length }} 轮</span></summary>
                <div class="debug-trace-body">
                  <details v-for="(turn, turnIndex) in runtimeDebugTurns(itemDetail(summary))" :key="turnIndex" class="debug-turn">
                    <summary>第 {{ turn.index + 1 }} 轮 · {{ turn.message || "问答" }}</summary>
                    <div class="debug-steps">
                      <details v-for="(step, stepIndex) in turn.debug_trace" :key="stepIndex" class="debug-step" :class="'debug-step-' + (step.type || 'step')">
                        <summary><strong>{{ step.type === 'model' ? '模型步骤' : step.type === 'tool' ? '工具步骤' : step.type === 'judge' ? '评判步骤' : step.type }}</strong><span>{{ step.status || '' }}</span></summary>
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
            <p>按题检查对话设计、参考回答、GT 图片和证据元数据。</p>
          </div>
          <div class="qa-browser-count"><strong>{{ qaBrowserItems.length }}</strong><span>道题目</span></div>
        </header>
        <div class="qa-browser-toolbar">
          <label><span>相册</span><select v-model="qaBrowserAlbum" @change="qaBrowserSet = (qaBrowserOptions[0] || 'compact-10q'); loadQaBrowser()">
            <option v-for="manifest in manifests" :key="manifest.album_id" :value="manifest.album_id">{{ manifest.album_name }} · {{ manifest.face_count }} 人 / {{ manifest.photo_count }} 图</option>
          </select></label>
          <label><span>QA 数据集</span><select v-model="qaBrowserSet" @change="loadQaBrowser">
            <option v-for="qa in qaBrowserOptions" :key="qa" :value="qa">{{ qa }}</option>
          </select></label>
          <button class="btn ghost" @click="loadQaBrowser">刷新数据</button>
        </div>
        <p v-if="qaBrowserError" class="error">{{ qaBrowserError }}</p>
        <div v-if="qaBrowserLoading" class="qa-browser-empty">正在加载数据集…</div>
        <div v-else class="qa-browser-list">
          <article v-for="(item, idx) in qaBrowserItems" :key="item.qa_id || idx" class="qa-review-card">
            <header class="qa-review-header">
              <div class="qa-review-index">{{ String(idx + 1).padStart(2, '0') }}</div>
              <div class="qa-review-title">
                <div class="qa-browser-meta">
                  <span class="qa-badge" :class="'badge-' + (item.expected_action || 'answer')">{{ qaActionBadge(item.expected_action) }}</span>
                  <span class="qa-type-tag">{{ qaTypeLabel(item.question_type) }}</span>
                  <span class="qa-answerability-tag">{{ qaAnswerabilityLabel(item.answerability) }}</span>
                  <span v-if="item.difficulty" class="qa-difficulty-tag">{{ item.difficulty }}</span>
                  <span v-if="qaConversationTurns(item).length > 1" class="multi-turn-badge">{{ qaConversationTurns(item).length }} 轮</span>
                </div>
                <span class="qa-review-id">{{ item.qa_id || `qa-${idx + 1}` }}</span>
              </div>
            </header>
            <div class="qa-review-layout" :class="{ 'no-evidence': !(item.retrieval_image_ids && item.retrieval_image_ids.length) }">
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
              <aside v-if="item.retrieval_image_ids && item.retrieval_image_ids.length" class="qa-evidence-panel">
                <div class="qa-evidence-head"><div><span>RETRIEVAL GROUND TRUTH</span><strong>检索 GT 图片</strong><small>同一事件内允许召回的相关图片</small></div><b>{{ item.retrieval_image_ids.length }}</b></div>
                <div class="gt-gallery">
                  <button v-for="imgId in item.retrieval_image_ids" :key="imgId" class="gt-thumb-card" :class="{ 'direct-evidence': isDirectEvidence(item, imgId) }" type="button" @click="lightbox = { url: qaPhotoUrl(qaBrowserAlbum, imgId), name: imgId.split('/').pop() }">
                    <img :src="qaPhotoUrl(qaBrowserAlbum, imgId)" :alt="imgId" loading="lazy" />
                    <span>{{ imgId.split('/').pop() }}<em>{{ isDirectEvidence(item, imgId) ? '直接证据' : '事件相关' }}</em></span>
                  </button>
                </div>
              </aside>
            </div>
            <div class="qa-review-foot">
              <details v-if="item.answer_claims && item.answer_claims.length" class="qa-detail-block">
                <summary>证据声明 <b>{{ item.answer_claims.length }}</b></summary>
                <div v-for="(claim, ci) in item.answer_claims" :key="ci" class="qa-claim"><span class="claim-type">{{ claim.support_type || claim.claim_id }}</span><span class="claim-text">{{ claim.text }}</span><span v-if="claim.evidence_image_ids?.length" class="claim-evidence">{{ claim.evidence_image_ids.map(x => x.split('/').pop()).join(' · ') }}</span></div>
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
