<script setup>
import { computed, nextTick, onMounted, onUnmounted, reactive, ref, watch } from "vue";
import * as echarts from "echarts";

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
const runPage = ref({ page: 1, page_size: 20, total: 0, pages: 1, has_previous: false, has_next: false, active_count: 0 });
const activeRunId = ref(null);
const activeRun = ref(null);
const keyframeAnalysis = ref(null);
const keyframeAnalysisLoading = ref(false);
const keyframeAnalysisError = ref("");
const memoryEffectiveness = ref(null);
const memoryAttribution = computed(() => memoryEffectiveness.value?.details?.memory_answer_attribution || null);
const errorChainDiagnosis = computed(() => memoryEffectiveness.value?.details?.error_chain_diagnosis || null);
const memoryValidityArchitecture = computed(() => {
  const levels = memoryEffectiveness.value?.levels || [];
  const level = (key) => levels.find((item) => item.key === key) || {};
  const raw = level("L0");
  const scene = level("L1");
  const events = level("L2");
  const rules = level("L3");
  const answers = level("L4+");
  const details = memoryEffectiveness.value?.details || {};
  const analysis = keyframeAnalysis.value?.status === "ready" ? keyframeAnalysis.value : null;
  const runSummary = effectiveRunSummary(activeRun.value);
  const visual = details.visual_calls || {};
  const ocr = details.ocr_calls || {};
  const correctness = details.answer_correctness || {};
  const ratio = (value, total) => total ? fmtPct(Number(value || 0) / total) : "-";
  return [
    {
      key: "L0", name: "原始媒体与检索", stage: "记忆库规模 → 测评检索结果",
      nodes: [
        { label: "原始媒体创建量", value: raw.created || 0, note: "记忆库输入规模，不代表测评有效" },
        { label: "原始图片交付 Recall", value: runSummary.image_retrieval_recall_micro == null ? "未计算" : fmtPct(runSummary.image_retrieval_recall_micro), note: `${fmtPct(runSummary.image_retrieval_precision_micro)} Precision · 按模型使用（交付）口径；工具召回维度见下方两列矩阵` },
        { label: "关键帧父级 Recall", value: analysis ? ratio(analysis.retrieval.video_gt_targets_hit_by_keyframe, analysis.retrieval.video_gt_targets) : "未计算", note: analysis ? `${analysis.retrieval.video_gt_targets_hit_by_keyframe || 0}/${analysis.retrieval.video_gt_targets || 0} 个 GT 源视频被覆盖` : "关键帧接口尚未返回" },
        { label: "关键帧返回 Precision", value: analysis ? ratio(analysis.retrieval.predicted_relevant_keyframes, analysis.retrieval.predicted_keyframes) : "未计算", note: analysis ? `${analysis.retrieval.predicted_relevant_keyframes || 0}/${analysis.retrieval.predicted_keyframes || 0} 个返回关键帧与 GT 父视频一致` : "关键帧接口尚未返回" },
      ],
    },
    {
      key: "L1", name: "场景解析记忆", stage: "视觉 / OCR → 场景节点",
      nodes: [
        { label: "解析轨迹覆盖率", value: ratio(scene.effective, scene.created), note: `${scene.effective || 0}/${scene.created || 0} 个解析结果在测评轨迹出现；不是语义正确率` },
        { label: "视觉工具成功率", value: ratio(visual.effective, visual.total), note: `${visual.effective || 0}/${visual.total || 0} 次 inspect_photo 成功` },
        { label: "OCR 工具成功率", value: ratio(ocr.effective, ocr.total), note: `${ocr.effective || 0}/${ocr.total || 0} 次 read_photo_text 成功` },
      ],
    },
    {
      key: "L2", name: "事件记忆", stage: "场景 → 时间 / 地点 / 事件",
      nodes: [
        { label: "事件图谱覆盖率", value: ratio(events.effective, events.created), note: `${events.effective || 0}/${events.created || 0} 个事件在测评工具图谱中出现；不是事实正确率` },
        { label: "跨媒体关联", value: memoryEffectiveness.value?.data_quality?.graph_ready ? "已校验" : "不完整", note: "事件 ID → 媒体 → 时间 / 地点字段" },
      ],
    },
    {
      key: "L3", name: "证据规则", stage: "事件记忆 → 目标证据",
      nodes: [
        { label: "证据闭合率（代理）", value: ratio(rules.effective, rules.created), note: `${rules.effective || 0}/${rules.created || 0} 项需求满足；仅表示证据链闭合` },
        { label: "未闭合需求", value: rules.ineffective || 0, note: "open / running / failed，需继续形成证据" },
      ],
    },
    {
      key: "L4+", name: "回答结果", stage: "证据 → 最终回答 Judge",
      nodes: [
        { label: "Agent 完整闭环率", value: ratio(answers.effective, answers.created), note: `${answers.effective || 0}/${answers.created || 0} 条回答 complete` },
        { label: "Exact Accuracy", value: ratio(correctness.correct, answers.created), note: `正确 ${correctness.correct || 0} · 部分 ${correctness.partial || 0} · 错误 ${correctness.incorrect || 0}` },
      ],
    },
  ];
});
const memoryRecallMatrices = computed(() => {
  const attribution = memoryAttribution.value || {};
  const evidenceBlocks = memoryEffectiveness.value?.details?.evidence_effectiveness || {};
  const correctness = memoryEffectiveness.value?.details?.answer_correctness || {};
  const judgedTotal = Number(correctness.correct || 0) + Number(correctness.partial || 0) + Number(correctness.incorrect || 0);
  const pct = (value, total) => total ? fmtPct(value / total) : "-";
  const DIMS = [
    {
      key: "tool", title: "工具召回命中矩阵",
      note: "工具召回命中：GT 媒体进入检索 / 工具返回集合（视频按关键帧父级覆盖）",
      hitLabel: "工具召回命中", missLabel: "工具召回未命中", stageL1: "GT 是否进入检索候选",
    },
    {
      key: "usage", title: "模型使用命中矩阵",
      note: "模型使用命中：GT 媒体被模型显式用于回答（严格交付口径，与逐题指标一致）",
      hitLabel: "模型使用命中", missLabel: "模型使用未命中", stageL1: "GT 是否被模型用于回答",
    },
  ];
  return DIMS.map((dim) => {
    const scope = attribution[dim.key] || {};
    const matrix = scope.matrix || {};
    const hit = Number(scope.hit_items || 0);
    const miss = Number(scope.miss_items || 0);
    const eligible = Number(scope.eligible || 0);
    const evBlock = evidenceBlocks[dim.key] || {};
    const evRow = (k) => evBlock[k] || {};
    const imageEv = evRow("image");
    const videoEv = evRow("video_keyframe");
    const eventEv = evidenceBlocks.event || {};
    const hitWrong = Number(matrix.hit_incorrect || 0);
    const missWrong = Number(matrix.miss_incorrect || 0);
    return {
      key: dim.key, title: dim.title, note: dim.note, eligible, hit, miss,
      layers: [
        {
          key: "L0", name: "评测题目范围", stage: "回答结果与 GT 媒体对齐",
          nodes: [
            { label: "全量已评分题", value: pct(judgedTotal, judgedTotal), note: `${judgedTotal} 题；正确 ${correctness.correct || 0} · 部分 ${correctness.partial || 0} · 错误 ${correctness.incorrect || 0}` },
            { label: "可归因题占比", value: pct(eligible, judgedTotal), note: `${eligible}/${judgedTotal} 题有 GT 媒体；不可回答题不进归因` },
          ],
        },
        {
          key: "L1", name: "命中结果", stage: dim.stageL1,
          nodes: [
            { label: dim.hitLabel, value: "已找到", note: `${hit}/${eligible} 题` },
            { label: dim.missLabel, value: "未找到", note: `${miss}/${eligible} 题` },
          ],
        },
        {
          key: "L2", name: "回答评分结果", stage: "在命中 / 未命中分支内看 Judge",
          nodes: [
            { label: "命中 → 正确", value: matrix.hit_correct || 0, note: `${pct(matrix.hit_correct, eligible)} 总体 · ${pct(matrix.hit_correct, hit)} 命中分支` },
            { label: "命中 → 部分", value: matrix.hit_partial || 0, note: `${pct(matrix.hit_partial, eligible)} 总体 · ${pct(matrix.hit_partial, hit)} 命中分支` },
            { label: "命中 → 错误", value: hitWrong, note: `${pct(hitWrong, eligible)} 总体 · ${pct(hitWrong, hit)} 命中分支` },
            { label: "未命中 → 正确", value: matrix.miss_correct || 0, note: `${pct(matrix.miss_correct, eligible)} 总体 · ${pct(matrix.miss_correct, miss)} 未命中分支` },
            { label: "未命中 → 部分", value: matrix.miss_partial || 0, note: `${pct(matrix.miss_partial, eligible)} 总体 · ${pct(matrix.miss_partial, miss)} 未命中分支` },
            { label: "未命中 → 错误", value: missWrong, note: `${pct(missWrong, eligible)} 总体 · ${pct(missWrong, miss)} 未命中分支` },
          ],
          groups: [
            {
              label: dim.hitLabel, total: hit, note: "命中分支",
              nodes: [
                { label: "回答正确", value: pct(matrix.hit_correct, eligible), note: `${pct(matrix.hit_correct, hit)} 本分支；${matrix.hit_correct || 0}/${eligible} 全部` },
                { label: "部分正确", value: pct(matrix.hit_partial, eligible), note: `${pct(matrix.hit_partial, hit)} 本分支；${matrix.hit_partial || 0}/${eligible} 全部` },
                { label: "回答错误", value: pct(hitWrong, eligible), note: `${pct(hitWrong, hit)} 本分支；${hitWrong}/${eligible} 全部` },
              ],
            },
            {
              label: dim.missLabel, total: miss, note: "未命中分支",
              nodes: [
                { label: "回答正确", value: pct(matrix.miss_correct, eligible), note: `${pct(matrix.miss_correct, miss)} 本分支；${matrix.miss_correct || 0}/${eligible} 全部` },
                { label: "部分正确", value: pct(matrix.miss_partial, eligible), note: `${pct(matrix.miss_partial, miss)} 本分支；${matrix.miss_partial || 0}/${eligible} 全部` },
                { label: "回答错误", value: pct(missWrong, eligible), note: `${pct(missWrong, miss)} 本分支；${missWrong}/${eligible} 全部` },
              ],
            },
          ],
        },
        {
          key: "L3", name: "证据有效性", stage: "原图 / 关键帧 / 事件 → Judge 验证",
          nodes: [
            { label: "原图片证据有效率", value: imageEv.effective_rate_on_returned == null ? "未计算" : fmtPct(imageEv.effective_rate_on_returned), note: `${imageEv.effective_items || 0}/${imageEv.returned_items || 0} 有效 · 无效 ${imageEv.ineffective_items || 0}` },
            { label: "视频 / 关键帧有效率", value: videoEv.effective_rate_on_returned == null ? "未计算" : fmtPct(videoEv.effective_rate_on_returned), note: `${videoEv.effective_items || 0}/${videoEv.returned_items || 0} 有效 · 无效 ${videoEv.ineffective_items || 0}` },
            { label: "事件上下文有效率", value: eventEv.effective_rate_on_returned == null ? "未计算" : fmtPct(eventEv.effective_rate_on_returned), note: `${eventEv.effective_items || 0}/${eventEv.returned_items || 0} 有效 · 无效 ${eventEv.ineffective_items || 0}` },
          ],
        },
      ],
    };
  });
});
function memoryLayerSpecialMetrics(level) {
  const analysis = keyframeAnalysis.value?.status === "ready" ? keyframeAnalysis.value : null;
  const details = memoryEffectiveness.value?.details || {};
  const quality = memoryEffectiveness.value?.data_quality || {};
  const correctness = details.answer_correctness || {};
  if (level.key === "L0") {
    return [
      { label: "关键帧父级召回", value: analysis?.retrieval?.video_parent_recall == null ? "未计算" : fmtPct(analysis.retrieval.video_parent_recall), note: analysis ? `${analysis.retrieval.video_gt_targets_hit_by_keyframe || 0}/${analysis.retrieval.video_gt_targets || 0} 个源视频被覆盖` : "关键帧接口尚未返回" },
      { label: "候选 / 最终精确率", value: analysis?.retrieval?.candidate_precision == null ? "未计算" : `${fmtPct(analysis.retrieval.candidate_precision)} / ${fmtPct(analysis.retrieval.predicted_precision)}`, note: analysis ? `候选 ${analysis.retrieval.candidate_relevant_keyframes || 0}/${analysis.retrieval.candidate_keyframes || 0} · 最终 ${analysis.retrieval.predicted_relevant_keyframes || 0}/${analysis.retrieval.predicted_keyframes || 0}` : "候选池 / 最终回答返回" },
      { label: "关键帧压缩率", value: analysis?.processing?.compression_ratio == null ? "未保存原始帧" : fmtPct(analysis.processing.compression_ratio), note: analysis?.processing?.raw_frame_count ? `${analysis.assets.keyframes}/${analysis.processing.raw_frame_count} 原始帧` : "本次为复用记忆，未保存原始帧总数" },
    ];
  }
  if (level.key === "L1") {
    const visual = details.visual_calls || {};
    const ocr = details.ocr_calls || {};
    return [
      { label: "视觉解析成功率", value: visual.total ? fmtPct(visual.effective / visual.total) : "-", note: `${visual.effective || 0}/${visual.total || 0} 次 inspect_photo 成功` },
      { label: "OCR 成功率", value: ocr.total ? fmtPct(ocr.effective / ocr.total) : "-", note: `${ocr.effective || 0}/${ocr.total || 0} 次 read_photo_text 成功` },
      { label: "图谱加载", value: quality.graph_ready ? "已校验" : "不完整", note: `${quality.preview_count || 0} 条媒体预览 · ${quality.event_context_count || 0} 个事件上下文` },
    ];
  }
  if (level.key === "L2") {
    return [
      { label: "事件图谱覆盖率", value: level.created ? fmtPct(level.effective / level.created) : "-", note: `${level.effective}/${level.created} 个事件在工具图谱预览出现；不是事件事实正确率` },
      { label: "跨媒体关联", value: quality.graph_ready ? "可追溯" : "待校验", note: "事件 ID → 媒体引用 → 时间/地点字段" },
    ];
  }
  if (level.key === "L3") {
    const status = details.requirement_status_counts || {};
    return [
      { label: "高层证据闭合率（代理）", value: level.created ? fmtPct(level.effective / level.created) : "-", note: `${status.satisfied || 0} satisfied / ${level.created} 项证据需求；不是回答正确率` },
      { label: "未闭合需求", value: `${(status.open || 0) + (status.running || 0) + (status.failed || 0)}`, note: `open ${status.open || 0} · running ${status.running || 0} · failed ${status.failed || 0}` },
    ];
  }
  return [
    { label: "Agent 完整闭环率", value: level.created ? fmtPct(level.effective / level.created) : "-", note: `${level.effective}/${level.created} 条 complete` },
    { label: "回答 Exact Accuracy", value: fmtPct(correctness.exact_accuracy), note: `正确 ${correctness.correct || 0} · 部分 ${correctness.partial || 0} · 错误 ${correctness.incorrect || 0}` },
  ];
}
const qaPage = ref({ items: [], page: 1, page_size: 20, total: 0, pages: 1 });
const qaDetails = reactive({});
const openQaItems = reactive(new Set());
const loadingQaItems = reactive(new Set());
const qaPageSize = ref(20);
const qaFilters = reactive({ search: "", score: "", task_type: "", tag: "", angle: "", difficulty: "", answerability: "", agent_status: "", primary: "" });
const reviewDrafts = reactive({});
const selectedAlbum = ref("album3-14");
const albumCountLabel = (manifest) => {
  const videos = Number(manifest?.video_count || 0);
  const base = `${manifest.face_count}人 / ${manifest.photo_count}图`;
  return videos ? `${base} / ${videos}视频` : base;
};
const selectedQa = ref("compact-10q");
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
const qaBrowserAlbum = ref("album3");
const qaBrowserSet = ref("full-album3-38q");
const qaBrowserItems = ref([]);
const qaBrowserSearch = ref("");
const qaBrowserTag = ref("");
const qaBrowserLoading = ref(false);
const qaBrowserError = ref("");
const qaBrowserMediaResolution = ref(null);
const error = ref("");
const lightbox = ref(null);
const judgeModal = ref(null);
const chainMode = ref("creation");
const chainDetail = ref(null);
let pollTimer = null;
let destroyed = false;
const telemetryChartEl = ref(null);
let telemetryChartInstance = null;

const api = async (path, options = {}) => {
  const { timeoutMs = 10000, retries, ...requestOptions } = options;
  const method = String(requestOptions.method || "GET").toUpperCase();
  const retryCount = retries ?? (method === "GET" ? 1 : 0);
  let lastError;
  for (let attempt = 0; attempt <= retryCount; attempt += 1) {
    const controller = new AbortController();
    const timer = window.setTimeout(() => controller.abort(), timeoutMs);
    try {
      const response = await fetch(path, {
        headers: { "content-type": "application/json", ...(requestOptions.headers || {}) },
        ...requestOptions,
        signal: controller.signal,
      });
      const text = await response.text();
      let data = {};
      try { data = text ? JSON.parse(text) : {}; }
      catch { throw new Error(`响应不是有效 JSON：${path}`); }
      if (!response.ok) {
        const httpError = new Error(data.error || `HTTP ${response.status}`);
        httpError.retryable = response.status >= 500;
        throw httpError;
      }
      return data;
    } catch (requestError) {
      lastError = requestError;
      const retryable = requestError?.name === "AbortError" || requestError instanceof TypeError || requestError?.retryable;
      if (attempt >= retryCount || !retryable) break;
      await new Promise((resolve) => window.setTimeout(resolve, 250 * (attempt + 1)));
    } finally {
      window.clearTimeout(timer);
    }
  }
  throw new Error(lastError?.name === "AbortError" ? `请求超时：${path}` : (lastError?.message || `请求失败：${path}`));
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
const hasRunning = computed(() => Number(runPage.value.active_count || 0) > 0
  || runs.value.some((run) => ["running", "pending", "cancelling"].includes(run.status) || run.rejudge?.status === "running"));
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
  return image?.asset_id ? `/api/assets/${encodeURIComponent(image.asset_id)}/file` : (image?.media_url || "");
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
function pauseVideoAtEvidenceFrame(media, event) {
  const seconds = Number(media?.source_timestamp_sec);
  const player = event?.target;
  if (!player || !Number.isFinite(seconds) || seconds < 0) return;
  player.pause();
  player.currentTime = seconds;
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
  // 交付口径（E）：模型"召回/回答来源"= 模型显式交付的图（selected/predicted），
  // 不再把上游 evidence 全量候选冒充回答来源。历史 run 只有 evidence/retrieved 字段时再回退。
  if (item.predicted_media?.length) return decorateMedia(item.predicted_media);
  if (item.predicted_images?.length) return decorateMedia(item.predicted_images);
  if (item.predicted_file_names?.length) {
    return item.predicted_file_names.map((file_name) => ({ file_name, media_type: inferMediaType(file_name), media_url: albumLocalUrl(file_name) }));
  }
  if (item.evidence_source_media?.length) return decorateMedia(item.evidence_source_media);
  if (item.evidence_source_images?.length) return decorateMedia(item.evidence_source_images);
  if (item.evidence_source_file_names?.length) {
    return decorateMedia(item.evidence_source_file_names.map((file_name) => ({ file_name, media_type: inferMediaType(file_name), media_url: albumLocalUrl(file_name) })));
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
function toolRecallMedia(item) {
  // 工具召回图片 = search 等找图工具返回的检索候选集（retrieved），与"模型使用图片"严格区分。
  if (item.retrieved_candidate_media?.length) return decorateMedia(item.retrieved_candidate_media);
  if (item.retrieved_candidate_images?.length) return decorateMedia(item.retrieved_candidate_images);
  if (item.retrieved_file_names?.length) {
    return item.retrieved_file_names.map((file_name) => ({ file_name, media_type: inferMediaType(file_name), media_url: albumLocalUrl(file_name) }));
  }
  return [];
}
function itemEvidenceMedia(item) {
  // 模型使用图片 = 模型显式交付图（predicted/selected）；未显式交付 → 空（"回答依据图片为空"），
  // 绝不回退到 evidence/retrieved 候选（否则又变成"回答来源==完整候选集"）。
  const media = item?.predicted_media || item?.predicted_images || [];
  if (media.length) return decorateMedia(media);
  const names = item?.predicted_file_names || [];
  if (names.length) return decorateMedia(names.map((file_name) => ({ file_name, media_type: inferMediaType(file_name), media_url: albumLocalUrl(file_name) })));
  return [];
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
  if (["orin_ssh_pss", "jetson_local_pss"].includes(phase.source)) return [
    ["Orin 模型进程物理内存峰值（PSS）", fmtMemory(modelMemory.peak), `均值 ${fmtMemory(modelMemory.mean)} · P95 ${fmtMemory(modelMemory.p95)} · 统一物理内存，非独立显存`, true],
    ["KV Cache 已用 token", fmtNumber(phase.kv_cache_used_tokens?.mean), `峰值 ${fmtNumber(phase.kv_cache_used_tokens?.peak)}`],
    ["KV Cache 使用率", fmtNumber(kvCache.mean, "%"), `峰值 ${fmtNumber(kvCache.peak, "%")}`],
    ["GPU 利用率", fmtNumber(util.mean, "%"), `tegrastats GR3D 峰值 ${fmtNumber(util.peak, "%")}`],
    ["GPU/SOC 功耗", fmtNumber(power.mean, "W"), `VDD_GPU_SOC 峰值 ${fmtNumber(power.peak, "W")}（非纯 GPU）`],
    ["采样数量", phase.samples_count == null ? "-" : `${phase.samples_count} 次`, `PSS 独立采样 ${phase.pss_samples_count ?? 0} 次（目标约 5 秒间隔）；KV 采样随主循环`],
  ];
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
function liveTelemetryRows(run) {
  const live = run?.telemetry_live || {};
  const latest = live.latest || {};
  const peak = live.peak || {};
  // Older persisted runs may have latest values but no peak object.
  const effectivePeak = Object.keys(peak).length ? peak : latest;
  const unit = live.source === "jetson_local_pss" || live.source === "orin_ssh_pss" ? "PSS" : "显存";
  const fmtLive = (value, suffix = "") => value == null ? "-" : `${Number(value).toFixed(2)}${suffix}`;
  const fmtGiB = (value) => value == null ? "-" : `${(Number(value) / 1024).toFixed(2)} GiB`;
  return [
    ["当前阶段", ["completed", "failed", "cancelled"].includes(run?.status) ? statusLabel(run.status) : (EXECUTION_PHASES.find((item) => item.key === (live.current_phase || run?.current_phase))?.label || "运行中"), `已采样 ${live.samples_count || 0} 次`],
    [`模型进程${unit}`, fmtGiB(latest.model_process_memory_used_mib), `峰值 ${fmtGiB(effectivePeak.model_process_memory_used_mib)} · 仅可归因到模型的进程`],
    ["整卡显存", fmtGiB(latest.memory_used_mib), `峰值 ${fmtGiB(effectivePeak.memory_used_mib)} · NVIDIA GPU 总占用`],
    ["整机 RAM", fmtGiB(latest.system_memory_used_mib), `峰值 ${fmtGiB(effectivePeak.system_memory_used_mib)} · ${latest.system_memory_scope === "host_all_processes" ? "宿主机全部进程" : "未标注范围"}`],
    ["全部 GPU 进程", fmtGiB(latest.all_processes_memory_mib), `峰值 ${fmtGiB(effectivePeak.all_processes_memory_mib)} · nvidia-smi 可见进程总和`],
    ["其他 GPU 进程", fmtGiB(latest.other_processes_memory_used_mib ?? latest.other_processes_memory_mib), `峰值 ${fmtGiB(effectivePeak.other_processes_memory_mib)} · 除模型进程外`],
    ["GPU 利用率", fmtLive(latest.gpu_utilization_pct, "%"), `峰值 ${fmtLive(effectivePeak.gpu_utilization_pct, "%")}`],
    ["温度 / 功耗", `${fmtLive(latest.temperature_c, " °C")} / ${fmtLive(latest.power_draw_w, " W")}`, `峰值功耗 ${fmtLive(effectivePeak.power_draw_w, " W")}`],
    ["KV Cache", latest.kv_cache_usage_pct == null ? "未提供" : fmtLive(latest.kv_cache_usage_pct, "%"), latest.kv_cache_used_tokens == null ? "当前框架未暴露运行时 KV 指标" : `峰值 token ${fmtLive(peak.kv_cache_used_tokens)}`],
  ];
}
function telemetryChart(run) {
  let history = Array.isArray(run?.telemetry_live?.history) ? run.telemetry_live.history : [];
  // Legacy runs may only have phase snapshots. Render a compact phase trend
  // instead of leaving the chart area blank.
  if (history.length < 2) {
    const phases = run?.telemetry_live?.phase_snapshots || {};
    history = Object.values(phases).map((phase, i) => ({
      t: i,
      ...(phase?.latest || phase?.peak || {}),
    })).filter((item) => Object.keys(item).some((key) => key.endsWith("_mib")));
  }
  if (history.length < 2) return null;
  return { history };
}
function renderTelemetryChart() {
  const chartData = telemetryChart(activeRun.value);
  if (!telemetryChartEl.value || !chartData) {
    if (telemetryChartInstance) {
      telemetryChartInstance.dispose();
      telemetryChartInstance = null;
    }
    return;
  }
  telemetryChartInstance ||= echarts.init(telemetryChartEl.value);
  const history = chartData.history;
  const definitions = [
    { key: "memory_used_mib", name: "整卡显存", color: "#4f7cff" },
    { key: "system_memory_used_mib", name: "整机 RAM", color: "#20a36a" },
    { key: "model_process_memory_used_mib", name: "模型进程", color: "#ef8a4b" },
    { key: "all_processes_memory_mib", name: "全部 GPU 进程", color: "#d9488b" },
    { key: "other_processes_memory_mib", name: "其他 GPU 进程", color: "#8b6de8" },
  ];
  const available = definitions.filter((definition) => history.some((item) => Number.isFinite(Number(item[definition.key]))));
  const firstTimestamp = Number(history[0]?.t);
  const labels = history.map((item, index) => {
    const elapsed = Number(item?.t) - firstTimestamp;
    return Number.isFinite(elapsed) ? `+${(elapsed / 60).toFixed(1)} min` : `#${index + 1}`;
  });
  telemetryChartInstance.setOption({
    animation: false,
    color: available.map((item) => item.color),
    grid: { left: 62, right: 28, top: 48, bottom: 64 },
    legend: { top: 8, left: 8, type: "scroll", selectedMode: "multiple", textStyle: { color: "#59627c", fontSize: 12 } },
    tooltip: {
      trigger: "axis",
      axisPointer: { type: "cross", label: { backgroundColor: "#667085" } },
      formatter(params) {
        const rows = (Array.isArray(params) ? params : [params]).filter((item) => item.value != null);
        const title = rows[0]?.axisValueLabel || "";
        return `<b>${title}</b><br/>${rows.map((item) => `${item.marker}${item.seriesName}: <b>${Number(item.value).toFixed(2)} GiB</b>`).join("<br/>")}`;
      },
    },
    xAxis: { type: "category", boundaryGap: false, data: labels, axisLabel: { color: "#7a849e", hideOverlap: true }, axisLine: { lineStyle: { color: "#d8deea" } } },
    yAxis: { type: "value", name: "GiB", nameTextStyle: { color: "#7a849e", padding: [0, 0, 8, 0] }, min: 0, axisLabel: { color: "#7a849e", formatter: (value) => `${value} GiB` }, splitLine: { lineStyle: { color: "#edf0f5" } } },
    dataZoom: [{ type: "inside", filterMode: "none" }, { type: "slider", height: 18, bottom: 14, borderColor: "#d8deea", fillerColor: "rgba(79,124,255,.14)", handleStyle: { color: "#4f7cff" } }],
    series: available.map((definition) => ({
      name: definition.name,
      type: "line",
      smooth: 0.18,
      showSymbol: false,
      connectNulls: false,
      lineStyle: { width: 2.5 },
      emphasis: { focus: "series", lineStyle: { width: 4 } },
      markPoint: { symbolSize: 42, label: { formatter: (params) => `峰值\n${Number(params.value).toFixed(2)} GiB`, color: "#344054", fontSize: 10 }, data: [{ type: "max", name: "峰值" }] },
      data: history.map((item) => Number.isFinite(Number(item[definition.key])) ? Number(item[definition.key]) / 1024 : null),
    })),
  }, true);
  telemetryChartInstance.resize();
}
function liveTelemetryPhaseRows(run) {
  const phases = run?.telemetry_live?.phase_snapshots || {};
  return Object.entries(phases).map(([key, value]) => {
    const label = EXECUTION_PHASES.find((item) => item.key === key)?.label || key;
    const peak = value?.peak || {};
    const model = peak.model_process_memory_used_mib == null ? "-" : `${(Number(peak.model_process_memory_used_mib) / 1024).toFixed(2)} GiB`;
    const card = peak.memory_used_mib == null ? "-" : `${(Number(peak.memory_used_mib) / 1024).toFixed(2)} GiB`;
    const host = peak.system_memory_used_mib == null ? "-" : `${(Number(peak.system_memory_used_mib) / 1024).toFixed(2)} GiB`;
    return [label, model, `模型进程峰值 · 整卡 ${card} · 整机 RAM ${host} · ${value?.samples_count || 0} 次`];
  });
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
function isOrinPssRun(run) {
  return ["orin_ssh_pss", "jetson_local_pss"].includes(run?.telemetry_source)
    || ["orin_ssh_pss", "jetson_local_pss"].includes(run?.telemetry_live?.source)
    || ["orin_ssh_pss", "jetson_local_pss"].includes(run?.phases?.gpu_metrics?.source)
    || run?.phases?.gpu_metrics?.memory_profile?.method === "orin_process_pss_uma_v1"
    || comparableMemoryProfile(run)?.memory_profile?.method === "orin_process_pss_uma_v1";
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
  if (memory.method === "orin_process_pss_uma_v1") return [
    ["进程物理内存峰值（PSS）", fmtMemory(processMemory.peak), "Orin 统一物理内存平台的进程峰值；不是独立显存，也不是工作负载估算", true],
    ["工作负载内存估算", "不可计算", "当前 llama.cpp 未提供可靠的 KV Cache 实际使用量"],
    ["KV Cache 已用峰值", memory.kv_cache_used_peak_tokens == null ? "-" : `${Number(memory.kv_cache_used_peak_tokens).toLocaleString("en-US")} token`, `使用率峰值 ${fmtNumber(memory.kv_cache_usage_peak_pct, "%")}`],
    ["KV 实际字节数", "不可用", "未验证模型每 token KV 字节数；不以 token 数伪造 GiB"],
    ["数据来源", "118 本机 PSS + tegrastats", "KV 仅在 llama.cpp metrics 确实提供时记录"],
  ];
  return [
    ["估算工作负载显存", memory.comparable_workload_memory_gib == null ? "-" : `${Number(memory.comparable_workload_memory_gib).toFixed(2)} GiB`, "固定基础占用 + KV 逻辑使用峰值；非实测显存", true],
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
  return [
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
function memoryLayerToolMetric(summary, name) {
  const metric = summary?.tool_performance?.[name] || {};
  return { calls: Number(metric.calls || 0), okRate: metric.ok_rate == null ? null : Number(metric.ok_rate) };
}
function memoryLayerMetrics() {
  const run = activeRun.value || {};
  const summary = effectiveRunSummary(run);
  const manifest = manifests.value.find((item) => item.album_id === run.album_id) || {};
  const photoCount = Number(manifest.photo_count || 0);
  const videoCount = Number(manifest.video_count || 0);
  const rawCount = photoCount + videoCount;
  const progress = run.phases?.qa_eval?.progress || {};
  const completed = Number(progress.completed ?? summary.completed ?? run.item_count ?? 0);
  const total = Number(progress.total ?? summary.total ?? run.qa_count ?? 0);
  const agent2 = summary.agent2_trace || {};
  const requirementCounts = agent2.requirement_status_counts || {};
  const requirementTotal = Object.values(requirementCounts).reduce((sum, value) => sum + Number(value || 0), 0);
  const satisfiedRequirements = Number(requirementCounts.satisfied || 0);
  // 优先用 memory-effectiveness 接口实测的调用次数（新老 run 都有）；汇总缺 tool_performance 时回退 0。
  const meCalls = (kind) => {
    const row = memoryEffectiveness.value?.details?.[kind] || {};
    const total = Number(row.total || 0);
    return { calls: total, okRate: total ? (Number(row.effective || 0) / total) : null };
  };
  const visual = meCalls("visual_calls");
  const ocr = meCalls("ocr_calls");
  const summaryCalls = memoryLayerToolMetric(summary, "get_core_memory");
  const analysis = keyframeAnalysis.value?.status === "ready" ? keyframeAnalysis.value : {};
  const analysisAssets = analysis.assets || {};
  const analysisProcessing = analysis.processing || {};
  const analysisRetrieval = analysis.retrieval || {};
  const analysisUsefulness = analysis.usefulness || {};
  const expanded = Object.values(qaDetails).filter(Boolean);
  const previewRows = expanded.flatMap((item) => (item.tool_trace || []).flatMap((trace) => trace.observation?.preview || []));
  const keyframes = new Set(previewRows.filter((row) => row.media_kind === "video_keyframe").map((row) => row.asset_id || row.file_name).filter(Boolean));
  const events = new Set(previewRows.map((row) => row.event_context?.id).filter(Boolean));
  const eventEvidenceRows = previewRows.filter((row) => row.event_context?.id).length;
  const qaSampleCount = expanded.length || (qaPage.value.items || []).length;
  const summaryTaskSampleCount = (qaPage.value.items || []).filter((item) => item.task_type === "T5_event_summary").length;
  const layers = [
    { key: "L0", name: "原始流", color: "raw", metric: rawCount ? `${rawCount} 个媒体` : "待读取", detail: `${photoCount} 图 · ${videoCount} 视频`, evidence: analysisAssets.keyframes == null ? "等待全量分析" : `${analysisAssets.source_videos} 个源视频 → ${analysisAssets.keyframes} 帧；保留 parent_asset_id 与时间戳`, note: "关键帧是视频派生媒体，仍归 L0" },
    { key: "L1", name: "场景图", color: "scene", metric: visual.okRate == null ? "未记录" : `视觉 ${fmtPct(visual.okRate)}`, detail: `观察 ${visual.calls} 次 · OCR ${ocr.calls} 次`, evidence: `视觉成功 ${visual.calls ? fmtPct(visual.okRate) : "-"} · OCR 调用 ${ocr.calls} 次 · 场景 ${analysis.temporal?.scene_count ?? "-"} 个`, note: "画面、文字、人物、动作、空间关系" },
    { key: "L2", name: "事件层", color: "event", metric: events.size ? `${events.size} 个事件` : (analysisUsefulness.event_context_count ? `${analysisUsefulness.event_context_count} 个事件上下文` : "待展开"), detail: `${eventEvidenceRows} 条证据带事件上下文`, evidence: `全量 QA 轨迹带回 ${analysisUsefulness.event_context_count ?? 0} 个事件上下文；当前展开 ${events.size} 个`, note: "同一事件聚合多个媒体与时间地点" },
    { key: "L3", name: "目标/规则", color: "goal", metric: requirementTotal ? `${satisfiedRequirements}/${requirementTotal} 满足` : "未记录", detail: `规划 ${agent2.planner_decision_count || 0} 题 · 目标证据需求`, evidence: requirementTotal ? `证据账本满足 ${satisfiedRequirements} / ${requirementTotal}；规划决策 ${agent2.planner_decision_count || 0} 次` : "当前 run 未记录目标账本", note: "时间、事件、地点、人物、场景等查询要求" },
    { key: "L4+", name: "总结/回答", color: "summary", metric: summaryCalls.calls ? `总结访问 ${summaryCalls.calls} 次` : "按题回答", detail: `当前页总结题 ${summaryTaskSampleCount} · 已完成 ${completed}/${total || "-"}`, evidence: `Judge 均分 ${summary.answer_quality_mean ?? "-"} / 2 · exact ${summary.exact_accuracy == null ? "-" : fmtPct(summary.exact_accuracy)} · 已完成 ${completed}/${total || "-"}`, note: "事件/相册级递归摘要与最终回答" },
  ];
  return { layers, rawMediaCount: rawCount, keyframes: keyframes.size, expandedCount: qaSampleCount, completed, total, visual, ocr, summaryCalls, analysisProcessing };
}
const memoryLayerView = computed(memoryLayerMetrics);
function memoryEvidenceGraph() {
  const view = memoryLayerView.value || { layers: [] };
  const analysis = keyframeAnalysis.value?.status === "ready" ? keyframeAnalysis.value : {};
  const assets = analysis.assets || {};
  const processing = analysis.processing || {};
  const retrieval = analysis.retrieval || {};
  const temporal = analysis.temporal || {};
  const usefulness = analysis.usefulness || {};
  const rawAssets = Number(view.rawMediaCount || activeRun.value?.input_integrity?.files_checked || 0);
  const keyframeCount = assets.keyframes == null ? null : Number(assets.keyframes);
  const visualCalls = Number(view.visual?.calls || 0);
  const ocrCalls = Number(view.ocr?.calls || 0);
  const sceneCount = Number(temporal.scene_count || 0);
  const eventCount = Number(usefulness.event_context_count || 0);
  const requirements = getAgent2Requirements(Object.values(qaDetails).find(Boolean) || {});
  const requirementCounts = effectiveRunSummary(activeRun.value).agent2_trace?.requirement_status_counts || {};
  const requirementTotal = Object.values(requirementCounts).reduce((sum, value) => sum + Number(value || 0), 0);
  const completed = view.completed || 0;
  const total = view.total || 0;
  const recorded = (value, fallback = "未记录") => value == null || value === 0 ? fallback : String(value);
  const sourceEvidence = rawAssets ? `${rawAssets} 个原始媒体 · asset_id / file_name · 输入完整性校验` : "接口未返回原始媒体数量，暂不能证明记忆已入库";
  return [
    {
      key: "L0", name: "原始流", color: "raw", stage: "输入与派生媒体",
      nodes: [
        { kind: "原始媒体", value: recorded(rawAssets), evidence: sourceEvidence },
        { kind: "关键帧", value: keyframeCount == null ? "未记录" : String(keyframeCount), evidence: keyframeCount == null ? "待获取 parent_asset_id" : "parent_asset_id · source_timestamp_sec" },
      ],
      transition: "asset_id + 时间戳 → 场景观察",
    },
    {
      key: "L1", name: "场景图", color: "scene", stage: "视觉、文字、人物、动作",
      nodes: [
        { kind: "视觉观察", value: recorded(visualCalls), evidence: "inspect_photo → observation.preview" },
        { kind: "文字 / OCR", value: recorded(ocrCalls), evidence: "read_photo_text → text + media_id" },
        { kind: "场景节点", value: recorded(sceneCount), evidence: "source_scene_index · 人物/动作/空间关系" },
      ],
      transition: "观察 + OCR + 时间地点 → 事件上下文",
    },
    {
      key: "L2", name: "事件层", color: "event", stage: "跨媒体聚合",
      nodes: [
        { kind: "事件上下文", value: recorded(eventCount), evidence: "event_context.id · related_media_ids" },
        { kind: "时间 / 地点", value: eventCount ? "已挂接" : "未记录", evidence: "event_context 的时间、地点、实体字段" },
      ],
      transition: "事件 ID + 媒体引用 → 查询证据需求",
    },
    {
      key: "L3", name: "目标 / 规则", color: "goal", stage: "通用查询槽位",
      nodes: [
        { kind: "查询槽位", value: "时间 · 事件 · 地点 · 人物 · 场景", evidence: "query_anchors → evidence_refs" },
        { kind: "证据账本", value: requirementTotal ? `${requirementTotal} 项` : (requirements.length ? `${requirements.length} 项` : "未记录"), evidence: requirementTotal ? "满足状态 + provenance_refs" : "本 run 未返回 Agent 证据账本" },
      ],
      transition: "evidence_refs + provenance → 总结与回答",
    },
    {
      key: "L4+", name: "总结 / 回答", color: "summary", stage: "可验证输出",
      nodes: [
        { kind: "回答", value: total ? `${completed}/${total}` : "未记录", evidence: "final_answer · answer_claims" },
        { kind: "证据核验", value: activeRun.value ? "Judge" : "未记录", evidence: "judge / evidence_judge → 分数与理由" },
      ],
      transition: "输出保留证据链，可回溯到 L0",
    },
  ];
}
const memoryEvidenceGraphView = computed(memoryEvidenceGraph);
function effectiveEvidenceGraph() {
  const item = Object.values(qaDetails).find(Boolean) || {};
  const traces = item.tool_trace || [];
  const previews = traces.flatMap((trace) => trace.observation?.preview || []);
  const candidates = item.retrieved_candidate_media || item.retrieved_candidate_images || [];
  const evidence = item.evidence_source_media || item.evidence_source_images || [];
  const requirements = getAgent2Requirements(item);
  const answer = item.final_answer || item.answer;
  const recorded = (value, fallback = "未展开 QA") => value == null || value === 0 ? fallback : String(value);
  return [
    { key: "L0", name: "问题 / 原始证据", color: "raw", stage: "本题输入范围", nodes: [
      { kind: "问题", value: item.question ? "1 条" : "未展开 QA", evidence: item.question ? item.question : "请先展开一条 QA" },
      { kind: "GT 媒体", value: recorded(item.gt_media?.length, "未记录"), evidence: "gt_media · file_name / asset_id" },
    ], transition: "问题槽位 → 工具查询" },
    { key: "L1", name: "工具 / 检索", color: "scene", stage: "测评时实际调用", nodes: [
      { kind: "工具调用", value: recorded(traces.length, "未记录"), evidence: "tool_trace · name · arguments" },
      { kind: "候选媒体", value: recorded(candidates.length, "未记录"), evidence: "retrieved_candidate_media · result_set_id" },
      { kind: "观察", value: recorded(previews.length, "未记录"), evidence: "observation.preview · evidence_summary" },
    ], transition: "候选 + 观察 → 事件与证据" },
    { key: "L2", name: "事件 / 记忆", color: "event", stage: "有效记忆上下文", nodes: [
      { kind: "事件上下文", value: recorded(new Set(previews.map((row) => row.event_context?.id).filter(Boolean)).size, "未记录"), evidence: "event_context.id · title · summary" },
      { kind: "时间 / 地点", value: previews.some((row) => row.captured_at || row.place || row.location_context) ? "已返回" : "未记录", evidence: "captured_at · place · location_context" },
    ], transition: "事件引用 → 证据需求" },
    { key: "L3", name: "目标 / 证据", color: "goal", stage: "本题判定依据", nodes: [
      { kind: "查询锚点", value: item.query_anchors ? "已记录" : "未记录", evidence: item.query_anchors || "query_anchors 未返回" },
      { kind: "证据账本", value: recorded(requirements.length, "未记录"), evidence: requirements.length ? requirements.map((req) => `${req.id}:${agent2RequirementStatusLabel(req.status)}`).join(" · ") : "evidence_refs 未返回" },
      { kind: "实际证据", value: recorded(evidence.length, "未记录"), evidence: "evidence_source_media · answer_evidence" },
    ], transition: "证据引用 → 回答与 Judge" },
    { key: "L4+", name: "回答 / 评判", color: "summary", stage: "测评结果", nodes: [
      { kind: "回答", value: answer ? "已形成" : "未形成", evidence: answer || "final_answer 未返回" },
      { kind: "声明", value: recorded(item.answer_claims?.length, "未记录"), evidence: "answer_claims · claim evidence" },
      { kind: "Judge", value: item.judge?.score == null ? "未记录" : `${item.judge.score} 分`, evidence: item.judge?.reason || "judge.reason 未返回" },
    ], transition: "结果可回溯到原始媒体" },
  ];
}
const effectiveEvidenceGraphView = computed(effectiveEvidenceGraph);
const activeEvidenceGraphView = computed(() => chainMode.value === "effective" ? effectiveEvidenceGraphView.value : memoryEvidenceGraphView.value);
function chainDetailMedia(item, previews) {
  const values = [
    ...(item?.gt_media || []), ...(item?.evidence_source_media || []),
    ...(item?.predicted_media || []),
    ...previews.map((row) => ({ ...row, file_name: row.file_name || row.media_id, media_type: row.media_kind === "video" ? "video" : "image" })),
  ];
  const unique = new Map();
  for (const media of decorateMedia(values)) {
    const key = mediaKey(media);
    if (key && !unique.has(key)) unique.set(key, media);
  }
  return [...unique.values()].slice(0, 12);
}
function chainDetailStatements(item, previews, layer, node) {
  const statements = [];
  if (item?.question) statements.push({ label: "用户问题", value: item.question });
  if (node?.evidence) statements.push({ label: `${layer.key} · ${node.kind}`, value: node.evidence });
  for (const row of previews.slice(0, 8)) {
    const event = row.event_context || {};
    const text = row.evidence_summary || event.summary || row.description || row.text;
    if (text) statements.push({ label: `${row.media_id || row.file_name || row.asset_id || "媒体"} · 观察`, value: text });
  }
  if (item?.answer) statements.push({ label: "模型回答", value: item.answer });
  if (item?.judge?.reason) statements.push({ label: "Judge 理由", value: item.judge.reason });
  return statements.slice(0, 12);
}
async function openChainNode(mode, layer, node) {
  const pageItems = qaPage.value.items || [];
  const indices = pageItems.map((summary) => summary.index).filter((index) => index != null);
  await Promise.allSettled(indices.map((index) => loadQaDetail(index)));
  const records = indices.map((index) => {
    const item = qaDetails[index];
    if (!item) return null;
    const previews = (item.tool_trace || []).flatMap((trace) => trace.observation?.preview || []);
    const traceRows = (item.tool_trace || []).slice(0, 12).map((trace, traceIndex) => ({
      label: `${traceIndex + 1}. ${trace.tool_name || trace.name || trace.tool || "工具调用"}`,
      value: JSON.stringify(trace, null, 2),
    }));
    return {
      key: `${item.qa_id || "qa"}-${index}`,
      index,
      question: item.question || pageItems.find((summary) => summary.index === index)?.question || "未记录问题",
      media: chainDetailMedia(item, previews),
      statements: chainDetailStatements(item, previews, layer, node),
      traceRows,
      reasoning: JSON.stringify({ agent2_trace: item.agent2_trace, execution_trace: item.execution_trace, answer_grounding: item.answer_grounding }, null, 2),
    };
  }).filter(Boolean);
  chainDetail.value = {
    mode,
    layer,
    node,
    records,
    page: qaPage.value.page,
    pageSize: qaPage.value.page_size,
    total: qaPage.value.total,
  };
}
function closeChainDetail() { chainDetail.value = null; }
function itemLayerChain(item) {
  const traces = item?.tool_trace || [];
  const previews = traces.flatMap((trace) => trace.observation?.preview || []);
  const keyframes = new Set(previews.filter((row) => row.media_kind === "video_keyframe").map((row) => row.asset_id || row.file_name).filter(Boolean));
  const events = new Set(previews.map((row) => row.event_context?.id).filter(Boolean));
  const observations = previews.length;
  const requirements = getAgent2Requirements(item);
  const satisfied = requirements.filter((req) => req.status === "satisfied").length;
  const mediaNames = itemMedia(item).slice(0, 2).map((media) => media.file_name || media.asset_id || media.media_id).filter(Boolean);
  const previewNames = previews.slice(0, 2).map((row) => row.asset_id || row.file_name || row.media_id).filter(Boolean);
  const eventNames = [...events].slice(0, 2);
  const requirementNames = requirements.slice(0, 2).map((req) => req.id).filter(Boolean);
  const answer = item.final_answer || item.answer;
  return [
    { key: "L0", name: "原始流", status: itemMedia(item).length ? "pass" : "na", detail: `${itemMedia(item).length} 媒体${keyframes.size ? ` · 关键帧 ${keyframes.size}` : ""}`, evidence: mediaNames.join("、") || "无 asset_id / media_id" },
    { key: "L1", name: "场景图", status: observations ? "pass" : "na", detail: observations ? `${observations} 条视觉/文字观察` : "未返回观察", evidence: previewNames.join("、") || "无 observation.preview" },
    { key: "L2", name: "事件", status: events.size ? "pass" : "na", detail: events.size ? `${events.size} 个事件上下文` : "未带回事件上下文", evidence: eventNames.join("、") || "无 event_context.id" },
    { key: "L3", name: "目标", status: requirements.length ? (satisfied === requirements.length ? "pass" : "fail") : "na", detail: requirements.length ? `${satisfied}/${requirements.length} 项证据需求` : "未记录规划", evidence: requirementNames.join("、") || "无 evidence_refs" },
    { key: "L4+", name: "回答", status: answer ? "pass" : "fail", detail: answer ? "已形成回答" : "未形成回答", evidence: answer ? `answer_claims ${item.answer_claims?.length || 0} · Judge ${item.judge?.score ?? "-"}` : "无 final_answer" },
  ];
}
function keyframeDeltaLabel(value) {
  if (value == null || !Number.isFinite(Number(value))) return "-";
  const number = Number(value);
  return `${number >= 0 ? "+" : ""}${number.toFixed(2)} / 2`;
}
function keyframeReasonLabel(value) {
  return Object.entries(value || {}).map(([key, count]) => `${key}×${count}`).join(" · ") || "-";
}
const keyframeMetricDefinitions = [
  { name: "保留率 / 压缩率", formula: "保留率 = 关键帧数 ÷ 原始视频帧数；压缩率 = 1 − 保留率", meaning: "衡量视频被压缩成关键帧后，计算量减少了多少；压缩越高不代表质量越高，必须结合父视频召回率观察。" },
  { name: "源视频父级召回", formula: "被任一关键帧覆盖的 GT 源视频 ÷ GT 源视频目标", meaning: "回答需要某段视频时，关键帧是否至少代表了这段视频；允许返回关键帧而不是原视频。" },
  { name: "候选关键帧精确率", formula: "与本题 GT 父视频一致的候选关键帧 ÷ 候选关键帧总数", meaning: "进入 Agent 候选池的关键帧有多少真正相关，反映检索噪声。" },
  { name: "最终返回精确率", formula: "与本题 GT 父视频一致的最终关键帧 ÷ 最终返回关键帧总数", meaning: "模型实际用于回答的关键帧相关程度，反映重排和工具选择后的有效性。" },
  { name: "时间间隔 / 覆盖跨度", formula: "同一视频关键帧相邻时间差的均值 / 最早至最晚时间差的均值", meaning: "间隔反映采样密度，跨度反映单个视频被覆盖的时间范围；二者都不是答案准确率。" },
  { name: "回答质量差值", formula: "有关键帧候选题 Judge 均分 − 无关键帧候选题 Judge 均分", meaning: "用于观察关键帧与回答质量的关联；受题型、样本量和模型状态影响，不能单独视为因果结论。" },
  { name: "视频处理速度", formula: "原始视频总时长 ÷ 视频处理耗时总和（real-time factor）", meaning: "1.0× 表示处理 1 秒视频约需 1 秒；低于 1.0× 表示处理慢于实时。页面同时给出总耗时和每视频均值。" },
];
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

async function loadRuns(page = runPage.value.page || 1) {
  const payload = await api(`/api/runs?page=${Math.max(1, Number(page) || 1)}&page_size=${runPage.value.page_size}`, {
    timeoutMs: 8000,
    retries: 2,
  });
  runs.value = payload.runs || [];
  runPage.value = {
    page: payload.page || 1,
    page_size: payload.page_size || 20,
    total: payload.total ?? runs.value.length,
    pages: payload.pages || 1,
    has_previous: Boolean(payload.has_previous),
    has_next: Boolean(payload.has_next),
    active_count: Number(payload.active_count || 0),
  };
}
async function changeRunPage(page) {
  await loadRuns(page);
  document.querySelector("#runs-region")?.scrollIntoView({ behavior: "smooth", block: "start" });
}
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
async function loadKeyframeAnalysis(runId) {
  keyframeAnalysisLoading.value = true;
  keyframeAnalysisError.value = "";
  try {
    const payload = await api(`/api/runs/${encodeURIComponent(runId)}/keyframe-analysis`);
    if (activeRunId.value === runId) keyframeAnalysis.value = payload;
  } catch (error) {
    if (activeRunId.value === runId) keyframeAnalysisError.value = error.message || "关键帧分析失败";
  } finally {
    keyframeAnalysisLoading.value = false;
  }
}
async function loadMemoryEffectiveness(runId) {
  try {
    const payload = await api(`/api/runs/${encodeURIComponent(runId)}/memory-effectiveness`);
    if (activeRunId.value === runId) memoryEffectiveness.value = payload;
  } catch (error) {
    if (activeRunId.value === runId) memoryEffectiveness.value = { status: "unavailable", error: error.message || "记忆有效性分析失败" };
  }
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
    keyframeAnalysis.value = null;
    keyframeAnalysisError.value = "";
    memoryEffectiveness.value = null;
    qaPage.value = { items: [], page: 1, page_size: qaPageSize.value, total: 0, pages: 1 };
    Object.keys(qaDetails).forEach((key) => delete qaDetails[key]);
    openQaItems.clear();
    Object.keys(reviewDrafts).forEach((key) => delete reviewDrafts[key]);
    const reviewPayload = await api(`/api/runs/${encodeURIComponent(runId)}/reviews`);
    Object.assign(reviewDrafts, reviewPayload.reviews || {});
  }
  await Promise.all([loadKeyframeAnalysis(runId), loadMemoryEffectiveness(runId), loadQaPage(resetPage ? 1 : qaPage.value.page)]);
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
function scrollToRunDetail() {
  nextTick(() => {
    requestAnimationFrame(() => {
      const target = document.querySelector("#qa-results");
      if (!target) return;
      const top = target.getBoundingClientRect().top + window.scrollY - 12;
      window.scrollTo({ top: Math.max(0, top), behavior: "smooth" });
    });
  });
}
function selectRun(run) {
  activeRunId.value = run.run_id;
  activeRun.value = { ...run };
  scrollToRunDetail();
  void loadActiveRun({ resetPage: true }).catch((err) => {
    if (activeRunId.value === run.run_id) error.value = err.message || "读取评测详情失败";
  });
}
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
function exportTraces() {
  if (!activeRunId.value) return;
  const scores = exportScores.value;
  if (!scores.length) { window.alert("请至少勾选一个评分再导出"); return; }
  // 每题只含两项：planner 完整输入/输出 + 完整轨迹
  window.open(`/api/runs/${encodeURIComponent(activeRunId.value)}/export-trace?scores=${scores.join(",")}`, "_blank");
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
    await loadRuns(1); await loadActiveRun({ resetPage: true }); startPolling();
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
  if (!sentrixUrl.value) return false;
  try {
    const payload = await api("/api/arbiter-status", { timeoutMs: 5000, retries: 0 });
    if (payload.supported === false) {
      arbiterStatus.value = null;
      return false;
    }
    if (payload.status) arbiterStatus.value = payload.status;
    return true;
  } catch (_) {
    return true; // transient failure: retain the previous value and retry later
  }
}
function arbStateLabel(s) {
  return ({idle: "空闲", import_running: "导入运行中", agent_active: "Agent 活跃", preempting: "抢占中"})[s] || s || "-";
}
function thermalStateLabel(v) {
  return ({0: "nominal", 1: "fair", 2: "serious", 3: "critical"})[v] ?? "-";
}
async function init() {
  let current = null;
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
    judgeProviderId.value = runtimeConfig.judge_provider_id || config.value.default_judge_provider_id || (config.value.judge_providers?.[0]?.id || "");
    connectionConfigState.value = "saved";
    connectionConfigMessage.value = "已读取配置文件";
    const [, manifestPayload] = await Promise.all([
      loadJudgePrompts(),
      api("/api/manifests"),
      loadRuns(1),
      vllmManagerUrl.value.trim() ? loadProfiles() : Promise.resolve(),
    ]);
    manifests.value = manifestPayload.manifests || [];
    current = runs.value.find((run) => ["running", "pending"].includes(run.status));
  } catch (e) { error.value = e.message; } finally { loading.value = false; }
  if (modelEndpoint.value.trim()) void loadCurrentModel({ openPopover: false });
  if (current) {
    activeRunId.value = current.run_id;
    void loadActiveRun({ resetPage: true });
    startPolling();
  }
  if (await loadArbiterStatus()) arbiterTimer = window.setInterval(loadArbiterStatus, 3000);
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
  return ({event_memory_qa:"事件记忆",single_evidence_memory_qa:"单图证据",relationship_qa:"关系问答",multi_turn_clarify:"多轮澄清",multi_turn_disambiguation:"多轮消歧",ambiguous_retrieval:"模糊检索",evidence_insufficient:"证据不足",unsupported_retrieval:"无依据检索",instruction_injection:"指令注入",prompt_injection:"提示注入",data_exfiltration:"数据泄露",authority_impersonation:"权限伪造",mixed_injection:"混合注入",indirect_injection:"间接注入",jailbreak_attempt:"越狱尝试"}[t]) || t || "未分类";
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
  const assetId = typeof media === "string" ? "" : media?.asset_id;
  return assetId ? `/api/assets/${encodeURIComponent(assetId)}/file`
    : (typeof media === "string" ? "" : (media?.media_url || ""));
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
watch(activeRun, async () => { await nextTick(); renderTelemetryChart(); }, { deep: true });
onUnmounted(() => { destroyed = true; if (pollTimer) clearTimeout(pollTimer); if (arbiterTimer) clearInterval(arbiterTimer); if (telemetryChartInstance) telemetryChartInstance.dispose(); });
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
              <strong>{{ currentModelInfo.served_model_name || `${(currentModelInfo.served_models || []).length} 个模型待选择` }}</strong>
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
                <option v-for="model in endpointModels" :key="model" :value="model">{{ model }}</option>
              </select>
            </label>
            <button class="btn ghost compact endpoint-test-button" type="button" :disabled="!selectedEndpointModel || modelTestState === 'testing'" @click="testEndpointModel">{{ modelTestState === 'testing' ? '测试中…' : '测试 POST' }}</button>
            <span v-if="modelTestMessage" class="model-test-feedback" :class="`state-${modelTestState}`">{{ modelTestMessage }}</span>
            <label class="check endpoint-reuse-check" :class="{ active: selectedModels.has('__current__') }">
              <input type="checkbox" :checked="selectedModels.has('__current__')" :disabled="!selectedEndpointModel || !currentModelInfo?.served_model_name" @change="setModelSelected('__current__', $event.target.checked)" />复用所选模型<span v-if="selectedEndpointModel">（{{ selectedEndpointModel }}，不启停）</span>
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
	<div class="pager" v-if="runPage.pages > 1">
	  <button class="btn ghost compact" :disabled="!runPage.has_previous" @click="changeRunPage(runPage.page - 1)">上一页</button>
	  <span>{{ runPage.page }} / {{ runPage.pages }}</span>
	  <button class="btn ghost compact" :disabled="!runPage.has_next" @click="changeRunPage(runPage.page + 1)">下一页</button>
	</div>
	<span class="muted">共 {{ runPage.total }} 条</span>
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
<td>{{ fmtPct(run.summary?.media_retrieval_recall_macro ?? run.summary?.retrieval_recall_macro ?? run.summary?.retrieval_recall_mean) }}</td>
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
            <button class="btn compact" @click="exportTraces" title="每题导出 planner 完整输入/输出 + 完整轨迹">导出轨迹 JSON</button>
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
      <section class="memory-layer-panel">
        <div class="memory-layer-head">
          <div><h3>记忆层级与有效证据链</h3><p>分别查看“原始创建链路”和“测评时生效链路”。图中节点都绑定实际字段；点击节点可查看图片、语句、工具记录和已保存的执行轨迹。</p></div>
          <span class="layer-scope-badge">通用层级 · 时间 / 事件 / 地点 / 人物 / 场景</span>
        </div>
        <div class="memory-chain-tabs" role="tablist" aria-label="证据链类型">
          <button type="button" role="tab" :aria-selected="chainMode === 'creation'" :class="{ active: chainMode === 'creation' }" @click="chainMode = 'creation'">原始创建链路</button>
          <button type="button" role="tab" :aria-selected="chainMode === 'effective'" :class="{ active: chainMode === 'effective' }" @click="chainMode = 'effective'">测评时生效链路</button>
          <span>点击节点查看具体图、语句、工具调用和保存的过程记录</span>
        </div>
        <div class="memory-tree-diagram" aria-label="记忆层级树状关系图">
          <div class="memory-tree-title"><strong>层级关系图 / 树状证据链</strong><span>从原始媒体向上追踪到回答；每个彩色节点都可以点击展开证据</span></div>
          <div class="memory-tree-levels">
            <template v-for="(layer, layerIndex) in activeEvidenceGraphView" :key="`tree-${layer.key}`">
              <div class="memory-tree-level" :class="`memory-tree-${layer.color}`">
                <div class="memory-tree-level-label"><b>{{ layer.key }}</b><strong>{{ layer.name }}</strong></div>
                <div class="memory-tree-nodes">
                  <button v-for="node in layer.nodes" :key="`tree-${layer.key}-${node.kind}`" type="button" class="memory-tree-node" @click="openChainNode(chainMode, layer, node)">
                    <small>{{ node.kind }}</small><b>{{ node.value }}</b><span>{{ node.evidence }}</span>
                  </button>
                </div>
              </div>
              <div v-if="layerIndex < activeEvidenceGraphView.length - 1" class="memory-tree-link" aria-hidden="true"><i></i><span>↓ {{ layer.transition }}</span></div>
            </template>
          </div>
        </div>
        <section class="memory-effectiveness-panel">
          <div class="memory-effectiveness-head">
            <div><strong>创建记忆 vs 测评覆盖（工程指标，非正确率）</strong><span>这张表只说明数据是否被工具返回、图谱是否挂接、证据需求是否闭合；不能直接代表记忆正确或回答正确。</span></div>
            <span v-if="memoryEffectiveness?.data_quality?.graph_ready" class="phase-status completed">工具 / 图谱已校验</span>
            <span v-else-if="memoryEffectiveness?.status === 'ready'" class="phase-status failed">数据不完整</span>
            <span v-else class="phase-status pending">计算中</span>
          </div>
          <div v-if="memoryEffectiveness?.status === 'ready'" class="memory-effect-quality">
            <span>工具轨迹 {{ memoryEffectiveness.data_quality.tool_trace_count }} 条</span>
            <span>成功 {{ memoryEffectiveness.data_quality.tool_success_count }} 条</span>
            <span>媒体预览 {{ memoryEffectiveness.data_quality.preview_count }} 条</span>
            <span>事件上下文 {{ memoryEffectiveness.data_quality.event_context_count }} 个</span>
          </div>
          <div v-if="memoryEffectiveness?.status === 'ready'" class="memory-effect-table-wrap">
            <table class="memory-effect-table">
              <thead><tr><th>层级</th><th>创建 / 可用基数</th><th>评测轨迹覆盖</th><th>未覆盖 / 未闭合</th><th>覆盖率（非正确率）</th><th>专项指标（正确性/可用性）</th><th>判定口径</th></tr></thead>
              <tbody>
                <tr v-for="level in memoryEffectiveness.levels" :key="level.key">
                  <td><b>{{ level.key }}</b><strong>{{ level.name }}</strong></td>
                  <td><b>{{ level.created }}</b><small>{{ level.created_detail }}</small></td>
                  <td class="effect-good"><b>{{ level.effective }}</b><small>{{ level.effective_detail }}</small></td>
                  <td class="effect-bad"><b>{{ level.ineffective }}</b><small>{{ level.ineffective_detail }}</small><ul v-if="level.ineffective_reasons?.length" class="memory-invalid-reasons"><li v-for="reason in level.ineffective_reasons" :key="`${level.key}-${reason.label}`"><strong>{{ reason.label }}：{{ reason.count }}</strong><span>{{ reason.meaning }}</span></li></ul></td>
                  <td><b>{{ fmtPct(level.rate) }}</b><div class="memory-effect-bar"><i :style="{ width: `${Math.min(100, Math.max(0, Number(level.rate || 0) * 100))}%` }"></i></div></td>
                  <td><div v-for="metric in memoryLayerSpecialMetrics(level)" :key="metric.label" class="memory-special-metric"><b>{{ metric.label }}：{{ metric.value }}</b><small>{{ metric.note }}</small></div></td>
                  <td><small>创建 → 生效：{{ level.name }} 下游实际可追溯引用</small></td>
                </tr>
              </tbody>
            </table>
          </div>
          <p v-if="memoryEffectiveness?.status === 'ready'" class="memory-effect-note">口径边界：L0/L1/L2/L3 的数字是工程覆盖、工具可用或证据闭合代理指标，不是语义正确率；只有关键帧相对 GT 的召回/精确率和最终回答 Judge 结果可以作为正确性依据。真正的“记忆是否导致答错”见下方归因矩阵。</p>
          <p v-else-if="memoryEffectiveness?.error" class="error">{{ memoryEffectiveness.error }}</p>
          <section v-if="memoryEffectiveness?.status === 'ready' && memoryEffectiveness.details?.answer_correctness" class="answer-correctness-panel">
            <div class="answer-correctness-head"><strong>最终回答正确性对比</strong><span>按 Judge 评分统计，不等同于工具调用成功率</span></div>
            <div class="answer-correctness-grid">
              <div class="answer-correct answer-correct-good"><small>回答正确 · 2 分</small><b>{{ memoryEffectiveness.details.answer_correctness.correct }}</b><span>{{ fmtPct(memoryEffectiveness.details.answer_correctness.exact_accuracy) }} · 共 {{ memoryEffectiveness.levels.find(level => level.key === 'L4+')?.created || 0 }} 题</span></div>
              <div class="answer-correct answer-correct-partial"><small>部分正确 · 1 分</small><b>{{ memoryEffectiveness.details.answer_correctness.partial }}</b><span>{{ fmtPct((memoryEffectiveness.details.answer_correctness.partial || 0) / (memoryEffectiveness.levels.find(level => level.key === 'L4+')?.created || 1)) }}</span></div>
              <div class="answer-correct answer-correct-bad"><small>回答错误 · 0 分</small><b>{{ memoryEffectiveness.details.answer_correctness.incorrect }}</b><span>{{ fmtPct((memoryEffectiveness.details.answer_correctness.incorrect || 0) / (memoryEffectiveness.levels.find(level => level.key === 'L4+')?.created || 1)) }}</span></div>
              <div class="answer-correct answer-correct-na"><small>未评分</small><b>{{ memoryEffectiveness.details.answer_correctness.unjudged }}</b><span>不纳入正确率</span></div>
            </div>
            <div class="answer-layer-architecture memory-validity-architecture">
              <div class="answer-layer-title"><strong>记忆有效性分层指标</strong><span>原始媒体与关键帧从 L0 逐层进入最终回答</span></div>
              <div v-for="(layer, layerIndex) in memoryValidityArchitecture" :key="layer.key" class="answer-layer-row" :class="`answer-layer-${layer.key.toLowerCase().replace('+','')}`">
                <div class="answer-layer-label"><b>{{ layer.key }}</b><strong>{{ layer.name }}</strong><small>{{ layer.stage }}</small></div>
                <div class="answer-layer-track">
                  <div v-for="node in layer.nodes" :key="`${layer.key}-${node.label}`" class="answer-layer-node"><small>{{ node.label }}</small><b>{{ node.value }}</b><span>{{ node.note }}</span></div>
                </div>
                <div v-if="layerIndex < memoryValidityArchitecture.length - 1" class="answer-layer-bridge"><i>↓</i><span>{{ layerIndex === 0 ? '解析并形成场景记忆' : layerIndex === 1 ? '聚合为事件记忆' : layerIndex === 2 ? '形成可引用证据' : '生成并判定回答' }}</span></div>
              </div>
      <p class="answer-layer-note">口径：L0 的创建量只是记忆库规模；原始图片用测评 Recall，关键帧用“父级视频 Recall / 返回关键帧 Precision”。L1–L3 是测评期间的解析、图谱和证据链覆盖代理，只有 L4+ 的 Exact Accuracy 才是最终回答正确率。</p>
            </div>
          </section>
          <section v-if="memoryRecallMatrices.length" class="memory-answer-attribution-panel">
            <div class="answer-correctness-head"><strong>命中 × 回答结果分层矩阵</strong><span>工具召回命中 / 模型使用命中分开统计；只统计有 GT 媒体的可归因题目（{{ memoryRecallMatrices[0].eligible }} 题），数字由当前 run 实时计算</span></div>
            <div class="recall-matrices">
              <div v-for="matrix in memoryRecallMatrices" :key="matrix.key" class="answer-layer-architecture recall-layer-architecture recall-matrix" :class="`recall-matrix-${matrix.key}`">
                <div class="answer-layer-title"><strong>{{ matrix.title }}</strong><span>{{ matrix.note }}</span></div>
                <div v-for="(layer, layerIndex) in matrix.layers" :key="`${matrix.key}-${layer.key}`" class="answer-layer-row" :class="`answer-layer-${layer.key.toLowerCase().replace('+','')}`">
                  <div class="answer-layer-label"><b>{{ layer.key }}</b><strong>{{ layer.name }}</strong><small>{{ layer.stage }}</small></div>
                  <div v-if="layer.key === 'L2'" class="answer-layer-track answer-layer-branch-track">
                    <div v-for="group in layer.groups" :key="group.label" class="answer-layer-branch-group">
                      <div class="answer-layer-branch-head"><strong>{{ group.label }}</strong><b>{{ fmtPct(group.total / matrix.eligible) }}</b><small>{{ group.total }} 题 · {{ group.note }}</small></div>
                      <div class="answer-layer-branch-nodes">
                        <div v-for="node in group.nodes" :key="`${matrix.key}-${group.label}-${node.label}`" class="answer-layer-node"><small>{{ node.label }}</small><b>{{ node.value }}</b><span>{{ node.note }}</span></div>
                      </div>
                    </div>
                  </div>
                  <div v-else class="answer-layer-track">
                    <div v-for="node in layer.nodes" :key="`${matrix.key}-${layer.key}-${node.label}`" class="answer-layer-node"><small>{{ node.label }}</small><b>{{ node.value }}</b><span>{{ node.note }}</span></div>
                  </div>
                  <div v-if="layerIndex < matrix.layers.length - 1" class="answer-layer-bridge"><i>↓</i><span>{{ layerIndex === 0 ? '进入命中判定' : layerIndex === 1 ? '进入 Judge 评分' : '汇总为有效性指标' }}</span></div>
                </div>
                <p class="answer-layer-note">口径：{{ matrix.note }}；命中后按 Judge 判定：2 或 1 的有效证据计有效、0 计无效，未命中单独计为未命中。样本量随当前 run 变化，不做预设。</p>
              </div>
            </div>
          </section>
        </section>
        <div class="memory-graph-caption">字段证据明细（与上方树状关系图对应）</div>
        <div class="memory-evidence-graph" role="img" aria-label="L0 到 L4+ 逐层证据链图">
          <div v-for="(layer, layerIndex) in activeEvidenceGraphView" :key="layer.key" class="memory-graph-lane" :class="`memory-graph-${layer.color}`">
            <div class="memory-graph-label"><b>{{ layer.key }}</b><strong>{{ layer.name }}</strong><span>{{ layer.stage }}</span></div>
            <div class="memory-graph-track">
              <button v-for="node in layer.nodes" :key="`${layer.key}-${node.kind}`" type="button" class="memory-graph-node" @click="openChainNode(chainMode, layer, node)">
                <small>{{ node.kind }}</small><b>{{ node.value }}</b><span>{{ node.evidence }}</span>
              </button>
            </div>
            <div v-if="layerIndex < activeEvidenceGraphView.length - 1" class="memory-graph-bridge"><i aria-hidden="true">↓</i><span>{{ layer.transition }}</span></div>
          </div>
        </div>
        <section class="keyframe-analysis-panel">
          <div class="keyframe-analysis-head"><div><strong>关键帧有效性评估</strong><span>只保留关键帧规模、覆盖/检索、时间采样、回答对照、压缩率和视频处理速度</span><span v-if="keyframeAnalysis?.asset_source_note" class="keyframe-source-note">证据源：{{ keyframeAnalysis.asset_source_note }}</span></div><span v-if="keyframeAnalysisLoading" class="phase-status running">分析中</span><span v-else-if="keyframeAnalysis?.status === 'ready'" class="phase-status completed">已计算</span></div>
          <p v-if="keyframeAnalysisError" class="error">{{ keyframeAnalysisError }}</p>
          <div v-else-if="keyframeAnalysis?.status === 'ready'" class="keyframe-analysis-grid">
            <div><small>关键帧规模</small><b>{{ keyframeAnalysis.assets.keyframes }}</b><span>{{ keyframeAnalysis.assets.source_videos }} 个源视频 · 平均 {{ keyframeAnalysis.assets.keyframes_per_video_mean ?? "-" }} 帧/视频</span></div>
            <div><small>源视频父级召回</small><b>{{ fmtPct(keyframeAnalysis.retrieval.video_parent_recall) }}</b><span>{{ keyframeAnalysis.retrieval.video_gt_targets }} 个 GT 视频目标中，{{ keyframeAnalysis.retrieval.video_gt_targets_hit_by_keyframe }} 个被关键帧覆盖</span></div>
            <div><small>候选关键帧精确率</small><b>{{ fmtPct(keyframeAnalysis.retrieval.candidate_precision) }}</b><span>{{ keyframeAnalysis.retrieval.candidate_relevant_keyframes }}/{{ keyframeAnalysis.retrieval.candidate_keyframes }} 个候选与 GT 父视频一致</span></div>
            <div><small>模型最终返回精确率</small><b>{{ fmtPct(keyframeAnalysis.retrieval.predicted_precision) }}</b><span>{{ keyframeAnalysis.retrieval.predicted_relevant_keyframes }}/{{ keyframeAnalysis.retrieval.predicted_keyframes }} 个返回关键帧有效</span></div>
            <div><small>时间采样</small><b>{{ keyframeAnalysis.temporal.mean_gap_sec == null ? "-" : `${keyframeAnalysis.temporal.mean_gap_sec.toFixed(1)}s` }}</b><span>平均间隔 · 平均覆盖跨度 {{ keyframeAnalysis.temporal.mean_span_sec == null ? "-" : `${keyframeAnalysis.temporal.mean_span_sec.toFixed(1)}s` }}</span></div>
            <div><small>回答质量对照</small><b>{{ keyframeDeltaLabel(keyframeAnalysis.usefulness.answer_quality_delta) }}</b><span>有关键帧 {{ keyframeAnalysis.usefulness.answer_quality_with_keyframe ?? "-" }} · 无关键帧 {{ keyframeAnalysis.usefulness.answer_quality_without_keyframe ?? "-" }}（非因果）</span></div>
            <div><small>全量帧保留 / 压缩</small><b>{{ keyframeAnalysis.processing?.retained_frame_ratio == null ? "未保存原始帧" : `${fmtPct(keyframeAnalysis.processing.retained_frame_ratio)} / ${fmtPct(keyframeAnalysis.processing.compression_ratio)}` }}</b><span>{{ keyframeAnalysis.processing?.raw_frame_count ? `${keyframeAnalysis.processing.keyframe_count ?? keyframeAnalysis.assets.keyframes} / ${keyframeAnalysis.processing.raw_frame_count} 原始帧` : "复用记忆未重新执行视频处理，无法计算原始帧压缩率" }}</span></div>
            <div><small>视频处理速度</small><b>{{ keyframeAnalysis.processing?.realtime_factor == null ? "未记录" : `${keyframeAnalysis.processing.realtime_factor.toFixed(3)}×` }}</b><span>{{ keyframeAnalysis.processing?.processing_seconds_total ? `总耗时 ${keyframeAnalysis.processing.processing_seconds_total}s · 均值 ${keyframeAnalysis.processing.processing_seconds_mean ?? "-"}s/视频` : "复用记忆没有视频处理耗时，不估算速度" }}</span></div>
          </div>
          <details v-if="keyframeAnalysis?.status === 'ready'" class="metric-definition-panel">
            <summary>指标定义、计算式与判读</summary>
            <div v-for="definition in keyframeMetricDefinitions" :key="definition.name" class="metric-definition-row">
              <strong>{{ definition.name }}</strong><span><b>计算式：</b>{{ definition.formula }}</span><span><b>含义：</b>{{ definition.meaning }}</span>
            </div>
          </details>
          <p v-if="keyframeAnalysis?.status === 'ready'" class="keyframe-analysis-note">选择策略：{{ keyframeReasonLabel(keyframeAnalysis.temporal.selection_reasons) }} · 已覆盖 {{ keyframeAnalysis.temporal.videos_sampled }} 个视频、{{ keyframeAnalysis.usefulness.event_context_count }} 个事件上下文。父级召回衡量“关键帧能否代表源视频”，候选精确率衡量“是否混入无关帧”，两者应同时观察。</p>
          <p v-else-if="!keyframeAnalysisLoading" class="keyframe-analysis-note">尚未取得完整 run 轨迹，暂不能计算关键帧有效性。</p>
        </section>
        <p class="memory-layer-footnote">指标口径：运行级指标来自本次 run 的汇总轨迹；“已展开轨迹”只对当前打开的 QA 逐题核验。这样可以直接定位是 L0 媒体、L1 解析、L2 事件聚合、L3 目标分解还是 L4+ 回答阶段出了问题。</p>
      </section>
      <h3 class="result-heading">结果指标</h3>
<div class="result-phase-list">
        <article v-if="activeRun?.telemetry_live?.samples_count" class="phase-card result-phase-card gpu-result-card live-telemetry-card">
<div class="phase-title"><b>实时资源遥测</b><span class="phase-status running">{{ activeRun.telemetry_live.status === 'running' ? '实时更新中' : '已停止' }}</span></div>
<p class="metric-calc-time">测评进行中持续采样；任务失败或取消时保留已采集的最后值与峰值。{{ activeRun.telemetry_live.source === 'jetson_local_pss' ? ' Orin 使用进程 PSS 表示统一物理内存。' : '' }}</p>
<div class="phase-metrics live-telemetry-metrics"><div v-for="row in liveTelemetryRows(activeRun)" :key="row[0]" class="phase-metric"><span>{{ row[0] }}</span><strong>{{ row[1] }}</strong><small>{{ row[2] }}</small></div></div>
<div v-if="telemetryChart(activeRun)" class="telemetry-chart"><div ref="telemetryChartEl" class="telemetry-chart-canvas" role="img" aria-label="资源占用趋势"></div><small class="muted">最近 {{ (activeRun.telemetry_live.history || []).length }} 个采样点；悬浮查看每个时刻的 GiB，图例可单独隐藏曲线，底部可拖动缩放。整机 RAM 为宿主机全部进程，模型进程/全部 GPU 进程按可归因范围记录。</small></div>
<div v-if="liveTelemetryPhaseRows(activeRun).length" class="phase-metrics"><div v-for="row in liveTelemetryPhaseRows(activeRun)" :key="`live-${row[0]}`" class="phase-metric"><span>{{ row[0] }}阶段峰值</span><strong>{{ row[1] }}</strong><small>{{ row[2] }}</small></div></div>
<details v-if="(activeRun.telemetry_live.all_processes || []).length" class="telemetry-processes">
  <summary>GPU 进程明细（{{ (activeRun.telemetry_live.all_processes || []).length }} 个）</summary>
  <div class="telemetry-process-grid">
    <div v-for="process in activeRun.telemetry_live.all_processes" :key="`${process.pid}-${process.process_name}`" class="telemetry-process-row">
      <span>{{ process.process_name || "未知进程" }}</span><small>PID {{ process.pid }}</small><b>{{ fmtMemory(process.used_memory_mib) }}</b>
    </div>
  </div>
</details>
</article>
        <article class="phase-card result-phase-card gpu-result-card">
<div class="phase-title">
<b>GPU 指标</b>
<span class="phase-status" :class="resultPhaseStatus(activeRun.phases?.gpu_metrics)">{{ statusLabel(resultPhaseStatus(activeRun.phases?.gpu_metrics)) }}</span>
</div>
<p class="metric-calc-time">指标计算耗时 {{ fmtSeconds(phaseSeconds(activeRun.phases?.gpu_metrics)) }} · {{ ['orin_ssh_pss', 'jetson_local_pss'].includes(activeRun.phases?.gpu_metrics?.source) ? 'Orin：进程 PSS 是统一物理内存峰值，不是独立显存；工作负载内存和 KV 实际用量未拆分' : activeRun.phases?.gpu_metrics?.memory_pressure ? "macOS 统一内存系统级采样（含模型 Metal 分配）" : "模型进程显存为 NVML 按 PID 汇总的实际占用，KV Cache 为 vLLM 逻辑使用率" }}{{ activeRun.qa_concurrency > 1 ? ` · QA 并发 ${activeRun.qa_concurrency}（时延含排队，勿与串行 run 直接对比）` : "" }}</p>
<div class="phase-metrics">
<div v-for="row in gpuMetricRows(activeRun.phases?.gpu_metrics)" :key="row[0]" :class="['phase-metric', { 'priority-metric': row[3] }]">
<span>{{ row[0] }}</span>
<strong>{{ row[1] }}</strong>
<small>{{ row[2] }}</small>
</div>
</div>
</article>
        <article v-if="!isOrinPssRun(activeRun)" class="phase-card result-phase-card gpu-result-card">
<div class="phase-title">
<b>{{ comparableMemoryProfile(activeRun)?.source === 'replay' ? '可比较显存复测' : '可比较显存' }}</b>
<span class="phase-status" :class="comparableMemoryProfile(activeRun)?.status || 'pending'">{{ statusLabel(comparableMemoryProfile(activeRun)?.status || 'pending') }}</span>
</div>
<p class="metric-calc-time">{{ comparableMemoryProfile(activeRun)?.memory_profile?.method === 'orin_process_pss_uma_v1' ? 'Orin 统一内存按进程 PSS 记录；KV 字节数尚不可用，不与独立显存直接比较。' : (comparableMemoryProfile(activeRun)?.source === 'gpu_metrics' ? '来自本次正式评测 GPU 采样；' : comparableMemoryProfile(activeRun)?.source === 'replay' ? '复用现有相册与问题，不运行 Benchmark/Judge，不保存本次回答；' : '本次 run 的 GPU 采样结束后生成；') + '工作负载显存为扣除预留 KV 后的估算值，不是硬件实测。' }}</p>
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
              <section class="qa-layer-chain">
                <div class="qa-layer-chain-head"><strong>本题证据链</strong><span>关键帧若存在，显示为 L0 媒体；不会直接标成 L2 事件</span></div>
                <div class="qa-layer-chain-row">
                  <div v-for="(node, nodeIndex) in itemLayerChain(itemDetail(summary))" :key="node.key" class="qa-layer-node-wrap">
                    <div class="qa-layer-node" :class="`qa-layer-node-${node.status}`">
                      <b>{{ node.key }}</b><strong>{{ node.name }}</strong><span>{{ node.detail }}</span><em>{{ node.evidence }}</em>
                    </div>
                    <i v-if="nodeIndex < itemLayerChain(itemDetail(summary)).length - 1" class="qa-layer-arrow" aria-hidden="true">→</i>
                  </div>
                </div>
              </section>
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
                  <h4>工具召回图片（{{ toolRecallMedia(itemDetail(summary)).length }}）</h4>
                  <div class="image-grid"><div v-for="media in toolRecallMedia(itemDetail(summary))" :key="media.asset_id || media.file_name" class="image-tile"><video v-if="isVideoMedia(media) && imageUrl(media)" :src="imageUrl(media)" controls playsinline preload="metadata" @loadedmetadata="pauseVideoAtEvidenceFrame(media, $event)"></video><img v-else-if="imageUrl(media)" :src="imageUrl(media)" :alt="media.file_name" loading="lazy" @click="openImage(media)" /><span v-else class="image-empty">无媒体</span><span class="image-label">{{ media.file_name || media.media_id || media.image_id }}</span></div><span v-if="!toolRecallMedia(itemDetail(summary)).length" class="muted small">检索未返回可识别媒体</span></div>
                  <h4>模型使用图片（{{ itemEvidenceMedia(itemDetail(summary)).length }}）</h4>
                  <div class="image-grid"><div v-for="media in itemEvidenceMedia(itemDetail(summary)).slice(0, 3)" :key="`used-${media.asset_id || media.file_name}`" class="image-tile"><video v-if="isVideoMedia(media) && imageUrl(media)" :src="imageUrl(media)" controls playsinline preload="metadata" @loadedmetadata="pauseVideoAtEvidenceFrame(media, $event)"></video><img v-else-if="imageUrl(media)" :src="imageUrl(media)" :alt="media.file_name" loading="lazy" @click="openImage(media)" /><span v-else class="image-empty">无媒体</span><span class="image-label">{{ media.file_name || media.media_id || media.image_id }}</span></div><span v-if="!itemEvidenceMedia(itemDetail(summary)).length" class="muted small">回答依据图片为空（模型未选择交付图）</span></div>
                  <details v-if="itemEvidenceMedia(itemDetail(summary)).length > 3" class="qa-detail-block"><summary>查看更多使用图片（{{ itemEvidenceMedia(itemDetail(summary)).length - 3 }}）</summary><div class="image-grid"><div v-for="media in itemEvidenceMedia(itemDetail(summary)).slice(3)" :key="`used-more-${media.asset_id || media.file_name}`" class="image-tile"><video v-if="isVideoMedia(media) && imageUrl(media)" :src="imageUrl(media)" controls playsinline preload="metadata" @loadedmetadata="pauseVideoAtEvidenceFrame(media, $event)"></video><img v-else-if="imageUrl(media)" :src="imageUrl(media)" :alt="media.file_name" loading="lazy" @click="openImage(media)" /><span v-else class="image-empty">无媒体</span><span class="image-label">{{ media.file_name || media.media_id || media.image_id }}</span></div></div></details>
                </div>
                <div>
                  <h4>正确答案</h4><p>{{ itemDetail(summary).reference_answer }}</p>
                  <h4>检索 GT 媒体（{{ itemMedia(itemDetail(summary), true).length }}）</h4>
                  <div class="image-grid"><div v-for="media in itemMedia(itemDetail(summary), true)" :key="media.asset_id || `${media.media_type}-${media.media_id}`" class="image-tile"><video v-if="isVideoMedia(media) && imageUrl(media)" :src="imageUrl(media)" controls playsinline preload="metadata" @loadedmetadata="pauseVideoAtEvidenceFrame(media, $event)"></video><img v-else-if="imageUrl(media)" :src="imageUrl(media)" :alt="media.file_name" loading="lazy" @click="openImage(media)" /><span v-else class="image-empty">无媒体</span><span class="image-label">{{ media.file_name || media.media_id || media.image_id }}<em v-if="media.matched === false"> · 未召回</em></span></div></div>
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
  <Teleport to="body">
    <div v-if="chainDetail" class="chain-detail-backdrop" @click.self="closeChainDetail">
      <section class="chain-detail-modal" role="dialog" aria-modal="true" aria-label="证据链节点详情">
        <header class="chain-detail-head">
          <div><small>{{ chainDetail.mode === 'creation' ? 'ORIGINAL CREATION CHAIN' : 'EFFECTIVE EVALUATION CHAIN' }}</small><h3>{{ chainDetail.layer.key }} · {{ chainDetail.layer.name }} · {{ chainDetail.node.kind }}</h3><span>{{ chainDetail.node.value }} · {{ chainDetail.node.evidence }}</span></div>
          <button class="judge-modal-close" type="button" aria-label="关闭" @click="closeChainDetail">×</button>
        </header>
        <div class="chain-detail-body">
          <div class="chain-detail-summary">当前展示第 {{ chainDetail.page }} 页的 {{ chainDetail.records.length }} 条 QA 链路（每页 {{ chainDetail.pageSize }} 条，共 {{ chainDetail.total }} 条）。切换 QA 分页后，可继续查看其他链路。</div>
          <details v-for="(record, recordIndex) in chainDetail.records" :key="record.key" class="chain-record" :open="recordIndex === 0">
            <summary><b>QA {{ record.index + 1 }}</b><span>{{ record.question }}</span><em>{{ record.media.length }} 媒体 · {{ record.statements.length }} 语句 · {{ record.traceRows.length }} 调用</em></summary>
            <div class="chain-record-body">
              <section class="chain-detail-section"><h4>图像 / 媒体证据 <small>{{ record.media.length }} 项</small></h4><div class="chain-detail-media-grid"><div v-for="media in record.media" :key="`${record.key}-${mediaKey(media)}`" class="chain-detail-media"><video v-if="isVideoMedia(media) && imageUrl(media)" :src="imageUrl(media)" controls playsinline preload="metadata" @loadedmetadata="pauseVideoAtEvidenceFrame(media, $event)"></video><button v-else-if="imageUrl(media)" type="button" @click="openImage(media)"><img :src="imageUrl(media)" :alt="media.file_name || media.media_id" loading="lazy" /></button><span v-else class="image-empty">媒体文件未能解析</span><small>{{ media.file_name || media.media_id || media.asset_id }}</small></div><p v-if="!record.media.length" class="muted small">该节点没有保存可展示的媒体引用。</p></div></section>
              <section class="chain-detail-section"><h4>结构化语句 / 观察 <small>{{ record.statements.length }} 项</small></h4><div class="chain-detail-statements"><article v-for="statement in record.statements" :key="`${record.key}-${statement.label}-${statement.value}`"><b>{{ statement.label }}</b><p>{{ statement.value }}</p></article><p v-if="!record.statements.length" class="muted small">没有保存文字观察或回答声明。</p></div></section>
              <section class="chain-detail-section"><h4>工具调用与返回 <small>{{ record.traceRows.length }} 项</small></h4><details v-for="row in record.traceRows" :key="`${record.key}-${row.label}`" class="chain-trace-row"><summary>{{ row.label }}</summary><pre>{{ row.value }}</pre></details><p v-if="!record.traceRows.length" class="muted small">该节点没有保存工具调用轨迹。</p></section>
              <section class="chain-detail-section"><h4>已保存的执行过程 / 证据归因</h4><p class="chain-detail-disclaimer">以下只展示系统实际保存的 execution trace、Agent 证据账本和 grounding 信息，不把未保存的模型隐藏思维链伪装成可复现推理。</p><pre class="chain-reasoning-pre">{{ record.reasoning }}</pre></section>
            </div>
          </details>
          <p v-if="!chainDetail.records.length" class="muted small">当前页没有成功读取到可展开的 QA 记录，请先确认 QA 接口可用。</p>
        </div>
      </section>
    </div>
  </Teleport>
  <Teleport to="body"><div v-if="judgeModal" class="judge-modal-backdrop" @click.self="closeJudgeInput"><section class="judge-modal" role="dialog" aria-modal="true" aria-label="Judge 模型原始输入"><header><div><h3>JUDGE 模型原始输入</h3><span class="muted small">{{ judgeModal.qaId }}</span></div><button class="judge-modal-close" type="button" aria-label="关闭" @click="closeJudgeInput">×</button></header><pre v-if="judgeModal.complete">{{ judgeModal.rawJson }}</pre><p v-else class="judge-input-note">该历史结果在运行时未保存 Judge 原始请求 JSON，无法恢复。</p></section></div></Teleport>
</template>
