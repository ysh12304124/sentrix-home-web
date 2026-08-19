(function () {
  const app = document.getElementById("app");
  const benchmarkModelProfiles = [
    { id: "gemma4-12b-it", label: "Gemma-4-12B (默认)" },
    { id: "gemma4-e2b-it", label: "Gemma-4-E2B 蒸馏前" },
    { id: "gemma4-e2b-it-lora-v2", label: "Gemma-4-E2B 蒸馏后+LoRA" },
    { id: "qwen3.5-0.8b-it", label: "Qwen-3.5-0.8B" },
    { id: "qwen3-instruct", label: "Qwen-3-4B Instruct" },
    { id: "qwen3-8b", label: "Qwen-3-8B" },
  ];

  function modelProfileOptions(payload) {
    const profiles = new Map((payload?.profiles || []).map((profile) => [profile.id, profile]));
    const current = payload?.current || {};
    const candidate = String(current.profile || "");
    const active = current.status === "running" && benchmarkModelProfiles.some((profile) => profile.id === candidate) ? candidate : "";
    const models = Object.fromEntries(benchmarkModelProfiles.map(({ id, label }) => {
      const profile = profiles.get(id);
      return [id, {
        available: Boolean(profile?.available),
        loaded: id === active,
        model: label,
        url: id === active ? current.base_url : "vLLM profile",
      }];
    }));
    return {
      backend: active,
      status: current.status || "unmanaged",
      error: current.error || "",
      available_backends: benchmarkModelProfiles.map((profile) => profile.id),
      models,
    };
  }

  const state = {
    view: "overview",
    query: "",
    conversationId: "",
    searchResult: null,
    assistantMessages: [],
    conversations: [],
    activeConversationSummary: "",
    searchLoading: false,
    liveProgress: [],
    selectedAsset: null,
    photoInspector: null,
    loading: true,
    backendError: "",
    toast: "",
    spaces: [],
    scopeId: "",
    scopeInitialized: false,
    queue: [],
    dashboard: null,
    events: [],
    assets: [],
    persons: [],
    entities: [],
    entityGroups: [],
    geoPlaces: [],
    geoBreadcrumb: [],
    knowledge: { profiles: [], claims: [] },
    clusters: [],
    relationships: [],
    entityMergeCandidates: [],
    stories: [],
    trips: [],
    health: null,
    vlmBackendOptions: null,
    ocrSettings: null,
    modal: null,
    modalHistory: [],
    storyGenerating: false,
    storyError: false,
    storyDraftEventIds: [],
    eventFilter: "all",
    eventDate: "",
    assetFilter: "all",
    assetSort: "newest",
    personFilter: "all",
    saving: false,
    expandedEntityTypes: {},
  };

  // RX-6: admin/debug presentation is opt-in (URL ?debug=1 or localStorage).
  // Normal users never see internal ids, retrieval traces or raw JSON.
  function adminDebug() {
    const enabled = new URLSearchParams(window.location.search).has("debug")
      || window.localStorage?.getItem("sentrix.adminDebug") === "1";
    document.body.classList.toggle("admin", enabled);
    return enabled;
  }

  function capability(name) {
    return Boolean(state.health && state.health.agent && state.health.agent.capabilities
      && state.health.agent.capabilities[name]);
  }

  const navItems = [
    { id: "overview", icon: "⌂", label: "家庭概览" },
    { id: "search", icon: "⌕", label: "家庭记忆助手" },
    { id: "timeline", icon: "↕", label: "事件时间线" },
    { id: "people", icon: "◎", label: "人物与关系" },
    { id: "knowledge", icon: "◇", label: "实体与知识" },
    { id: "library", icon: "▦", label: "资料库" },
    { id: "stories", icon: "▶", label: "故事工作室" },
    { id: "imports", icon: "↓", label: "导入队列" },
  ];

  function escapeHtml(value) {
    return String(value ?? "").replace(/[&<>'"]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" }[char]));
  }

  function icon(value, extra = "") { return `<span class="icon ${extra}" aria-hidden="true">${value}</span>`; }

  function formatDate(value) {
    if (!value) return "未标注时间";
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? String(value).slice(0, 10).replace(/-/g, ".") : date.toISOString().slice(0, 10).replace(/-/g, ".");
  }

  function formatDateTime(value) {
    if (!value) return "未标注时间";
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? String(value) : `${formatDate(value)} ${date.toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" })}`;
  }

  function formatVideoTime(value) {
    const seconds = Math.max(0, Number(value) || 0);
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const rest = (seconds % 60).toFixed(seconds % 1 ? 1 : 0).padStart(2, "0");
    return hours ? `${hours}:${String(minutes).padStart(2, "0")}:${rest.padStart(4, "0")}` : `${minutes}:${rest}`;
  }

  function eventPeople(event) {
    return event.participants || event.participants_json || [];
  }

  function eventViewModel(event, index = 0) {
    const rolePeople = (event.participant_roles || []).map((person) => person.person_name).filter(Boolean);
    const people = (rolePeople.length ? rolePeople : eventPeople(event)).map((person) => typeof person === "string" ? person : person.name || person.display_name).filter(Boolean);
    return {
      ...event,
      date: formatDate(event.time_start),
      coverAssetId: event.cover_asset_id || event.asset_ids?.[0] || "",
      placeLabel: [event.place || "未标注地点", people.length ? people.join("、") : "暂无已确认人物"].join(" · "),
      albumLabel: albumLabel(event.scope_id),
      typeLabel: event.event_type || "家庭事件",
      countLabel: `${(event.observation_ids || []).length} 项证据`,
      isVideoScene: event.source_type === "video_scene",
      sceneRange: `${formatVideoTime(event.source_start_sec)}~${formatVideoTime(event.source_end_sec)}`,
      sceneDuration: Math.max(0, Number(event.source_end_sec || 0) - Number(event.source_start_sec || 0)),
      tone: ["mint", "peach", "blue", "lime", "lavender"][index % 5],
    };
  }

  function albumLabel(scopeId) {
    if (!scopeId) return "全部相册";
    return state.spaces.find((space) => space.id === scopeId)?.name || scopeId;
  }

  function visibleSpaces() {
    return [
      { id: "", name: "全部相册", kind: "all" },
      ...state.spaces.filter((space) => space.kind === "benchmark"),
    ];
  }

  function albumBadge(scopeId) {
    return `<span class="album-badge">${escapeHtml(albumLabel(scopeId))}</span>`;
  }

  function mediaLabel(type) {
    return type === "image" ? "图片" : type === "audio" ? "音频" : type === "video" ? "视频" : "文本";
  }

  function assetStatusLabel(status) {
    if (status === "processed") return "已完成语义整理";
    if (status === "semantic_enriching") return "基础证据可用，语义整理中";
    if (status === "processing") return "正在建立基础证据";
    if (status === "video-queued") return "视频等待处理";
    if (status === "video-metadata") return "读取视频元数据";
    if (status === "video-keyframe-extracting") return "正在提取关键帧并切分场景";
    if (status === "video-scene-importing") return "场景关键帧正在进入长期记忆";
    if (status === "video-processing-failed") return "视频处理失败，可重试";
    return status || "排队中";
  }

  function stats() {
    return state.dashboard?.stats || { assets: 0, observations: 0, events: 0, facts: 0, persons: 0 };
  }

  function filteredEvents() {
    let source = state.events.map(eventViewModel);
    if (state.eventDate) {
      source = source.filter((event) => [event.time_start, event.time_end].some((value) => String(value || "").slice(0, 10) === state.eventDate));
    }
    if (state.eventFilter === "all") return source;
    if (state.eventFilter === "people") return source.filter((event) => eventPeople(event).length);
    if (state.eventFilter === "place") return source.filter((event) => event.place);
    return source.filter((event) => event.event_type === state.eventFilter);
  }

  function filteredAssets() {
    let source = state.assets.slice();
    if (state.assetFilter !== "all") source = source.filter((asset) => asset.media_type === state.assetFilter);
    source.sort((left, right) => state.assetSort === "oldest" ? left.created_at.localeCompare(right.created_at) : right.created_at.localeCompare(left.created_at));
    return source;
  }

  function pageHeader(kicker, title, description, action = "") {
    return `<div class="page-heading"><div><p class="eyebrow">${escapeHtml(kicker)}</p><h1>${escapeHtml(title)}</h1><p class="page-description">${escapeHtml(description)}</p></div>${action}</div>`;
  }

  function searchBar(placeholder = "问人物、地点、事件或想回顾的片段…") {
    return `<form class="search-bar assistant-composer" id="search-form"><span class="search-symbol">⌕</span><input id="search-input" value="${escapeHtml(state.query)}" placeholder="${escapeHtml(placeholder)}" /><kbd>Enter</kbd><button type="submit" aria-label="发送给家庭记忆助手">→</button></form>`;
  }

  function memoryPill(type, label, stateLabel = "已建立") {
    return `<span class="memory-pill ${type}"><i></i>${escapeHtml(label)}<small>${escapeHtml(stateLabel)}</small></span>`;
  }

  function emptyState(title, description, action = "") {
    return `<div class="empty-queue"><span>⌁</span><strong>${escapeHtml(title)}</strong><p>${escapeHtml(description)}</p>${action}</div>`;
  }

  function assetThumb(asset, compact = false) {
    const source = `/api/assets/${encodeURIComponent(asset.id)}/file`;
    const body = asset.media_type === "image" ? `<img src="${source}" alt="原始图片证据" loading="lazy" />` : asset.media_type === "video" ? `<video src="${source}" preload="metadata"></video>` : `<span class="asset-type-mark">${mediaLabel(asset.media_type)}</span>`;
    return `<div class="asset-thumb ${compact ? "compact" : ""} ${asset.media_type}">${body}<b>${escapeHtml(mediaLabel(asset.media_type))}</b></div>`;
  }

  function videoSceneStack(event) {
    const frames = (event.keyframe_assets || []).slice(0, 4);
    return `<div class="video-scene-cover ${event.tone}"><div class="scene-frame-stack">${frames.map((frame, index) => `<img src="/api/assets/${encodeURIComponent(frame.id)}/file" alt="视频场景关键帧 ${index + 1}" loading="lazy" style="--stack-index:${index}" />`).join("")}</div><div class="event-cover-label">${albumBadge(event.scope_id)}<span>${escapeHtml(event.title)}</span></div><b>${frames.length || (event.observation_ids || []).length} 张关键帧</b></div>`;
  }

  function faceAvatar(faceInstanceId, label = "?", tone = "gray") {
    if (faceInstanceId) {
      return `<img class="face-avatar ${tone}" src="/api/face-instances/${encodeURIComponent(faceInstanceId)}/crop" alt="${escapeHtml(label)}的人脸证据" loading="lazy" />`;
    }
    return `<span class="avatar person-avatar ${tone}">${escapeHtml((label || "?").slice(0, 1))}</span>`;
  }

  function assetCard(asset) {
    return `<button class="asset-card" data-action="open-asset" data-asset-id="${escapeHtml(asset.id)}">${assetThumb(asset, true)}<div class="asset-info"><div class="asset-info-head"><strong>原始${escapeHtml(mediaLabel(asset.media_type))}证据</strong>${albumBadge(asset.source_album_id || asset.scope_id)}</div><small>${formatDateTime(asset.created_at)} · ${escapeHtml(mediaLabel(asset.media_type))}</small><span>${escapeHtml(assetStatusLabel(asset.status))}</span></div></button>`;
  }

  function eventRow(event) {
    return `<button class="event-row" data-action="open-event" data-event-id="${escapeHtml(event.id)}"><span class="event-date">${event.date}</span><span class="event-dot ${event.tone}"></span><span class="event-summary"><strong>${escapeHtml(event.title)}</strong><small>${escapeHtml(event.placeLabel)}</small></span><span class="event-meta">${albumBadge(event.scope_id)} ${escapeHtml(event.typeLabel)} · ${escapeHtml(event.countLabel)}</span>${icon("→", "row-arrow")}</button>`;
  }

  function shell() {
    const activeSpace = visibleSpaces().find((space) => space.id === state.scopeId) || visibleSpaces()[0];
    const spaceOptions = visibleSpaces().map((space) => `<option value="${escapeHtml(space.id)}" ${space.id === state.scopeId ? "selected" : ""}>${escapeHtml(space.name || space.id)}</option>`).join("");
    const activeScopeLabel = activeSpace?.kind === "all" ? "三相册合并视图" : "独立相册空间";
    const currentLabel = state.view === "settings" ? "设备与隐私" : (navItems.find((item) => item.id === state.view)?.label || "");
    app.innerHTML = `<aside class="sidebar"><div class="brand-lockup"><span class="brand-mark">S</span><div><strong>Sentrix</strong><small>Home Memory</small></div></div><label class="space-switcher"><span class="avatar tiny">S</span><span><b>当前相册</b><select id="space-select" aria-label="切换全部相册或独立相册">${spaceOptions}</select><small>${escapeHtml(activeScopeLabel)}</small></span></label><div class="side-label">家庭记忆</div><nav class="main-nav">${navItems.map((item) => `<button class="nav-item ${state.view === item.id ? "active" : ""}" data-view="${item.id}">${icon(item.icon)}<span>${item.label}</span></button>`).join("")}</nav><div class="side-label lower">空间与系统</div><nav class="main-nav"><button class="nav-item ${state.view === "settings" ? "active" : ""}" data-view="settings">${icon("◌")}<span>设备与隐私</span></button><button class="nav-item" data-action="open-help">${icon("?")}<span>使用帮助</span></button><button class="nav-item" data-action="open-qa-dashboard">${icon("▤")}<span>QA 测评</span></button></nav><div class="sidebar-footer"><div class="local-pulse"><i></i><span>${state.backendError ? "本地服务不可用" : "本地 AI 正常运行"}</span></div><small>Sentrix Home · 0.2.0</small></div></aside><main class="main-content"><header class="topbar"><div class="breadcrumbs">${state.view !== "overview" ? `<button class="crumb-back" data-action="back">${icon("←")}返回</button><b>/</b>` : ""}<button data-action="home">Sentrix Home</button>${currentLabel ? `<b>/</b><strong>${escapeHtml(currentLabel)}</strong>` : ""}</div><div class="top-actions"><button class="icon-button" data-action="command" aria-label="打开命令搜索">⌘</button><button class="top-user" data-action="open-space"><span class="avatar tiny">S</span><span>${escapeHtml(activeSpace?.name || "全部相册")}</span>${icon("⌄", "muted")}</button></div></header><div id="view-root" class="view-root"></div></main><div id="toast-root" aria-live="polite"></div><div id="modal-root"></div>`;
    renderView();
  }

  function overview() {
    const count = stats();
    const events = state.events.slice(0, 3).map(eventViewModel);
    return `${pageHeader("家庭记忆 / 真实数据", "把家里的记忆，重新放在一起。", "这里展示已经导入并完成处理的本地资料。没有资料时，Sentrix 不会用示例内容填充。", `<button class="button primary" data-view="imports">${icon("＋")}导入资料</button>`)}${state.backendError ? `<div class="error-banner">${escapeHtml(state.backendError)}</div>` : ""}<div class="album-context"><span class="section-kicker">当前数据范围</span><strong>${escapeHtml(albumLabel(state.scopeId))}</strong><small>${state.scopeId ? "当前正在查看单个相册" : "当前正在查看 album1、album2、album3 的全部优化内容"}</small></div><section class="overview-search"><div><p class="section-kicker">问 Sentrix</p><h2>你想找回哪一段记忆？</h2><p>从人物、时间、地点、物体或一句描述开始，答案会带回原始证据。</p></div>${searchBar()}</section><section class="stats-grid"><article class="stat-card"><span>已整理内容</span><strong>${count.assets}</strong><small>本地 Asset</small></article><article class="stat-card"><span>已形成事件</span><strong>${count.events}</strong><small>可回到 Observation</small></article><article class="stat-card"><span>待确认事实</span><strong>${state.dashboard?.pendingFacts ?? 0}</strong><small>版本维护队列</small></article><article class="stat-card accent"><span>本地 AI 状态</span><strong>${state.health?.status === "ok" ? "正常" : "未知"}</strong><small>${escapeHtml(state.health?.models?.llm?.name || "等待服务")}</small></article></section><section class="content-section"><div class="section-head"><div><p class="section-kicker">三类记忆</p><h2>同一家庭，不同的记忆入口</h2></div><button class="text-button" data-view="settings">查看系统状态 ${icon("→")}</button></div><div class="memory-grid"><article class="memory-card episodic-card"><div class="card-top">${memoryPill("episodic", "事件记忆")}<span class="card-index">01</span></div><h3>把分散的资料聚成共同经历</h3><p>图片、音频和文本共同参与人物、时间、地点与事件整理。</p><div class="card-metric"><strong>${count.events}</strong><span>个已建立事件</span></div></article><article class="memory-card semantic-card"><div class="card-top">${memoryPill("semantic", "语义记忆")}<span class="card-index">02</span></div><h3>让事实持续生长且保留修订</h3><p>每条事实都保留来源、置信度和人工确认历史。</p><div class="card-metric"><strong>${count.facts}</strong><span>条本地事实</span></div></article><article class="memory-card visual-card"><div class="card-top">${memoryPill("visual", "视频场景记忆", "已启用")}<span class="card-index">03</span></div><h3>视频自动整理为可回溯的场景记忆</h3><p>WorldMM 提取关键帧并划分 Scene，关键帧继续进入图片理解、人物、OCR、向量与检索。</p><div class="reserved-line">${icon("◌")} video_memory_adapter <span>WorldMM 已接入</span></div></article></div></section><section class="content-section two-column"><div><div class="section-head"><div><p class="section-kicker">最近事件</p><h2>家里的时间线</h2></div><button class="text-button" data-view="timeline">查看全部 ${icon("→")}</button></div>${events.length ? `<div class="event-list">${events.map(eventRow).join("")}</div>` : emptyState("还没有事件", "导入图片、音频或文本后，处理完成的 Observation 会在这里形成事件。", `<button class="button small primary" data-view="imports">${icon("＋")}导入第一份资料</button>`)}</div><div><div class="section-head"><div><p class="section-kicker">需要你的确认</p><h2>让记忆更准确</h2></div><button class="text-button" data-view="settings">查看事实 ${icon("→")}</button></div>${(state.dashboard?.pendingFacts || 0) ? `<div class="review-panel"><div class="review-face-pair"><span class="avatar large gray">?</span></div><div><strong>${state.dashboard.pendingFacts} 条事实等待确认</strong><p>确认或驳回前，原始 Observation 会一直保留。</p></div><div class="review-actions"><button class="button small primary" data-view="settings">处理</button></div></div>` : emptyState("目前没有待确认事实", "新资料产生矛盾信息时，会进入版本维护队列。")}</div></section>`;
  }

  function evidenceCard(evidence) {
    const isAdmin = adminDebug();
    const sourceAction = evidence.kind === "observation" && evidence.asset_id ? `data-action="open-asset" data-asset-id="${escapeHtml(evidence.asset_id)}"` : evidence.event_id ? `data-action="open-event" data-event-id="${escapeHtml(evidence.event_id)}"` : evidence.kind === "fact" && evidence.evidence_ids?.[0] ? `data-action="open-observation" data-observation-id="${escapeHtml(evidence.evidence_ids[0])}"` : "";
    const title = evidence.kind === "fact" ? `${evidence.subject} ${evidence.predicate} ${evidence.object}` : evidence.summary || evidence.caption || "原始图片证据";
    const text = evidence.kind === "observation" ? evidence.caption || evidence.transcript || "无文字摘要" : evidence.summary || evidence.status || "";
    const media = evidence.kind === "observation" && evidence.asset_id ? `<button class="evidence-media" data-action="open-asset" data-asset-id="${escapeHtml(evidence.asset_id)}" aria-label="打开原始证据">${assetThumb({ id: evidence.asset_id, media_type: evidence.media_type || "image" }, true)}</button>` : "";
    const assetAction = evidence.kind === "asset" && evidence.id ? `data-action="open-asset" data-asset-id="${escapeHtml(evidence.id)}"` : "";
    const main = sourceAction || assetAction ? `<button class="evidence-main" ${sourceAction || assetAction}><strong>${escapeHtml(title)}</strong><p>${escapeHtml(text)}</p><small>${escapeHtml(evidence.captured_at || evidence.place || evidence.media_type || "证据记录")}</small></button>` : `<div class="evidence-main static"><strong>${escapeHtml(title)}</strong><p>${escapeHtml(text)}</p><small>${escapeHtml(evidence.captured_at || evidence.place || evidence.media_type || "证据记录")}</small></div>`;
    // RX-6: internal ids and raw JSON only in the admin debug layer.
    const idLine = isAdmin && (evidence.asset_id || evidence.observation_id) ? `<small class="admin-only">${escapeHtml(evidence.asset_id || "")}${evidence.observation_id ? " · " + escapeHtml(evidence.observation_id) : ""}</small>` : "";
    const debugRaw = isAdmin && evidence.raw ? `<details class="admin-only"><summary>查看模型原始 JSON</summary><pre>${escapeHtml(JSON.stringify(evidence.raw, null, 2))}</pre></details>` : "";
    return `<article class="evidence-card"><div class="evidence-head"><span class="evidence-kind">${evidence.kind === "observation" ? "图片观察" : evidence.kind === "asset" ? "原始资料" : evidence.kind === "fact" ? "人物事实" : "记忆证据"}</span></div>${idLine}${media}${main}${debugRaw}</article>`;
  }

  function evidenceLayer(title, values) {
    if (!values?.length) return "";
    return `<section class="evidence-layer"><div class="section-head"><div><p class="section-kicker">${escapeHtml(title)}</p><h3>${values.length} 项</h3></div></div><div class="evidence-list">${values.slice(0, 12).map(evidenceCard).join("")}</div></section>`;
  }

  function imageResults(result) {
    const images = result?.image_results || [];
    if (!images.length) return "";
    const rows = images.map((item) => {
      const label = item.display_handle || "原始图片";
      const aspects = [
        ...(item.supported_aspects || []).map((aspect) => `对上了：${aspect}`),
        ...(item.uncertain_aspects || []).map((aspect) => `还不能确认：${aspect}`),
      ];
      const caption = aspects.length
        ? aspects.map(escapeHtml).join(" · ")
        : (item.captured_at || item.caption || "可回看的原始证据");
      const dup = item.near_duplicate_size > 1 ? `<small class="image-dup">另有 ${item.near_duplicate_size - 1} 张相似照片</small>` : "";
      return `<button class="image-result" data-action="open-asset" data-asset-id="${escapeHtml(item.asset_id)}"><img src="${escapeHtml(item.media_url)}" alt="${escapeHtml(label)}" loading="lazy" /><span><strong>${escapeHtml(label)}</strong><small>${escapeHtml(String(caption))}</small>${dup}</span></button>`;
    }).join("");
    return `<section class="evidence-layer image-results"><div class="section-head"><div><p class="section-kicker">相关图片</p><h3>${images.length} 张</h3></div></div><div class="image-result-grid">${rows}</div></section>`;
  }

  function traceLabel(item) {
    if (typeof item === "string") return item;
    return item?.stage || "检索阶段";
  }

  function traceDetail(item) {
    if (typeof item === "string") return "已完成检索阶段";
    const counts = Object.entries(item?.counts || {}).map(([key, value]) => `${key} ${value}`).join(" · ");
    return counts || item?.status || "已完成检索阶段";
  }

  function algorithmEvidence(result) {
    if (!adminDebug()) return "";
    const trace = result.retrievalTrace || result.retrieval_trace || [];
    const modelEvidence = result.modelEvidence || [];
    return `<details class="algorithm-evidence admin-only"><summary>算法判断依据</summary><div class="algorithm-evidence-body"><p>回答只使用本地语义、事件和原始观察；结构化命中时会跳过向量检索。</p><dl>${trace.map((item) => `<div><dt>${escapeHtml(traceLabel(item))}</dt><dd>${escapeHtml(item.status || "complete")} · ${escapeHtml(traceDetail(item))}</dd></div>`).join("")}</dl><p>模型引用校验：${modelEvidence.length} 项候选引用，未通过证据 ID 校验时自动降级为本地证据回答。</p></div></details>`;
  }

  function toolTrace(result) {
    if (!adminDebug()) return "";
    const trace = result.toolTrace || result.tool_trace || [];
    if (!trace.length) return "";
    return `<details class="algorithm-evidence admin-only"><summary>本轮判断与工具</summary><div class="algorithm-evidence-body"><dl>${trace.map((item) => `<div><dt>${escapeHtml(item.tool || "memory_tool")}</dt><dd>${escapeHtml(item.permission || "read")} · ${escapeHtml(item.status || "complete")}${item.reason ? ` · ${escapeHtml(item.reason)}` : ""}</dd></div>`).join("")}</dl></div></details>`;
  }

  function guardDebug(result) {
    if (!adminDebug()) return "";
    const gd = result.guard_debug || result.guardDebug || {};
    const l1 = gd.l1_codes || gd.l1Codes || [];
    const judge = gd.judge || [];
    const rec = gd.recovery_attempts || 0;
    const rows = [];
    rows.push(`<div><dt>L1 规则</dt><dd>${l1.length ? escapeHtml(l1.join(" · ")) : "通过"}</dd></div>`);
    judge.forEach((j, i) => {
      const problems = (j.problems || []).join(" · ") || "通过";
      rows.push(`<div><dt>L2 评审 #${i + 1}</dt><dd>${j.faithful ? "faithful" : "unfaithful"} · ${escapeHtml(problems)}</dd></div>`);
    });
    rows.push(`<div><dt>恢复步数</dt><dd>${rec}</dd></div>`);
    if (gd.status) rows.push(`<div><dt>最终状态</dt><dd>${escapeHtml(gd.status)}${gd.reason ? " · " + escapeHtml(gd.reason) : ""}</dd></div>`);
    return `<details class="algorithm-evidence admin-only"${gd.status === "blocked_by_guard" ? " open" : ""}><summary>Guard 校验明细</summary><div class="algorithm-evidence-body"><dl>${rows.join("")}</dl></div></details>`;
  }

  const TOOL_LABELS = {
    search_memories: "搜索记忆",
    query_memory_facts: "查询记忆",
    get_original_photos: "获取原始照片",
    inspect_photo: "检查照片",
    result_page: "翻看更多结果",
  };

  const STAGE_LABELS = {
    thinking: "理解问题",
    gate: "判断意图",
    retrieval: "检索记忆",
    channels: "合并检索结果",
    inspecting: "检查照片",
    recovering: "核对事实",
    finalizing: "组织回答",
    answer: "组织回答",
    tool_result: "调用工具",
    tool_error: "调用工具",
  };

  function thinkingStepLabel(step) {
    if (step.type === "tool") return TOOL_LABELS[step.tool] || step.tool || "调用工具";
    return STAGE_LABELS[step.stage] || (step.stage ? String(step.stage) : "思考");
  }

  function buildThinkingSteps(result, liveProgress = null) {
    const progress = liveProgress || (result && (result.public_progress || [])) || [];
    const tools = (result && (result.toolTrace || result.tool_trace)) || [];
    const taskTools = (result && result.task_state && result.task_state.tool_results) || [];
    let toolIndex = 0;
    let taskIndex = 0;
    return progress.map((item) => {
      const stage = item.stage || "";
      const status = item.status || "running";
      if (stage === "tool_result" || stage === "tool_error") {
        const tool = tools[toolIndex] || {};
        const fallback = taskTools[taskIndex] || {};
        toolIndex += 1;
        taskIndex += 1;
        const denied = tool.status === "denied" || stage === "tool_error";
        return {
          type: "tool",
          tool: tool.tool || fallback.tool,
          text: item.text || tool.reason || (denied ? "工具调用被拒绝" : "正在处理…"),
          status: denied ? "blocked" : "complete",
          latency: tool.latency_s,
        };
      }
      return {
        type: "stage",
        stage,
        text: item.text || "",
        status: status === "ok" ? "complete" : status,
      };
    });
  }

  function agentStepHtml(step) {
    const running = step.status === "running";
    const blocked = step.status === "blocked" || step.status === "denied" || step.status === "error";
    const stateClass = running ? "running" : blocked ? "blocked" : "complete";
    const mark = step.type === "tool" ? "🔧" : "💭";
    const statusMark = running ? "…" : blocked ? "!" : "✓";
    const latency = step.latency != null ? `<small>${escapeHtml(String(step.latency))}s</small>` : "";
    return `<div class="agent-step ${stateClass}"><span class="agent-step-mark">${mark}</span><div class="agent-step-body"><strong>${escapeHtml(thinkingStepLabel(step))}</strong><span>${escapeHtml(step.text || "")}</span>${latency}</div><span class="agent-step-status">${statusMark}</span></div>`;
  }

  function assistantAnswer(result) {
    const segments = result.segments || [{ type: "text", text: result.answer || "" }];
    return segments.map((segment) => {
      const text = escapeHtml(segment.text || "").replace(/\n/g, "<br />");
      if (segment.type !== "claim") return text;
      return `<span class="assistant-claim assistant-claim-${escapeHtml(segment.status || "unverified")}" data-claim-id="${escapeHtml(segment.claim_id || "")}">${text}</span>`;
    }).join("");
  }

  function claimEvidence(result) {
    const claims = result.claims || [];
    const index = result.claim_evidence_index || result.claimEvidenceIndex || {};
    const evidenceById = new Map((result.evidence || []).map((item) => [item.id, item]));
    const bundles = new Map((result.evidence_bundles || result.evidenceBundles || []).map((item) => [item.claim_id, item]));
    if (!claims.length) return "";
    const rows = claims.map((claim) => {
      const link = index[claim.claim_id] || {};
      const bundle = bundles.get(claim.claim_id) || {};
      const evidence = (link.evidence_ids || []).map((id) => {
        const original = evidenceById.get(id);
        if (original) return original;
        const canonical = (bundle.canonical_evidence || []).find((item) => item.evidence_id === id);
        if (!canonical) return null;
        return { id, kind: canonical.type, event_id: canonical.type === "event" ? id : undefined, observation_id: canonical.type === "observation" ? id : undefined, asset_id: canonical.asset_id, summary: canonical.source_text, caption: canonical.source_text, time_start: canonical.time };
      }).filter(Boolean);
      const verification = (result.claim_verifications || result.claimVerifications || []).find((item) => item.claim_id === claim.claim_id) || {};
      return `<article class="claim-evidence-row" data-claim-id="${escapeHtml(claim.claim_id)}"><div class="claim-evidence-head"><strong>${escapeHtml(claim.text || "")}</strong><small>${escapeHtml(verification.status || link.status || "未校验")}</small></div>${evidence.length ? `<div class="claim-evidence-list">${evidence.slice(0, 8).map(evidenceCard).join("")}</div>` : `<p class="claim-evidence-empty">这条表述没有绑定到可展示的依据。</p>`}</article>`;
    }).join("");
    return `<details class="claim-evidence"><summary>逐句查看依据</summary><div class="claim-evidence-body">${rows}</div></details>`;
  }

  function proactiveRecall(result) {
    const recall = result.proactive_recall;
    if (!recall || !result.proactivity_candidate_found) return "";
    return `<section class="proactive-recall"><p>${escapeHtml(recall.entry_text || "我想到一段相关回忆。")}</p><div class="proactive-actions"><button class="text-button" data-action="accept-proactive" data-scene-key="${escapeHtml(recall.scene_key)}">看看这段回忆 ${icon("→")}</button><button class="text-button" data-action="dismiss-proactive" data-scene-key="${escapeHtml(recall.scene_key)}">暂不查看</button><button class="text-button muted" data-action="disable-proactive" data-scene-key="${escapeHtml(recall.scene_key)}">关闭主动回忆</button></div></section>`;
  }

  function toolLoopEvidence(result) {
    const samples = [];
    for (const tr of ((result.task_state || {}).tool_results) || []) {
      const samples = (tr.observation && tr.observation.samples) || tr.samples || [];
      for (const s of samples) {
        if (!s || !s.asset_id) continue;
        samples.push({
          kind: "observation",
          asset_id: s.asset_id,
          media_type: s.media_type || "image",
          caption: s.caption || s.transcript || "",
          captured_at: s.captured_at,
        });
      }
    }
    const seen = new Set();
    return samples.filter((item) => {
      if (seen.has(item.asset_id)) return false;
      seen.add(item.asset_id);
      return true;
    }).slice(0, 6);
  }

  function assistantEvidence(result) {
    const isAdmin = adminDebug();
    const layers = result.evidence_layers || {};
    const presentation = result.evidence_presentation || {};
    const grounding = result.answerGrounding || result.answer_grounding || {};
    const displayMode = grounding.display_mode || "";
    const primary = [...(layers.events || []), ...(layers.observations || []), ...(layers.claims || [])].sort((left, right) => (right.relevance || 0) - (left.relevance || 0));
    const candidates = result.clarification_candidates || [];
    const evidence = primary.length ? evidenceLayer("本次依据（按相关度）", primary.slice(0, 6)) : "";
    const gaps = layers.gaps || [];
    const gapContent = gaps.length ? `<section class="evidence-gap"><div class="section-head"><div><p class="section-kicker">证据缺口</p><h3>当前没有足够的原始依据</h3></div></div><p>${escapeHtml(gaps[0].reason || "请补充人物、地点、日期或其他线索。")}</p></section>` : "";
    const followups = candidates.length ? `<div class="assistant-followups"><p>请选择你指的人物：</p>${candidates.map((item) => `<button class="assistant-identity-choice" data-action="continue-assistant" data-query="${escapeHtml(item.name)}" data-entity-id="${escapeHtml(item.id)}">${item.preview_asset_id ? `<img src="/api/assets/${encodeURIComponent(item.preview_asset_id)}/file" alt="" loading="lazy" />` : `<span class="assistant-choice-placeholder">${escapeHtml((item.name || "?").slice(0, 1))}</span>`}<span><strong>${escapeHtml(item.name || "已确认成员")}</strong><small>${escapeHtml(item.family_role || "已确认人物")} · ${item.evidence_count || 0} 条依据</small></span></button>`).join("")}</div>` : "";
    const ordered = result.evidence_order || [];
    const order = ordered.length && isAdmin ? `<details class="algorithm-evidence admin-only"><summary>证据顺序与可信度</summary><div class="algorithm-evidence-body"><dl>${ordered.map((item, index) => `<div><dt>${String(index + 1).padStart(2, "0")} · ${escapeHtml(item.source_level)}</dt><dd>${escapeHtml(item.time || "时间未标注")} · 可信度 ${Math.round((item.confidence || 0) * 100)}%</dd></div>`).join("")}</dl></div></details>` : "";
    const directEvidence = Boolean(result.original_evidence_requested || presentation.direct_original_evidence);
    const directOriginal = directEvidence ? `<section class="assistant-original-evidence"><div class="section-head"><div><p class="section-kicker">直接查看原始证据</p><h3>与本次回答相关的原始资料</h3></div></div>${imageResults(result) || evidence || gapContent}</section>` : "";
    const optionalImages = directEvidence ? "" : imageResults(result);
    const debugBlock = isAdmin ? `${guardDebug(result)}${toolTrace(result)}${algorithmEvidence(result)}` : "";
    const toolSamples = toolLoopEvidence(result);
    const toolEvidence = toolSamples.length ? `<section class="evidence-layer"><div class="section-head"><div><p class="section-kicker">本次依据（工具结果）</p><h3>${toolSamples.length} 项</h3></div></div><div class="evidence-list">${toolSamples.map(evidenceCard).join("")}</div></section>` : "";
    const evidenceCount = grounding.evidence_count != null ? grounding.evidence_count
      : (primary.length + (result.image_results || []).length + toolSamples.length);
    const hasToolEvidence = toolSamples.length > 0;
    const hasResultSet = Boolean((result.task_state || {}).current_result_set && (result.task_state || {}).result_total > 0);
    // RX-6: a chat turn (memory_used === false) never shows an evidence entry; tool-loop turns use task_state evidence.
    const requiresEvidence = (result.memory_used !== false && presentation.required !== false) || hasToolEvidence || hasResultSet;
    const hasGap = result.evidence_status === "gap" || (!evidenceCount && result.tool_loop_status === "complete");
    const resultSetBlock = displayMode === "collapsed" ? resultSetCard(result) : "";
    const basisOpen = displayMode === "result_grid" || hasGap || (displayMode !== "collapsed" && evidenceCount > 0);
    const basis = requiresEvidence ? `<details class="assistant-basis"${basisOpen ? " open" : ""}><summary>原始证据${evidenceCount ? ` · ${evidenceCount} 项` : ""}</summary><div class="assistant-basis-body">${resultSetBlock}${claimEvidence(result)}${optionalImages}${toolEvidence}${evidence}${gapContent}${order}${debugBlock}</div></details>` : "";
    if (displayMode === "none") return `${followups}${gapContent}`;
    return `${followups}${proactiveRecall(result)}${directOriginal}${basis}`;
  }

  function evidenceStatusLabel(result) {
    const mode = result.response_mode || "";
    if (mode === "asset_delivery" || result.original_evidence_requested) return "已找到并展示";
    if (result.evidence_status === "not_applicable") return "";
    if (result.evidence_status === "gap") return "暂时没有足够依据";
    if (result.evidence_status === "clarify") return "需要补充一点线索";
    if (result.evidence_status === "anchored") return "已找到相关记忆";
    return "";
  }

  function resultSetCard(result) {
    const ts = result.task_state || {};
    const rid = ts.current_result_set;
    if (!rid) return "";
    // Phase C C7：total 未知显示"找到一批相关结果"；remaining=0 / has_more=false 不显示下一页。
    const totalKnown = ts.result_total != null;
    const total = totalKnown ? ts.result_total : null;
    const remaining = ts.result_remaining != null ? ts.result_remaining : 0;
    const hasMore = Boolean(ts.has_more) && remaining > 0;
    const head = totalKnown ? `共 ${total} 张${hasMore ? ` · 还有 ${remaining} 张` : ""}` : "找到一批相关结果";
    const handles = (ts.result_preview || []).slice(0, 6);
    const selected = state.selectedAsset && state.selectedAsset.result_set_id === rid ? state.selectedAsset.handle : "";
    // C8：本轮的 inspect_photo 复核结果与 handle 对应展示（已复核徽标 + 复核观察）
    const inspectRows = (ts.tool_results || []).filter((tr) => tr.tool === "inspect_photo" && tr.inspect_handle);
    const inspected = new Set(inspectRows.map((tr) => tr.inspect_handle));
    const inspectedNotes = inspectRows.filter((tr) => tr.inspect_text)
      .map((tr) => `<span>${escapeHtml(tr.inspect_handle)} · 复核：${escapeHtml(tr.inspect_text)}</span>`).join("");
    const thumbs = handles.length ? `<div class="result-set-thumbs">${handles.map((h) => {
      const active = h === selected ? " selected" : "";
      const checked = inspected.has(h) ? " inspected" : "";
      return `<div class="result-set-thumb-wrap${active}${checked}"><button class="result-set-thumb" data-action="open-photo-inspector" data-result-set-id="${escapeHtml(rid)}" data-handle="${escapeHtml(h)}" title="打开照片检查器"><img src="${escapeHtml(window.sentrixApi.resultSetPhoto(rid, h, state.scopeId))}" alt="${escapeHtml(h)}" loading="lazy" />${inspected.has(h) ? `<span class="result-set-check inspected">已复核</span>` : ""}</button><button class="result-set-select" data-action="select-result-photo" data-result-set-id="${escapeHtml(rid)}" data-handle="${escapeHtml(h)}" title="在主对话中选中这张">${h === selected ? "✓" : "＋"}</button></div>`;
    }).join("")}</div>` : "";
    const inspectBlock = inspectedNotes ? `<div class="result-set-inspect-notes">${inspectedNotes}</div>` : "";
    const originalButton = selected ? `<button class="text-button" data-action="open-selected-original" data-result-set-id="${escapeHtml(rid)}" data-handle="${escapeHtml(selected)}">查看原图 ${icon("→")}</button>` : "";
    const next = hasMore ? `<button class="text-button" data-action="result-next-page">还有 ${remaining} 张 · 看下一页 ${icon("→")}</button>` : "";
    return `<section class="result-set-card"><div class="result-set-head"><span class="section-kicker">结果集</span><strong>${escapeHtml(head)}</strong></div>${thumbs}${inspectBlock}${originalButton}${next}</section>`;
  }

  function assistantMessage(message) {
    if (message.role === "user") return `<article class="assistant-message user"><div class="assistant-bubble"><p>${escapeHtml(message.text)}</p></div></article>`;
    const result = message.result || {};
    const status = evidenceStatusLabel(result);
    const plan = result.dialogue_plan || {};
    const agentPlan = result.agent_plan || {};
    const mode = plan.mode === "contextual_follow_up" ? "沿用上一段记忆" : plan.style === "narrative" ? "回忆叙事" : plan.style === "clarifying" ? "等待补充线索" : "事实回答";
    const failureStatus = ["partial", "timeout", "error", "blocked_by_guard"].includes(result.tool_loop_status || "");
    const traceSteps = buildThinkingSteps(result);
    const trace = traceSteps.length ? `<details class="agent-trace-box"${failureStatus ? " open" : ""}><summary>思考过程 · ${traceSteps.length} 步</summary><div>${traceSteps.map(agentStepHtml).join("")}</div></details>` : "";
    const grounding = result.answerGrounding || result.answer_grounding || {};
    const gridVisible = ["result_grid", "inline_images"].includes(grounding.display_mode);
    return `<article class="assistant-message steward"><div class="assistant-ident"><span class="assistant-mark">S</span><span>家庭助手</span>${status ? `<small>${escapeHtml(status)}</small>` : ""}</div><div class="assistant-bubble"><p>${assistantAnswer(result) || "我在。"}</p>${trace}${gridVisible ? resultSetCard(result) : ""}${assistantEvidence(result)}</div></article>`;
  }

  function updateLiveProgress() {
    const host = document.querySelector("[data-live-progress]");
    if (!host) return;
    host.innerHTML = buildThinkingSteps(null, state.liveProgress).map(agentStepHtml).join("");
  }

  function conversationRail() {
    if (!capability("conversation_management")) return "";
    const convs = state.conversations || [];
    const items = convs.length ? convs.map((conv) => {
      const active = conv.conversation_id === state.conversationId;
      const when = conv.last_message_at || conv.updated_at || "";
      const whenLabel = when ? String(when).slice(5, 16).replace("T", " ") : "";
      return `<div class="conversation-item${active ? " active" : ""}"><button class="conversation-open" data-action="open-conversation" data-conversation-id="${escapeHtml(conv.conversation_id)}"><span class="conversation-title">${escapeHtml(conv.title || "新对话")}</span><small>${escapeHtml(whenLabel)}</small></button><button class="conversation-delete" data-action="delete-conversation" data-conversation-id="${escapeHtml(conv.conversation_id)}" aria-label="删除对话" title="删除对话">✕</button></div>`;
    }).join("") : `<div class="conversation-empty">还没有历史对话</div>`;
    return `<aside class="conversation-rail"><div class="conversation-rail-head"><strong>对话</strong><button class="text-button" data-action="new-conversation">${icon("＋")}新对话</button></div><div class="conversation-list">${items}</div></aside>`;
  }

  function searchView() {
    const messages = state.assistantMessages;
    const introduction = `<section class="assistant-intro"><div><span class="assistant-mark">S</span><p class="section-kicker">FAMILY COMPANION</p><h2>家庭助手</h2><p>我记得这座家庭相册中整理出的成员、共同经历与生活细节。我们可以自然聊聊；谈到家里的往事时，我会在需要时调取记忆，并保留可查看的依据。</p></div><div class="assistant-scope"><span>当前相册</span><strong>${escapeHtml(albumLabel(state.scopeId))}</strong></div></section>`;
    const suggestions = `<div class="assistant-suggestions"><button data-query="介绍一下明哥">介绍一位家人</button><button data-query="明哥的时间线">查看人物时间线</button><button data-query="推荐一些明哥的回忆">推荐有依据的回忆</button></div>`;
    const summary = state.activeConversationSummary ? `<details class="conversation-summary"><summary>本会话摘要</summary><p>${escapeHtml(state.activeConversationSummary).replace(/\n/g, "<br />")}</p></details>` : "";
    const rail = conversationRail();
    const inner = `${introduction}${summary}<section class="assistant-conversation">${messages.length ? messages.map(assistantMessage).join("") : `<div class="assistant-welcome"><p>今天想聊什么？</p>${suggestions}</div>`}${state.searchLoading ? `<article class="assistant-message steward loading"><div class="assistant-ident"><span class="assistant-mark">S</span><span>家庭助手</span></div><div class="assistant-bubble"><p>我在想，正在整理这段记忆。</p><div class="agent-trace live" data-live-progress>${buildThinkingSteps(null, state.liveProgress).map(agentStepHtml).join("")}</div></div></article>` : ""}</section>${searchBar("和家庭助手聊聊，或问起家里的任何一段经历…")}`;
    return `${pageHeader("家庭对话", "家庭助手", "一个中性的本地数字人，带着这座家庭相册形成的长期记忆。")}${rail ? `<div class="assistant-layout">${rail}<div class="assistant-main">${inner}</div></div>` : inner}`;
  }

  function timelineView() {
    const events = filteredEvents();
    const card = (event) => `<article class="timeline-event ${event.isVideoScene ? "video-scene-event" : ""}"><div class="timeline-marker ${event.tone}"></div><div class="timeline-date">${event.date}<small>${escapeHtml(event.typeLabel)}</small></div><div class="timeline-event-body">${event.isVideoScene ? videoSceneStack(event) : `<div class="event-cover ${event.tone}">${event.coverAssetId ? `<img src="/api/assets/${encodeURIComponent(event.coverAssetId)}/file" alt="${escapeHtml(event.title)}的事件证据" loading="lazy" />` : ""}<div class="event-cover-label">${albumBadge(event.scope_id)}<span>${escapeHtml(event.title)}</span></div><b>${escapeHtml(event.countLabel)}</b></div>`}<div class="timeline-event-copy"><div class="card-top"><span class="event-kind">${escapeHtml(event.isVideoScene ? `视频场景 ${(event.source_scene_index || 0) + 1}` : event.status || "active")}</span><span class="confidence-label">${event.isVideoScene ? escapeHtml(event.sceneRange) : `revision ${event.revision || 1}`}</span></div><h2>${escapeHtml(event.title)}</h2><p>${escapeHtml(event.summary || "暂无事件摘要")}</p><div class="event-facts"><span>${icon("◎")} ${escapeHtml(event.placeLabel)}</span><span>${icon("◷")} ${event.isVideoScene ? `${escapeHtml(event.sceneRange)} · ${escapeHtml(event.countLabel)}` : escapeHtml(event.countLabel)}</span><span>${icon("↗")} ${event.isVideoScene ? "可回到原始视频" : "可回到原始证据"}</span></div><div class="event-actions"><button class="button small ghost" data-action="open-event" data-event-id="${escapeHtml(event.id)}">查看证据</button>${event.isVideoScene ? "" : `<button class="text-button" data-action="edit-event" data-event-id="${escapeHtml(event.id)}">修正事件 ${icon("→")}</button>`}</div></div></div></article>`;
    const groups = [];
    const byDate = new Map();
    events.forEach((event) => {
      const date = String(event.time_start || "").slice(0, 10) || "unknown";
      if (!byDate.has(date)) { const group = { date, events: [] }; byDate.set(date, group); groups.push(group); }
      byDate.get(date).events.push(event);
    });
    const groupedTimeline = groups.map((group) => `<details class="timeline-day" open><summary><span>${escapeHtml(group.date === "unknown" ? "未标注日期" : formatDate(group.date))}</span><small>${group.events.length} 个事件</small><b>⌄</b></summary><div class="timeline-day-events">${group.events.map(card).join("")}</div></details>`).join("");
    return `${pageHeader("记忆组织 / 事件", "家里的时间线，不只是文件列表。", `当前范围：${albumLabel(state.scopeId)}。照片和视频片段按真实拍摄时间统一整理。`, `<button class="button ghost" data-action="create-event">${icon("＋")}新建事件</button>`)}<div class="filter-row"><button class="filter-chip ${state.eventFilter === "all" ? "active" : ""}" data-event-filter="all">全部事件</button><button class="filter-chip ${state.eventFilter === "people" ? "active" : ""}" data-event-filter="people">有人物</button><button class="filter-chip ${state.eventFilter === "place" ? "active" : ""}" data-event-filter="place">有地点</button><label class="timeline-date-filter"><span>查看日期</span><input type="date" data-event-date value="${escapeHtml(state.eventDate)}" aria-label="按日期查看时间线" /></label>${state.eventDate ? `<button class="text-button" data-action="clear-event-date">清除日期</button>` : ""}<span class="filter-spacer"></span><button class="icon-button bordered" data-action="reload" aria-label="刷新时间线">↻</button></div><section class="timeline-layout"><div class="timeline-main">${events.length ? groupedTimeline : emptyState(state.eventDate ? "这一天还没有事件" : "还没有事件", state.eventDate ? "请选择其他日期，或清除日期筛选查看全部记忆。" : "导入并处理资料后，记忆会自动出现在时间线。", state.eventDate ? `<button class="button small ghost" data-action="clear-event-date">查看全部事件</button>` : `<button class="button small primary" data-view="imports">${icon("＋")}导入资料</button>`)}</div><aside class="side-inspector"><p class="section-kicker">视频记忆</p><h2>一段视频，也能成为清晰的家庭回忆</h2><p>系统会选出有代表性的画面，按场景整理，并继承拍摄时间与地点。</p><div class="inspector-note">场景图片堆 <strong>已启用</strong><small>展开后可点击画面，直接跳回原视频对应时刻。</small></div></aside></section>`;
  }

  function peopleView() {
    const people = state.persons.filter((person) => state.personFilter === "all" || !person.confirmed);
    const pending = state.persons.filter((person) => !person.confirmed);
    return `${pageHeader("家庭治理 / 人物", "先确认人物，再让关系长出来。", "人脸模型只生成候选。单张样本会明确标注，仍可查看原图后确认或驳回。", `<button class="button primary" data-action="invite">${icon("＋")}生成邀请</button>`)}<div class="people-toolbar"><div class="segmented"><button class="${state.personFilter === "all" ? "active" : ""}" data-person-filter="all">全部人物</button><button class="${state.personFilter === "pending" ? "active" : ""}" data-person-filter="pending">待确认 <b>${pending.length}</b></button><button data-action="relationship-graph">关系图</button></div><button class="button ghost" data-action="reload">${icon("↻")}刷新</button></div><section class="people-grid">${people.length ? people.map((person, index) => { const name = person.confirmed ? (person.display_name || person.name) : `待命名成员 ${index + 1}`; const caution = !person.confirmed && person.single_sample ? `<small>单张样本，需谨慎确认</small>` : ""; return `<article class="person-card ${person.confirmed ? "" : "needs-review"}"><div class="person-head">${faceAvatar(person.avatar_face_instance_id, name, person.confirmed ? "green" : "gray")}${person.confirmed ? `<span class="confirmed">✓ 已确认</span>` : `<span class="needs-label">待确认</span>`}</div><h2>${escapeHtml(name)}</h2><p>${escapeHtml(person.status)} · 置信度 ${Math.round((person.confidence || 0) * 100)}%</p>${caution}<div class="person-stats">${person.confirmed ? `<span><strong>${person.mention_count || 0}</strong> 次出现</span><span><strong>✓</strong> 已确认</span>` : `<span><strong>${person.cluster_count || 0}</strong> 个人物簇</span><span>待确认</span>`}</div><div class="person-actions"><button class="button small ghost" data-action="open-person" data-person-id="${escapeHtml(person.id)}">查看证据</button>${person.confirmed ? "" : `<button class="button small primary" data-action="confirm-person" data-person-id="${escapeHtml(person.id)}">确认</button><button class="button small ghost" data-action="delete-person" data-person-id="${escapeHtml(person.id)}">不是人物</button>`}</div></article>`; }).join("") : emptyState("还没有人物候选", "导入包含人脸的图片后，InsightFace 会生成待确认候选；不会凭空创建家庭成员。", `<button class="button small primary" data-view="imports">${icon("＋")}导入图片</button>`)}</section>`;
  }

  function knowledgeView() {
    const pendingClusters = state.clusters.filter((cluster) => cluster.status === "pending" && cluster.reviewable !== false);
    const visibleEntities = state.entities.filter((entity) => entity.status === "confirmed" || entity.reviewable !== false);
    const confirmed = visibleEntities.filter((entity) => entity.status === "confirmed");
    const pending = visibleEntities.filter((entity) => entity.status === "pending");
    const entityCard = (entity) => `<button class="entity-card" data-action="open-entity" data-entity-id="${escapeHtml(entity.id)}"><div class="entity-card-head">${faceAvatar(entity.avatar_face_instance_id, entity.canonical_name, entity.status === "confirmed" ? "green" : "gray")}<span class="needs-label">${escapeHtml(entity.status)}</span></div><h2>${escapeHtml(entity.canonical_name)}</h2><p>${escapeHtml(entity.family_role || "未确认家庭角色")}</p><div class="entity-stats"><span><strong>${entity.mention_count || 0}</strong> 次出现</span><span><strong>${entity.cluster_count || 0}</strong> 个人物簇</span><span><strong>${entity.relationship_count || 0}</strong> 条关系</span></div></button>`;
    const clusterCard = (cluster) => `<article class="cluster-card"><div class="cluster-head"><div><span class="section-kicker">人物识别</span><strong>候选人物簇</strong></div><span class="needs-label">${cluster.member_count || cluster.samples?.length || 0} 张样本</span></div><div class="cluster-samples">${(cluster.samples || []).filter((s) => (s.quality || 0) >= 0.5).slice(0, 6).map((sample) => `<div class="cluster-sample"><button data-action="open-asset" data-asset-id="${escapeHtml(sample.asset_id)}" title="质量 ${Math.round((sample.quality || 0) * 100)}%">${faceAvatar(sample.id, "人脸样本")}</button><button class="sample-split" data-action="split-face" data-cluster-id="${escapeHtml(cluster.id)}" data-face-instance-id="${escapeHtml(sample.id)}" aria-label="从人物簇拆出样本">×</button></div>`).join("")}</div><p>聚类置信度 ${Math.round((cluster.confidence || 0) * 100)}% · ${cluster.entity_status === "confirmed" ? "已绑定人物" : "待确认身份"} · 显示 ${(cluster.samples || []).filter((s) => (s.quality || 0) >= 0.5).length} 张高质量样本</p><div class="person-actions"><button class="button small primary" data-action="confirm-cluster" data-cluster-id="${escapeHtml(cluster.id)}">确认实体</button><button class="button small ghost" data-action="merge-cluster" data-cluster-id="${escapeHtml(cluster.id)}">合并到其他簇</button><button class="button small ghost" data-action="delete-cluster" data-cluster-id="${escapeHtml(cluster.id)}">不是人物</button></div></article>`;
    return `${pageHeader("语义记忆 / 实体治理", "看见家庭记忆里稳定存在的实体。", "人物簇先由 buffalo_l 聚类，再由你确认名称和角色；确认结果会回写观察、事件、人物画像和关系图。", `<button class="button ghost" data-action="reload">${icon("↻")}刷新状态</button>`)}<section class="knowledge-summary"><article><strong>${confirmed.length}</strong><span>已确认实体</span></article><article><strong>${pending.length + pendingClusters.length}</strong><span>待维护候选</span></article><article><strong>${state.relationships.filter((item) => item.status === "active").length}</strong><span>已确认关系</span></article><article><strong>${state.relationships.filter((item) => item.status === "pending").length}</strong><span>待确认关系</span></article></section><section class="content-section"><div class="section-head"><div><p class="section-kicker">实体总览</p><h2>跨事件维护的家庭实体</h2></div><button class="text-button" data-action="relationship-graph">查看关系图 ${icon("→")}</button></div><div class="entity-grid">${visibleEntities.length ? visibleEntities.map(entityCard).join("") : emptyState("还没有实体", "完成一轮图片导入和人脸聚类后，实体会出现在这里。")}</div></section>${(() => { const multi = pendingClusters.filter((c) => (c.member_count || 0) >= 2); const single = pendingClusters.filter((c) => (c.member_count || 0) < 2); return `<section class="content-section"><div class="section-head"><div><p class="section-kicker">人物聚类 / 待确认</p><h2>先确认身份，再进入长期记忆</h2></div><span class="result-count">${multi.length} 簇</span></div><div class="cluster-grid">${multi.length ? multi.map(clusterCard).join("") : emptyState("没有待确认人物簇", "相似人脸会被合并为簇；单张样本折叠在下方。")}</div>${single.length ? `<details class="single-clusters" style="margin-top:14px;border:1px solid var(--line);border-radius:10px;padding:12px;"><summary style="cursor:pointer;color:var(--muted);font-size:12px;">单张样本（${single.length}）· 可能是小脸或误检，展开后谨慎确认</summary><div class="cluster-grid" style="margin-top:12px;">${single.map(clusterCard).join("")}</div></details>` : ""}</section>`; })()}`;
  }

  function libraryView() {
    const assets = filteredAssets();
    const albumCounts = state.assets.reduce((result, asset) => { const key = asset.source_album_id || asset.scope_id || "unknown"; result[key] = (result[key] || 0) + 1; return result; }, {});
    const albumSummary = Object.entries(albumCounts).map(([id, count]) => `<span>${albumBadge(id)} ${count} 张</span>`).join("");
    return `${pageHeader("家庭资料 / 全部内容", "所有资料，都有回到记忆的路径。", `当前范围：${albumLabel(state.scopeId)}。这里展示真实导入的原始 Asset。点击任意条目可查看文件、处理状态、Observation 和模型原始输出。`, `<button class="button primary" data-view="imports">${icon("↓")}导入资料</button>`)}<div class="album-summary-row"><strong>相册内容</strong>${albumSummary || `<span>${albumBadge(state.scopeId)} 暂无资料</span>`}</div><div class="library-summary"><div><strong>${state.assets.length}</strong><span>当前范围资产</span></div><div><strong>${state.assets.filter((a) => a.media_type === "image").length}</strong><span>图片</span></div><div><strong>${state.assets.filter((a) => ["audio", "text"].includes(a.media_type)).length}</strong><span>音频 / 文本</span></div><div class="muted-stat"><strong>${state.assets.filter((a) => a.media_type === "video").length}</strong><span>视频与场景</span></div></div><div class="filter-row">${[["all", "全部"], ["image", "图片"], ["audio", "音频"], ["text", "文本"], ["video", "视频"]].map(([key, label]) => `<button class="filter-chip ${state.assetFilter === key ? "active" : ""}" data-asset-filter="${key}">${label}</button>`).join("")}<span class="filter-spacer"></span><button class="sort-label" data-action="toggle-sort">按${state.assetSort === "newest" ? "最近" : "最早"}导入 ↕</button></div><section class="asset-grid library-grid">${assets.length ? assets.map(assetCard).join("") : emptyState("没有匹配的资料", "调整筛选条件或导入一份新的原始文件。", `<button class="button small primary" data-view="imports">${icon("＋")}导入资料</button>`)}</section>`;
  }

  function storiesView() {
    const _evtTimes = ((state.stories[0]||{}).event_ids||[]).map(id => (state.events||[]).find(x=>x.id===id)).filter(Boolean).filter(e=>e?.time_start).map(e=>e.time_start).sort();
    let _spanText = "";
    if (_evtTimes.length >= 2) { const _d = (new Date(_evtTimes[_evtTimes.length-1]) - new Date(_evtTimes[0])) / 86400000; if (_d > 180) _spanText = "跨越 " + (_d/365).toFixed(1) + " 年"; else if (_d > 30) _spanText = "跨越 " + Math.ceil(_d/30) + " 个月"; else _spanText = "跨越 " + Math.ceil(_d) + " 天"; }
    const _places = {}; const _people = {};
    ((state.stories[0]||{}).event_ids||[]).forEach(id => { const ev = (state.events||[]).find(x=>x.id===id); if (!ev) return; if (ev?.place && ev.place !== "其他或不确定") _places[ev.place] = (_places[ev.place]||0)+1; (ev.participants||[]).forEach(p => { const eid = (typeof p === 'object' && p) ? (p.entity_id||null) : null; const ent = eid ? (state.entities||[]).find(e=>e.id===eid) : null; if (ent?.is_self) return; const nm = typeof p === 'string' ? p : (p?.canonical_name || p?.name || "未知"); _people[nm] = (_people[nm]||0)+1; }); });
    if (state.storyGenerating) {
      const _ids = state.storyDraftEventIds || [];
      const _evs = _ids.map(id => (state.events||[]).find(x=>x.id===id)).filter(Boolean);
      const _days = new Set();
      _evs.forEach(e => { if (e?.time_start) _days.add(String(e.time_start).slice(0,10)); if (e?.time_end) _days.add(String(e.time_end).slice(0,10)); });
      const _photos = _evs.reduce((s,e)=>s+((e?.observations||[]).length||((e?.asset_ids||[]).length)||0),0);
      const _shimmer = (w) => `<div style="height:14px;width:${w};border-radius:6px;background:linear-gradient(90deg,#ecece6,#f6f6f0,#ecece6);background-size:200% 100%;animation:storyShimmer 1.2s infinite;margin-top:8px;"></div>`;
      const _hint = state.storyError
        ? "故事生成失败，请稍后重试。"
        : `AI 正在根据 ${_ids.length} 个事件撰写故事，约需 10-30 秒…`;
      return `${pageHeader("家庭表达 / 故事工作室", "把真实事件整理成家人愿意一起看的故事。", "故事只引用你选择的事件和证据；标题、章节和内容保存为本地草稿。", `<button class="button primary" data-action="create-story">${icon("＋")}新建故事</button>`)}<style>@keyframes storyShimmer{to{background-position:-200% 0}}</style><section class="story-layout"><div class="story-canvas"><div class="story-canvas-label">${state.storyError ? "生成失败" : "正在生成故事"}</div><div class="story-meta" style="font-size:12px;color:#9A9486;margin:4px 0 12px;letter-spacing:0.5px;">${_days.size} 天 · ${_ids.length} 个事件 · ${_photos} 张照片</div><div style="margin:20px 0 8px;"><div style="height:22px;width:68%;border-radius:6px;background:linear-gradient(90deg,#ecece6,#f6f6f0,#ecece6);background-size:200% 100%;animation:storyShimmer 1.2s infinite;"></div>${_shimmer("94%")}${_shimmer("87%")}${_shimmer("91%")}${_shimmer("60%")}</div><div style="margin-top:18px;font-size:13px;color:#9A9486;">${_hint}${state.storyError ? ` <button class="button primary" data-action="retry-story" style="margin-left:10px;padding:6px 14px;">重试</button>` : ""}</div></div><aside class="story-editor"><div class="panel-title"><span>STORY DRAFTS</span><span class="draft-badge">${state.stories.length} 个</span></div>${state.stories.map((story) => `<div class="chapter"><button class="chapter-open" data-action="edit-story" data-story-id="${escapeHtml(story.id)}"><span>●</span><strong>${escapeHtml(story.title)}</strong>${icon("→", "muted")}</button><button class="icon-button bordered" data-action="delete-story" data-story-id="${escapeHtml(story.id)}" aria-label="删除故事">×</button></div>`).join("")}<button class="button primary full" data-action="create-story">新建本地草稿 ${icon("→")}</button></aside></section>`;
    }
    return `${pageHeader("家庭表达 / 故事工作室", "把真实事件整理成家人愿意一起看的故事。", "故事只引用你选择的事件和证据；标题、章节和内容保存为本地草稿。", `<button class="button primary" data-action="create-story">${icon("＋")}新建故事</button>`)}<section class="story-layout">${state.stories.length ? `<div class="story-report"><div class="report-topbar"><div class="report-title">${escapeHtml(state.stories[0].title || "未命名故事")}</div><div class="report-actions"><button class="button ghost" data-action="edit-story" data-story-id="${escapeHtml(state.stories[0].id)}">编辑</button><button class="button primary" data-action="create-story">${icon("＋")}生成</button></div></div><div class="report-meta">${(()=>{const es=(state.stories[0].event_ids||[]).map(id=>(state.events||[]).find(x=>x.id===id)).filter(Boolean);const ds=new Set();es.forEach(e=>{if(e?.time_start)ds.add(String(e.time_start).slice(0,10));if(e?.time_end)ds.add(String(e.time_end).slice(0,10));});return ds.size;})()} 天 · ${(state.stories[0].event_ids||[]).length} 个事件 · ${(state.stories[0].event_ids||[]).reduce((s,eid)=>{const ev=(state.events||[]).find(x=>x.id===eid);return s+((ev?.observations||[]).length||((ev?.asset_ids||[]).length)||0);},0)} 张照片${_spanText ? " · " + _spanText : ""}</div><div class="report-capsules">${Object.entries(_places).sort((a,b)=>b[1]-a[1]).slice(0,5).map(([k,v])=>`<span class="place-capsule">${escapeHtml(k)} ${v}</span>`).join("")}</div><div class="report-body"><div class="report-narrative">${escapeHtml(state.stories[0].content || "这个故事还没有内容。")}</div><div class="report-stats"><div class="stat-card"><div class="stat-num">${Object.keys(_places).length}</div><div class="stat-label">场景数</div></div><div class="stat-card"><div class="stat-num">${(state.stories[0].event_ids||[]).reduce((s,eid)=>{const ev=(state.events||[]).find(x=>x.id===eid);return s+((ev?.observations||[]).length||((ev?.asset_ids||[]).length)||0);},0)}</div><div class="stat-label">照片数</div></div><div class="stat-card"><div class="stat-num">${(state.stories[0].event_ids||[]).length}</div><div class="stat-label">事件数</div></div><div class="stat-card"><div class="stat-num">${(()=>{const es=(state.stories[0].event_ids||[]).map(id=>(state.events||[]).find(x=>x.id===id)).filter(Boolean);const ds=new Set();es.forEach(e=>{if(e?.time_start)ds.add(String(e.time_start).slice(0,10));if(e?.time_end)ds.add(String(e.time_end).slice(0,10));});return ds.size;})()}</div><div class="stat-label">天数</div></div></div></div>${Object.keys(_places).length || Object.keys(_people).length ? `<div class="report-bottom">${Object.keys(_places).length ? `<div class="report-places"><div class="report-sec-title">最常去的地方</div>${(()=>{const ps=Object.entries(_places).sort((a,b)=>b[1]-a[1]).slice(0,5);const mx=Math.max(1,...ps.map(x=>x[1]));return ps.map(([k,v])=>`<div class="place-bar-row"><span class="place-bar-name">${escapeHtml(k)}</span><div class="place-bar-bg"><div class="place-bar" style="width:${Math.round(v/mx*100)}%"></div></div><span class="place-bar-count">${v}</span></div>`).join("");})()}</div>` : ''}${Object.keys(_people).length ? `<div class="report-people"><div class="report-sec-title">陪伴你的人</div>${Object.entries(_people).sort((a,b)=>b[1]-a[1]).slice(0,6).map(([k,v])=>`<div class="person-card"><span class="person-avatar">${escapeHtml((k||"?")[0])}</span><div><div class="person-name">${escapeHtml(k)}</div><div class="person-count">${v} 次</div></div></div>`).join("")}</div>` : ''}</div>` : ''}</div><aside class="story-editor"><div class="panel-title"><span>STORY DRAFTS</span><span class="draft-badge">${state.stories.length} 个</span></div>${state.stories.map((story) => `<div class="chapter"><button class="chapter-open" data-action="edit-story" data-story-id="${escapeHtml(story.id)}"><span>●</span><strong>${escapeHtml(story.title)}</strong>${icon("→", "muted")}</button><button class="icon-button bordered" data-action="delete-story" data-story-id="${escapeHtml(story.id)}" aria-label="删除故事">×</button></div>`).join("")}<button class="button primary full" data-action="create-story">新建本地草稿 ${icon("→")}</button></aside></section>` : `<section class="empty-search"><div class="empty-symbol">▤</div><h2>还没有故事草稿</h2><p>先导入并形成事件，再选择真实事件生成故事草稿。</p><button class="button primary" data-action="create-story">${icon("＋")}创建空白故事</button></section>`}`;
  }

  function renderUploadQueue() {
    if (!state.queue.length) return "";
    const failedStatuses = new Set(["metadata-failed", "upload-failed", "failed", "rejected"]);
    const uploaded = state.queue.filter((item) => !["reading-metadata", "ready", ...failedStatuses].includes(item.status)).length;
    const failed = state.queue.filter((item) => failedStatuses.has(item.status)).length;
    const active = state.queue.length - uploaded - failed;
    const progress = Math.round(((uploaded + failed) / state.queue.length) * 100);
    const statusLabel = (status) => ({
      "reading-metadata": "读取元数据",
      ready: "等待上传",
      "metadata-failed": "元数据失败",
      "upload-failed": "上传失败",
      queued: "已进入队列",
      processing: "处理中",
      processed: "已完成",
      failed: "处理失败",
      rejected: "已拒绝",
    }[status] || status || "等待处理");
    const rows = state.queue.slice(0, 20).map((item) => `<div class="queue-row"><span class="queue-type image">图</span><div><strong>${escapeHtml(item.fileName)}</strong><small>${escapeHtml(item.error || statusLabel(item.status))}</small></div><span class="queue-status ${failedStatuses.has(item.status) ? "reserved" : "queued"}">${escapeHtml(statusLabel(item.status))}</span></div>`).join("");
    return `<section class="content-section upload-progress"><div class="section-head"><div><p class="section-kicker">本次上传</p><h2>${state.queue.length} 份资料</h2></div><span class="result-count">${uploaded} 已上传 · ${active} 进行中 · ${failed} 失败</span></div><div class="health-bar"><i style="width:${progress}%"></i></div><div class="queue-list">${rows}</div>${state.queue.length > 20 ? `<p class="upload-more">仅展示最近20条，完整状态会同步到下方处理任务。</p>` : ""}</section>`;
  }

  function importsView() {
    const videoStatuses = ["video-queued", "video-metadata", "video-keyframe-extracting", "video-scene-importing", "video-processing-failed"];
    const assets = state.assets.filter((asset) => ["queued", "processing", "semantic_enriching", "failed", ...videoStatuses].includes(asset.status));
    const videos = state.assets.filter((asset) => asset.media_type === "video");
    const activeVideo = videos.filter((asset) => videoStatuses.includes(asset.status) && asset.status !== "video-processing-failed").length;
    const pipelineRows = [
      ["接收与去重", `${state.assets.length} 个 Asset`, "done"],
      ["读取视频元数据", `${videos.length} 个视频`, activeVideo ? "active" : "done"],
      ["关键帧与场景切分", `${videos.reduce((sum, item) => sum + Number(item.metadata_json?.worldmm_scene_count || 0), 0)} 个场景`, activeVideo ? "active" : "done"],
      ["场景图片语义理解", `${state.assets.filter((a) => a.derived_kind === "video_keyframe" && a.status === "processed").length} 张关键帧`, activeVideo ? "active" : "done"],
      ["事件记忆构建", `${stats().events} 个事件 · ${stats().facts} 条事实`, "done"],
    ];
    return `${pageHeader("资料入口 / 本地导入", "把资料带回家，剩下的交给本地 AI。", "视频会在后台读取拍摄信息、提取关键画面并整理进家庭时间线。", `<button class="button ghost" data-action="open-folder">${icon("▦")}选择图片文件夹</button>`)}<section class="import-layout"><div><label class="dropzone" for="file-input"><input id="file-input" type="file" multiple accept="image/*,.heic,.heif,image/heic,image/heif,audio/*,text/*,video/*" /><input id="folder-input" type="file" webkitdirectory directory multiple accept=".jpg,.jpeg,.png,.heic,.heif,image/jpeg,image/png,image/heic,image/heif" /><span class="drop-icon">↓</span><strong>拖入照片或视频，或点击选择文件</strong><small>视频支持 MOV / MP4，并在后台构建场景记忆</small><span class="button primary">选择资料</span></label><div class="import-notice"><span class="notice-mark">i</span><div><strong>原始资料不会被覆盖</strong><p>每张关键画面都能跳回原视频的准确时刻。</p></div></div></div><aside class="import-status"><div class="panel-title"><span>本地处理</span><span class="live-label"><i></i>真实状态</span></div><h2>当前处理</h2>${pipelineRows.map((row) => `<div class="pipeline-row"><span class="pipeline-state ${row[2]}">${row[2] === "done" ? "✓" : "•"}</span><div><strong>${row[0]}</strong><small>${row[1]}</small></div><em>${row[2] === "done" ? "完成" : "运行中"}</em></div>`).join("")}</aside></section>${renderUploadQueue()}<section class="content-section"><div class="section-head"><div><p class="section-kicker">导入记录</p><h2>最近处理任务</h2></div><button class="text-button" data-action="reload">刷新状态 ${icon("↻")}</button></div><div class="queue-list">${assets.length ? assets.map((asset) => `<div class="queue-row"><span class="queue-type ${asset.media_type}">${escapeHtml(mediaLabel(asset.media_type).slice(0, 3))}</span><div><strong>原始${escapeHtml(mediaLabel(asset.media_type))}资料</strong><small>${formatDateTime(asset.updated_at)} · ${escapeHtml(assetStatusLabel(asset.status))}</small></div><span class="queue-status ${asset.status.includes("failed") ? "reserved" : "queued"}">${escapeHtml(assetStatusLabel(asset.status))}</span></div>`).join("") : emptyState("没有待处理任务", "处理中的资料会显示在这里。")}</div></section>`;
  }

  function ocrSettingsCard() {
    const ocr = state.ocrSettings || {};
    const available = ocr.small_ocr_available === true;
    const enabled = ocr.small_ocr_enabled === true;
    const statusLabel = available ? "可用" : "当前不可用";
    return `<section class="content-section ocr-settings-card"><div class="section-head"><div><p class="section-kicker">图片文字识别</p><h2>菜单、价格、电话、招牌等文字</h2></div><span class="result-count">OCR 小模型 · ${statusLabel}</span></div><div class="ocr-setting-row"><div><strong>优先使用 OCR 小模型</strong><small>开启后读取菜单、价格、电话号码、招牌等文字优先使用轻量 OCR（CPU、更快、零显存）；必要时再用多模态模型补充核对。关闭则直接使用当前多模态模型。</small></div><label class="switch"><input type="checkbox" data-action="toggle-ocr-small" ${enabled ? "checked" : ""} ${available ? "" : "disabled"} /><span></span></label></div>${available ? "" : `<p class="ocr-unavailable-note">OCR 小模型未安装，当前使用多模态模型读取文字。</p>`}</section>`;
  }

  function settingsView() {
    const facts = state.dashboard?.facts || [];
    const pending = facts.filter((fact) => fact.status === "pending");
    const router = state.vlmBackendOptions || {};
    const activeModel = router.models?.[router.backend];
    const routerReady = router.status === "running" && Boolean(router.backend);
    const routerStatus = modelSwitchInFlight ? "SWITCHING" : routerReady ? "RUNNING" : "UNMANAGED";
    const routerStatusClass = routerReady || modelSwitchInFlight ? "" : "warn";
    const unmanagedOption = router.backend ? "" : `<option value="" selected disabled>未托管模型</option>`;
    const face = state.health.models?.face || {};
    const faceStatusText = face.identityFallback
      ? `人物检测已启用 · AdaFace 未加载，身份向量已回退为 ${escapeHtml(face.identityFallbackModel || "InsightFace buffalo_l")}`
      : face.identityReady
        ? `人物识别已启用 · ${escapeHtml(face.identityModel || "InsightFace")}`
        : face.detectionReady
          ? "人物检测已启用 · AdaFace 未加载，身份向量暂不可用"
          : "人物识别不可用";
    return `${pageHeader("系统 / 本地状态", "你的记忆，运行在自己的家里。", "服务、模型、存储和事实修订状态都来自当前本地后端。")}${state.health ? `<section class="health-grid"><article class="health-card dark"><div class="health-title"><span>Sentrix Home</span><span class="online-pill"><i></i>在线</span></div><strong>本地服务正常</strong><p>健康接口返回正常</p><div class="health-line"><span>数据资产</span><b>${stats().assets}</b></div><div class="health-bar"><i style="width:100%"></i></div></article><article class="health-card"><div class="health-title"><span>AI MODEL ROUTER</span><span class="ready-label ${routerStatusClass}">${routerStatus}</span></div><label class="model-switcher"><span>主推理</span><select data-action="switch-vlm" ${modelSwitchInFlight ? 'disabled' : ''}>${unmanagedOption}${(router.available_backends || []).map(id => { const info = router.models?.[id] || {}; return `<option value="${escapeHtml(id)}" ${id === router.backend ? 'selected' : ''} ${!info.available ? 'disabled' : ''}>${escapeHtml(info.model || id)}${info.available ? '' : ' · 离线'}${info.loaded ? ' · 当前运行' : ''}</option>` }).join('')}</select><small>${escapeHtml(modelSwitchInFlight ? `正在切换到 ${requestedModelProfile}` : activeModel?.url || router.error || '未托管模型')}</small></label><div class="model-row"><span>语音转写</span><strong>FunASR</strong><small>${escapeHtml(state.health.models?.asr?.name || "未连接")}</small></div><div class="model-row"><span>人物识别</span><strong>InsightFace</strong><small>${faceStatusText}</small></div></article><article class="health-card"><div class="health-title"><span>MEMORY INDEX</span><span class="ready-label">LOCAL</span></div><strong>${stats().facts} <small>条事实</small></strong><p>SQLite 事实库 · 原生语义图与向量索引</p><div class="index-list"><span>${icon("●")}事件记忆 <b>${stats().events}</b></span><span>${icon("●")}观察证据 <b>${stats().observations}</b></span><span class="dim">${icon("—")}视频场景记忆 <b>WorldMM</b></span></div></article></section>` : emptyState("正在读取本地状态", "请稍候或刷新页面。")}${ocrSettingsCard()}${`<section class="content-section fact-review">`}<div class="section-head"><div><p class="section-kicker">语义记忆 / 版本维护</p><h2>需要确认的事实</h2></div><span class="result-count">${pending.length} 条</span></div>${pending.length ? `<div class="fact-review-list">${pending.map((fact) => `<div class="fact-review-row"><div><strong>${escapeHtml(fact.subject)} ${escapeHtml(fact.predicate)} ${escapeHtml(fact.object)}</strong><small>${escapeHtml(fact.id)} · 置信度 ${Math.round((fact.confidence || 0) * 100)}% · 证据 ${(fact.evidence_ids_json || []).join(", ")}</small></div><div class="review-actions"><button class="button small primary" data-action="confirm-fact" data-fact="${escapeHtml(fact.id)}">${icon("✓")}确认</button><button class="button small ghost" data-action="reject-fact" data-fact="${escapeHtml(fact.id)}">${icon("×")}驳回</button></div></div>`).join("")}</div>` : emptyState("没有待确认事实", "冲突事实出现后会进入这里，旧版本不会被删除。")}</section><section class="content-section two-column settings-lower"><div><div class="section-head"><div><p class="section-kicker">隐私边界</p><h2>数据只在本地流动</h2></div></div><div class="privacy-list"><div><span>原始媒体</span><b>本地存储</b></div><div><span>人物特征</span><b>本地处理</b></div><div><span>原生记忆索引</span><b>本地实体与向量检索</b></div><div><span>视频场景</span><b>本地 WorldMM 已启用</b></div></div></div><div><div class="section-head"><div><p class="section-kicker">审计入口</p><h2>可操作的系统动作</h2></div></div><div class="audit-list"><div><button class="button small ghost" data-action="reload">刷新服务状态 ${icon("↻")}</button><small>重新读取后端、模型和数据库状态</small></div><div><button class="button small ghost" data-action="recheck">重新检查失败任务 ${icon("→")}</button><small>只重试 queued 或 failed Asset</small></div><div><button class="button small ghost" data-action="open-help">查看接口与隐私说明 ${icon("?")}</button><small>当前部署边界和证据规则</small></div><div><button class="button small ghost" data-action="open-qa-dashboard">查看 QA 测评 Dashboard ${icon("▤")}</button><small>Agent 基准测评与历史 run 对比</small></div></div></div></section>`;
  }

  function semanticDetails(group) {
    const values = Array.isArray(group?.semantic_details) ? group.semantic_details : [];
    return [...new Set(values.map((value) => String(value || "").trim()).filter(Boolean))];
  }

  function semanticSummary(group) {
    const type = group?.entity_type === "place" ? "场景" : group?.entity_type === "object" ? "物品" : "画面氛围";
    return `这些原始图片中可以观察到与${group?.canonical_name || "该语义"}有关的${type}特征。`;
  }

  function semanticGroupPreview(group) {
    if (group?.preview_asset_id && group.preview_media_type === "image") {
      return `<img class="entity-thumb" src="/api/assets/${encodeURIComponent(group.preview_asset_id)}/file" alt="${escapeHtml(group.canonical_name || "语义实体")}的原始证据" loading="lazy" />`;
    }
    return `<div class="entity-thumb fallback" aria-hidden="true">${escapeHtml((group?.canonical_name || "?").slice(0, 1))}</div>`;
  }

  function semanticGroupCard(group) {
    const details = semanticDetails(group);
    return `<button class="entity-card entity-media-card semantic-group-card" data-action="open-entity-group" data-entity-group-id="${escapeHtml(group.id)}">${semanticGroupPreview(group)}<div class="entity-card-copy"><div class="entity-card-head"><span class="semantic-type-label">${group.members?.length > 1 ? "相近场景" : "语义场景"}</span><span class="semantic-evidence-label">原始证据</span></div><h2>${escapeHtml(group.canonical_name)}</h2><div class="semantic-detail-tags">${(details.length ? details : ["由图片观察维护"]).map((detail) => `<span>${escapeHtml(detail)}</span>`).join("")}</div><p>${escapeHtml(semanticSummary(group))}</p><span class="semantic-open-link">查看相关证据 ${icon("→")}</span></div></button>`;
  }


  function geoPlaceSection() {
    const places = state.geoPlaces || [];
    if (!places.length) return emptyState("没有地点数据", "导入带GPS的照片后，系统会自动按城市和区县整理。");

    const breadcrumb = state.geoBreadcrumb || [];
    const selectedCity = breadcrumb[0] || null;
    const selectedDistrict = breadcrumb[1] || null;

    const breadcrumbHtml = breadcrumb.length
      ? `<nav class="geo-breadcrumb"><button class="text-button" data-action="geo-breadcrumb" data-geo-level="root">${icon("&#127968;")} 地点总览</button>${breadcrumb.map((item, idx) => `<span class="geo-sep">></span><button class="text-button" data-action="geo-breadcrumb" data-geo-level="${idx}">${escapeHtml(item === "unknown" ? "无法判断地点" : item)}</button>`).join("")}</nav>`
      : "";

    if (!selectedCity) {
      // City level
      const cards = places.map((city) => {
        if (city.level === "unknown") {
          return `<button class="geo-card geo-card-unknown" data-action="geo-select-city" data-geo-city="unknown"><strong>${escapeHtml(city.name)}</strong><span>${city.count} 张照片</span></button>`;
        }
        const districtCount = (city.children || []).length;
        return `<button class="geo-card" data-action="geo-select-city" data-geo-city="${escapeHtml(city.name)}" data-geo-province="${escapeHtml(city.province || "")}"><strong>${escapeHtml(city.province || "")} ${escapeHtml(city.name)}</strong><span>${city.count} 张照片 · ${districtCount} 个区县</span></button>`;
      }).join("");
      return `<section class="content-section"><div class="section-head"><div><p class="section-kicker">GPS 地理位置</p><h2>按拍摄地点整理的回忆</h2></div><span class="result-count">${places.length} 处</span></div><div class="geo-grid">${cards || emptyState("没有地点数据", "导入带GPS的照片后，系统会按城市整理。")}</div></section>`;
    }

    // Handle "unknown" location directly — it has photos, not districts
    if (selectedCity === "unknown") {
      const unknown = places.find((c) => c.level === "unknown") || null;
      if (!unknown || !unknown.photos || !unknown.photos.length) {
        return `<section class="content-section">${breadcrumbHtml}<div class="geo-photo-wall">${emptyState("没有无法判断地点的照片", "")}</div></section>`;
      }
      const photoGrid = unknown.photos.map((p) => {
        return `<button class="geo-photo-card" data-action="open-asset" data-asset-id="${escapeHtml(p.asset_id)}">
          <img src="/api/assets/${encodeURIComponent(p.asset_id)}/file" alt="${escapeHtml(p.caption || p.file_name)}" loading="lazy" />
          <div class="geo-photo-overlay">
            <span class="geo-photo-caption">${escapeHtml(p.caption || p.observation_place || "无描述")}</span>
            ${p.semantic_place ? `<span class="geo-photo-semantic">${escapeHtml(p.semantic_place)}</span>` : ""}
          </div>
        </button>`;
      }).join("");
      return `<section class="content-section">${breadcrumbHtml}<div class="section-head"><div><p class="section-kicker">无法判断地点</p><h2>照片墙</h2></div><span class="result-count">${unknown.count} 张</span></div><div class="geo-photo-wall">${photoGrid}</div></section>`;
    }

    const city = places.find((c) => c.name === selectedCity && c.level !== "unknown") || null;

    if (!selectedDistrict) {
      // District level
      if (!city || !city.children || !city.children.length) {
        return `<section class="content-section">${breadcrumbHtml}<div class="section-head"><div><p class="section-kicker">${escapeHtml(selectedCity)}</p></div></div><div class="geo-photo-wall">${emptyState("该地区暂无照片", "")}</div></section>`;
      }
      const districtCards = city.children.map((d) => {
        const previews = (d.photos || []).slice(0, 4).map((p) => `<div class="geo-district-thumb"><img src="/api/assets/${encodeURIComponent(p.asset_id)}/file" alt="${escapeHtml(p.caption || p.file_name)}" loading="lazy" /></div>`).join("");
        return `<button class="geo-card geo-card-district" data-action="geo-select-district" data-geo-district="${escapeHtml(d.name)}">${previews ? `<div class="geo-district-previews">${previews}</div>` : ""}<strong>${escapeHtml(d.name)}</strong><span>${d.count} 张照片</span></button>`;
      }).join("");
      return `<section class="content-section">${breadcrumbHtml}<div class="section-head"><div><p class="section-kicker">${escapeHtml(selectedCity)}</p><h2>各区县分布</h2></div></div><div class="geo-grid">${districtCards}</div></section>`;
    }

    // Photo wall level
    const district = (city && city.children || []).find((d) => d.name === selectedDistrict) || null;
    if (!district || !district.photos || !district.photos.length) {
      return `<section class="content-section">${breadcrumbHtml}<div class="geo-photo-wall">${emptyState("该区暂无照片", "")}</div></section>`;
    }
    const photoGrid = district.photos.map((p) => {
      return `<button class="geo-photo-card" data-action="open-asset" data-asset-id="${escapeHtml(p.asset_id)}">
        <img src="/api/assets/${encodeURIComponent(p.asset_id)}/file" alt="${escapeHtml(p.caption || p.file_name)}" loading="lazy" />
        <div class="geo-photo-overlay">
          <span class="geo-photo-caption">${escapeHtml(p.caption || p.observation_place || "无描述")}</span>
          ${p.semantic_place ? `<span class="geo-photo-semantic">${escapeHtml(p.semantic_place)}</span>` : ""}
        </div>
      </button>`;
    }).join("");
    return `<section class="content-section">${breadcrumbHtml}<div class="section-head"><div><p class="section-kicker">${escapeHtml(selectedDistrict)}</p><h2>照片墙</h2></div><span class="result-count">${district.count} 张</span></div><div class="geo-photo-wall">${photoGrid || emptyState("暂无照片", "")}</div></section>`;
  }

  function semanticKnowledgeView() {
    const people = state.persons.filter((person) => person.confirmed);
    const claims = (state.knowledge.claims || people.flatMap((person) => person.claims || [])).filter((claim) => claim.status !== "superseded");
    const groups = [["object", "物品", "物品语义"], ["atmosphere", "氛围", "画面氛围"]];
    const personCards = people.map((person) => "<article class=\"entity-card\"><div class=\"entity-card-head\">" + faceAvatar(person.avatar_face_instance_id, person.display_name, "green") + "<span class=\"confirmed\">已确认</span></div><h2>" + escapeHtml(person.display_name) + "</h2><p>" + escapeHtml(person.profile?.summary_zh || person.summary || "正在从事件和证据形成画像。") + "</p><div class=\"entity-stats\"><span><strong>" + (person.event_memory || []).length + "</strong> 个事件</span><span><strong>" + claims.filter((claim) => claim.person_id === person.id).length + "</strong> 条当前声明</span></div><button class=\"button small ghost\" data-action=\"open-person-profile\" data-person-id=\"" + escapeHtml(person.id) + "\">查看画像和证据</button></article>").join("");
    const entitySection = ([type, label, description]) => {
      const entities = state.entityGroups.filter((entity) => entity.entity_type === type && entity.evidence_count > 0);
      const expanded = Boolean(state.expandedEntityTypes[type]);
      const visible = expanded ? entities : entities.slice(0, 6);
      return `<section class="content-section"><div class="section-head"><div><p class="section-kicker">${label}</p><h2>${description}</h2></div><span class="result-count">${entities.length} 项</span></div><div class="entity-grid entity-grid-collapsed">${entities.length ? visible.map(semanticGroupCard).join("") : emptyState(`尚未形成${label}实体`, "等待带有可回溯观察证据的资料。")}</div>${entities.length > 6 ? `<div class="entity-expand"><button class="button small ghost" data-action="toggle-entity-type" data-entity-type="${type}">${expanded ? "收起" : `查看全部 ${entities.length} 项`}</button></div>` : ""}</section>`;
    };
    const tripCards = state.trips.map((trip) => `<article class="trip-candidate"><span class="needs-label">${escapeHtml(trip.status)}</span><h3>${escapeHtml(trip.name)}</h3><p>${escapeHtml(formatDate(trip.time_start))} 至 ${escapeHtml(formatDate(trip.time_end))}</p><small>${(trip.place_names_json || []).map(escapeHtml).join("、") || "地点待补充"} · ${(trip.event_ids_json || []).length} 个事件 · ${(trip.evidence_ids_json || []).length} 条证据</small>${trip.status === "pending" ? `<div class="person-actions"><button class="button small primary" data-action="confirm-trip" data-trip-id="${escapeHtml(trip.id)}">命名并确认</button><button class="button small ghost" data-action="reject-trip" data-trip-id="${escapeHtml(trip.id)}">不是行程</button></div>` : `<small>类型 · ${escapeHtml(trip.trip_type || "未分类")} · revision ${trip.revision || 1}</small>`}</article>`).join("");
    return pageHeader("语义记忆 / 实体目录", "人物、地点与细节共同组成回忆。", "相近描述会自动归到同一语义实体组；成员实体、事件和照片始终保留为可追溯的组成部分。", "<button class=\"button ghost\" data-action=\"reload\">" + icon("↻") + "刷新知识</button>") + "<section class=\"knowledge-summary\"><article><strong>" + people.length + "</strong><span>已确认人物</span></article><article><strong>" + claims.length + "</strong><span>当前人物声明</span></article><article><strong>" + state.entityGroups.length + "</strong><span>语义实体组</span></article><article><strong>" + (state.dashboard?.pendingFacts || 0) + "</strong><span>待维护事实</span></article></section><section class=\"content-section\"><div class=\"section-head\"><div><p class=\"section-kicker\">人物总结</p><h2>跨事件形成的熟人档案</h2></div><span class=\"result-count\">" + people.length + " 人</span></div><div class=\"entity-grid\">" + (personCards || emptyState("还没有已确认人物", "先在人物页面确认人脸簇，语义知识才会有稳定的中心。", "<button class=\"button small primary\" data-view=\"people\">打开人物</button>")) + "</div></section><section class=\"content-section\"><div class=\"section-head\"><div><p class=\"section-kicker\">行程候选</p><h2>跨事件的长线回忆</h2></div><span class=\"result-count\">" + state.trips.length + " 项</span></div><div class=\"trip-grid\">" + (tripCards || emptyState("暂无行程候选", "只有跨日或跨地点的连续事件才会成为待确认行程。")) + "</div></section>" + [geoPlaceSection(), entitySection(["object", "物品", "物品语义"]), entitySection(["atmosphere", "氛围", "画面氛围"])].join("");
  }

  function renderView() {
    const root = document.getElementById("view-root");
    const views = { overview, search: searchView, timeline: timelineView, people: peopleView, knowledge: semanticKnowledgeView, library: libraryView, stories: storiesView, imports: importsView, settings: settingsView };
    root.innerHTML = state.loading ? emptyState("正在读取本地记忆", "正在加载 Asset、Observation、Event、Fact 和故事。") : views[state.view]();
    renderModal();
    bindViewEvents();
    if (state.toast) showToast(state.toast);
  }

  function renderModal() {
    const root = document.getElementById("modal-root");
    if (!root || !state.modal) { if (root) root.innerHTML = ""; return; }
    const modal = state.modal;
    if (modal.type === "loading") { root.innerHTML = `<div class="modal-backdrop"><div class="modal-panel"><button class="modal-close" data-action="close-modal">×</button><div class="empty-search"><div class="empty-symbol">◌</div><h2>正在读取证据</h2></div></div></div>`; return; }
    let body = "";
    if (modal.type === "event") {
      const detail = modal.detail;
      const eventEntities = detail.entities || [];
      const coverSelection = detail.event.cover_selection || {};
      const coverObservation = (detail.observations || []).find((item) => item.id === coverSelection.evidence_observation_id);
      const coverEvidence = detail.event.cover_asset_id && coverSelection.source ? `<details class="algorithm-evidence"><summary>封面选择依据</summary><div class="algorithm-evidence-body"><p>${coverSelection.source === "user" ? "该封面由用户指定，不会被自动选择覆盖。" : "系统只在该事件的原始图片中选择封面。"}</p><dl><div><dt>封面原图</dt><dd>${escapeHtml(coverObservation?.asset?.file_name || detail.event.cover_asset_id)}</dd></div><div><dt>候选图片</dt><dd>${escapeHtml(String(coverSelection.criteria?.candidate_count || 1))} 项</dd></div><div><dt>观察置信度</dt><dd>${Math.round((coverSelection.criteria?.observation_confidence || 0) * 100)}%</dd></div></dl></div></details>` : "";
      const entityRows = eventEntities.map((entity) => `<button class="event-entity-row" data-action="open-entity" data-entity-id="${escapeHtml(entity.id)}"><span class="needs-label">${escapeHtml(entity.relation || "关联")}</span><strong>${escapeHtml(entity.canonical_name)}</strong><small>${escapeHtml(entity.entity_type)} · ${entity.evidence_count || 0} 条证据 · 置信度 ${Math.round((entity.confidence || 0) * 100)}%</small></button>`).join("");
      const isVideoScene = detail.event.source_type === "video_scene";
      const sourceVideo = detail.event.source_video || {};
      const sceneFrames = detail.event.keyframe_assets || [];
      const sceneEvidence = isVideoScene ? `<section class="video-scene-detail"><div class="detail-facts"><span>视频范围 · ${formatVideoTime(detail.event.source_start_sec)}~${formatVideoTime(detail.event.source_end_sec)}</span><span>场景 ${(detail.event.source_scene_index || 0) + 1}</span><span>${sceneFrames.length} 张关键帧</span></div><video id="scene-video-player" controls preload="metadata" src="/api/assets/${encodeURIComponent(detail.event.source_asset_id)}/file"></video><div class="scene-keyframe-grid">${sceneFrames.map((frame) => `<button class="scene-keyframe" data-action="seek-video" data-timestamp-sec="${Number(frame.source_timestamp_sec || 0)}"><img src="/api/assets/${encodeURIComponent(frame.id)}/file" alt="${formatVideoTime(frame.source_timestamp_sec)} 的关键帧" /><strong>${formatVideoTime(frame.source_timestamp_sec)}</strong></button>`).join("")}</div><button class="text-button" data-action="open-asset" data-asset-id="${escapeHtml(detail.event.source_asset_id)}">打开原始视频 ${icon("→")}</button></section>` : "";
      body = `<div class="modal-kicker">${isVideoScene ? "视频场景" : "事件详情"}</div><h2>${escapeHtml(detail.event.title)}</h2><p class="modal-lead">${escapeHtml(detail.event.summary || "暂无摘要")}</p><div class="detail-facts"><span>时间 · ${formatDateTime(detail.event.time_start)}</span><span>地点 · ${escapeHtml(detail.event.place || "未标注")}</span><span>${isVideoScene ? `原视频 · ${escapeHtml(sourceVideo.file_name || "视频证据")}` : `版本 · ${detail.event.revision || 1}`}</span></div>${sceneEvidence}${coverEvidence}<div class="section-head"><div><p class="section-kicker">记忆内容</p><h3>人物、地点与画面细节</h3></div></div><div class="event-entity-list">${entityRows || emptyState("暂时没有更多细节", "处理新资料后，这里会继续补充。")}</div><div class="section-head"><div><p class="section-kicker">原始资料</p><h3>${detail.observations.length} 条画面记录</h3></div>${isVideoScene ? "" : `<button class="button small ghost" data-action="edit-event" data-event-id="${escapeHtml(detail.event.id)}">修正事件</button>`}</div><div class="evidence-list event-evidence-list">${detail.observations.length ? detail.observations.map((observation) => evidenceCard({ kind: "observation", id: observation.id, observation_id: observation.id, asset_id: observation.asset_id, file_name: observation.asset?.file_name, media_type: observation.asset?.media_type, captured_at: observation.captured_at, caption: observation.caption, transcript: observation.transcript, raw: observation.raw_json })).join("") : emptyState("没有关联画面", "这是一个人工创建的事件。")}</div>${detail.facts?.length ? `<div class="section-head"><div><p class="section-kicker">语义记忆</p><h3>关联事实</h3></div></div><div class="evidence-list">${detail.facts.map((fact) => evidenceCard({ kind: "fact", id: fact.id, subject: fact.subject, predicate: fact.predicate, object: fact.object, status: fact.status, evidence_ids: fact.evidence_ids_json })).join("")}</div>` : ""}`;
    } else if (modal.type === "asset") {
      const asset = modal.asset;
      body = `<div class="modal-kicker">ORIGINAL EVIDENCE</div><h2>原始${escapeHtml(mediaLabel(asset.media_type))}证据</h2><div class="asset-modal-preview">${asset.media_type === "image" ? `<img src="/api/assets/${encodeURIComponent(asset.id)}/file" alt="原始图片证据" />` : asset.media_type === "audio" ? `<audio controls src="/api/assets/${encodeURIComponent(asset.id)}/file"></audio>` : asset.media_type === "video" ? `<video controls src="/api/assets/${encodeURIComponent(asset.id)}/file"></video>` : `<pre>${escapeHtml(asset.file_name)}</pre>`}</div><div class="detail-facts"><span>类型 · ${escapeHtml(mediaLabel(asset.media_type))}</span><span>状态 · ${escapeHtml(asset.status)}</span><span>大小 · ${asset.size_bytes || 0} bytes</span></div><details class="technical-evidence"><summary>查看资料技术信息</summary><div><span>文件名</span><small>${escapeHtml(asset.file_name)}</small></div><div><span>Asset</span><small>${escapeHtml(asset.id)}</small></div></details><div class="section-head"><div><p class="section-kicker">Observation</p><h3>这份资料产生的证据</h3></div></div><div class="evidence-list">${modal.observations.length ? modal.observations.map((observation) => evidenceCard({ kind: "observation", id: observation.id, asset_id: asset.id, file_name: asset.file_name, media_type: asset.media_type, captured_at: observation.captured_at, caption: observation.caption, transcript: observation.transcript, raw: observation.raw_json })).join("") : emptyState("还没有 Observation", "资料正在处理，刷新后查看结果。")}</div>`;
    } else if (modal.type === "event-edit" || modal.type === "event-create") {
      const event = modal.event || {};
      const coverOptions = (modal.observations || []).map((observation, index) => `<option value="${escapeHtml(observation.asset_id)}" ${observation.asset_id === event.cover_asset_id ? "selected" : ""}>原始证据 ${index + 1}</option>`).join("");
      body = `<form id="modal-form"><div class="modal-kicker">${modal.type === "event-create" ? "CREATE EVENT" : "EDIT EVENT"}</div><h2>${modal.type === "event-create" ? "新建人工事件" : "修正事件"}</h2><label>标题<input name="title" value="${escapeHtml(event.title || "")}" required /></label><label>事件类型<input name="event_type" value="${escapeHtml(event.event_type || "")}" placeholder="例如：旅行、生日、日常" /></label><label>摘要<textarea name="summary">${escapeHtml(event.summary || "")}</textarea></label><label>地点<input name="place" value="${escapeHtml(event.place || "")}" /></label><label>开始时间<input name="time_start" type="datetime-local" value="${event.time_start ? event.time_start.slice(0, 16) : ""}" /></label><label>结束时间<input name="time_end" type="datetime-local" value="${event.time_end ? event.time_end.slice(0, 16) : ""}" /></label>${coverOptions ? `<label>封面证据<select name="cover_asset_id">${coverOptions}</select></label>` : ""}<div class="modal-actions"><button type="button" class="button ghost" data-action="close-modal">取消</button><button type="submit" class="button primary">保存事件</button></div></form>`;
    } else if (modal.type === "trip-confirm") {
      const trip = modal.trip;
      body = `<form id="modal-form"><div class="modal-kicker">TRIP CONFIRMATION · ${escapeHtml(trip.id)}</div><h2>确认这段行程</h2><p class="modal-lead">确认后会保留现有事件、照片证据与候选 ID，并把行程写入稳定语义记忆。</p><label>行程名称<input name="name" value="" placeholder="例如：国庆深圳之行" required /></label><label>行程类型<select name="trip_type"><option value="">暂不分类</option><option>旅行</option><option>探亲</option><option>团建</option><option>日常出行</option></select></label><div class="detail-facts"><span>${escapeHtml(formatDate(trip.time_start))} 至 ${escapeHtml(formatDate(trip.time_end))}</span><span>${(trip.event_ids_json || []).length} 个事件</span><span>${(trip.evidence_ids_json || []).length} 条证据</span></div><div class="modal-actions"><button type="button" class="button ghost" data-action="close-modal">取消</button><button type="submit" class="button primary">确认行程</button></div></form>`;
    } else if (modal.type === "entity-merge-confirm") {
      const candidate = modal.candidate;
      const choices = state.entities.filter((entity) => (candidate.entity_ids || []).includes(entity.id));
      body = `<form id="modal-form"><div class="modal-kicker">SEMANTIC ENTITY MERGE</div><h2>确认相近实体归并</h2><p class="modal-lead">选择一个保留的稳定实体。其他描述会退出活动列表，原始观察、事件关联和修订记录会保留并迁移到该实体。</p><label>保留的实体<select name="target_entity_id" required>${choices.map((entity) => `<option value="${escapeHtml(entity.id)}">${escapeHtml(entity.canonical_name)} · ${entity.evidence_count || 0} 条证据</option>`).join("")}</select></label><div class="detail-facts"><span>建议名称 · ${escapeHtml(candidate.suggested_name)}</span><span>原始标签 · ${escapeHtml((candidate.rationale?.source_labels || []).join("、"))}</span><span>证据 · ${(candidate.evidence_ids || []).length} 条</span></div><div class="modal-actions"><button type="button" class="button ghost" data-action="close-modal">取消</button><button type="submit" class="button primary">确认归并</button></div></form>`;
    } else if (modal.type === "person") {
      const person = modal.person;
      body = `<form id="modal-form"><div class="modal-kicker">IDENTITY CONFIRMATION · ${escapeHtml(person.id)}</div><h2>确认人物身份</h2><p class="modal-lead">只在这里填写姓名和家庭角色。确认后会把该人物回写到相关事件、人物画像和语义记忆。</p><label>家庭成员名称<input name="name" value="" placeholder="确认时填写名称" required /></label><label>家庭角色<select name="family_role"><option value="">暂不确认</option><option>母亲</option><option>父亲</option><option>孩子</option><option>祖父母</option><option>其他家庭成员</option></select></label><div class="modal-actions"><button type="button" class="button ghost" data-action="close-modal">取消</button><button type="submit" class="button primary">确认并更新记忆</button></div></form>`;
    } else if (modal.type === "person-evidence") {
      const detail = modal.detail;
      const entity = detail.entity;
      const samples = detail.face_samples || [];
      const events = detail.events || [];
      body = `<div class="modal-kicker">PERSON EVIDENCE</div><div class="profile-heading">${faceAvatar(samples[0]?.id || entity.avatar_face_instance_id, entity.canonical_name, entity.status === "confirmed" ? "green" : "gray")}<div><h2>${escapeHtml(entity.canonical_name)}</h2><p class="modal-lead">${escapeHtml(entity.status === "confirmed" ? "已确认人物，下面是可回看的原始证据。" : "待确认人物候选，先检查人脸样本再命名。")}</p></div></div><div class="detail-facts"><span>状态 · ${escapeHtml(entity.status)}</span><span>人脸样本 · ${samples.length}</span><span>关联事件 · ${events.length}</span></div><div class="section-head"><div><p class="section-kicker">人脸样本</p><h3>用于判断身份的头像证据</h3></div></div><div class="face-evidence-grid">${samples.length ? samples.map((sample) => `<article class="face-evidence-item"><img src="${escapeHtml(sample.crop_url)}" alt="人脸样本" loading="lazy" /><button class="text-button" data-action="open-asset" data-asset-id="${escapeHtml(sample.asset_id)}">查看原图 ${icon("→")}</button></article>`).join("") : emptyState("没有可用人脸样本", "当前候选没有可回看的 face instance。")}</div><div class="section-head"><div><p class="section-kicker">关联事件</p><h3>该人物出现过的事件</h3></div></div><div class="evidence-list">${events.length ? events.map((event) => `<button class="evidence-main" data-action="open-event" data-event-id="${escapeHtml(event.id)}"><strong>${escapeHtml(event.title)}</strong><p>${escapeHtml(event.summary || "暂无事件摘要")}</p><small>${escapeHtml(formatDateTime(event.time_start))} · ${escapeHtml(event.place || "未标注地点")}</small></button>`).join("") : emptyState("还没有关联事件", "确认后，新的事件观察会继续维护人物画像。")}</div>${entity.status === "confirmed" ? `<div class="modal-actions"><button class="button ghost" data-action="close-modal">关闭</button><button class="button primary" data-action="open-person-profile" data-person-id="${escapeHtml(entity.id)}">查看画像与改名</button></div>` : `<div class="modal-actions"><button class="button ghost" data-action="close-modal">关闭</button><button class="button primary" data-action="confirm-person" data-person-id="${escapeHtml(entity.id)}">确认姓名和关系</button></div>`}`;
    } else if (modal.type === "cluster-confirm") {
      const cluster = modal.cluster;
      body = `<form id="modal-form"><div class="modal-kicker">人物候选</div><h2>确认这个人物实体</h2><p class="modal-lead">这组样本由本机人脸模型自动聚类。确认后，所有样本会统一绑定到同一个实体。</p><div class="cluster-samples modal-samples">${(cluster.samples || []).map((sample) => `<button type="button" data-action="open-asset" data-asset-id="${escapeHtml(sample.asset_id)}">${faceAvatar(sample.id, "人脸样本")}</button>`).join("")}</div><label>姓名或称呼<input name="name" placeholder="例如：妈妈" required /></label><label>家庭角色<select name="family_role"><option value="">暂不确认</option><option>母亲</option><option>父亲</option><option>孩子</option><option>祖父母</option><option>其他家庭成员</option></select></label><label>与已确认实体的关系（可选）<select name="relation_target"><option value="">暂不建立关系</option>${state.entities.filter((entity) => entity.status === "confirmed").map((entity) => `<option value="${escapeHtml(entity.id)}">${escapeHtml(entity.canonical_name)}</option>`).join("")}</select></label><label>关系类型（可选）<input name="relation_predicate" placeholder="例如：母亲、父亲、兄弟姐妹" /></label><div class="modal-actions"><button type="button" class="button ghost" data-action="close-modal">取消</button><button type="submit" class="button primary">确认并更新记忆</button></div></form>`;
    } else if (modal.type === "cluster-merge") {
      const choices = state.clusters.filter((cluster) => cluster.status === "pending" && cluster.id !== modal.cluster.id);
      body = `<form id="modal-form"><div class="modal-kicker">FACE CLUSTER MERGE</div><h2>选择目标人物簇</h2><p class="modal-lead">合并会保留原始样本，并写入人物实体 revision。两个已确认人物不会被合并。</p><label>目标簇<select name="target_cluster_id" required>${choices.map((cluster) => `<option value="${escapeHtml(cluster.id)}">${escapeHtml(cluster.id)} · ${cluster.member_count || 0} 张样本</option>`).join("")}</select></label><div class="modal-actions"><button type="button" class="button ghost" data-action="close-modal">取消</button><button type="submit" class="button primary" ${choices.length ? "" : "disabled"}>合并人物簇</button></div></form>`;
    } else if (modal.type === "cluster-split") {
      const sample = modal.sample;
      body = `<form id="modal-form"><div class="modal-kicker">FACE CLUSTER SPLIT</div><h2>确认拆出这张人脸样本</h2><p class="modal-lead">拆分会创建同一记忆空间中的新候选簇，原始图片、人脸证据和审计记录都会保留。</p><div class="split-review"><img src="${escapeHtml(`/api/face-instances/${sample.id}/crop`)}" alt="待拆分人脸样本" /><div><strong>原始人脸证据</strong></div></div><div class="modal-actions"><button type="button" class="button ghost" data-action="close-modal">取消</button><button type="submit" class="button primary">确认拆分</button></div></form>`;
    } else if (modal.type === "person-profile") {
      const detail = modal.detail;
      const entity = detail.entity;
      const claims = detail.claims || [];
      const properties = detail.properties || [];
      const evidenceById = new Map((detail.evidence_files || []).map((item) => [item.evidence_id, item]));
      const claimEvidence = (claim) => (claim.evidence_ids_json || []).map((id) => evidenceById.get(id)).filter(Boolean);
      const claimRows = claims.map((claim) => {
        const evidence = claimEvidence(claim);
        const evidenceLinks = evidence.length ? evidence.map((item) => `<button class="text-button" data-action="open-asset" data-asset-id="${escapeHtml(item.asset_id)}">打开原始证据 ${icon("→")}</button>`).join("") : "<small>暂无可直接打开的原始证据</small>";
        return `<div class="fact-review-row"><div><strong>${escapeHtml(claim.predicate)} · ${escapeHtml(claim.value_text)}</strong><small>${escapeHtml(claim.dimension)} · ${escapeHtml(claim.status)} · 置信度 ${Math.round((claim.confidence || 0) * 100)}%</small><div class="claim-evidence-links">${evidenceLinks}</div></div></div>`;
      }).join("");
      const identityRows = properties.filter((item) => ["is_self", "relation_to_user", "groups"].includes(item.property_key)).map((item) => {
        const value = typeof item.value === "boolean" ? (item.value ? "是" : "否") : Array.isArray(item.value) ? item.value.join("、") : String(item.value ?? "未设置");
        return `<div class="property-row"><strong>${escapeHtml(item.property_key)} · ${escapeHtml(value)}</strong><small>${escapeHtml(item.source)} · v${item.revision}</small></div>`;
      }).join("");
      const aliases = Array.isArray(entity.aliases) ? entity.aliases : [];
      const aliasLine = aliases.length ? `<div class="person-aliases"><strong>其他称呼</strong><span>${aliases.map(escapeHtml).join("、")}</span></div>` : "";
      body = "<div class=\"modal-kicker\">PERSON PROFILE · " + escapeHtml(entity.id) + "</div><div class=\"profile-heading\">" + faceAvatar(entity.avatar_face_instance_id, entity.canonical_name, "green") + "<div><h2>" + escapeHtml(entity.canonical_name) + "</h2><p class=\"modal-lead\">" + escapeHtml(detail.profile?.summary_zh || entity.summary || "暂无人物画像") + "</p>" + aliasLine + "</div></div><div class=\"detail-facts\"><span>家庭角色 · " + escapeHtml(entity.family_role || "未确认") + "</span><span>语义声明 · " + claims.length + "</span><span>人物簇 · " + detail.clusters.length + "</span></div><div class=\"section-head\"><div><p class=\"section-kicker\">用户维护档案</p><h3>身份、关系与圈子</h3></div><button class=\"button small ghost\" data-action=\"edit-person-name\">编辑名字</button><button class=\"button small ghost\" data-action=\"edit-person-properties\">修正档案</button></div><div class=\"property-list\">" + (identityRows || emptyState("尚未维护身份属性", "这些字段只由你维护，模型不会覆盖。")) + "</div><div class=\"fact-review-list\">" + (claimRows || emptyState("暂无语义声明", "确认人物后，相关事件会持续维护人物画像。")) + "</div>";
    } else if (modal.type === "person-property-edit") {
      const detail = modal.detail;
      const properties = new Map((detail.properties || []).map((item) => [item.property_key, item]));
      const isSelf = Boolean(properties.get("is_self")?.value);
      const relation = properties.get("relation_to_user")?.value || "";
      const canonicalName = detail.entity?.canonical_name || "";
      const groups = Array.isArray(properties.get("groups")?.value) ? properties.get("groups").value.join("、") : "";
      body = `<form id="modal-form"><div class="modal-kicker">PERSON PROPERTY EDIT</div><h2>修正人物档案</h2><p class="modal-lead">这些是用户维护字段，会保留版本且不会被模型推断覆盖。</p><label>名字<input name="canonical_name" value="${escapeHtml(canonicalName)}" placeholder="例如：小张、妈妈" /></label><label class="property-toggle"><input type="checkbox" name="is_self" ${isSelf ? "checked" : ""} />这是相册主人</label><label>与相册主人的关系<input name="relation_to_user" value="${escapeHtml(relation)}" placeholder="例如：本人、母亲、同事" /></label><label>所属圈子<input name="groups" value="${escapeHtml(groups)}" placeholder="例如：家人、大学同学" /></label><div class="modal-actions"><button type="button" class="button ghost" data-action="open-person-profile" data-person-id="${escapeHtml(detail.entity.id)}">取消</button><button type="submit" class="button primary">保存人物档案</button></div></form>`;
    } else if (modal.type === "person-name-edit") {
      const detail = modal.detail;
      const entity = detail.entity;
      const aliases = Array.isArray(entity.aliases) ? entity.aliases : [];
      body = `<form id="modal-form"><div class="modal-kicker">PERSON NAME</div><h2>编辑人物名字</h2><p class="modal-lead">主名是家人通常的称呼；别名用于检索和确认时自动归类为同一人。改名后所有历史记忆会同步使用新名字。</p><label>主名<input name="name" value="${escapeHtml(entity.canonical_name)}" placeholder="例如：明哥" required /></label><label>其他称呼（别名，用顿号或逗号分隔）<input name="aliases" value="${escapeHtml(aliases.join("、"))}" placeholder="例如：小明、阿明" /></label><div class="modal-actions"><button type="button" class="button ghost" data-action="open-person-profile" data-person-id="${escapeHtml(entity.id)}">取消</button><button type="submit" class="button primary">保存名字</button></div></form>`;
    } else if (modal.type === "entity") {
      const detail = modal.detail;
      const entity = detail.entity;
    } else if (modal.type === "photo-inspector") {
      const pi = state.photoInspector || {};
      const photoUrl = pi.result_set_id ? window.sentrixApi.resultSetPhoto(pi.result_set_id, pi.asset_handle, state.scopeId) : (pi.asset_id ? `/api/assets/${encodeURIComponent(pi.asset_id)}/file` : "");
      const originalUrl = pi.result_set_id ? window.sentrixApi.resultSetPhoto(pi.result_set_id, pi.asset_handle, state.scopeId, true) : (pi.asset_id ? `/api/assets/${encodeURIComponent(pi.asset_id)}/file?original=1` : "");
      const msgRows = (pi.messages || []).map((msg) => `<div class="photo-thread-msg ${msg.role}"><strong>${msg.role === "user" ? "你" : "助手"}</strong><p>${escapeHtml(msg.text || "")}</p></div>`).join("");
      const quick = ["这是什么地方？", "有几个人？", "穿什么颜色？", "桌上是什么？", "文字写了什么？"].map((q) => `<button class="photo-thread-quick" data-action="photo-inspector-quick" data-query="${escapeHtml(q)}">${escapeHtml(q)}</button>`).join("");
      body = `<div class="modal-kicker">PHOTO INSPECTOR</div><h2>照片检查</h2><div class="photo-inspector-layout"><div class="photo-inspector-media"><img src="${escapeHtml(photoUrl)}" alt="当前检查的照片" /><div class="detail-facts"><span>handle · ${escapeHtml(pi.asset_handle || "photo_1")}</span>${pi.result_set_id ? `<span>结果集 · ${escapeHtml(pi.result_set_id)}</span>` : ""}</div><div class="photo-inspector-actions"><a class="button small ghost" href="${escapeHtml(originalUrl)}" target="_blank" rel="noopener">查看原图 ${icon("→")}</a></div></div><div class="photo-inspector-chat"><div class="photo-thread-messages">${msgRows || `<div class="photo-thread-empty">关于这张照片，问我任何细节。</div>`}${pi.loading ? `<div class="photo-thread-msg assistant loading"><strong>助手</strong><p>正在检查这张照片…</p></div>` : ""}</div><div class="photo-thread-quick-row">${quick}</div><form id="photo-thread-form" class="photo-thread-form"><input id="photo-thread-input" value="${escapeHtml(pi.draft || "")}" placeholder="关于这张照片问我…" /><button type="submit">→</button></form></div></div>`;
      const observations = detail.observations || [];
      const properties = detail.properties || [];
      const propertyHistory = detail.property_history || properties;
      const evidenceById = new Map((detail.evidence_files || []).map((item) => [item.evidence_id, item]));
      const preview = observations[0]?.asset?.id && observations[0]?.asset?.media_type === "image" ? `<img class="entity-detail-thumb" src="/api/assets/${encodeURIComponent(observations[0].asset.id)}/file" alt="${escapeHtml(observations[0].asset.file_name || entity.canonical_name)}的证据缩略图" />` : "";
      const relationRows = detail.relationships.concat(detail.facts.map((fact) => ({ ...fact, subject_name: fact.subject, object_name: fact.object, predicate: fact.predicate, status: fact.status }))).map((item) => `<div class="fact-review-row"><div><strong>${escapeHtml(item.subject_name || item.subject)} ${escapeHtml(item.predicate)} ${escapeHtml(item.object_name || item.object)}</strong><small>${escapeHtml(item.status)}</small></div></div>`).join("");
      const evidenceRows = observations.map((observation) => `<button class="evidence-main entity-observation" data-action="open-asset" data-asset-id="${escapeHtml(observation.asset_id)}"><strong>原始证据</strong><p>${escapeHtml(observation.caption || observation.transcript || "原始观察证据")}</p><small>${escapeHtml(formatDateTime(observation.captured_at))}</small></button>`).join("");
      const propertyRows = properties.map((property) => {
        const evidenceLinks = (property.evidence_ids || []).map((id) => evidenceById.get(id)).filter(Boolean).map((item) => `<button class="text-button" data-action="open-asset" data-asset-id="${escapeHtml(item.asset_id)}">打开原始证据 ${icon("→")}</button>`).join("");
        const value = typeof property.value === "boolean" ? (property.value ? "已开启" : "未开启") : Array.isArray(property.value) ? property.value.join("、") : String(property.value ?? "未设置");
        return `<div class="property-row"><strong>${escapeHtml(property.property_key)} · ${escapeHtml(value)}</strong><small>${escapeHtml(property.source)} · ${escapeHtml(property.status)} · 置信度 ${Math.round((property.confidence || 0) * 100)}% · v${property.revision}</small>${evidenceLinks ? `<div class="claim-evidence-links">${evidenceLinks}</div>` : ""}</div>`;
      }).join("");
      const placeControls = entity.entity_type === "place" ? `<button class="button small ghost" data-action="edit-entity-properties">修正地点属性</button>` : "";
      body = `<div class="modal-kicker">ENTITY · ${escapeHtml(entity.entity_type)}</div><div class="profile-heading">${preview}<div><h2>${escapeHtml(entity.canonical_name)}</h2><p class="modal-lead">${escapeHtml(entity.summary || "这是跨多个事件维护的实体，不是单张图片的描述。")}</p></div></div><div class="detail-facts"><span>类型 · ${escapeHtml(entity.entity_type)}</span><span>原始证据 · ${observations.length}</span><span>属性版本 · ${propertyHistory.length}</span><span>置信度 · ${Math.round((entity.confidence || 0) * 100)}%</span></div><div class="section-head"><div><p class="section-kicker">当前属性</p><h3>来源、置信度与证据</h3></div>${placeControls}</div><div class="property-list">${propertyRows || emptyState("暂无维护属性", "模型或用户修正后，会在这里显示当前版本和可回溯证据。")}</div><div class="section-head"><div><p class="section-kicker">关联事件与关系</p><h3>${detail.relationships.length} 条关系 · ${detail.facts.length} 条事实</h3></div></div><div class="evidence-list">${relationRows || emptyState("暂无关系和事实", "新资料会继续补充实体关系。")}</div><div class="section-head"><div><p class="section-kicker">原始证据</p><h3>可打开的文件与观察描述</h3></div></div><div class="evidence-list">${evidenceRows || emptyState("暂无原始证据", "该实体尚未关联可查看的 Observation。")}</div>`;
    } else if (modal.type === "entity-property-edit") {
      const detail = modal.detail;
      const properties = new Map((detail.properties || []).map((property) => [property.property_key, property]));
      const alias = properties.get("alias")?.value || "";
      const privateFlag = Boolean(properties.get("private_flag")?.value);
      const evidenceOptions = (detail.observations || []).map((observation, index) => `<label class="property-evidence-option"><input type="checkbox" name="evidence_ids" value="${escapeHtml(observation.id)}" />原始证据 ${index + 1}</label>`).join("");
      body = `<form id="modal-form"><div class="modal-kicker">PLACE PROPERTY EDIT</div><h2>修正地点属性</h2><p class="modal-lead">保存后以你的值为准；后续模型推断只会形成待审核版本，不能覆盖此设置。</p><label>自定义别名<input name="alias" value="${escapeHtml(alias)}" placeholder="例如：我们的老地方" /></label><label class="property-toggle"><input type="checkbox" name="private_flag" ${privateFlag ? "checked" : ""} />在普通页面和问答中隐藏精确地点</label><div class="property-evidence"><strong>支撑本次修正的原始证据（可选）</strong>${evidenceOptions || "<small>当前没有可选 Observation。</small>"}</div><div class="modal-actions"><button type="button" class="button ghost" data-action="open-entity" data-entity-id="${escapeHtml(detail.entity.id)}">取消</button><button type="submit" class="button primary">保存地点属性</button></div></form>`;
    } else if (modal.type === "entity-group") {
      const detail = modal.detail;
      const group = detail.group || {};
      const details = semanticDetails(group);
      const evidenceRows = (detail.observations || []).slice(0, 12).map((observation) => `<button class="semantic-evidence-tile" data-action="open-asset" data-asset-id="${escapeHtml(observation.asset_id)}"><img src="/api/assets/${encodeURIComponent(observation.asset_id)}/file" alt="${escapeHtml(group.canonical_name || "语义实体")}的原始证据" loading="lazy" /><span>${escapeHtml(observation.caption || observation.transcript || "原始观察证据")}</span></button>`).join("");
      const technicalRows = (detail.observations || []).slice(0, 12).map((observation) => `<div><span>${escapeHtml(observation.asset?.file_name || observation.asset_id)}</span><small>${escapeHtml(observation.id)} · ${escapeHtml(formatDateTime(observation.captured_at))}</small></div>`).join("");
      body = `<div class="modal-kicker">SEMANTIC ENTITY GROUP</div><div class="profile-heading"><div><h2>${escapeHtml(group.canonical_name || "语义实体组")}</h2><p class="modal-lead">${escapeHtml(semanticSummary(group))}</p></div></div><section class="semantic-detail-section"><p class="section-kicker">语义摘要</p><p>${escapeHtml(semanticSummary(group))}</p></section><section class="semantic-detail-section"><p class="section-kicker">细节语义</p><div class="semantic-detail-tags">${(details.length ? details : ["由图片观察维护"]).map((item) => `<span>${escapeHtml(item)}</span>`).join("")}</div></section><section class="semantic-detail-section"><p class="section-kicker">原始证据</p><div class="semantic-evidence-grid">${evidenceRows || emptyState("暂无原始证据", "该组尚未关联可查看的 Observation。")}</div></section><details class="technical-evidence"><summary>查看证据技术信息</summary>${technicalRows || "<p>暂无技术信息</p>"}</details><div class="modal-actions"><button class="button primary" data-action="close-modal">关闭</button></div>`;
    } else if (modal.type === "story-create" || modal.type === "story-edit") {
      const story = modal.story || {};
      body = `<form id="modal-form"><div class="modal-kicker">STORY ${modal.type === "story-create" ? "DRAFT" : "EDITOR"}</div><h2>${modal.type === "story-create" ? "创建故事草稿" : "编辑故事"}</h2><label>标题<input name="title" value="${escapeHtml(story.title || "")}" placeholder="留空则AI根据所选事件生成" /></label><label>故事内容<textarea name="content" rows="5">${escapeHtml(story.content || "")}</textarea></label><label>自定义标签<input name="tags" value="${escapeHtml((story.tags||[]).join(", "))}" placeholder="温馨, 搞笑, 感动（逗号分隔）" /></label><div class="story-event-select"><strong>选择事件证据</strong>${state.events.map((event) => `<label><input type="checkbox" name="event_ids" value="${escapeHtml(event.id)}" ${(story.event_ids || []).includes(event.id) ? "checked" : ""} />${escapeHtml(event.title)}</label>`).join("") || `<small>当前没有事件，请先导入资料。</small>`}</div><div class="modal-actions"><button type="button" class="button ghost" data-action="close-modal">取消</button><button type="submit" class="button primary">保存故事</button></div></form>`;
    } else if (modal.type === "invite") {
      body = modal.invite ? `<div class="modal-kicker">FAMILY SPACE INVITE</div><h2>局域网邀请已生成</h2><p class="modal-lead">这是一个本地邀请 token。当前不会发送到云端或第三方服务。</p><code class="invite-code">${escapeHtml(modal.invite.invite_url)}</code><div class="modal-actions"><button class="button primary" data-action="close-modal">完成</button></div>` : `<form id="modal-form"><div class="modal-kicker">FAMILY SPACE</div><h2>生成家庭成员邀请</h2><label>邀请备注<input name="label" value="家庭成员" required /></label><div class="modal-actions"><button type="button" class="button ghost" data-action="close-modal">取消</button><button type="submit" class="button primary">生成邀请</button></div></form>`;
    } else if (modal.type === "command") {
      body = `<form id="modal-form"><div class="modal-kicker">COMMAND</div><h2>打开一个工作区</h2><label>输入页面或问题<input name="command" autofocus placeholder="例如：时间线、资料库、搜索冰箱" /></label><div class="command-links">${navItems.map((item) => `<button type="button" class="button ghost" data-view="${item.id}">${item.label}</button>`).join("")}</div></form>`;
    } else if (modal.type === "family-graph") {
      const graph = modal.graph || {};
      const nodes = (graph.nodes || []).filter((node) => node.status === "confirmed");
      const edges = (graph.edges || []).filter((edge) => nodes.some((node) => node.id === edge.source) && nodes.some((node) => node.id === edge.target));
      const relById = new Map((graph.relationships || []).map((item) => [item.id, item]));
      const personById = new Map(state.persons.filter((person) => person.confirmed).map((person) => [person.id, person]));
      const editing = modal.editing || null;
      const relationOptions = ["配偶", "丈夫", "妻子", "父亲", "母亲", "儿子", "女儿", "兄弟", "姐妹", "祖父", "祖母", "外祖父", "外祖母", "本人"];
      const width = 620, height = 400, cx = 310, cy = 195, radius = 145;
      const pos = new Map();
      nodes.forEach((node, index) => {
        const angle = (2 * Math.PI * index) / Math.max(nodes.length, 1) - Math.PI / 2;
        pos.set(node.id, { x: Math.round(cx + radius * Math.cos(angle)), y: Math.round(cy + radius * Math.sin(angle)) });
      });
      const edgeEls = edges.map((edge) => {
        const a = pos.get(edge.source), b = pos.get(edge.target);
        if (!a || !b) return "";
        const mx = Math.round((a.x + b.x) / 2), my = Math.round((a.y + b.y) / 2);
        return `<g class="family-edge" data-action="edit-family-relation" data-relation-id="${escapeHtml(edge.id)}" tabindex="0" aria-label="编辑关系"><line x1="${a.x}" y1="${a.y}" x2="${b.x}" y2="${b.y}"/><rect class="family-edge-hit" x="${mx - 44}" y="${my - 12}" width="88" height="24" rx="9"/><text class="family-edge-label" x="${mx}" y="${my + 3}">${escapeHtml(edge.label)}</text></g>`;
      }).join("");
      const nodeEls = nodes.map((node, index) => {
        const p = pos.get(node.id);
        const person = personById.get(node.id);
        const clipId = `family-clip-${index}`;
        const avatarUrl = person?.avatar_face_instance_id ? `/api/face-instances/${encodeURIComponent(person.avatar_face_instance_id)}/crop` : "";
        const img = avatarUrl ? `<image href="${escapeHtml(avatarUrl)}" x="${p.x - 20}" y="${p.y - 20}" width="40" height="40" preserveAspectRatio="xMidYMid slice" clip-path="url(#${clipId})"/>` : `<circle class="family-node-fallback" cx="${p.x}" cy="${p.y}" r="20"/>`;
        const scopeName = albumLabel(node.scope_id);
        return `<clipPath id="${clipId}"><circle cx="${p.x}" cy="${p.y}" r="20"/></clipPath><g class="family-node" data-action="open-person-profile" data-person-id="${escapeHtml(node.id)}" tabindex="0" aria-label="打开人物档案"><circle cx="${p.x}" cy="${p.y}" r="26"/><text class="family-node-text" x="${p.x}" y="${p.y + 36}">${escapeHtml(node.label)}</text>${node.scope_id ? `<text class="family-node-scope" x="${p.x}" y="${p.y + 50}">${escapeHtml(scopeName)}</text>` : ""}${img}</g>`;
      }).join("");
      const graphBody = nodes.length >= 2
        ? `<svg class="family-graph" viewBox="0 0 ${width} ${height}" role="img" aria-label="家庭关系图">${nodeEls}${edgeEls}</svg>`
        : emptyState("至少需要两位已确认人物", "先在人物页确认两位以上的家庭成员，再建立家庭关系。");
      const personOptions = nodes.map((node) => `<option value="${escapeHtml(node.id)}" ${editing && (editing.subject_entity_id === node.id || editing.object_entity_id === node.id) ? "selected" : ""}>${escapeHtml(node.label)}</option>`).join("");
      const relationSelect = relationOptions.map((role) => `<option value="${escapeHtml(role)}" ${editing && editing.predicate === role ? "selected" : ""}>${escapeHtml(role)}</option>`).join("");
      const relationRows = (graph.relationships || []).filter((item) => item.status !== "retracted").map((item) => `<div class="fact-review-row"><div><strong>${escapeHtml(item.subject_name)} ${escapeHtml(item.predicate)} ${escapeHtml(item.object_name)}</strong><small>${escapeHtml(item.status)} · 已由你维护</small></div><div class="review-actions"><button class="text-button" data-action="edit-family-relation" data-relation-id="${escapeHtml(item.id)}">编辑</button><button class="button small ghost" data-action="delete-family-relation" data-relation-id="${escapeHtml(item.id)}">删除</button></div></div>`).join("");
      const form = nodes.length >= 2 ? `<form id="modal-form" class="relation-form"><label>人物A<select name="person_a" required>${personOptions}</select></label><label>家庭关系<select name="relation">${relationSelect}</select><input name="relation_custom" placeholder="或自定义关系，如：养父" value="${editing && !relationOptions.includes(editing.predicate) ? escapeHtml(editing.predicate) : ""}" /></label><label>人物B<select name="person_b" required>${personOptions}</select></label><div class="modal-actions"><button type="button" class="button ghost" data-action="clear-relation-edit">取消</button><button type="submit" class="button primary">${editing ? "保存修改" : "添加关系"}</button></div></form>` : "";
      body = `<div class="modal-kicker">FAMILY GRAPH</div><h2>家庭关系图</h2><p class="modal-lead">这里只显示你已确认的人物与家庭关系。关系写入后会进入本地记忆，家庭助手也能回忆这些关系。</p>${graphBody}<div class="family-graph-toolbar"><div class="section-head"><div><p class="section-kicker">维护家庭关系</p><h3>${editing ? "编辑关系" : "添加关系"}</h3></div></div>${form}<div class="section-head" style="margin-top:18px"><div><p class="section-kicker">已建立的关系</p><h3>${(graph.relationships || []).filter((item) => item.status !== "retracted").length} 条</h3></div></div><div class="fact-review-list">${relationRows || emptyState("还没有家庭关系", "从上方选择两个人并填写家庭角色，关系会出现在这张图上。")}</div></div>`;
    } else if (modal.type === "import-picker") {
      body = `<div class="modal-kicker">IMPORT MEDIA</div><h2>选择导入方式</h2><p class="modal-lead">浏览器原生选择器不能在同一个窗口同时选择文件和文件夹，请选择一种导入方式。</p><div class="modal-actions"><button class="button primary" data-action="open-files">选择多个文件</button><button class="button ghost" data-action="open-folder">选择整个文件夹</button></div>`;
    } else if (modal.type === "space-manager") {
      const spaceRows = (state.spaces || []).map((sp) => {
        const isDefault = sp.id === "home-default";
        const isCurrent = sp.id === state.scopeId;
        const deleteBtn = isDefault
          ? `<span class="muted">系统默认,不可删除</span>`
          : `<button class="button small danger" data-action="ask-delete-space" data-scope-id="${escapeHtml(sp.id)}" data-scope-name="${escapeHtml(sp.name || sp.id)}">删除</button>`;
        return `<div class="space-row ${isCurrent ? "current" : ""}"><div><strong>${escapeHtml(sp.name || sp.id)}</strong><small>${escapeHtml(sp.id)} · ${escapeHtml(sp.kind || "")}</small></div><div>${deleteBtn}</div></div>`;
      }).join("");
      body = `<div class="modal-kicker">SPACE MANAGER</div><h2>相册管理</h2><p class="modal-lead">删除相册会同时清除该相册的图片、事件、人物、向量和 <code>data/media/</code> 下的物理文件,不可撤销。<code>home-default</code> 是系统默认相册,不能删除。</p><div class="space-list">${spaceRows || "<p class=\"muted\">暂无相册</p>"}</div><div class="modal-actions"><button class="button ghost" data-action="create-space">＋ 创建新相册</button><button class="button primary" data-action="close-modal">关闭</button></div>`;
    } else if (modal.type === "space-create") {
      body = `<form id="modal-form"><div class="modal-kicker">NEW MEMORY SPACE</div><h2>创建独立相册</h2><p class="modal-lead">创建后会自动切换到该相册，后续导入的图片和人物标注都会限制在这个范围。</p><label>相册名称<input name="name" autofocus maxlength="100" placeholder="例如：2025年旅行测试" required /></label><div class="modal-actions"><button type="button" class="button ghost" data-action="open-space">取消</button><button type="submit" class="button primary">创建并切换</button></div></form>`;
    } else if (modal.type === "space-delete-confirm") {
      const scopeId = modal.scopeId;
      const scopeName = modal.scopeName || scopeId;
      const stats = modal.stats;
      const summary = stats
        ? `将永久删除:<strong>${stats.assets || 0}</strong> 张图 / <strong>${stats.events || 0}</strong> 个事件 / <strong>${stats.persons || 0}</strong> 个人物 / <strong>${stats.vectors || 0}</strong> 条向量。`
        : `将永久删除此相册的全部内容。`;
      body = `<div class="modal-kicker">DELETE SPACE</div><h2>确认删除相册『${escapeHtml(scopeName)}』?</h2><p class="modal-lead">${summary}<br/><strong>此操作不可撤销,物理文件也会一同清理。</strong></p><div class="modal-actions"><button class="button ghost" data-action="open-space">取消</button><button class="button danger" data-action="confirm-delete-space" data-scope-id="${escapeHtml(scopeId)}" data-scope-name="${escapeHtml(scopeName)}">确认删除</button></div>`;
    } else if (modal.type === "help") {
      body = `<div class="modal-kicker">SENTRIX HOME / HELP</div><h2>当前可用能力</h2><div class="help-list"><div><strong>导入</strong><span>图片、音频、文本和视频关键帧会生成 Observation；视频 Scene 进入家庭时间线。</span></div><div><strong>证据</strong><span>事件和 Agent 回答都能打开 Asset、Observation 和模型原始 JSON。</span></div><div><strong>维护</strong><span>事实冲突进入 pending，确认后旧版本变为 superseded。</span></div><div><strong>隐私</strong><span>原始文件、人物候选、视频场景和 SQLite 都在当前本机运行。</span></div></div><div class="modal-actions"><button class="button primary" data-action="close-modal">关闭</button></div>`;
    }
    root.innerHTML = `<div class="modal-backdrop"><div class="modal-panel"><button class="modal-back" data-action="close-modal" aria-label="返回上一页">${icon("←")}返回</button><button class="modal-close" data-action="close-modal" aria-label="关闭">×</button>${body}</div></div>`;
  }

  function showToast(message) {
    const root = document.getElementById("toast-root");
    if (!root) return;
    root.innerHTML = `<div class="toast"><span>✓</span>${escapeHtml(message)}</div>`;
    setTimeout(() => { if (root) root.innerHTML = ""; }, 2600);
    state.toast = "";
  }

  function isUserEditing() {
    const active = document.activeElement;
    return Boolean(state.modal || active?.matches("input, textarea, select") || active?.closest("form"));
  }

  function updateLiveStats() {
    const values = {
      assets: state.dashboard?.stats?.assets ?? 0,
      events: state.dashboard?.stats?.events ?? 0,
      facts: state.dashboard?.stats?.facts ?? 0,
      pendingFacts: state.dashboard?.pendingFacts ?? 0,
    };
    document.querySelectorAll("[data-live-stat]").forEach((element) => {
      const key = element.dataset.liveStat;
      if (Object.prototype.hasOwnProperty.call(values, key)) element.textContent = values[key];
    });
  }

  let refreshInFlight = false;
  let modelSwitchInFlight = false;
  let requestedModelProfile = "";

  async function handleModelProfileChange(select) {
    if (modelSwitchInFlight) return;
    const target = String(select.value || "");
    if (!target) return;
    modelSwitchInFlight = true;
    requestedModelProfile = target;
    select.disabled = true;
    const selectedOption = select.selectedOptions[0];
    if (selectedOption) selectedOption.textContent = `${selectedOption.textContent.replace(/ · 切换中$/, "")} · 切换中`;
    try {
      await window.sentrixApi.switchModelProfile(target);
      const payload = await window.sentrixApi.getModelProfiles();
      const current = payload.current || {};
      if (current.profile !== target || current.status !== "running") {
        throw new Error(`后端未确认目标模型运行，当前状态: ${current.status || "unknown"}`);
      }
      state.vlmBackendOptions = modelProfileOptions(payload);
      state.backendError = "";
      state.toast = `主模型已切换为 ${state.vlmBackendOptions.models?.[target]?.model || target}`;
    } catch (error) {
      state.backendError = `模型切换失败：${error.message || error}`;
      try {
        state.vlmBackendOptions = modelProfileOptions(await window.sentrixApi.getModelProfiles());
      } catch (_) {
        state.vlmBackendOptions = null;
      }
    } finally {
      modelSwitchInFlight = false;
      requestedModelProfile = "";
      renderShellNavigation();
    }
  }

  async function refreshData(options = {}) {
    if (refreshInFlight) return;
    refreshInFlight = true;
    const silent = options.silent === true;
    state.loading = !silent;
    if (!silent) renderView();
    const spaceResult = await Promise.allSettled([window.sentrixApi.memorySpaces()]);
    if (spaceResult[0].status === "fulfilled") {
      state.spaces = spaceResult[0].value.spaces || [];
      if (!state.scopeInitialized) {
        const storedScopeId = window.localStorage?.getItem("sentrix.scopeId") || "";
        const storedSpace = state.spaces.find((space) => space.id === storedScopeId);
        // The web portal should open on the complete benchmark corpus. A
        // legacy household selection is intentionally migrated to all albums;
        // users can still choose an individual benchmark space explicitly.
        state.scopeId = storedSpace?.kind === "benchmark" ? storedSpace.id : "";
        state.scopeInitialized = true;
      } else if (state.scopeId && state.spaces.length && !state.spaces.some((space) => space.id === state.scopeId)) {
        state.scopeId = "";
      }
    }
    const scopeId = state.scopeId;
    const calls = await Promise.allSettled([
          window.sentrixApi.dashboard(scopeId), window.sentrixApi.events(scopeId), window.sentrixApi.assets("?limit=1000", scopeId), window.sentrixApi.people("", scopeId), window.sentrixApi.stories(), window.sentrixApi.health(), window.sentrixApi.entities("", scopeId), window.sentrixApi.faceClusters("", scopeId), window.sentrixApi.relationships(scopeId), window.sentrixApi.knowledge("", scopeId), window.sentrixApi.trips(scopeId, "pending"), window.sentrixApi.entityMergeCandidates(scopeId), window.sentrixApi.entityGroups(scopeId),
          window.sentrixApi.geoPlaces(scopeId),
    ]);
    state.dashboard = calls[0].status === "fulfilled" ? calls[0].value : null;
    state.events = calls[1].status === "fulfilled" ? calls[1].value.events || [] : [];
    state.assets = calls[2].status === "fulfilled" ? calls[2].value.assets || [] : [];
    state.persons = calls[3].status === "fulfilled" ? calls[3].value.people || [] : [];
    state.stories = calls[4].status === "fulfilled" ? calls[4].value.stories || [] : [];
    state.health = calls[5].status === "fulfilled" ? calls[5].value : null;
    loadConversations().catch(() => {});
    try {
      const profilePayload = await window.sentrixApi.getModelProfiles();
      if (!modelSwitchInFlight) state.vlmBackendOptions = modelProfileOptions(profilePayload);
    } catch (err) {
      state.vlmBackendOptions = null;
    }
    try {
      state.ocrSettings = await window.sentrixApi.getOcrSettings();
    } catch (err) {
      state.ocrSettings = null;
    }
    state.entities = calls[6].status === "fulfilled" ? calls[6].value.entities || [] : [];
    state.clusters = calls[7].status === "fulfilled" ? calls[7].value.clusters || [] : [];
    state.relationships = calls[8].status === "fulfilled" ? calls[8].value.relationships || [] : [];
        state.knowledge = calls[9].status === "fulfilled" ? calls[9].value : { profiles: [], claims: [] };
        state.trips = calls[10].status === "fulfilled" ? calls[10].value.trips || [] : [];
        state.entityMergeCandidates = calls[11].status === "fulfilled" ? calls[11].value.candidates || [] : [];
        state.entityGroups = calls[12].status === "fulfilled" ? calls[12].value.groups || [] : [];
        state.geoPlaces = calls[13].status === "fulfilled" ? calls[13].value.places || [] : [];
    const failed = calls.find((call) => call.status === "rejected");
    state.backendError = failed ? "本地后端暂时不可用，当前页面只显示已读取到的真实数据。" : "";
    state.loading = false;
    if (options.forceRender || !isUserEditing()) {
      if (silent) updateLiveStats();
      else renderShellNavigation();
    }
    refreshInFlight = false;
  }

  function renderShellNavigation() { shell(); }

  function navigate(view) {
    const target = view || "overview";
    if (state.view === target && !state.modal) return;
    const nextHash = `#/${target}`;
    if (window.location.hash !== nextHash) {
      window.location.hash = nextHash;
    } else {
      state.view = target;
      state.modal = null;
      state.modalHistory = [];
      renderShellNavigation();
    }
  }

  function goBack() {
    if (window.history.length > 1) {
      window.history.back();
    } else {
      state.view = "overview";
      state.modal = null;
      state.modalHistory = [];
      renderShellNavigation();
    }
  }

  function bindViewEvents() {
    document.querySelectorAll("[data-view]").forEach((element) => element.addEventListener("click", () => { navigate(element.dataset.view); }));
    document.querySelectorAll("[data-query]").forEach((element) => element.addEventListener("click", () => { state.query = element.dataset.query; state.view = "search"; renderShellNavigation(); submitSearch(); }));
    document.querySelectorAll("[data-event-filter]").forEach((element) => element.addEventListener("click", () => { state.eventFilter = element.dataset.eventFilter; renderView(); }));
    const eventDate = document.querySelector("[data-event-date]");
    if (eventDate) eventDate.addEventListener("change", (event) => { state.eventDate = event.target.value || ""; renderView(); });
    document.querySelectorAll("[data-asset-filter]").forEach((element) => element.addEventListener("click", () => { state.assetFilter = element.dataset.assetFilter; renderView(); }));
    document.querySelectorAll("[data-person-filter]").forEach((element) => element.addEventListener("click", () => { state.personFilter = element.dataset.personFilter; renderView(); }));
    const spaceSelect = document.getElementById("space-select");
    if (spaceSelect) spaceSelect.addEventListener("change", async (event) => { state.scopeId = event.target.value; window.localStorage?.setItem("sentrix.scopeId", state.scopeId); state.modal = null; state.conversationId = ""; state.searchResult = null; state.assistantMessages = []; state.activeConversationSummary = ""; state.selectedAsset = null; await refreshData(); });
    const form = document.getElementById("search-form");
    if (form) form.addEventListener("submit", submitSearch);
    const modalForm = document.getElementById("modal-form");
    if (modalForm) modalForm.addEventListener("submit", handleModalSubmit);
    const photoThreadForm = document.getElementById("photo-thread-form");
    if (photoThreadForm) photoThreadForm.addEventListener("submit", (event) => {
      event.preventDefault();
      const input = document.getElementById("photo-thread-input");
      if (input && state.photoInspector) state.photoInspector.draft = input.value;
      submitPhotoInspector();
    });
    document.querySelectorAll("[data-action]").forEach((element) => element.addEventListener("click", () => handleAction(element.dataset.action, element)));
    const fileInput = document.getElementById("file-input");
    if (fileInput) fileInput.addEventListener("change", handleFiles);
    if (state.view === "imports") document.querySelector('.page-heading [data-action="open-folder"]')?.remove();
    const folderInput = document.getElementById("folder-input");
    if (folderInput) folderInput.addEventListener("change", handleFiles);
    const dropzone = fileInput?.closest(".dropzone");
    if (dropzone) {
      const hint = dropzone.querySelector("small");
      if (hint) hint.textContent = "选择文件或文件夹，系统会自动导入其中的 JPG/JPEG/PNG 图片";
      const chooseButton = dropzone.querySelector(".button.primary");
      if (chooseButton && chooseButton.tagName !== "BUTTON") {
        const unifiedButton = document.createElement("button");
        unifiedButton.type = "button";
        unifiedButton.className = chooseButton.className;
        unifiedButton.dataset.action = "open-import-picker";
        unifiedButton.textContent = "选择文件或文件夹";
        chooseButton.replaceWith(unifiedButton);
        unifiedButton.addEventListener("click", (event) => { event.preventDefault(); event.stopPropagation(); handleAction("open-import-picker", unifiedButton); });
      }
    }
    const topUser = document.querySelector(".top-user");
    const topUserLabel = topUser?.querySelector("span:not(.avatar)");
    if (topUserLabel) topUserLabel.textContent = "相册管理";
  }

  async function submitSearch(event, selectedEntityId = "") {
    if (event?.preventDefault) event.preventDefault();
    const input = document.getElementById("search-input");
    state.query = input ? input.value.trim() : state.query.trim();
    if (!state.query) return;
    state.view = "search";
    state.assistantMessages.push({ role: "user", text: state.query });
    state.searchLoading = true;
    state.liveProgress = [];
    renderShellNavigation();
    try {
      const { result, conversationId } = await runAssistantTurn(state.query, state.conversationId, null, state.scopeId, selectedEntityId, state.selectedAsset);
      state.conversationId = conversationId;
      state.searchResult = result;
    } catch (error) {
      state.searchResult = { answer: "当前无法读取本地记忆，请稍后重试。", confidence: 0, evidence: [], retrievalTrace: [], error: error.message, insufficient_evidence: true };
    }
    state.assistantMessages.push({ role: "steward", result: state.searchResult });
    state.query = "";
    state.searchLoading = false;
    state.liveProgress = [];
    loadConversations().catch(() => {});
    renderShellNavigation();
  }

  async function loadConversations() {
    try {
      const data = await window.sentrixApi.conversations(state.scopeId);
      state.conversations = (data && data.conversations) || [];
      if (state.conversationId && !state.conversations.some((conv) => conv.conversation_id === state.conversationId)) {
        try {
          const one = await window.sentrixApi.conversation(state.conversationId);
          if (one && one.conversation) state.conversations.unshift(one.conversation);
        } catch { /* 已删除或不可用 */ }
      }
    } catch { state.conversations = state.conversations || []; }
  }

  async function newConversation() {
    try {
      const data = await window.sentrixApi.createConversation(state.scopeId);
      state.conversationId = data.conversation_id;
    } catch { state.conversationId = `conversation_${Date.now()}`; }
    state.assistantMessages = [];
    state.searchResult = null;
    state.selectedAsset = null;
    state.activeConversationSummary = "";
    await loadConversations();
    renderShellNavigation();
  }

  async function openConversation(id) {
    state.conversationId = id;
    state.selectedAsset = null;
    try {
      const data = await window.sentrixApi.conversation(id);
      state.activeConversationSummary = (data.conversation && data.conversation.summary) || "";
      state.assistantMessages = (data.messages || []).map((msg) => {
        const text = (msg.content && (msg.content.text || msg.content.content)) || "";
        if (msg.role === "user") return { role: "user", text };
        return { role: "steward", result: { answer: text, conversation_id: id, answer_grounding: { display_mode: "none" }, tool_loop_status: "complete" } };
      });
    } catch { state.assistantMessages = []; }
    renderShellNavigation();
  }

  async function deleteConversationAction(id) {
    const target = (state.conversations || []).find((conv) => conv.conversation_id === id);
    const title = (target && target.title) || "这个对话";
    if (!window.confirm(`删除「${title}」？\n这会删除聊天记录和处理过程，但不会删除已经保存的家庭照片和记忆。`)) return;
    try {
      await window.sentrixApi.deleteConversation(id);
      if (state.conversationId === id) {
        state.conversationId = "";
        state.assistantMessages = [];
        state.selectedAsset = null;
        state.activeConversationSummary = "";
      }
      await loadConversations();
    } catch (error) { state.toast = `删除失败：${error.message}`; }
    renderShellNavigation();
  }

  async function submitProactiveOutcome(element, outcome) {
    const sceneKey = element.dataset.sceneKey || "";
    const message = outcome === "accepted" ? "看看这段回忆" : "好的";
    state.assistantMessages.push({ role: "user", text: message });
    state.searchLoading = true;
    renderShellNavigation();
    try {
      const { result, conversationId } = await runAssistantTurn(
        message, state.conversationId,
        { proactivity_outcome: outcome, proactivity_scene_key: sceneKey },
        state.scopeId, "",
      );
      state.searchResult = result;
      state.conversationId = conversationId;
      state.assistantMessages.push({ role: "steward", result: state.searchResult });
    } catch (error) {
      state.toast = `主动回忆状态未更新：${error.message}`;
    }
    state.searchLoading = false;
    renderShellNavigation();
  }

  async function runAssistantTurn(message, conversationId = "", feedback = null, scopeId = "home-default", selectedEntityId = "", selectedAsset = null) {
    const start = await window.sentrixApi.assistantTurnAsync(message, conversationId, feedback, scopeId, selectedEntityId, "owner", selectedAsset);
    if (start && start.turn_id && start.status === "running") {
      const nextConversationId = start.conversation_id || conversationId;
      // Phase C C13：优先 SSE 实时事件；EventSource 不可用时回退 700ms 轮询。
      let done = await subscribeTurnEvents(start.turn_id);
      if (!done) done = await pollTurnEvents(start.turn_id);
      return { result: done || { answer: "执行超时，请重试。", evidence_status: "error" }, conversationId: nextConversationId };
    }
    return { result: start, conversationId: (start && start.conversation_id) || conversationId };
  }

  function mergeLiveProgress(event) {
    const idx = event && event.step_index != null ? event.step_index : null;
    if (idx != null && Array.isArray(state.liveProgress)) {
      const existing = state.liveProgress.findIndex((p) => p.step_index === idx);
      if (existing >= 0) state.liveProgress[existing] = event;
      else state.liveProgress.push(event);
    } else {
      state.liveProgress.push(event);
    }
  }

  function subscribeTurnEvents(turnId) {
    return new Promise((resolve) => {
      let es = null;
      try { es = new EventSource(window.sentrixApi.assistantTurnEventsUrl(turnId)); } catch (_) { resolve(null); return; }
      const finish = (value) => { try { es.close(); } catch (_) { /* noop */ } resolve(value); };
      es.addEventListener("progress", (e) => {
        try {
          const event = JSON.parse(e.data);
          if (event && event.text) { mergeLiveProgress(event); updateLiveProgress(); }
        } catch (_) { /* 忽略坏事件 */ }
      });
      es.addEventListener("complete", (e) => {
        try { finish(JSON.parse(e.data).result || JSON.parse(e.data)); } catch (_) { finish(null); }
      });
      es.onerror = () => finish(null); // 交给轮询兜底
      setTimeout(() => finish(null), 600_000); // 超时保护
    });
  }

  async function pollTurnEvents(turnId) {
    let done = null;
    for (let i = 0; i < 150 && !done; i += 1) {
      await new Promise((resolve) => setTimeout(resolve, 700));
      try {
        const poll = await window.sentrixApi.assistantTurnPoll(turnId);
        if (Array.isArray(poll.public_progress)) state.liveProgress = poll.public_progress;
        updateLiveProgress();
        if (poll.status === "complete") done = poll.result;
        else if (poll.status === "error") done = { answer: "执行过程中出错。", error: poll.error, evidence_status: "error" };
      } catch (pollError) { /* 单次轮询失败继续等待 */ }
    }
    return done;
  }

  async function handleFiles(event) {
    let files = Array.from(event.target.files || []);
    if (event.target.id === "folder-input") files = files.filter((file) => /\.(jpe?g|png)$/i.test(file.name || ""));
    if (!files.length) { state.toast = "所选目录中没有 JPG/JPEG/PNG 图片"; renderShellNavigation(); return; }
    const queueEntries = files.map((file) => ({ fileName: file.name, status: "reading-metadata" }));
    state.queue.unshift(...queueEntries);
    state.toast = `已读取 ${files.length} 张图片，正在解析元数据...`;
    renderShellNavigation();
    const items = [];
    for (let index = 0; index < files.length; index += 1) {
      const file = files[index];
      try {
        const metadata = await window.sentrixImageMetadata.extract(file);
        queueEntries[index].metadata = metadata;
        queueEntries[index].status = "ready";
        items.push({ file, metadata });
      } catch (error) {
        queueEntries[index].status = "metadata-failed";
        queueEntries[index].error = error.message || String(error);
      }
      if ((index + 1) % 10 === 0 || index === files.length - 1) {
        state.toast = `正在解析图片元数据：${index + 1}/${files.length}`;
        renderShellNavigation();
      }
    }
    const chunkSize = 20;
    let accepted = 0;
    try {
      for (let offset = 0; offset < items.length; offset += chunkSize) {
        const chunk = items.slice(offset, offset + chunkSize);
        const result = await window.sentrixApi.importAssets(chunk, { scopeId: state.scopeId });
        (result.items || []).forEach((item, index) => {
          const entry = queueEntries.find((candidate) => candidate.fileName === chunk[index].file.name && candidate.status === "ready");
          if (!entry) return;
          entry.assetId = item.assetId || item.asset_id;
          entry.status = item.status || (item.accepted ? "queued" : "failed");
          entry.error = item.error || "";
          if (item.accepted) accepted += 1;
        });
        state.toast = `正在上传图片：${Math.min(offset + chunk.length, items.length)}/${items.length}`;
        renderShellNavigation();
      }
    } catch (error) {
      queueEntries.filter((entry) => entry.status === "ready").forEach((entry) => { entry.status = "upload-failed"; entry.error = error.message || String(error); });
      state.toast = `上传失败：${error.message || error}`;
      renderShellNavigation();
    }
    state.toast = `上传完成：${accepted}/${files.length} 张进入本地处理队列`;
    await refreshData();
    state.view = "imports";
    renderShellNavigation();
  }

  function openModal(modal, options = {}) {
    if (options.push && state.modal && state.modal.type !== "loading") {
      state.modalHistory.push(state.modal);
    }
    state.modal = modal;
    renderShellNavigation();
  }

  function closeCurrentModal() {
    if (state.modal && state.modal.type === "photo-inspector" && state.photoInspector && state.photoInspector.thread_id) {
      const tid = state.photoInspector.thread_id;
      state.photoInspector = null;
      window.sentrixApi.deletePhotoThread(tid).catch(() => {});
    }
    const previous = state.modalHistory.pop();
    state.modal = previous || null;
    renderShellNavigation();
  }

  async function openEvent(eventId, edit = false) {
    openModal({ type: "loading" }, { push: true });
    try { const detail = await window.sentrixApi.event(eventId); openModal(edit ? { type: "event-edit", event: detail.event, observations: detail.observations || [] } : { type: "event", detail }); } catch { state.toast = "无法读取事件证据"; state.modal = null; renderShellNavigation(); }
  }

  async function openAsset(assetId) {
    openModal({ type: "loading" }, { push: true });
    try { const [asset, result] = await Promise.all([window.sentrixApi.asset(assetId), window.sentrixApi.observations(`?assetId=${encodeURIComponent(assetId)}`)]); openModal({ type: "asset", asset, observations: result.observations || [] }); } catch { state.toast = "无法读取原始资料"; state.modal = null; renderShellNavigation(); }
  }

  async function openPhotoInspector(resultSetId, handle) {
    openModal({ type: "loading" }, { push: true });
    try {
      const thread = await window.sentrixApi.createPhotoThread({
        asset_handle: handle, result_set_id: resultSetId,
        scope_id: state.scopeId, parent_conversation_id: state.conversationId || null,
      });
      state.photoInspector = { ...thread, messages: [], draft: "", loading: false };
      openModal({ type: "photo-inspector" });
      await loadPhotoInspectorMessages();
    } catch (error) {
      state.toast = `无法打开照片检查器：${error.message}`;
      state.modal = null;
      renderShellNavigation();
    }
  }

  async function loadPhotoInspectorMessages() {
    const pi = state.photoInspector;
    if (!pi || !pi.thread_id) return;
    try {
      const data = await window.sentrixApi.photoThreadMessages(pi.thread_id, 30);
      pi.messages = (data.messages || []).map((msg) => {
        const text = (msg.content && (msg.content.text || msg.content.content)) || "";
        return { role: msg.role, text };
      });
    } catch { /* 保持现有消息 */ }
    renderShellNavigation();
  }

  async function submitPhotoInspector() {
    const pi = state.photoInspector;
    if (!pi || !pi.thread_id) return;
    const message = (pi.draft || "").trim();
    if (!message || pi.loading) return;
    pi.draft = "";
    pi.loading = true;
    pi.messages = pi.messages || [];
    pi.messages.push({ role: "user", text: message });
    renderShellNavigation();
    try {
      const start = await window.sentrixApi.photoThreadTurnAsync(pi.thread_id, message, "owner");
      let done = null;
      if (start && start.turn_id && start.status === "running") {
        done = await subscribeTurnEvents(start.turn_id);
        if (!done) done = await pollTurnEvents(start.turn_id);
      }
      const answer = (done && (done.answer || done.result?.answer)) || "没有收到回答。";
      pi.messages.push({ role: "assistant", text: answer });
    } catch (error) {
      pi.messages.push({ role: "assistant", text: `处理失败：${error.message}` });
    }
    pi.loading = false;
    renderShellNavigation();
  }

  async function openEntity(entityId) {
    openModal({ type: "loading" }, { push: true });
    try { const detail = await window.sentrixApi.entity(entityId); openModal({ type: "entity", detail }); } catch { state.toast = "无法读取实体记忆"; state.modal = null; renderShellNavigation(); }
  }

  async function openEntityGroup(groupId) {
    openModal({ type: "loading" }, { push: true });
    try { const detail = await window.sentrixApi.entityGroup(groupId, state.scopeId); openModal({ type: "entity-group", detail }); } catch { state.toast = "无法读取语义实体组"; state.modal = null; renderShellNavigation(); }
  }

  async function handleModalSubmit(event) {
    event.preventDefault();
    if (state.saving) return;
    const form = new FormData(event.target);
    const modal = state.modal;
    state.saving = true;
    const submitButton = event.target.querySelector("button[type='submit']");
    if (submitButton) { submitButton.disabled = true; submitButton.textContent = "正在更新记忆..."; }
    try {
      if (modal.type === "event-edit") await window.sentrixApi.updateEvent(modal.event.id, { title: form.get("title"), event_type: form.get("event_type"), summary: form.get("summary"), place: form.get("place"), time_start: form.get("time_start") ? new Date(form.get("time_start")).toISOString() : modal.event.time_start, time_end: form.get("time_end") ? new Date(form.get("time_end")).toISOString() : modal.event.time_end, cover_asset_id: form.get("cover_asset_id") || modal.event.cover_asset_id });
      if (modal.type === "event-create") await window.sentrixApi.createEvent({ title: form.get("title"), summary: form.get("summary"), place: form.get("place"), time_start: form.get("time_start") ? new Date(form.get("time_start")).toISOString() : null });
      if (modal.type === "trip-confirm") { await window.sentrixApi.confirmTrip(modal.trip.id, { name: form.get("name"), trip_type: form.get("trip_type") }); state.toast = "行程已确认，原始事件和照片证据已保留"; }
      if (modal.type === "entity-merge-confirm") { await window.sentrixApi.confirmEntityMergeCandidate(modal.candidate.id, form.get("target_entity_id")); state.toast = "实体已按你的选择归并，原始证据和修订记录已保留"; }
      if (modal.type === "entity-property-edit") {
        const evidenceIds = form.getAll("evidence_ids");
        await window.sentrixApi.setEntityProperty(modal.detail.entity.id, "alias", String(form.get("alias") || "").trim(), evidenceIds);
        await window.sentrixApi.setEntityProperty(modal.detail.entity.id, "private_flag", form.get("private_flag") === "on", evidenceIds);
        state.toast = "地点属性已按你的修正保存，并保留版本和证据";
      }
      if (modal.type === "person-property-edit") {
        await window.sentrixApi.setEntityProperty(modal.detail.entity.id, "canonical_name", String(form.get("canonical_name") || "").trim());
        await window.sentrixApi.setEntityProperty(modal.detail.entity.id, "is_self", form.get("is_self") === "on");
        await window.sentrixApi.setEntityProperty(modal.detail.entity.id, "relation_to_user", String(form.get("relation_to_user") || "").trim());
        await window.sentrixApi.setEntityProperty(modal.detail.entity.id, "groups", String(form.get("groups") || "").split(/[、,，]/).map((item) => item.trim()).filter(Boolean));
        state.toast = "人物档案已按你的修正保存";
      }
      if (modal.type === "person-name-edit") {
        const renamed = await window.sentrixApi.renamePerson(modal.detail.entity.id, { name: form.get("name"), aliases: String(form.get("aliases") || "").split(/[、,，]/).map((item) => item.trim()).filter(Boolean) });
        state.toast = `已更新名字，历史记忆已同步`;
        if (renamed?.entity) {
          state.modal = null;
          const detail = await window.sentrixApi.personProfile(renamed.entity.id);
          return openModal({ type: "person-profile", detail });
        }
      }
      if (modal.type === "person") {
        const confirmed = await window.sentrixApi.confirmPerson(modal.person.id, form.get("name"), form.get("family_role"));
        if (confirmed.merged_into) {
          state.toast = `「${form.get("name")}」已合并到 ${confirmed.canonical_name || "已有成员"}，人脸证据已归并`;
        } else {
          const counts = confirmed.refresh_counts || {};
          state.toast = `已确认${form.get("name")}，已更新 ${counts.events || 0} 个事件、${counts.patterns || 0} 个模式、${counts.claims || 0} 条语义声明`;
        }
      }
      if (modal.type === "cluster-confirm") {
        const confirmed = await window.sentrixApi.confirmFaceCluster(modal.cluster.id, { name: form.get("name"), family_role: form.get("family_role") });
        const target = String(form.get("relation_target") || "");
        const predicate = String(form.get("relation_predicate") || "").trim();
        if (target && predicate && confirmed.entity?.id) await window.sentrixApi.createRelationship({ subject_entity_id: confirmed.entity.id, predicate, object_entity_id: target, evidence_ids: (modal.cluster.samples || []).map((sample) => sample.observation_id), confidence: 1, status: "active" });
        if (confirmed.merged_into) {
          state.toast = `「${form.get("name")}」已合并到 ${confirmed.canonical_name || "已有成员"}，人脸证据已归并`;
        } else {
          const counts = confirmed.refresh_counts || {};
          state.toast = `已确认${form.get("name")}，已更新 ${counts.events || 0} 个事件、${counts.patterns || 0} 个模式、${counts.claims || 0} 条语义声明`;
        }
      }
      if (modal.type === "cluster-merge") {
        await window.sentrixApi.mergeFaceClusters(form.get("target_cluster_id"), modal.cluster.id);
      }
      if (modal.type === "cluster-split") {
        await window.sentrixApi.splitFaceCluster(modal.cluster.id, modal.sample.id);
        state.toast = "已拆出新人物簇，原始证据仍保留";
      }
      if (modal.type === "story-create") {
        const storyEventIds = form.getAll("event_ids");
        state.storyGenerating = true;
        state.storyError = false;
        state.storyDraftEventIds = storyEventIds;
        state.modal = null;
        state.modalHistory = [];
        state.view = "stories";
        renderShellNavigation();
        try {
          await window.sentrixApi.createStory({ title: form.get("title") || "", content: "", event_ids: storyEventIds, tags: (form.get("tags")||"").split(",").map(s=>s.trim()).filter(Boolean) });
          await refreshData({ forceRender: true });
          state.toast = "故事已生成";
        } catch (error) {
          state.storyError = true;
          state.toast = `故事生成失败：${error.message}`;
        } finally {
          state.storyGenerating = false;
          renderShellNavigation();
        }
        return;
      }
      if (modal.type === "story-edit") await window.sentrixApi.updateStory(modal.story.id, { title: form.get("title"), content: form.get("content"), event_ids: form.getAll("event_ids"), tags: (form.get("tags")||"").split(",").map(s=>s.trim()).filter(Boolean) });
      if (modal.type === "command") {
        const command = String(form.get("command") || "").trim();
        const target = navItems.find((item) => command.includes(item.label) || command.toLowerCase().includes(item.id));
        if (target) { state.modal = null; state.view = target.id; renderShellNavigation(); return; }
        state.modal = null; state.query = command; state.view = "search"; renderShellNavigation(); return submitSearch();
      }
      if (modal.type === "invite") { const invite = await window.sentrixApi.createInvite(form.get("label")); openModal({ type: "invite", invite }); return; }
      if (modal.type === "space-create") {
        const space = await window.sentrixApi.createMemorySpace(String(form.get("name") || "").trim());
        state.scopeId = space.id;
        window.localStorage?.setItem("sentrix.scopeId", state.scopeId);
        state.modal = null;
        state.modalHistory = [];
        state.toast = `已创建并切换到相册“${space.name}”`;
        await refreshData({ forceRender: true });
        renderShellNavigation();
        return;
      }
      if (modal.type === "family-graph") {
        const subject = String(form.get("person_a") || "").trim();
        const object = String(form.get("person_b") || "").trim();
        const predicate = String(form.get("relation_custom") || form.get("relation") || "").trim();
        if (!subject || !object || !predicate) { state.toast = "请选择人物A、关系类型和人物B"; renderShellNavigation(); return; }
        const scopeOf = (id) => (modal.graph?.nodes || []).find((node) => node.id === id)?.scope_id;
        if (scopeOf(subject) && scopeOf(object) && scopeOf(subject) !== scopeOf(object)) {
          state.toast = "保持相册隔离：这两个人物属于不同相册，请在同一相册内建立家庭关系";
          renderShellNavigation(); return;
        }
        if (modal.editing) {
          try { await window.sentrixApi.retractRelationship(modal.editing.id); } catch { /* 新关系直接创建 */ }
        }
        const created = await window.sentrixApi.createRelationship({ subject_entity_id: subject, predicate, object_entity_id: object, status: "active", confidence: 1 });
        state.toast = `已保存家庭关系：${predicate}，并写入长期记忆`;
        try { const graph = await window.sentrixApi.relationships(state.scopeId, "person"); state.modal = { type: "family-graph", graph }; } catch { state.modal = null; }
        await refreshData({ silent: true });
        renderShellNavigation();
        return;
      }
      state.modal = null;
      state.modalHistory = [];
      await refreshData({ forceRender: true });
      state.toast = state.toast || "已保存到本地记忆";
      renderShellNavigation();
    } catch (error) {
      state.toast = `保存失败：${error.message}`;
      if (submitButton) { submitButton.disabled = false; submitButton.textContent = "重试保存"; }
      renderShellNavigation();
    } finally { state.saving = false; }
  }

  async function handleAction(action, element) {
    if (action === "seek-video") {
      const player = document.getElementById("scene-video-player");
      if (!player) return;
      // Pause first so seeking is deterministic even when autoplay is blocked
      // or the browser has only buffered part of the generated MP4 preview.
      const seek = () => { player.pause(); player.currentTime = Number(element.dataset.timestampSec || 0); };
      if (player.readyState >= 1) seek(); else player.addEventListener("loadedmetadata", seek, { once: true });
      return;
    }
    if (action === "toggle-ocr-small") {
      const next = element.checked;
      try {
        state.ocrSettings = await window.sentrixApi.setOcrSettings(next);
        state.toast = next ? "已开启 OCR 小模型优先" : "已关闭 OCR 小模型，使用多模态模型";
      } catch (error) {
        state.backendError = `OCR 设置保存失败：${error.message || error}`;
      }
      renderShellNavigation();
      return;
    }
    if (action === "result-next-page") { state.query = "下一页"; state.view = "search"; renderShellNavigation(); submitSearch(); return; }
    if (action === "new-conversation") { await newConversation(); return; }
    if (action === "open-conversation") { await openConversation(element.dataset.conversationId); return; }
    if (action === "delete-conversation") { await deleteConversationAction(element.dataset.conversationId); return; }
    if (action === "open-photo-inspector") { await openPhotoInspector(element.dataset.resultSetId, element.dataset.handle); return; }
    if (action === "photo-inspector-quick") {
      if (state.photoInspector) state.photoInspector.draft = element.dataset.query || "";
      renderShellNavigation();
      const input = document.getElementById("photo-thread-input");
      if (input) input.focus();
      return;
    }
    if (action === "select-result-photo") {
      state.selectedAsset = { result_set_id: element.dataset.resultSetId, handle: element.dataset.handle };
      renderShellNavigation();
      state.toast = "已选中这张照片，可以问它里面的内容或要原图。";
      return;
    }
    if (action === "open-selected-original") {
      window.open(window.sentrixApi.resultSetPhoto(element.dataset.resultSetId, element.dataset.handle, state.scopeId, true), "_blank");
      return;
    }
    if (action === "close-modal") { closeCurrentModal(); return; }
    if (action === "back") { state.modal = null; goBack(); return; }
    if (action === "home") { state.modal = null; navigate("overview"); return; }
    if (action === "open-event") return openEvent(element.dataset.eventId);
    if (action === "edit-event") return openEvent(element.dataset.eventId, true);
    if (action === "open-asset") return openAsset(element.dataset.assetId);
    if (action === "open-entity") return openEntity(element.dataset.entityId);
    if (action === "open-entity-group") return openEntityGroup(element.dataset.entityGroupId);
    if (action === "geo-select-city") { state.geoBreadcrumb = [element.dataset.geoCity]; renderView(); return; }
    if (action === "geo-select-district") { state.geoBreadcrumb = [state.geoBreadcrumb[0], element.dataset.geoDistrict]; renderView(); return; }
    if (action === "geo-breadcrumb") { const level = element.dataset.geoLevel; state.geoBreadcrumb = level === "root" ? [] : state.geoBreadcrumb.slice(0, parseInt(level) + 1); renderView(); return; }
    if (action === "edit-entity-properties") return openModal({ type: "entity-property-edit", detail: state.modal.detail });
    if (action === "open-observation") { const observation = await window.sentrixApi.observation(element.dataset.observationId); return openAsset(observation.asset_id); }
    if (action === "create-event") return openModal({ type: "event-create", event: {} });
    if (action === "create-story") return openModal({ type: "story-create", story: {} });
    if (action === "retry-story") {
      const ids = state.storyDraftEventIds || [];
      state.storyGenerating = true;
      state.storyError = false;
      renderShellNavigation();
      try {
        await window.sentrixApi.createStory({ title: "", content: "", event_ids: ids, tags: [] });
        await refreshData({ forceRender: true });
        state.toast = "故事已重新生成";
      } catch (error) {
        state.storyError = true;
        state.toast = `故事生成失败：${error.message}`;
      } finally {
        state.storyGenerating = false;
        renderShellNavigation();
      }
      return;
    }
    if (action === "confirm-trip") { const trip = state.trips.find((item) => item.id === element.dataset.tripId); return trip && openModal({ type: "trip-confirm", trip }); }
    if (action === "reject-trip") { await window.sentrixApi.rejectTrip(element.dataset.tripId); state.toast = "已标记为非行程，原始事件和照片证据仍保留"; return refreshData(); }
    if (action === "edit-story") { const story = state.stories.find((item) => item.id === element.dataset.storyId); return openModal({ type: "story-edit", story }); }
    if (action === "delete-story") { await window.sentrixApi.deleteStory(element.dataset.storyId); state.toast = "故事草稿已删除"; return refreshData(); }
    if (action === "open-person") { openModal({ type: "loading" }, { push: true }); try { const detail = await window.sentrixApi.personEvidence(element.dataset.personId, state.scopeId); return openModal({ type: "person-evidence", detail }); } catch (error) { state.modal = null; state.toast = `无法读取人物证据：${error.message}`; return renderShellNavigation(); } }
    if (action === "open-person-profile") { openModal({ type: "loading" }, { push: true }); const detail = await window.sentrixApi.personProfile(element.dataset.personId); return openModal({ type: "person-profile", detail }); }
    if (action === "edit-person-properties") return openModal({ type: "person-property-edit", detail: state.modal.detail });
    if (action === "edit-person-name") return openModal({ type: "person-name-edit", detail: state.modal.detail });
    if (action === "confirm-person") { const person = state.persons.find((item) => item.id === element.dataset.personId) || { id: element.dataset.personId, name: "待确认人物" }; return openModal({ type: "person", person }); }
    if (action === "confirm-cluster") { const cluster = state.clusters.find((item) => item.id === element.dataset.clusterId); return openModal({ type: "cluster-confirm", cluster }); }
    if (action === "merge-cluster") { const cluster = state.clusters.find((item) => item.id === element.dataset.clusterId); return openModal({ type: "cluster-merge", cluster }); }
    if (action === "split-face") { const cluster = state.clusters.find((item) => item.id === element.dataset.clusterId); const sample = cluster?.samples?.find((item) => item.id === element.dataset.faceInstanceId); return openModal({ type: "cluster-split", cluster, sample }); }
    if (action === "reject-cluster") { try { await window.sentrixApi.rejectFaceCluster(element.dataset.clusterId); state.toast = "已删除该候选簇，原始图片保留"; return refreshData(); } catch (error) { state.toast = `删除失败：${error.message}`; renderShellNavigation(); return; } }
    if (action === "split-person") { try { await window.sentrixApi.rejectPerson(element.dataset.personId); state.toast = "已删除该候选，原始图片保留"; return refreshData(); } catch (error) { state.toast = `删除失败：${error.message}`; renderShellNavigation(); return; } }
    if (action === "delete-person") { try { await window.sentrixApi.rejectPerson(element.dataset.personId); state.toast = "已删除「不是人物」的候选，原始图片保留"; return refreshData(); } catch (error) { state.toast = `删除失败：${error.message}`; renderShellNavigation(); return; } }
    if (action === "delete-cluster") { try { await window.sentrixApi.rejectFaceCluster(element.dataset.clusterId); state.toast = "已删除「不是人物」的簇，原始图片保留"; return refreshData(); } catch (error) { state.toast = `删除失败：${error.message}`; renderShellNavigation(); return; } }
    if (action === "confirm-fact") { await window.sentrixApi.confirmFact(element.dataset.fact); state.toast = "事实已确认并生成修订记录"; return refreshData(); }
    if (action === "reject-fact") { await window.sentrixApi.rejectFact(element.dataset.fact); state.toast = "事实已驳回并保留证据记录"; return refreshData(); }
    if (action === "confirm-relationship") { await window.sentrixApi.confirmRelationship(element.dataset.relationshipId); state.toast = "关系已确认并进入语义记忆"; return refreshData(); }
    if (action === "edit-family-relation") {
      const relation = (state.modal?.graph?.relationships || []).find((item) => item.id === element.dataset.relationId);
      if (!relation) return;
      const graph = state.modal.graph;
      state.modal = null;
      return openModal({ type: "family-graph", graph, editing: relation });
    }
    if (action === "delete-family-relation") {
      try { await window.sentrixApi.retractRelationship(element.dataset.relationId); state.toast = "关系已删除，原始证据与修订记录保留"; } catch (error) { state.toast = `删除失败：${error.message}`; }
      try { const graph = await window.sentrixApi.relationships(state.scopeId, "person"); state.modal = { type: "family-graph", graph }; } catch { state.modal = null; }
      return renderShellNavigation();
    }
    if (action === "clear-relation-edit") {
      const graph = state.modal?.graph;
      state.modal = null;
      return openModal({ type: "family-graph", graph });
    }
    if (action === "invite") return openModal({ type: "invite" });
    if (action === "open-help") return openModal({ type: "help" });
    if (action === "open-qa-dashboard") {
      const benchUrl = `${window.location.protocol}//${window.location.hostname}:8771/`;
      showToast("正在启动评测服务，请稍候…");
      try {
        await fetch("/api/photobench/ensure", { method: "POST" });
      } catch (_) { /* 评测服务不可达时仍打开，让用户看到连接状态 */ }
      window.open(benchUrl, "_blank");
      return;
    }
    if (action === "command") return openModal({ type: "command" });
    if (action === "open-space") return openModal({ type: "space-manager" });
    if (action === "ask-delete-space") {
      const scopeId = element.dataset.scopeId;
      const scopeName = element.dataset.scopeName || scopeId;
      try {
        const dash = await window.sentrixApi.dashboard(scopeId);
        const stats = {
          assets: dash?.stats?.assets ?? 0,
          events: dash?.stats?.events ?? 0,
          persons: dash?.stats?.persons ?? 0,
          vectors: dash?.stats?.vectors ?? 0,
        };
        return openModal({ type: "space-delete-confirm", scopeId, scopeName, stats });
      } catch {
        return openModal({ type: "space-delete-confirm", scopeId, scopeName, stats: null });
      }
    }
    if (action === "confirm-delete-space") {
      const scopeId = element.dataset.scopeId;
      const scopeName = element.dataset.scopeName || scopeId;
      if (element.disabled) return;
      element.disabled = true;
      element.textContent = "删除中…";
      try {
        const result = await window.sentrixApi.deleteMemorySpace(scopeId);
        const r = (result && result.removed) || {};
        state.toast = `相册『${scopeName}』已删除:${r.assets || 0} 图 / ${r.events || 0} 事件 / ${r.persons || 0} 人物 / ${r.files_removed || 0} 物理文件`;
        if (state.scopeId === scopeId) {
          state.scopeId = "";
          window.localStorage?.removeItem("sentrix.scopeId");
        }
        state.modal = null;
        state.modalHistory = [];
        await refreshData({ forceRender: true });
      } catch (err) {
        state.toast = `删除失败:${err.message || err}`;
        element.disabled = false;
        element.textContent = "确认删除";
        renderShellNavigation();
      }
      return;
    }
    if (action === "open-import-picker") return openModal({ type: "import-picker" });
    if (action === "open-files") { state.modal = null; renderShellNavigation(); document.getElementById("file-input")?.click(); return; }
    if (action === "create-space") return openModal({ type: "space-create" });
    if (action === "select-space") {
      state.scopeId = element.dataset.spaceId || "";
      window.localStorage?.setItem("sentrix.scopeId", state.scopeId);
      state.modal = null;
      state.conversationId = "";
      state.searchResult = null;
      state.assistantMessages = [];
      return refreshData({ forceRender: true });
    }
    if (action === "open-folder") {
      if (element?.classList.contains("top-user")) return openModal({ type: "space-manager" });
      state.modal = null;
      renderShellNavigation();
      document.getElementById("folder-input")?.click();
      return;
    }
    if (action === "toggle-sort") { state.assetSort = state.assetSort === "newest" ? "oldest" : "newest"; renderView(); return; }
    if (action === "toggle-entity-type") { const type = element.dataset.entityType; state.expandedEntityTypes[type] = !state.expandedEntityTypes[type]; renderView(); return; }
    if (action === "continue-assistant") { state.query = element.dataset.query || ""; return submitSearch(null, element.dataset.entityId || ""); }
    if (action === "accept-proactive") return submitProactiveOutcome(element, "accepted");
    if (action === "dismiss-proactive") return submitProactiveOutcome(element, "dismissed");
    if (action === "disable-proactive") return submitProactiveOutcome(element, "disabled");
    if (action === "derive-entity-merge-candidates") { await window.sentrixApi.deriveEntityMergeCandidates(state.scopeId); state.toast = "已生成待审核的语义归并候选，实体尚未合并"; return refreshData(); }
    if (action === "review-entity-merge-candidate") { const candidate = state.entityMergeCandidates.find((item) => item.id === element.dataset.candidateId); return candidate && openModal({ type: "entity-merge-confirm", candidate }); }
    if (action === "reject-entity-merge-candidate") { await window.sentrixApi.rejectEntityMergeCandidate(element.dataset.candidateId); state.toast = "已保留原有实体，不会再次显示同一归并候选"; return refreshData(); }
    if (action === "reload") return refreshData();
    if (action === "clear-event-date") { state.eventDate = ""; renderView(); return; }
    if (action === "recheck") { await fetch("/api/maintenance/recheck", { method: "POST" }); state.toast = "已提交失败任务重试"; return refreshData(); }
    if (action === "relationship-graph") { openModal({ type: "loading" }, { push: true }); try { const graph = await window.sentrixApi.relationships(state.scopeId, "person"); return openModal({ type: "family-graph", graph }); } catch (error) { state.modal = null; state.toast = `无法读取家庭关系：${error.message}`; return renderShellNavigation(); } }
  }

  const initialHash = window.location.hash.replace(/^#\/?/, "");
  if (initialHash && (navItems.some((item) => item.id === initialHash) || initialHash === "settings")) {
    state.view = initialHash;
  }
  shell();
  document.addEventListener("change", (event) => {
    const select = event.target?.closest?.('[data-action="switch-vlm"]');
    if (select) handleModelProfileChange(select);
  });
  refreshData();
  window.addEventListener("hashchange", () => {
    const hashView = window.location.hash.replace(/^#\/?/, "");
    if (hashView && hashView !== state.view && (navItems.some((item) => item.id === hashView) || hashView === "settings")) {
      state.view = hashView;
      state.modal = null;
      state.modalHistory = [];
      renderShellNavigation();
    }
  });
  window.setInterval(() => {
    if (document.visibilityState === "hidden" || isUserEditing()) return;
    refreshData({ silent: true });
  }, 5000);
})();
