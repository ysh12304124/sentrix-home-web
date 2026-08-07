(function () {
  const app = document.getElementById("app");
  const state = {
    view: "overview",
    query: "",
    conversationId: "",
    searchResult: null,
    assistantMessages: [],
    searchLoading: false,
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
    knowledge: { profiles: [], claims: [] },
    clusters: [],
    relationships: [],
    entityMergeCandidates: [],
    stories: [],
    trips: [],
    health: null,
    modal: null,
    eventFilter: "all",
    assetFilter: "all",
    assetSort: "newest",
    personFilter: "all",
    saving: false,
    expandedEntityTypes: {},
  };

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
    return status || "排队中";
  }

  function stats() {
    return state.dashboard?.stats || { assets: 0, observations: 0, events: 0, facts: 0, persons: 0 };
  }

  function filteredEvents() {
    const source = state.events.map(eventViewModel);
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
    app.innerHTML = `<aside class="sidebar"><div class="brand-lockup"><span class="brand-mark">S</span><div><strong>Sentrix</strong><small>Home Memory</small></div></div><label class="space-switcher"><span class="avatar tiny">S</span><span><b>当前相册</b><select id="space-select" aria-label="切换全部相册或独立相册">${spaceOptions}</select><small>${escapeHtml(activeScopeLabel)}</small></span></label><div class="side-label">家庭记忆</div><nav class="main-nav">${navItems.map((item) => `<button class="nav-item ${state.view === item.id ? "active" : ""}" data-view="${item.id}">${icon(item.icon)}<span>${item.label}</span></button>`).join("")}</nav><div class="side-label lower">空间与系统</div><nav class="main-nav"><button class="nav-item ${state.view === "settings" ? "active" : ""}" data-view="settings">${icon("◌")}<span>设备与隐私</span></button><button class="nav-item" data-action="open-help">${icon("?")}<span>使用帮助</span></button></nav><div class="sidebar-footer"><div class="local-pulse"><i></i><span>${state.backendError ? "本地服务不可用" : "本地 AI 正常运行"}</span></div><small>Sentrix Home · 0.2.0</small></div></aside><main class="main-content"><header class="topbar"><div class="breadcrumbs"><span>Sentrix Home</span>${state.view !== "overview" ? `<b>/</b><strong>${escapeHtml(navItems.find((item) => item.id === state.view)?.label || "设备与隐私")}</strong>` : ""}</div><div class="top-actions"><button class="icon-button" data-action="command" aria-label="打开命令搜索">⌘</button><button class="top-user" data-action="open-space"><span class="avatar tiny">S</span><span>${escapeHtml(activeSpace?.name || "全部相册")}</span>${icon("⌄", "muted")}</button></div></header><div id="view-root" class="view-root"></div></main><div id="toast-root" aria-live="polite"></div><div id="modal-root"></div>`;
    renderView();
  }

  function overview() {
    const count = stats();
    const events = state.events.slice(0, 3).map(eventViewModel);
    return `${pageHeader("家庭记忆 / 真实数据", "把家里的记忆，重新放在一起。", "这里展示已经导入并完成处理的本地资料。没有资料时，Sentrix 不会用示例内容填充。", `<button class="button primary" data-view="imports">${icon("＋")}导入资料</button>`)}${state.backendError ? `<div class="error-banner">${escapeHtml(state.backendError)}</div>` : ""}<div class="album-context"><span class="section-kicker">当前数据范围</span><strong>${escapeHtml(albumLabel(state.scopeId))}</strong><small>${state.scopeId ? "当前正在查看单个相册" : "当前正在查看 album1、album2、album3 的全部优化内容"}</small></div><section class="overview-search"><div><p class="section-kicker">问 Sentrix</p><h2>你想找回哪一段记忆？</h2><p>从人物、时间、地点、物体或一句描述开始，答案会带回原始证据。</p></div>${searchBar()}</section><section class="stats-grid"><article class="stat-card"><span>已整理内容</span><strong>${count.assets}</strong><small>本地 Asset</small></article><article class="stat-card"><span>已形成事件</span><strong>${count.events}</strong><small>可回到 Observation</small></article><article class="stat-card"><span>待确认事实</span><strong>${state.dashboard?.pendingFacts ?? 0}</strong><small>版本维护队列</small></article><article class="stat-card accent"><span>本地 AI 状态</span><strong>${state.health?.status === "ok" ? "正常" : "未知"}</strong><small>${escapeHtml(state.health?.models?.gamma4_12B?.name || "等待服务")}</small></article></section><section class="content-section"><div class="section-head"><div><p class="section-kicker">三类记忆</p><h2>同一家庭，不同的记忆入口</h2></div><button class="text-button" data-view="settings">查看系统状态 ${icon("→")}</button></div><div class="memory-grid"><article class="memory-card episodic-card"><div class="card-top">${memoryPill("episodic", "事件记忆")}<span class="card-index">01</span></div><h3>把分散的资料聚成共同经历</h3><p>图片、音频和文本共同参与人物、时间、地点与事件整理。</p><div class="card-metric"><strong>${count.events}</strong><span>个已建立事件</span></div></article><article class="memory-card semantic-card"><div class="card-top">${memoryPill("semantic", "语义记忆")}<span class="card-index">02</span></div><h3>让事实持续生长且保留修订</h3><p>每条事实都保留来源、置信度和人工确认历史。</p><div class="card-metric"><strong>${count.facts}</strong><span>条本地事实</span></div></article><article class="memory-card visual-card"><div class="card-top">${memoryPill("visual", "视频编码记忆", "接口预留")}<span class="card-index">03</span></div><h3>视频先归档，编码接口独立接入</h3><p>视频不会在第一版生成动作、片段或向量记忆。</p><div class="reserved-line">${icon("◌")} video_memory_adapter <span>未启用</span></div></article></div></section><section class="content-section two-column"><div><div class="section-head"><div><p class="section-kicker">最近事件</p><h2>家里的时间线</h2></div><button class="text-button" data-view="timeline">查看全部 ${icon("→")}</button></div>${events.length ? `<div class="event-list">${events.map(eventRow).join("")}</div>` : emptyState("还没有事件", "导入图片、音频或文本后，处理完成的 Observation 会在这里形成事件。", `<button class="button small primary" data-view="imports">${icon("＋")}导入第一份资料</button>`)}</div><div><div class="section-head"><div><p class="section-kicker">需要你的确认</p><h2>让记忆更准确</h2></div><button class="text-button" data-view="settings">查看事实 ${icon("→")}</button></div>${(state.dashboard?.pendingFacts || 0) ? `<div class="review-panel"><div class="review-face-pair"><span class="avatar large gray">?</span></div><div><strong>${state.dashboard.pendingFacts} 条事实等待确认</strong><p>确认或驳回前，原始 Observation 会一直保留。</p></div><div class="review-actions"><button class="button small primary" data-view="settings">处理</button></div></div>` : emptyState("目前没有待确认事实", "新资料产生矛盾信息时，会进入版本维护队列。")}</div></section>`;
  }

  function evidenceCard(evidence) {
    const sourceAction = evidence.kind === "observation" && evidence.asset_id ? `data-action="open-asset" data-asset-id="${escapeHtml(evidence.asset_id)}"` : evidence.event_id ? `data-action="open-event" data-event-id="${escapeHtml(evidence.event_id)}"` : evidence.kind === "fact" && evidence.evidence_ids?.[0] ? `data-action="open-observation" data-observation-id="${escapeHtml(evidence.evidence_ids[0])}"` : "";
    const title = evidence.kind === "fact" ? `${evidence.subject} ${evidence.predicate} ${evidence.object}` : evidence.summary || evidence.caption || "原始图片证据";
    const text = evidence.kind === "observation" ? evidence.caption || evidence.transcript || "无文字摘要" : evidence.summary || evidence.status || "";
    const media = evidence.kind === "observation" && evidence.asset_id ? `<button class="evidence-media" data-action="open-asset" data-asset-id="${escapeHtml(evidence.asset_id)}" aria-label="打开原始证据">${assetThumb({ id: evidence.asset_id, media_type: evidence.media_type || "image" }, true)}</button>` : "";
    const assetAction = evidence.kind === "asset" && evidence.id ? `data-action="open-asset" data-asset-id="${escapeHtml(evidence.id)}"` : "";
    const main = sourceAction || assetAction ? `<button class="evidence-main" ${sourceAction || assetAction}><strong>${escapeHtml(title)}</strong><p>${escapeHtml(text)}</p><small>${escapeHtml(evidence.captured_at || evidence.place || evidence.media_type || "证据记录")}</small></button>` : `<div class="evidence-main static"><strong>${escapeHtml(title)}</strong><p>${escapeHtml(text)}</p><small>${escapeHtml(evidence.captured_at || evidence.place || evidence.media_type || "证据记录")}</small></div>`;
    return `<article class="evidence-card"><div class="evidence-head"><span class="evidence-kind">${evidence.kind === "observation" ? "图片观察" : evidence.kind === "asset" ? "原始资料" : evidence.kind === "fact" ? "人物事实" : "记忆证据"}</span></div>${media}${main}${evidence.raw ? `<details><summary>查看模型原始 JSON</summary><pre>${escapeHtml(JSON.stringify(evidence.raw, null, 2))}</pre></details>` : ""}</article>`;
  }

  function evidenceLayer(title, values) {
    if (!values?.length) return "";
    return `<section class="evidence-layer"><div class="section-head"><div><p class="section-kicker">${escapeHtml(title)}</p><h3>${values.length} 项</h3></div></div><div class="evidence-list">${values.slice(0, 12).map(evidenceCard).join("")}</div></section>`;
  }

  function imageResults(result) {
    const images = result?.image_results || [];
    if (!images.length) return "";
    return `<section class="evidence-layer image-results"><div class="section-head"><div><p class="section-kicker">相关原始图片</p><h3>${images.length} 张已通过相关度门槛的证据</h3></div></div><div class="image-result-grid">${images.map((item) => `<button class="image-result" data-action="open-asset" data-asset-id="${escapeHtml(item.asset_id)}"><img src="${escapeHtml(item.media_url)}" alt="原始图片证据" loading="lazy" /><span><strong>原始图片证据</strong><small>${escapeHtml(item.captured_at || item.caption || "可回看的原始证据")}</small></span></button>`).join("")}</div></section>`;
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
    const trace = result.retrievalTrace || result.retrieval_trace || [];
    const modelEvidence = result.modelEvidence || [];
    return `<details class="algorithm-evidence"><summary>算法判断依据</summary><div class="algorithm-evidence-body"><p>回答只使用本地语义、事件和原始观察；结构化命中时会跳过向量检索。</p><dl>${trace.map((item) => `<div><dt>${escapeHtml(traceLabel(item))}</dt><dd>${escapeHtml(item.status || "complete")} · ${escapeHtml(traceDetail(item))}</dd></div>`).join("")}</dl><p>模型引用校验：${modelEvidence.length} 项候选引用，未通过证据 ID 校验时自动降级为本地证据回答。</p></div></details>`;
  }

  function toolTrace(result) {
    const trace = result.toolTrace || result.tool_trace || [];
    if (!trace.length) return "";
    return `<details class="algorithm-evidence"><summary>本轮判断与工具</summary><div class="algorithm-evidence-body"><dl>${trace.map((item) => `<div><dt>${escapeHtml(item.tool || "memory_tool")}</dt><dd>${escapeHtml(item.permission || "read")} · ${escapeHtml(item.status || "complete")}${item.reason ? ` · ${escapeHtml(item.reason)}` : ""}</dd></div>`).join("")}</dl></div></details>`;
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

  function assistantEvidence(result) {
    const layers = result.evidence_layers || {};
    const presentation = result.evidence_presentation || {};
    const primary = [...(layers.events || []), ...(layers.observations || []), ...(layers.claims || [])].sort((left, right) => (right.relevance || 0) - (left.relevance || 0));
    const candidates = result.clarification_candidates || [];
    const evidence = primary.length ? evidenceLayer("本次依据（按相关度）", primary.slice(0, 6)) : "";
    const gaps = layers.gaps || [];
    const gapContent = gaps.length ? `<section class="evidence-gap"><div class="section-head"><div><p class="section-kicker">证据缺口</p><h3>当前没有足够的原始依据</h3></div></div><p>${escapeHtml(gaps[0].reason || "请补充人物、地点、日期或其他线索。")}</p></section>` : "";
    const followups = candidates.length ? `<div class="assistant-followups"><p>请选择你指的人物：</p>${candidates.map((item) => `<button class="assistant-identity-choice" data-action="continue-assistant" data-query="${escapeHtml(item.name)}" data-entity-id="${escapeHtml(item.id)}">${item.preview_asset_id ? `<img src="/api/assets/${encodeURIComponent(item.preview_asset_id)}/file" alt="" loading="lazy" />` : `<span class="assistant-choice-placeholder">${escapeHtml((item.name || "?").slice(0, 1))}</span>`}<span><strong>${escapeHtml(item.name || "已确认成员")}</strong><small>${escapeHtml(item.family_role || "已确认人物")} · ${item.evidence_count || 0} 条依据</small></span></button>`).join("")}</div>` : "";
    const ordered = result.evidence_order || [];
    const order = ordered.length ? `<details class="algorithm-evidence"><summary>证据顺序与可信度</summary><div class="algorithm-evidence-body"><dl>${ordered.map((item, index) => `<div><dt>${String(index + 1).padStart(2, "0")} · ${escapeHtml(item.source_level)}</dt><dd>${escapeHtml(item.time || "时间未标注")} · 可信度 ${Math.round((item.confidence || 0) * 100)}%</dd></div>`).join("")}</dl></div></details>` : "";
    const directEvidence = Boolean(result.original_evidence_requested || presentation.direct_original_evidence);
    const requiresEvidence = result.intent !== "chat" && result.memory_used !== false && presentation.required !== false;
    const directOriginal = directEvidence ? `<section class="assistant-original-evidence"><div class="section-head"><div><p class="section-kicker">直接查看原始证据</p><h3>与本次回答相关的原始资料</h3></div></div>${imageResults(result) || evidence || gapContent}</section>` : "";
    const optionalImages = directEvidence ? "" : imageResults(result);
    const basis = requiresEvidence ? `<details class="assistant-basis"${result.evidence_status === "gap" ? " open" : ""}><summary>查看这次回答的依据</summary><div class="assistant-basis-body">${claimEvidence(result)}${optionalImages}${evidence}${gapContent}${order}${toolTrace(result)}${algorithmEvidence(result)}</div></details>` : "";
    return `${followups}${proactiveRecall(result)}${directOriginal}${basis}`;
  }

  function assistantMessage(message) {
    if (message.role === "user") return `<article class="assistant-message user"><div class="assistant-bubble"><p>${escapeHtml(message.text)}</p></div></article>`;
    const result = message.result || {};
    const status = result.insufficient_evidence ? "需要补充线索" : `${Math.round((result.confidence || 0) * 100)}% 证据置信度`;
    const plan = result.dialogue_plan || {};
    const agentPlan = result.agent_plan || {};
    const mode = plan.mode === "contextual_follow_up" ? "沿用上一段记忆" : plan.style === "narrative" ? "回忆叙事" : plan.style === "clarifying" ? "等待补充线索" : "事实回答";
    return `<article class="assistant-message steward"><div class="assistant-ident"><span class="assistant-mark">S</span><span>家庭助手</span><small>${escapeHtml(status)}</small></div><div class="assistant-bubble"><p>${assistantAnswer(result) || "我在。"}</p>${assistantEvidence(result)}</div></article>`;
  }

  function searchView() {
    const messages = state.assistantMessages;
    const introduction = `<section class="assistant-intro"><div><span class="assistant-mark">S</span><p class="section-kicker">FAMILY COMPANION</p><h2>家庭助手</h2><p>我记得这座家庭相册中整理出的成员、共同经历与生活细节。我们可以自然聊聊；谈到家里的往事时，我会在需要时调取记忆，并保留可查看的依据。</p></div><div class="assistant-scope"><span>当前相册</span><strong>${escapeHtml(albumLabel(state.scopeId))}</strong></div></section>`;
    const suggestions = `<div class="assistant-suggestions"><button data-query="介绍一下明哥">介绍一位家人</button><button data-query="明哥的时间线">查看人物时间线</button><button data-query="推荐一些明哥的回忆">推荐有依据的回忆</button></div>`;
    return `${pageHeader("家庭对话", "家庭助手", "一个中性的本地数字人，带着这座家庭相册形成的长期记忆。")}${introduction}<section class="assistant-conversation">${messages.length ? messages.map(assistantMessage).join("") : `<div class="assistant-welcome"><p>今天想聊什么？</p>${suggestions}</div>`}${state.searchLoading ? `<article class="assistant-message steward loading"><div class="assistant-ident"><span class="assistant-mark">S</span><span>家庭助手</span></div><div class="assistant-bubble"><p>我在想一想，也在整理这段家庭记忆。</p></div></article>` : ""}</section>${searchBar("和家庭助手聊聊，或问起家里的任何一段经历…")}`;
  }

  function timelineView() {
    const events = filteredEvents();
    return `${pageHeader("记忆组织 / 事件", "家里的时间线，不只是文件列表。", `当前范围：${albumLabel(state.scopeId)}。每个事件都能回到原始 Observation、文件和模型输出；人工修改会增加 revision。`, `<button class="button ghost" data-action="create-event">${icon("＋")}新建事件</button>`)}<div class="filter-row"><button class="filter-chip ${state.eventFilter === "all" ? "active" : ""}" data-event-filter="all">全部事件</button><button class="filter-chip ${state.eventFilter === "people" ? "active" : ""}" data-event-filter="people">有人物</button><button class="filter-chip ${state.eventFilter === "place" ? "active" : ""}" data-event-filter="place">有地点</button><span class="filter-spacer"></span><button class="icon-button bordered" data-action="reload" aria-label="刷新时间线">↻</button></div><section class="timeline-layout"><div class="timeline-main">${events.length ? events.map((event) => `<article class="timeline-event"><div class="timeline-marker ${event.tone}"></div><div class="timeline-date">${event.date}<small>${escapeHtml(event.typeLabel)}</small></div><div class="timeline-event-body"><div class="event-cover ${event.tone}">${event.coverAssetId ? `<img src="/api/assets/${encodeURIComponent(event.coverAssetId)}/file" alt="${escapeHtml(event.title)}的事件证据" loading="lazy" />` : ""}<div class="event-cover-label">${albumBadge(event.scope_id)}<span>${escapeHtml(event.title)}</span></div><b>${escapeHtml(event.countLabel)}</b></div><div class="timeline-event-copy"><div class="card-top"><span class="event-kind">${escapeHtml(event.status || "active")}</span><span class="confidence-label">revision ${event.revision || 1}</span></div><h2>${escapeHtml(event.title)}</h2><p>${escapeHtml(event.summary || "暂无事件摘要")}</p><div class="event-facts"><span>${icon("◎")} ${escapeHtml(event.placeLabel)}</span><span>${icon("◷")} ${escapeHtml(event.countLabel)}</span><span>${icon("↗")} 可回到原始证据</span></div><div class="event-actions"><button class="button small ghost" data-action="open-event" data-event-id="${escapeHtml(event.id)}">查看证据</button><button class="text-button" data-action="edit-event" data-event-id="${escapeHtml(event.id)}">修正事件 ${icon("→")}</button></div></div></div></article>`).join("") : emptyState("还没有事件", "导入并处理资料后，时间线会由真实 Observation 自动生成。", `<button class="button small primary" data-view="imports">${icon("＋")}导入资料</button>`)}</div><aside class="side-inspector"><p class="section-kicker">事件记忆</p><h2>事件是可维护的记忆单元</h2><p>新资料先成为 Observation，再根据时间、地点和活动合并到事件。所有变更保留证据。</p><div class="mini-flow"><span>Observation</span><b>→</b><span>Event</span><b>→</b><span>Revision</span></div><div class="inspector-note">视频事件提取 <strong>接口预留</strong><small>第一版只保存视频原始 Asset。</small></div></aside></section>`;
  }

  function peopleView() {
    const people = state.persons.filter((person) => state.personFilter === "all" || !person.confirmed);
    const pending = state.persons.filter((person) => !person.confirmed);
    return `${pageHeader("家庭治理 / 人物", "先确认人物，再让关系长出来。", "人脸模型只生成候选。单张样本会明确标注，仍可查看原图后确认或驳回。", `<button class="button primary" data-action="invite">${icon("＋")}生成邀请</button>`)}<div class="people-toolbar"><div class="segmented"><button class="${state.personFilter === "all" ? "active" : ""}" data-person-filter="all">全部人物</button><button class="${state.personFilter === "pending" ? "active" : ""}" data-person-filter="pending">待确认 <b>${pending.length}</b></button><button data-action="relationship-graph">关系图</button></div><button class="button ghost" data-action="reload">${icon("↻")}刷新</button></div><section class="people-grid">${people.length ? people.map((person, index) => { const name = person.confirmed ? (person.display_name || person.name) : `待命名成员 ${index + 1}`; const caution = !person.confirmed && person.single_sample ? `<small>单张样本，需谨慎确认</small>` : ""; return `<article class="person-card ${person.confirmed ? "" : "needs-review"}"><div class="person-head">${faceAvatar(person.avatar_face_instance_id, name, person.confirmed ? "green" : "gray")}${person.confirmed ? `<span class="confirmed">✓ 已确认</span>` : `<span class="needs-label">待确认</span>`}</div><h2>${escapeHtml(name)}</h2><p>${escapeHtml(person.status)} · 置信度 ${Math.round((person.confidence || 0) * 100)}%</p>${caution}<div class="person-stats"><span><strong>${person.mention_count || 0}</strong> 次出现</span><span><strong>${person.cluster_count || 0}</strong> 个人物簇</span></div><div class="person-actions"><button class="button small ghost" data-action="open-person" data-person-id="${escapeHtml(person.id)}">查看证据</button>${person.confirmed ? "" : `<button class="button small primary" data-action="confirm-person" data-person-id="${escapeHtml(person.id)}">确认</button><button class="button small ghost" data-action="split-person" data-person-id="${escapeHtml(person.id)}">驳回候选</button>`}</div></article>`; }).join("") : emptyState("还没有人物候选", "导入包含人脸的图片后，InsightFace 会生成待确认候选；不会凭空创建家庭成员。", `<button class="button small primary" data-view="imports">${icon("＋")}导入图片</button>`)}</section>`;
  }

  function knowledgeView() {
    const pendingClusters = state.clusters.filter((cluster) => cluster.status === "pending" && cluster.reviewable !== false);
    const visibleEntities = state.entities.filter((entity) => entity.status === "confirmed" || entity.reviewable !== false);
    const confirmed = visibleEntities.filter((entity) => entity.status === "confirmed");
    const pending = visibleEntities.filter((entity) => entity.status === "pending");
    const entityCard = (entity) => `<button class="entity-card" data-action="open-entity" data-entity-id="${escapeHtml(entity.id)}"><div class="entity-card-head">${faceAvatar(entity.avatar_face_instance_id, entity.canonical_name, entity.status === "confirmed" ? "green" : "gray")}<span class="needs-label">${escapeHtml(entity.status)}</span></div><h2>${escapeHtml(entity.canonical_name)}</h2><p>${escapeHtml(entity.family_role || "未确认家庭角色")}</p><div class="entity-stats"><span><strong>${entity.mention_count || 0}</strong> 次出现</span><span><strong>${entity.cluster_count || 0}</strong> 个人物簇</span><span><strong>${entity.relationship_count || 0}</strong> 条关系</span></div></button>`;
    const clusterCard = (cluster) => `<article class="cluster-card"><div class="cluster-head"><div><span class="section-kicker">人物识别</span><strong>候选人物簇</strong></div><span class="needs-label">${cluster.member_count || cluster.samples?.length || 0} 张样本</span></div><div class="cluster-samples">${(cluster.samples || []).slice(0, 6).map((sample) => `<div class="cluster-sample"><button data-action="open-asset" data-asset-id="${escapeHtml(sample.asset_id)}">${faceAvatar(sample.id, "人脸样本")}</button><button class="sample-split" data-action="split-face" data-cluster-id="${escapeHtml(cluster.id)}" data-face-instance-id="${escapeHtml(sample.id)}" aria-label="从人物簇拆出样本">×</button></div>`).join("")}</div><p>聚类置信度 ${Math.round((cluster.confidence || 0) * 100)}% · ${cluster.entity_status === "confirmed" ? "已绑定人物" : "待确认身份"}</p><div class="person-actions"><button class="button small primary" data-action="confirm-cluster" data-cluster-id="${escapeHtml(cluster.id)}">确认实体</button><button class="button small ghost" data-action="merge-cluster" data-cluster-id="${escapeHtml(cluster.id)}">合并到其他簇</button><button class="button small ghost" data-action="reject-cluster" data-cluster-id="${escapeHtml(cluster.id)}">驳回簇</button></div></article>`;
    return `${pageHeader("语义记忆 / 实体治理", "看见家庭记忆里稳定存在的实体。", "人物簇先由 buffalo_l 聚类，再由你确认名称和角色；确认结果会回写观察、事件、人物画像和关系图。", `<button class="button ghost" data-action="reload">${icon("↻")}刷新状态</button>`)}<section class="knowledge-summary"><article><strong>${confirmed.length}</strong><span>已确认实体</span></article><article><strong>${pending.length + pendingClusters.length}</strong><span>待维护候选</span></article><article><strong>${state.relationships.filter((item) => item.status === "active").length}</strong><span>已确认关系</span></article><article><strong>${state.relationships.filter((item) => item.status === "pending").length}</strong><span>待确认关系</span></article></section><section class="content-section"><div class="section-head"><div><p class="section-kicker">实体总览</p><h2>跨事件维护的家庭实体</h2></div><button class="text-button" data-action="relationship-graph">查看关系图 ${icon("→")}</button></div><div class="entity-grid">${visibleEntities.length ? visibleEntities.map(entityCard).join("") : emptyState("还没有实体", "完成一轮图片导入和人脸聚类后，实体会出现在这里。")}</div></section><section class="content-section"><div class="section-head"><div><p class="section-kicker">人物聚类 / 待确认</p><h2>先确认身份，再进入长期记忆</h2></div><span class="result-count">${pendingClusters.length} 簇</span></div><div class="cluster-grid">${pendingClusters.length ? pendingClusters.map(clusterCard).join("") : emptyState("没有待确认人物簇", "新的图片出现人物后，系统会把相似人脸合并为簇并保留样本证据。")}</div></section>`;
  }

  function libraryView() {
    const assets = filteredAssets();
    const albumCounts = state.assets.reduce((result, asset) => { const key = asset.source_album_id || asset.scope_id || "unknown"; result[key] = (result[key] || 0) + 1; return result; }, {});
    const albumSummary = Object.entries(albumCounts).map(([id, count]) => `<span>${albumBadge(id)} ${count} 张</span>`).join("");
    return `${pageHeader("家庭资料 / 全部内容", "所有资料，都有回到记忆的路径。", `当前范围：${albumLabel(state.scopeId)}。这里展示真实导入的原始 Asset。点击任意条目可查看文件、处理状态、Observation 和模型原始输出。`, `<button class="button primary" data-view="imports">${icon("↓")}导入资料</button>`)}<div class="album-summary-row"><strong>相册内容</strong>${albumSummary || `<span>${albumBadge(state.scopeId)} 暂无资料</span>`}</div><div class="library-summary"><div><strong>${state.assets.length}</strong><span>当前范围资产</span></div><div><strong>${state.assets.filter((a) => a.media_type === "image").length}</strong><span>图片</span></div><div><strong>${state.assets.filter((a) => ["audio", "text"].includes(a.media_type)).length}</strong><span>音频 / 文本</span></div><div class="muted-stat"><strong>${state.assets.filter((a) => a.media_type === "video").length}</strong><span>视频 · 接口预留</span></div></div><div class="filter-row">${[["all", "全部"], ["image", "图片"], ["audio", "音频"], ["text", "文本"], ["video", "视频"]].map(([key, label]) => `<button class="filter-chip ${state.assetFilter === key ? "active" : ""}" data-asset-filter="${key}">${label}</button>`).join("")}<span class="filter-spacer"></span><button class="sort-label" data-action="toggle-sort">按${state.assetSort === "newest" ? "最近" : "最早"}导入 ↕</button></div><section class="asset-grid library-grid">${assets.length ? assets.map(assetCard).join("") : emptyState("没有匹配的资料", "调整筛选条件或导入一份新的原始文件。", `<button class="button small primary" data-view="imports">${icon("＋")}导入资料</button>`)}</section>`;
  }

  function storiesView() {
    const _evtTimes = ((state.stories[0]||{}).event_ids||[]).map(id => (state.events||[]).find(x=>x.id===id)).filter(Boolean).flatMap(e => (e.observations||[]).map(o => o.captured_at)).filter(Boolean).sort();
    let _spanText = "";
    if (_evtTimes.length >= 2) { const _d = (new Date(_evtTimes[_evtTimes.length-1]) - new Date(_evtTimes[0])) / 86400000; if (_d > 180) _spanText = "跨越 " + (_d/365).toFixed(1) + " 年"; else if (_d > 30) _spanText = "跨越 " + Math.ceil(_d/30) + " 个月"; else _spanText = "跨越 " + Math.ceil(_d) + " 天"; }
    const _places = {}; const _people = {};
    ((state.stories[0]||{}).event_ids||[]).forEach(id => { const ev = (state.events||[]).find(x=>x.id===id); if (!ev) return; (ev.place_names||[]).forEach(p => _places[p] = (_places[p]||0)+1); (ev.companion_ids||[]).forEach(pid => { const ent = (state.entities||[]).find(e=>e.id===pid); const nm = ent?.canonical_name || "未知"; _people[nm] = (_people[nm]||0)+1; }); });
    return `${pageHeader("家庭表达 / 故事工作室", "把真实事件整理成家人愿意一起看的故事。", "故事只引用你选择的事件和证据；标题、章节和内容保存为本地草稿。", `<button class="button primary" data-action="create-story">${icon("＋")}新建故事</button>`)}<section class="story-layout">${state.stories.length ? `<div class="story-canvas"><div class="story-canvas-label">选择一个故事查看</div><div class="story-title">${escapeHtml(state.stories[0].title)}</div><div class="story-meta" style="font-size:12px;color:#9A9486;margin:4px 0 12px;letter-spacing:0.5px;">${(state.stories[0].event_ids||[]).length} 个事件 · ${(state.stories[0].event_ids||[]).reduce((s,eid)=>{const ev=(state.events||[]).find(x=>x.id===eid);return s+((ev?.observations||[]).length||((ev?.asset_ids||[]).length)||0);},0)} 张照片 · ${(state.stories[0].outline||[]).length} 个章节${_spanText ? " · " + _spanText : ""}</div>${(Object.keys(_places).length||Object.keys(_people).length) ? `<div class="story-keywords" style="margin:8px 0 12px;display:flex;flex-wrap:wrap;gap:6px;">${Object.entries(_places).sort((a,b)=>b[1]-a[1]).slice(0,5).map(([k,v])=>`<span style="font-size:11px;padding:2px 8px;background:rgba(94,122,24,0.12);color:#5E7A18;border-radius:10px;">${escapeHtml(k)} ${v}</span>`).join("")}${Object.entries(_people).sort((a,b)=>b[1]-a[1]).slice(0,5).map(([k,v])=>`<span style="font-size:11px;padding:2px 8px;background:rgba(199,112,14,0.12);color:#C7700E;border-radius:10px;">${escapeHtml(k)} ${v}</span>`).join("")}</div>` : ""}${((state.stories[0].tags||[]).length) ? `<div class="story-user-tags" style="margin:4px 0 12px;display:flex;flex-wrap:wrap;gap:6px;">${(state.stories[0].tags||[]).map(t=>`<span style="font-size:11px;padding:2px 8px;background:rgba(28,29,25,0.08);color:#1c1d19;border-radius:10px;">${escapeHtml(t)}</span>`).join("")}</div>` : ""}<div class="story-caption">${escapeHtml(state.stories[0].content || "这个故事还没有内容。")}</div></div><aside class="story-editor"><div class="panel-title"><span>STORY DRAFTS</span><span class="draft-badge">${state.stories.length} 个</span></div>${state.stories.map((story) => `<div class="chapter"><button class="chapter-open" data-action="edit-story" data-story-id="${escapeHtml(story.id)}"><span>●</span><strong>${escapeHtml(story.title)}</strong>${icon("→", "muted")}</button><button class="icon-button bordered" data-action="delete-story" data-story-id="${escapeHtml(story.id)}" aria-label="删除故事">×</button></div>`).join("")}<button class="button primary full" data-action="create-story">新建本地草稿 ${icon("→")}</button></aside></section>` : `<section class="empty-search"><div class="empty-symbol">▤</div><h2>还没有故事草稿</h2><p>先导入并形成事件，再选择真实事件生成故事草稿。</p><button class="button primary" data-action="create-story">${icon("＋")}创建空白故事</button></section>`}`;
  }

  function importsView() {
    const assets = state.assets.filter((asset) => ["queued", "processing", "semantic_enriching", "failed", "video-extraction-reserved"].includes(asset.status));
      return `${pageHeader("资料入口 / 本地导入", "把资料带回家，剩下的交给本地 AI。", "上传后会创建稳定 Asset ID，并在后台生成 Observation、Event 和 Fact；视频只建立原始资产。", `<button class="button ghost" data-action="open-folder">${icon("▦")}选择资料</button>`)}<section class="import-layout"><div><label class="dropzone" for="file-input"><input id="file-input" type="file" multiple accept="image/*,audio/*,text/*,video/*" /><span class="drop-icon">↓</span><strong>拖入资料，或点击选择文件</strong><small>支持图片、音频、文本和视频 · 原始文件不会离开本机</small><span class="button primary">选择资料</span></label><div class="import-notice"><span class="notice-mark">i</span><div><strong>原始证据不会被覆盖</strong><p>每个 Asset 都可以追溯到 Observation 和模型原始 JSON。</p></div></div></div><aside class="import-status"><div class="panel-title"><span>LOCAL PIPELINE</span><span class="live-label"><i></i>真实状态</span></div><h2>当前处理</h2>${[["接收与去重", `${state.assets.length} 个 Asset`, "done"], ["图片理解", `${state.assets.filter((a) => a.media_type === "image" && a.status === "processed").length} 个已完成 · ${state.assets.filter((a) => a.status === "semantic_enriching").length} 个语义整理中`, "done"], ["音频转写", `${state.assets.filter((a) => a.media_type === "audio").length} 个音频`, "active"], ["事件与事实", `${stats().events} 个事件 · ${stats().facts} 条事实`, "active"], ["视频编码", `${state.assets.filter((a) => a.media_type === "video").length} 个视频`, "reserved"]].map((row) => `<div class="pipeline-row"><span class="pipeline-state ${row[2]}">${row[2] === "done" ? "✓" : row[2] === "active" ? "•" : "—"}</span><div><strong>${row[0]}</strong><small>${row[1]}</small></div><em>${row[2] === "done" ? "完成" : row[2] === "active" ? "运行中" : "预留"}</em></div>`).join("")}</aside></section><section class="content-section"><div class="section-head"><div><p class="section-kicker">导入记录</p><h2>最近处理任务</h2></div><button class="text-button" data-action="reload">刷新状态 ${icon("↻")}</button></div><div class="queue-list">${assets.length ? assets.map((asset) => `<div class="queue-row"><span class="queue-type ${asset.media_type}">${escapeHtml(mediaLabel(asset.media_type).slice(0, 3))}</span><div><strong>原始${escapeHtml(mediaLabel(asset.media_type))}证据</strong><small>${formatDateTime(asset.updated_at)} · ${escapeHtml(assetStatusLabel(asset.status))}</small></div><span class="queue-status ${asset.status === "video-extraction-reserved" ? "reserved" : "queued"}">${escapeHtml(assetStatusLabel(asset.status))}</span></div>`).join("") : emptyState("没有待处理任务", "处理中的 Asset 会显示在这里。")}</div></section>`;
  }

  function settingsView() {
    const facts = state.dashboard?.facts || [];
    const pending = facts.filter((fact) => fact.status === "pending");
    return `${pageHeader("系统 / 本地状态", "你的记忆，运行在自己的家里。", "服务、模型、存储和事实修订状态都来自当前本地后端。")}${state.health ? `<section class="health-grid"><article class="health-card dark"><div class="health-title"><span>Sentrix Home</span><span class="online-pill"><i></i>在线</span></div><strong>本地服务正常</strong><p>健康接口返回正常</p><div class="health-line"><span>数据资产</span><b>${stats().assets}</b></div><div class="health-bar"><i style="width:100%"></i></div></article><article class="health-card"><div class="health-title"><span>AI MODEL ROUTER</span><span class="ready-label">READY</span></div><div class="model-row"><span>主推理</span><strong>${escapeHtml(state.health.models?.gamma4_12B?.name || "未知")}</strong><small>${escapeHtml(state.health.models?.gamma4_12B?.endpoint || "未连接")}</small></div><div class="model-row"><span>语音转写</span><strong>FunASR</strong><small>${escapeHtml(state.health.models?.asr?.name || "未连接")}</small></div><div class="model-row"><span>人物识别</span><strong>InsightFace</strong><small>${state.health.models?.face?.ready ? "已启用" : "不可用"}</small></div></article><article class="health-card"><div class="health-title"><span>MEMORY INDEX</span><span class="ready-label">LOCAL</span></div><strong>${stats().facts} <small>条事实</small></strong><p>SQLite 事实库 · 原生语义图与向量索引</p><div class="index-list"><span>${icon("●")}事件记忆 <b>${stats().events}</b></span><span>${icon("●")}观察证据 <b>${stats().observations}</b></span><span class="dim">${icon("—")}视频编码记忆 <b>预留</b></span></div></article></section>` : emptyState("正在读取本地状态", "请稍候或刷新页面。")}<section class="content-section fact-review"><div class="section-head"><div><p class="section-kicker">语义记忆 / 版本维护</p><h2>需要确认的事实</h2></div><span class="result-count">${pending.length} 条</span></div>${pending.length ? `<div class="fact-review-list">${pending.map((fact) => `<div class="fact-review-row"><div><strong>${escapeHtml(fact.subject)} ${escapeHtml(fact.predicate)} ${escapeHtml(fact.object)}</strong><small>${escapeHtml(fact.id)} · 置信度 ${Math.round((fact.confidence || 0) * 100)}% · 证据 ${(fact.evidence_ids_json || []).join(", ")}</small></div><div class="review-actions"><button class="button small primary" data-action="confirm-fact" data-fact="${escapeHtml(fact.id)}">${icon("✓")}确认</button><button class="button small ghost" data-action="reject-fact" data-fact="${escapeHtml(fact.id)}">${icon("×")}驳回</button></div></div>`).join("")}</div>` : emptyState("没有待确认事实", "冲突事实出现后会进入这里，旧版本不会被删除。")}</section><section class="content-section two-column settings-lower"><div><div class="section-head"><div><p class="section-kicker">隐私边界</p><h2>数据只在本地流动</h2></div></div><div class="privacy-list"><div><span>原始媒体</span><b>本地存储</b></div><div><span>人物特征</span><b>本地处理</b></div><div><span>原生记忆索引</span><b>本地实体与向量检索</b></div><div><span>视频编码</span><b>接口关闭</b></div></div></div><div><div class="section-head"><div><p class="section-kicker">审计入口</p><h2>可操作的系统动作</h2></div></div><div class="audit-list"><div><button class="button small ghost" data-action="reload">刷新服务状态 ${icon("↻")}</button><small>重新读取后端、模型和数据库状态</small></div><div><button class="button small ghost" data-action="recheck">重新检查失败任务 ${icon("→")}</button><small>只重试 queued 或 failed Asset</small></div><div><button class="button small ghost" data-action="open-help">查看接口与隐私说明 ${icon("?")}</button><small>当前部署边界和证据规则</small></div></div></div></section>`;
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

  function semanticKnowledgeView() {
    const people = state.persons.filter((person) => person.confirmed);
    const claims = (state.knowledge.claims || people.flatMap((person) => person.claims || [])).filter((claim) => claim.status !== "superseded");
    const groups = [["place", "地点", "地点语义"], ["object", "物品", "物品语义"], ["atmosphere", "氛围", "画面氛围"]];
    const personCards = people.map((person) => "<article class=\"entity-card\"><div class=\"entity-card-head\">" + faceAvatar(person.avatar_face_instance_id, person.display_name, "green") + "<span class=\"confirmed\">已确认</span></div><h2>" + escapeHtml(person.display_name) + "</h2><p>" + escapeHtml(person.profile?.summary_zh || person.summary || "正在从事件和证据形成画像。") + "</p><div class=\"entity-stats\"><span><strong>" + (person.event_memory || []).length + "</strong> 个事件</span><span><strong>" + claims.filter((claim) => claim.person_id === person.id).length + "</strong> 条当前声明</span></div><button class=\"button small ghost\" data-action=\"open-person-profile\" data-person-id=\"" + escapeHtml(person.id) + "\">查看画像和证据</button></article>").join("");
    const entitySection = ([type, label, description]) => {
      const entities = state.entityGroups.filter((entity) => entity.entity_type === type && entity.evidence_count > 0);
      const expanded = Boolean(state.expandedEntityTypes[type]);
      const visible = expanded ? entities : entities.slice(0, 6);
      return `<section class="content-section"><div class="section-head"><div><p class="section-kicker">${label}</p><h2>${description}</h2></div><span class="result-count">${entities.length} 项</span></div><div class="entity-grid entity-grid-collapsed">${entities.length ? visible.map(semanticGroupCard).join("") : emptyState(`尚未形成${label}实体`, "等待带有可回溯观察证据的资料。")}</div>${entities.length > 6 ? `<div class="entity-expand"><button class="button small ghost" data-action="toggle-entity-type" data-entity-type="${type}">${expanded ? "收起" : `查看全部 ${entities.length} 项`}</button></div>` : ""}</section>`;
    };
    const tripCards = state.trips.map((trip) => `<article class="trip-candidate"><span class="needs-label">${escapeHtml(trip.status)}</span><h3>${escapeHtml(trip.name)}</h3><p>${escapeHtml(formatDate(trip.time_start))} 至 ${escapeHtml(formatDate(trip.time_end))}</p><small>${(trip.place_names_json || []).map(escapeHtml).join("、") || "地点待补充"} · ${(trip.event_ids_json || []).length} 个事件 · ${(trip.evidence_ids_json || []).length} 条证据</small>${trip.status === "pending" ? `<div class="person-actions"><button class="button small primary" data-action="confirm-trip" data-trip-id="${escapeHtml(trip.id)}">命名并确认</button><button class="button small ghost" data-action="reject-trip" data-trip-id="${escapeHtml(trip.id)}">不是行程</button></div>` : `<small>类型 · ${escapeHtml(trip.trip_type || "未分类")} · revision ${trip.revision || 1}</small>`}</article>`).join("");
    return pageHeader("语义记忆 / 实体目录", "人物、地点与细节共同组成回忆。", "相近描述会自动归到同一语义实体组；成员实体、事件和照片始终保留为可追溯的组成部分。", "<button class=\"button ghost\" data-action=\"reload\">" + icon("↻") + "刷新知识</button>") + "<section class=\"knowledge-summary\"><article><strong>" + people.length + "</strong><span>已确认人物</span></article><article><strong>" + claims.length + "</strong><span>当前人物声明</span></article><article><strong>" + state.entityGroups.length + "</strong><span>语义实体组</span></article><article><strong>" + (state.dashboard?.pendingFacts || 0) + "</strong><span>待维护事实</span></article></section><section class=\"content-section\"><div class=\"section-head\"><div><p class=\"section-kicker\">人物总结</p><h2>跨事件形成的熟人档案</h2></div><span class=\"result-count\">" + people.length + " 人</span></div><div class=\"entity-grid\">" + (personCards || emptyState("还没有已确认人物", "先在人物页面确认人脸簇，语义知识才会有稳定的中心。", "<button class=\"button small primary\" data-view=\"people\">打开人物</button>")) + "</div></section><section class=\"content-section\"><div class=\"section-head\"><div><p class=\"section-kicker\">行程候选</p><h2>跨事件的长线回忆</h2></div><span class=\"result-count\">" + state.trips.length + " 项</span></div><div class=\"trip-grid\">" + (tripCards || emptyState("暂无行程候选", "只有跨日或跨地点的连续事件才会成为待确认行程。")) + "</div></section>" + groups.map(entitySection).join("");
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
      body = `<div class="modal-kicker">EVENT · ${escapeHtml(detail.event.id)}</div><h2>${escapeHtml(detail.event.title)}</h2><p class="modal-lead">${escapeHtml(detail.event.summary || "暂无摘要")}</p><div class="detail-facts"><span>时间 · ${formatDateTime(detail.event.time_start)}</span><span>地点 · ${escapeHtml(detail.event.place || "未标注")}</span><span>revision · ${detail.event.revision || 1}</span></div>${coverEvidence}<div class="section-head"><div><p class="section-kicker">事件实体</p><h3>人物、地点与画面细节</h3></div></div><div class="event-entity-list">${entityRows || emptyState("尚未投影实体", "该事件等待 Observation 实体索引。")}</div><div class="section-head"><div><p class="section-kicker">原始证据媒体</p><h3>${detail.observations.length} 条 Observation</h3></div><button class="button small ghost" data-action="edit-event" data-event-id="${escapeHtml(detail.event.id)}">修正事件</button></div><div class="evidence-list event-evidence-list">${detail.observations.length ? detail.observations.map((observation) => evidenceCard({ kind: "observation", id: observation.id, observation_id: observation.id, asset_id: observation.asset_id, file_name: observation.asset?.file_name, media_type: observation.asset?.media_type, captured_at: observation.captured_at, caption: observation.caption, transcript: observation.transcript, raw: observation.raw_json })).join("") : emptyState("没有关联 Observation", "这是一个人工创建的事件。")}</div>${detail.facts?.length ? `<div class="section-head"><div><p class="section-kicker">语义记忆</p><h3>关联事实</h3></div></div><div class="evidence-list">${detail.facts.map((fact) => evidenceCard({ kind: "fact", id: fact.id, subject: fact.subject, predicate: fact.predicate, object: fact.object, status: fact.status, evidence_ids: fact.evidence_ids_json })).join("")}</div>` : ""}`;
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
      body = `<form id="modal-form"><div class="modal-kicker">FACE CLUSTER · ${escapeHtml(cluster.id)}</div><h2>确认这个人物实体</h2><p class="modal-lead">这组样本由 153 上的 buffalo_l embedding 聚类得到。确认后，所有样本会统一绑定到同一个实体。</p><div class="cluster-samples modal-samples">${(cluster.samples || []).map((sample) => `<button type="button" data-action="open-asset" data-asset-id="${escapeHtml(sample.asset_id)}">${faceAvatar(sample.id, "人脸样本")}</button>`).join("")}</div><label>姓名或称呼<input name="name" placeholder="例如：妈妈" required /></label><label>家庭角色<select name="family_role"><option value="">暂不确认</option><option>母亲</option><option>父亲</option><option>孩子</option><option>祖父母</option><option>其他家庭成员</option></select></label><label>与已确认实体的关系（可选）<select name="relation_target"><option value="">暂不建立关系</option>${state.entities.filter((entity) => entity.status === "confirmed").map((entity) => `<option value="${escapeHtml(entity.id)}">${escapeHtml(entity.canonical_name)}</option>`).join("")}</select></label><label>关系类型（可选）<input name="relation_predicate" placeholder="例如：母亲、父亲、兄弟姐妹" /></label><div class="modal-actions"><button type="button" class="button ghost" data-action="close-modal">取消</button><button type="submit" class="button primary">确认并更新记忆</button></div></form>`;
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
      body = "<div class=\"modal-kicker\">PERSON PROFILE · " + escapeHtml(entity.id) + "</div><div class=\"profile-heading\">" + faceAvatar(entity.avatar_face_instance_id, entity.canonical_name, "green") + "<div><h2>" + escapeHtml(entity.canonical_name) + "</h2><p class=\"modal-lead\">" + escapeHtml(detail.profile?.summary_zh || entity.summary || "暂无人物画像") + "</p></div></div><div class=\"detail-facts\"><span>家庭角色 · " + escapeHtml(entity.family_role || "未确认") + "</span><span>语义声明 · " + claims.length + "</span><span>人物簇 · " + detail.clusters.length + "</span></div><div class=\"section-head\"><div><p class=\"section-kicker\">用户维护档案</p><h3>身份、关系与圈子</h3></div><button class=\"button small ghost\" data-action=\"edit-person-properties\">修正档案</button></div><div class=\"property-list\">" + (identityRows || emptyState("尚未维护身份属性", "这些字段只由你维护，模型不会覆盖。")) + "</div><div class=\"fact-review-list\">" + (claimRows || emptyState("暂无语义声明", "确认人物后，相关事件会持续维护人物画像。")) + "</div>";
    } else if (modal.type === "person-property-edit") {
      const detail = modal.detail;
      const properties = new Map((detail.properties || []).map((item) => [item.property_key, item]));
      const isSelf = Boolean(properties.get("is_self")?.value);
      const relation = properties.get("relation_to_user")?.value || "";
      const canonicalName = detail.entity?.canonical_name || "";
      const groups = Array.isArray(properties.get("groups")?.value) ? properties.get("groups").value.join("、") : "";
      body = `<form id="modal-form"><div class="modal-kicker">PERSON PROPERTY EDIT</div><h2>修正人物档案</h2><p class="modal-lead">这些是用户维护字段，会保留版本且不会被模型推断覆盖。</p><label>名字<input name="canonical_name" value="${escapeHtml(canonicalName)}" placeholder="例如：小张、妈妈" /></label><label class="property-toggle"><input type="checkbox" name="is_self" ${isSelf ? "checked" : ""} />这是相册主人</label><label>与相册主人的关系<input name="relation_to_user" value="${escapeHtml(relation)}" placeholder="例如：本人、母亲、同事" /></label><label>所属圈子<input name="groups" value="${escapeHtml(groups)}" placeholder="例如：家人、大学同学" /></label><div class="modal-actions"><button type="button" class="button ghost" data-action="open-person-profile" data-person-id="${escapeHtml(detail.entity.id)}">取消</button><button type="submit" class="button primary">保存人物档案</button></div></form>`;
    } else if (modal.type === "entity") {
      const detail = modal.detail;
      const entity = detail.entity;
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
    } else if (modal.type === "relation") {
      const relationships = modal.graph?.relationships || [];
      const candidateRows = relationships.map((item) => `<div class="fact-review-row"><div><strong>${escapeHtml(item.subject_name)} ${escapeHtml(item.predicate)} ${escapeHtml(item.object_name)}</strong><small>${escapeHtml(item.status)} · 置信度 ${Math.round((item.confidence || 0) * 100)}% · ${(item.evidence_ids_json || []).length} 条原始证据</small></div>${item.status === "pending" ? `<button class="button small primary" data-action="confirm-relationship" data-relationship-id="${escapeHtml(item.id)}">确认关系</button>` : ""}</div>`).join("");
      body = `<div class="modal-kicker">RELATIONSHIP GRAPH</div><h2>实体关系图</h2><p class="modal-lead">候选只代表共现证据，不会自动推断亲属、同事等关系。确认后才会进入长期语义记忆。</p><div class="relation-graph">${modal.graph?.nodes?.length ? modal.graph.nodes.map((node) => `<div class="relation-node"><span class="avatar ${node.status === "confirmed" ? "green" : "gray"}">${escapeHtml((node.label || "?").slice(0, 1))}</span><strong>${escapeHtml(node.label)}</strong><small>${escapeHtml(node.status)}</small>${modal.graph.edges.filter((edge) => edge.source === node.id).map((edge) => `<em>${escapeHtml(edge.label)} · ${escapeHtml(edge.status)}</em>`).join("")}</div>`).join("") : emptyState("没有人物关系", "先确认人物实体，再创建关系候选。")}</div><div class="section-head"><div><p class="section-kicker">关系候选与证据</p><h3>${relationships.length} 条关系</h3></div></div><div class="fact-review-list">${candidateRows || emptyState("暂无关系候选", "确认人物在同一事件中出现后，系统会提示共同出现候选。")}</div><div class="modal-actions"><button class="button primary" data-action="close-modal">关闭</button></div>`;
    } else if (modal.type === "help") {
      body = `<div class="modal-kicker">SENTRIX HOME / HELP</div><h2>当前可用能力</h2><div class="help-list"><div><strong>导入</strong><span>图片、音频、文本会生成 Observation；视频只建立 Asset。</span></div><div><strong>证据</strong><span>事件和 Agent 回答都能打开 Asset、Observation 和模型原始 JSON。</span></div><div><strong>维护</strong><span>事实冲突进入 pending，确认后旧版本变为 superseded。</span></div><div><strong>隐私</strong><span>原始文件、人物候选和 SQLite 都在 153 本地运行。</span></div></div><div class="modal-actions"><button class="button primary" data-action="close-modal">关闭</button></div>`;
    }
    root.innerHTML = `<div class="modal-backdrop"><div class="modal-panel"><button class="modal-close" data-action="close-modal" aria-label="关闭">×</button>${body}</div></div>`;
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
    ]);
    state.dashboard = calls[0].status === "fulfilled" ? calls[0].value : null;
    state.events = calls[1].status === "fulfilled" ? calls[1].value.events || [] : [];
    state.assets = calls[2].status === "fulfilled" ? calls[2].value.assets || [] : [];
    state.persons = calls[3].status === "fulfilled" ? calls[3].value.people || [] : [];
    state.stories = calls[4].status === "fulfilled" ? calls[4].value.stories || [] : [];
    state.health = calls[5].status === "fulfilled" ? calls[5].value : null;
    state.entities = calls[6].status === "fulfilled" ? calls[6].value.entities || [] : [];
    state.clusters = calls[7].status === "fulfilled" ? calls[7].value.clusters || [] : [];
    state.relationships = calls[8].status === "fulfilled" ? calls[8].value.relationships || [] : [];
        state.knowledge = calls[9].status === "fulfilled" ? calls[9].value : { profiles: [], claims: [] };
        state.trips = calls[10].status === "fulfilled" ? calls[10].value.trips || [] : [];
        state.entityMergeCandidates = calls[11].status === "fulfilled" ? calls[11].value.candidates || [] : [];
        state.entityGroups = calls[12].status === "fulfilled" ? calls[12].value.groups || [] : [];
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

  function bindViewEvents() {
    document.querySelectorAll("[data-view]").forEach((element) => element.addEventListener("click", () => { state.view = element.dataset.view; state.modal = null; renderShellNavigation(); }));
    document.querySelectorAll("[data-query]").forEach((element) => element.addEventListener("click", () => { state.query = element.dataset.query; state.view = "search"; renderShellNavigation(); submitSearch(); }));
    document.querySelectorAll("[data-event-filter]").forEach((element) => element.addEventListener("click", () => { state.eventFilter = element.dataset.eventFilter; renderView(); }));
    document.querySelectorAll("[data-asset-filter]").forEach((element) => element.addEventListener("click", () => { state.assetFilter = element.dataset.assetFilter; renderView(); }));
    document.querySelectorAll("[data-person-filter]").forEach((element) => element.addEventListener("click", () => { state.personFilter = element.dataset.personFilter; renderView(); }));
    const spaceSelect = document.getElementById("space-select");
    if (spaceSelect) spaceSelect.addEventListener("change", async (event) => { state.scopeId = event.target.value; window.localStorage?.setItem("sentrix.scopeId", state.scopeId); state.modal = null; state.conversationId = ""; state.searchResult = null; state.assistantMessages = []; await refreshData(); });
    const form = document.getElementById("search-form");
    if (form) form.addEventListener("submit", submitSearch);
    const modalForm = document.getElementById("modal-form");
    if (modalForm) modalForm.addEventListener("submit", handleModalSubmit);
    document.querySelectorAll("[data-action]").forEach((element) => element.addEventListener("click", () => handleAction(element.dataset.action, element)));
    const fileInput = document.getElementById("file-input");
    if (fileInput) fileInput.addEventListener("change", handleFiles);
  }

  async function submitSearch(event, selectedEntityId = "") {
    if (event?.preventDefault) event.preventDefault();
    const input = document.getElementById("search-input");
    state.query = input ? input.value.trim() : state.query.trim();
    if (!state.query) return;
    state.view = "search";
    state.assistantMessages.push({ role: "user", text: state.query });
    state.searchLoading = true;
    renderShellNavigation();
    try {
      state.searchResult = await window.sentrixApi.assistantTurn(state.query, state.conversationId, null, state.scopeId, selectedEntityId);
      state.conversationId = state.searchResult.conversation_id || state.conversationId;
    } catch (error) {
      state.searchResult = { answer: "当前无法读取本地记忆，请稍后重试。", confidence: 0, evidence: [], retrievalTrace: [], error: error.message, insufficient_evidence: true };
    }
    state.assistantMessages.push({ role: "steward", result: state.searchResult });
    state.query = "";
    state.searchLoading = false;
    renderShellNavigation();
  }

  async function submitProactiveOutcome(element, outcome) {
    const sceneKey = element.dataset.sceneKey || "";
    const message = outcome === "accepted" ? "看看这段回忆" : "好的";
    state.assistantMessages.push({ role: "user", text: message });
    state.searchLoading = true;
    renderShellNavigation();
    try {
      const result = await window.sentrixApi.assistantTurn(
        message, state.conversationId,
        { proactivity_outcome: outcome, proactivity_scene_key: sceneKey },
        state.scopeId, "", "owner",
      );
      state.searchResult = result;
      state.conversationId = result.conversation_id || state.conversationId;
      state.assistantMessages.push({ role: "steward", result });
    } catch (error) {
      state.toast = `主动回忆状态未更新：${error.message}`;
    }
    state.searchLoading = false;
    renderShellNavigation();
  }

  async function handleFiles(event) {
    const files = Array.from(event.target.files || []);
    for (const file of files) {
      state.queue.unshift({ fileName: file.name, status: "uploading" });
      try { const result = await window.sentrixApi.importAsset(file); state.queue[0].assetId = result.assetId; state.queue[0].status = result.status; } catch { state.queue[0].status = "failed"; }
    }
    state.toast = `${files.length} 个资料已进入本地处理队列`;
    await refreshData();
    state.view = "imports";
    renderShellNavigation();
  }

  function openModal(modal) { state.modal = modal; renderShellNavigation(); }

  async function openEvent(eventId, edit = false) {
    openModal({ type: "loading" });
    try { const detail = await window.sentrixApi.event(eventId); openModal(edit ? { type: "event-edit", event: detail.event, observations: detail.observations || [] } : { type: "event", detail }); } catch { state.toast = "无法读取事件证据"; state.modal = null; renderShellNavigation(); }
  }

  async function openAsset(assetId) {
    openModal({ type: "loading" });
    try { const [asset, result] = await Promise.all([window.sentrixApi.asset(assetId), window.sentrixApi.observations(`?assetId=${encodeURIComponent(assetId)}`)]); openModal({ type: "asset", asset, observations: result.observations || [] }); } catch { state.toast = "无法读取原始资料"; state.modal = null; renderShellNavigation(); }
  }

  async function openEntity(entityId) {
    openModal({ type: "loading" });
    try { const detail = await window.sentrixApi.entity(entityId); openModal({ type: "entity", detail }); } catch { state.toast = "无法读取实体记忆"; state.modal = null; renderShellNavigation(); }
  }

  async function openEntityGroup(groupId) {
    openModal({ type: "loading" });
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
      if (modal.type === "person") {
        const confirmed = await window.sentrixApi.confirmPerson(modal.person.id, form.get("name"), form.get("family_role"));
        const counts = confirmed.refresh_counts || {};
        state.toast = `已确认${form.get("name")}，已更新 ${counts.events || 0} 个事件、${counts.patterns || 0} 个模式、${counts.claims || 0} 条语义声明`;
      }
      if (modal.type === "cluster-confirm") {
        const confirmed = await window.sentrixApi.confirmFaceCluster(modal.cluster.id, { name: form.get("name"), family_role: form.get("family_role") });
        const target = String(form.get("relation_target") || "");
        const predicate = String(form.get("relation_predicate") || "").trim();
        if (target && predicate && confirmed.entity?.id) await window.sentrixApi.createRelationship({ subject_entity_id: confirmed.entity.id, predicate, object_entity_id: target, evidence_ids: (modal.cluster.samples || []).map((sample) => sample.observation_id), confidence: 1, status: "active" });
        const counts = confirmed.refresh_counts || {};
        state.toast = `已确认${form.get("name")}，已更新 ${counts.events || 0} 个事件、${counts.patterns || 0} 个模式、${counts.claims || 0} 条语义声明`;
      }
      if (modal.type === "cluster-merge") {
        await window.sentrixApi.mergeFaceClusters(form.get("target_cluster_id"), modal.cluster.id);
      }
      if (modal.type === "cluster-split") {
        await window.sentrixApi.splitFaceCluster(modal.cluster.id, modal.sample.id);
        state.toast = "已拆出新人物簇，原始证据仍保留";
      }
      if (modal.type === "story-create") await window.sentrixApi.createStory({ title: form.get("title"), content: form.get("content"), event_ids: form.getAll("event_ids"), tags: (form.get("tags")||"").split(",").map(s=>s.trim()).filter(Boolean) });
      if (modal.type === "story-edit") await window.sentrixApi.updateStory(modal.story.id, { title: form.get("title"), content: form.get("content"), event_ids: form.getAll("event_ids"), tags: (form.get("tags")||"").split(",").map(s=>s.trim()).filter(Boolean) });
      if (modal.type === "command") {
        const command = String(form.get("command") || "").trim();
        const target = navItems.find((item) => command.includes(item.label) || command.toLowerCase().includes(item.id));
        if (target) { state.modal = null; state.view = target.id; renderShellNavigation(); return; }
        state.modal = null; state.query = command; state.view = "search"; renderShellNavigation(); return submitSearch();
      }
      if (modal.type === "invite") { const invite = await window.sentrixApi.createInvite(form.get("label")); openModal({ type: "invite", invite }); return; }
      state.modal = null;
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
    if (action === "close-modal") { state.modal = null; renderShellNavigation(); return; }
    if (action === "open-event") return openEvent(element.dataset.eventId);
    if (action === "edit-event") return openEvent(element.dataset.eventId, true);
    if (action === "open-asset") return openAsset(element.dataset.assetId);
    if (action === "open-entity") return openEntity(element.dataset.entityId);
    if (action === "open-entity-group") return openEntityGroup(element.dataset.entityGroupId);
    if (action === "edit-entity-properties") return openModal({ type: "entity-property-edit", detail: state.modal.detail });
    if (action === "open-observation") { const observation = await window.sentrixApi.observation(element.dataset.observationId); return openAsset(observation.asset_id); }
    if (action === "create-event") return openModal({ type: "event-create", event: {} });
    if (action === "create-story") return openModal({ type: "story-create", story: {} });
    if (action === "confirm-trip") { const trip = state.trips.find((item) => item.id === element.dataset.tripId); return trip && openModal({ type: "trip-confirm", trip }); }
    if (action === "reject-trip") { await window.sentrixApi.rejectTrip(element.dataset.tripId); state.toast = "已标记为非行程，原始事件和照片证据仍保留"; return refreshData(); }
    if (action === "edit-story") { const story = state.stories.find((item) => item.id === element.dataset.storyId); return openModal({ type: "story-edit", story }); }
    if (action === "delete-story") { await window.sentrixApi.deleteStory(element.dataset.storyId); state.toast = "故事草稿已删除"; return refreshData(); }
    if (action === "open-person") { openModal({ type: "loading" }); try { const detail = await window.sentrixApi.personEvidence(element.dataset.personId, state.scopeId); return openModal({ type: "person-evidence", detail }); } catch (error) { state.modal = null; state.toast = `无法读取人物证据：${error.message}`; return renderShellNavigation(); } }
    if (action === "open-person-profile") { openModal({ type: "loading" }); const detail = await window.sentrixApi.personProfile(element.dataset.personId); return openModal({ type: "person-profile", detail }); }
    if (action === "edit-person-properties") return openModal({ type: "person-property-edit", detail: state.modal.detail });
    if (action === "confirm-person") { const person = state.persons.find((item) => item.id === element.dataset.personId) || { id: element.dataset.personId, name: "待确认人物" }; return openModal({ type: "person", person }); }
    if (action === "confirm-cluster") { const cluster = state.clusters.find((item) => item.id === element.dataset.clusterId); return openModal({ type: "cluster-confirm", cluster }); }
    if (action === "merge-cluster") { const cluster = state.clusters.find((item) => item.id === element.dataset.clusterId); return openModal({ type: "cluster-merge", cluster }); }
    if (action === "split-face") { const cluster = state.clusters.find((item) => item.id === element.dataset.clusterId); const sample = cluster?.samples?.find((item) => item.id === element.dataset.faceInstanceId); return openModal({ type: "cluster-split", cluster, sample }); }
    if (action === "reject-cluster") { try { await window.sentrixApi.rejectFaceCluster(element.dataset.clusterId); state.toast = "人物候选已驳回，原始人脸证据仍保留"; return refreshData(); } catch (error) { state.toast = `驳回失败：${error.message}`; renderShellNavigation(); return; } }
    if (action === "split-person") { await window.sentrixApi.rejectPerson(element.dataset.personId); state.toast = "候选人物已驳回，原始人脸证据仍保留"; return refreshData(); }
    if (action === "confirm-fact") { await window.sentrixApi.confirmFact(element.dataset.fact); state.toast = "事实已确认并生成修订记录"; return refreshData(); }
    if (action === "reject-fact") { await window.sentrixApi.rejectFact(element.dataset.fact); state.toast = "事实已驳回并保留证据记录"; return refreshData(); }
    if (action === "confirm-relationship") { await window.sentrixApi.confirmRelationship(element.dataset.relationshipId); state.toast = "关系已确认并进入语义记忆"; return refreshData(); }
    if (action === "invite") return openModal({ type: "invite" });
    if (action === "open-help") return openModal({ type: "help" });
    if (action === "command") return openModal({ type: "command" });
    if (action === "open-space") return openModal({ type: "help" });
    if (action === "open-folder") { document.getElementById("file-input")?.click(); return; }
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
    if (action === "recheck") { await fetch("/api/maintenance/recheck", { method: "POST" }); state.toast = "已提交失败任务重试"; return refreshData(); }
    if (action === "relationship-graph") { openModal({ type: "loading" }); const graph = await window.sentrixApi.relationships(); return openModal({ type: "relation", graph }); }
  }

  shell();
  refreshData();
  window.setInterval(() => {
    if (document.visibilityState === "hidden" || isUserEditing()) return;
    refreshData({ silent: true });
  }, 5000);
})();
