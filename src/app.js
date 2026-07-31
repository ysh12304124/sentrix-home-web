(function () {
  const app = document.getElementById("app");
  const state = {
    view: "overview",
    query: "",
    conversationId: "",
    searchResult: null,
    searchLoading: false,
    loading: true,
    backendError: "",
    toast: "",
    spaces: [],
    scopeId: window.localStorage?.getItem("sentrix.scopeId") || "home-default",
    queue: [],
    dashboard: null,
    events: [],
    assets: [],
    persons: [],
    entities: [],
    knowledge: { profiles: [], claims: [] },
    clusters: [],
    relationships: [],
    stories: [],
    health: null,
    modal: null,
    eventFilter: "all",
    assetFilter: "all",
    assetSort: "newest",
    personFilter: "all",
  };

  const navItems = [
    { id: "overview", icon: "⌂", label: "家庭概览" },
    { id: "search", icon: "⌕", label: "记忆搜索" },
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
      placeLabel: [event.place || "未标注地点", people.length ? people.join("、") : "暂无已确认人物"].join(" · "),
      typeLabel: event.event_type || "家庭事件",
      countLabel: `${(event.observation_ids || []).length} 项证据`,
      tone: ["mint", "peach", "blue", "lime", "lavender"][index % 5],
    };
  }

  function mediaLabel(type) {
    return type === "image" ? "图片" : type === "audio" ? "音频" : type === "video" ? "视频" : "文本";
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

  function searchBar(placeholder = "搜索人物、事件、地点或原始资料…") {
    return `<form class="search-bar" id="search-form"><span class="search-symbol">⌕</span><input id="search-input" value="${escapeHtml(state.query)}" placeholder="${escapeHtml(placeholder)}" /><kbd>Enter</kbd><button type="submit" aria-label="执行搜索">→</button></form>`;
  }

  function memoryPill(type, label, stateLabel = "已建立") {
    return `<span class="memory-pill ${type}"><i></i>${escapeHtml(label)}<small>${escapeHtml(stateLabel)}</small></span>`;
  }

  function emptyState(title, description, action = "") {
    return `<div class="empty-queue"><span>⌁</span><strong>${escapeHtml(title)}</strong><p>${escapeHtml(description)}</p>${action}</div>`;
  }

  function assetThumb(asset, compact = false) {
    const source = `/api/assets/${encodeURIComponent(asset.id)}/file`;
    const body = asset.media_type === "image" ? `<img src="${source}" alt="${escapeHtml(asset.file_name)}" loading="lazy" />` : asset.media_type === "video" ? `<video src="${source}" preload="metadata"></video>` : `<span class="asset-type-mark">${mediaLabel(asset.media_type)}</span>`;
    return `<div class="asset-thumb ${compact ? "compact" : ""} ${asset.media_type}">${body}<b>${escapeHtml(mediaLabel(asset.media_type))}</b></div>`;
  }

  function faceAvatar(faceInstanceId, label = "?", tone = "gray") {
    if (faceInstanceId) {
      return `<img class="face-avatar ${tone}" src="/api/face-instances/${encodeURIComponent(faceInstanceId)}/crop" alt="${escapeHtml(label)}的人脸证据" loading="lazy" />`;
    }
    return `<span class="avatar person-avatar ${tone}">${escapeHtml((label || "?").slice(0, 1))}</span>`;
  }

  function assetCard(asset) {
    return `<button class="asset-card" data-action="open-asset" data-asset-id="${escapeHtml(asset.id)}">${assetThumb(asset, true)}<div class="asset-info"><strong>${escapeHtml(asset.file_name)}</strong><small>${formatDateTime(asset.created_at)} · ${escapeHtml(mediaLabel(asset.media_type))}</small><span>${escapeHtml(asset.status)}</span></div></button>`;
  }

  function eventRow(event) {
    return `<button class="event-row" data-action="open-event" data-event-id="${escapeHtml(event.id)}"><span class="event-date">${event.date}</span><span class="event-dot ${event.tone}"></span><span class="event-summary"><strong>${escapeHtml(event.title)}</strong><small>${escapeHtml(event.placeLabel)}</small></span><span class="event-meta">${escapeHtml(event.typeLabel)} · ${escapeHtml(event.countLabel)}</span>${icon("→", "row-arrow")}</button>`;
  }

  function shell() {
    const activeSpace = state.spaces.find((space) => space.id === state.scopeId);
    const spaceOptions = (state.spaces.length ? state.spaces : [{ id: state.scopeId, name: state.scopeId }]).map((space) => `<option value="${escapeHtml(space.id)}" ${space.id === state.scopeId ? "selected" : ""}>${escapeHtml(space.name || space.id)}</option>`).join("");
    app.innerHTML = `<aside class="sidebar"><div class="brand-lockup"><span class="brand-mark">S</span><div><strong>Sentrix</strong><small>Home Memory</small></div></div><label class="space-switcher"><span class="avatar tiny">S</span><span><b>记忆空间</b><select id="space-select" aria-label="切换记忆空间">${spaceOptions}</select><small>${escapeHtml(activeSpace?.kind === "benchmark" ? "独立相册空间" : "本地家庭数据")}</small></span></label><div class="side-label">家庭记忆</div><nav class="main-nav">${navItems.map((item) => `<button class="nav-item ${state.view === item.id ? "active" : ""}" data-view="${item.id}">${icon(item.icon)}<span>${item.label}</span></button>`).join("")}</nav><div class="side-label lower">空间与系统</div><nav class="main-nav"><button class="nav-item ${state.view === "settings" ? "active" : ""}" data-view="settings">${icon("◌")}<span>设备与隐私</span></button><button class="nav-item" data-action="open-help">${icon("?")}<span>使用帮助</span></button></nav><div class="sidebar-footer"><div class="local-pulse"><i></i><span>${state.backendError ? "本地服务不可用" : "本地 AI 正常运行"}</span></div><small>Sentrix Home · 0.2.0</small></div></aside><main class="main-content"><header class="topbar"><div class="breadcrumbs"><span>Sentrix Home</span>${state.view !== "overview" ? `<b>/</b><strong>${escapeHtml(navItems.find((item) => item.id === state.view)?.label || "设备与隐私")}</strong>` : ""}</div><div class="top-actions"><button class="icon-button" data-action="command" aria-label="打开命令搜索">⌘</button><button class="top-user" data-action="open-space"><span class="avatar tiny">S</span><span>${escapeHtml(activeSpace?.name || "本地家庭空间")}</span>${icon("⌄", "muted")}</button></div></header><div id="view-root" class="view-root"></div></main><div id="toast-root" aria-live="polite"></div><div id="modal-root"></div>`;
    renderView();
  }

  function overview() {
    const count = stats();
    const events = state.events.slice(0, 3).map(eventViewModel);
    return `${pageHeader("家庭记忆 / 真实数据", "把家里的记忆，重新放在一起。", "这里展示已经导入并完成处理的本地资料。没有资料时，Sentrix 不会用示例内容填充。", `<button class="button primary" data-view="imports">${icon("＋")}导入资料</button>`)}${state.backendError ? `<div class="error-banner">${escapeHtml(state.backendError)}</div>` : ""}<section class="overview-search"><div><p class="section-kicker">问 Sentrix</p><h2>你想找回哪一段记忆？</h2><p>从人物、时间、地点、物体或一句描述开始，答案会带回原始证据。</p></div>${searchBar()}</section><section class="stats-grid"><article class="stat-card"><span>已整理内容</span><strong>${count.assets}</strong><small>本地 Asset</small></article><article class="stat-card"><span>已形成事件</span><strong>${count.events}</strong><small>可回到 Observation</small></article><article class="stat-card"><span>待确认事实</span><strong>${state.dashboard?.pendingFacts ?? 0}</strong><small>版本维护队列</small></article><article class="stat-card accent"><span>本地 AI 状态</span><strong>${state.health?.status === "ok" ? "正常" : "未知"}</strong><small>${escapeHtml(state.health?.models?.gamma4_12B?.name || "等待服务")}</small></article></section><section class="content-section"><div class="section-head"><div><p class="section-kicker">三类记忆</p><h2>同一家庭，不同的记忆入口</h2></div><button class="text-button" data-view="settings">查看系统状态 ${icon("→")}</button></div><div class="memory-grid"><article class="memory-card episodic-card"><div class="card-top">${memoryPill("episodic", "事件记忆")}<span class="card-index">01</span></div><h3>把分散的资料聚成共同经历</h3><p>图片、音频和文本共同参与人物、时间、地点与事件整理。</p><div class="card-metric"><strong>${count.events}</strong><span>个已建立事件</span></div></article><article class="memory-card semantic-card"><div class="card-top">${memoryPill("semantic", "语义记忆")}<span class="card-index">02</span></div><h3>让事实持续生长且保留修订</h3><p>每条事实都保留来源、置信度和人工确认历史。</p><div class="card-metric"><strong>${count.facts}</strong><span>条本地事实</span></div></article><article class="memory-card visual-card"><div class="card-top">${memoryPill("visual", "视频编码记忆", "接口预留")}<span class="card-index">03</span></div><h3>视频先归档，编码接口独立接入</h3><p>视频不会在第一版生成动作、片段或向量记忆。</p><div class="reserved-line">${icon("◌")} video_memory_adapter <span>未启用</span></div></article></div></section><section class="content-section two-column"><div><div class="section-head"><div><p class="section-kicker">最近事件</p><h2>家里的时间线</h2></div><button class="text-button" data-view="timeline">查看全部 ${icon("→")}</button></div>${events.length ? `<div class="event-list">${events.map(eventRow).join("")}</div>` : emptyState("还没有事件", "导入图片、音频或文本后，处理完成的 Observation 会在这里形成事件。", `<button class="button small primary" data-view="imports">${icon("＋")}导入第一份资料</button>`)}</div><div><div class="section-head"><div><p class="section-kicker">需要你的确认</p><h2>让记忆更准确</h2></div><button class="text-button" data-view="settings">查看事实 ${icon("→")}</button></div>${(state.dashboard?.pendingFacts || 0) ? `<div class="review-panel"><div class="review-face-pair"><span class="avatar large gray">?</span></div><div><strong>${state.dashboard.pendingFacts} 条事实等待确认</strong><p>确认或驳回前，原始 Observation 会一直保留。</p></div><div class="review-actions"><button class="button small primary" data-view="settings">处理</button></div></div>` : emptyState("目前没有待确认事实", "新资料产生矛盾信息时，会进入版本维护队列。")}</div></section>`;
  }

  function evidenceCard(evidence) {
    const sourceAction = evidence.kind === "observation" && evidence.asset_id ? `data-action="open-asset" data-asset-id="${escapeHtml(evidence.asset_id)}"` : evidence.event_id ? `data-action="open-event" data-event-id="${escapeHtml(evidence.event_id)}"` : evidence.kind === "fact" && evidence.evidence_ids?.[0] ? `data-action="open-observation" data-observation-id="${escapeHtml(evidence.evidence_ids[0])}"` : "";
    const title = evidence.kind === "fact" ? `${evidence.subject} ${evidence.predicate} ${evidence.object}` : evidence.file_name || evidence.summary || evidence.caption || evidence.id;
    const text = evidence.kind === "observation" ? evidence.caption || evidence.transcript || "无文字摘要" : evidence.summary || evidence.status || "";
    const media = evidence.kind === "observation" && evidence.asset_id ? `<button class="evidence-media" data-action="open-asset" data-asset-id="${escapeHtml(evidence.asset_id)}" aria-label="打开 ${escapeHtml(evidence.file_name || "原始媒体")}">${assetThumb({ id: evidence.asset_id, file_name: evidence.file_name, media_type: evidence.media_type || "image" }, true)}</button>` : "";
    const assetAction = evidence.kind === "asset" && evidence.id ? `data-action="open-asset" data-asset-id="${escapeHtml(evidence.id)}"` : "";
    const main = sourceAction || assetAction ? `<button class="evidence-main" ${sourceAction || assetAction}><strong>${escapeHtml(title)}</strong><p>${escapeHtml(text)}</p><small>${escapeHtml(evidence.captured_at || evidence.place || evidence.media_type || "证据记录")}</small></button>` : `<div class="evidence-main static"><strong>${escapeHtml(title)}</strong><p>${escapeHtml(text)}</p><small>${escapeHtml(evidence.captured_at || evidence.place || evidence.media_type || "证据记录")}</small></div>`;
    return `<article class="evidence-card"><div class="evidence-head"><span class="evidence-kind">${escapeHtml(evidence.kind)}</span><code>${escapeHtml(evidence.id)}</code></div>${media}${main}${evidence.raw ? `<details><summary>查看模型原始 JSON</summary><pre>${escapeHtml(JSON.stringify(evidence.raw, null, 2))}</pre></details>` : ""}</article>`;
  }

  function evidenceLayer(title, values) {
    if (!values?.length) return "";
    return `<section class="evidence-layer"><div class="section-head"><div><p class="section-kicker">${escapeHtml(title)}</p><h3>${values.length} 项</h3></div></div><div class="evidence-list">${values.slice(0, 12).map(evidenceCard).join("")}</div></section>`;
  }

  function imageResults(result) {
    const images = result?.image_results || [];
    if (!images.length) return "";
    return `<section class="evidence-layer image-results"><div class="section-head"><div><p class="section-kicker">原始图片</p><h3>${images.length} 张可回看的证据</h3></div></div><div class="image-result-grid">${images.map((item) => `<button class="image-result" data-action="open-asset" data-asset-id="${escapeHtml(item.asset_id)}"><img src="${escapeHtml(item.media_url)}" alt="${escapeHtml(item.file_name || "原始图片")}" loading="lazy" /><span><strong>${escapeHtml(item.file_name || item.asset_id)}</strong><small>${escapeHtml(item.captured_at || item.caption || "图片证据")}</small></span></button>`).join("")}</div></section>`;
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

  function searchView() {
    const result = state.searchResult;
    return `${pageHeader("记忆搜索 / Agent", "问一句，找到一段有证据的记忆。", "答案只来自本地事件、事实、Observation 和原生向量索引；每条证据都可以打开原始资料。")}${searchBar("例如：哪张图片里有冰箱？")}${state.searchLoading ? `<section class="empty-search"><div class="empty-symbol">◌</div><h2>正在检索本地记忆</h2><p>正在召回事件、事实、原始观察和语义上下文。</p></section>` : result ? `<section class="search-layout"><div><div class="answer-card"><div class="answer-meta"><span class="status-dot"></span>Sentrix 已完成本地检索 <span class="confidence">置信度 ${Math.round((result.confidence || 0) * 100)}% · ${escapeHtml(result.model || "本地 Agent")}</span></div><h2>检索结果</h2><p>${escapeHtml(result.answer || "证据不足，无法回答。").replace(/\n/g, "<br />")}</p><div class="answer-tags">${memoryPill("episodic", "事件证据")}${memoryPill("semantic", "事实证据")}${memoryPill("visual", "原始资料")}</div></div>${imageResults(result)}${evidenceLayer("人物与事件", [...(result.evidence_layers?.people || []), ...(result.evidence_layers?.events || [])])}${evidenceLayer("语义声明", result.evidence_layers?.claims)}${evidenceLayer("观察证据", result.evidence_layers?.observations)}${evidenceLayer("原始 Asset", result.evidence_layers?.assets?.map((item) => ({ ...item, kind: "asset", file_name: item.id, summary: "打开原始资料" })))}${evidenceLayer("原始证据", !result.evidence_layers ? result.evidence : [])}${result.evidence_layers?.gaps?.length ? evidenceLayer("查询缺口", result.evidence_layers.gaps.map((gap) => ({ ...gap, kind: "query_gap", summary: `${gap.missing_dimension} · ${gap.status}` }))) : ""}</div><aside class="trace-panel"><div class="panel-title"><span>RETRIEVAL TRACE</span><span class="live-label"><i></i>本地</span></div><h3>这次回答经过了什么？</h3>${(result.retrievalTrace || []).map((item, i) => `<div class="trace-step"><span>${String(i + 1).padStart(2, "0")}</span><div><strong>${escapeHtml(traceLabel(item))}</strong><small>${escapeHtml(traceDetail(item))}</small></div><b>✓</b></div>`).join("")}<div class="trace-note">视频编码记忆 <span>接口预留</span></div></aside></section>` : `<section class="empty-search"><div class="empty-symbol">⌕</div><h2>从一个线索开始</h2><p>输入后会返回真实答案和原始证据，不会显示预填充结果。</p><div class="suggestions"><button data-query="图片里有什么？">图片里有什么？</button><button data-query="最近发生了什么？">最近发生了什么？</button><button data-query="哪些事实等待确认？">哪些事实等待确认？</button></div></section>`}`;
  }

  function timelineView() {
    const events = filteredEvents();
    return `${pageHeader("记忆组织 / 事件", "家里的时间线，不只是文件列表。", "每个事件都能回到原始 Observation、文件和模型输出；人工修改会增加 revision。", `<button class="button ghost" data-action="create-event">${icon("＋")}新建事件</button>`)}<div class="filter-row"><button class="filter-chip ${state.eventFilter === "all" ? "active" : ""}" data-event-filter="all">全部事件</button><button class="filter-chip ${state.eventFilter === "people" ? "active" : ""}" data-event-filter="people">有人物</button><button class="filter-chip ${state.eventFilter === "place" ? "active" : ""}" data-event-filter="place">有地点</button><span class="filter-spacer"></span><button class="icon-button bordered" data-action="reload" aria-label="刷新时间线">↻</button></div><section class="timeline-layout"><div class="timeline-main">${events.length ? events.map((event) => `<article class="timeline-event"><div class="timeline-marker ${event.tone}"></div><div class="timeline-date">${event.date}<small>${escapeHtml(event.typeLabel)}</small></div><div class="timeline-event-body"><div class="event-cover ${event.tone}"><span>${escapeHtml(event.title)}</span><b>${escapeHtml(event.countLabel)}</b></div><div class="timeline-event-copy"><div class="card-top"><span class="event-kind">${escapeHtml(event.status || "active")}</span><span class="confidence-label">revision ${event.revision || 1}</span></div><h2>${escapeHtml(event.title)}</h2><p>${escapeHtml(event.summary || "暂无事件摘要")}</p><div class="event-facts"><span>${icon("◎")} ${escapeHtml(event.placeLabel)}</span><span>${icon("◷")} ${escapeHtml(event.countLabel)}</span><span>${icon("↗")} 可回到原始证据</span></div><div class="event-actions"><button class="button small ghost" data-action="open-event" data-event-id="${escapeHtml(event.id)}">查看证据</button><button class="text-button" data-action="edit-event" data-event-id="${escapeHtml(event.id)}">修正事件 ${icon("→")}</button></div></div></div></article>`).join("") : emptyState("还没有事件", "导入并处理资料后，时间线会由真实 Observation 自动生成。", `<button class="button small primary" data-view="imports">${icon("＋")}导入资料</button>`)}</div><aside class="side-inspector"><p class="section-kicker">事件记忆</p><h2>事件是可维护的记忆单元</h2><p>新资料先成为 Observation，再根据时间、地点和活动合并到事件。所有变更保留证据。</p><div class="mini-flow"><span>Observation</span><b>→</b><span>Event</span><b>→</b><span>Revision</span></div><div class="inspector-note">视频事件提取 <strong>接口预留</strong><small>第一版只保存视频原始 Asset。</small></div></aside></section>`;
  }

  function peopleView() {
    const people = state.persons.filter((person) => state.personFilter === "all" || !person.confirmed);
    return `${pageHeader("家庭治理 / 人物", "先确认人物，再让关系长出来。", "人脸模型只生成候选。确认、驳回和命名都会写入本地人物状态。", `<button class="button primary" data-action="invite">${icon("＋")}生成邀请</button>`)}<div class="people-toolbar"><div class="segmented"><button class="${state.personFilter === "all" ? "active" : ""}" data-person-filter="all">全部人物</button><button class="${state.personFilter === "pending" ? "active" : ""}" data-person-filter="pending">待确认 <b>${state.persons.filter((person) => !person.confirmed).length}</b></button><button data-action="relationship-graph">关系图</button></div><button class="button ghost" data-action="reload">${icon("↻")}刷新</button></div><section class="people-grid">${people.length ? people.map((person) => `<article class="person-card ${person.confirmed ? "" : "needs-review"}"><div class="person-head">${faceAvatar(person.avatar_face_instance_id, person.display_name || person.name, person.confirmed ? "green" : "gray")}${person.confirmed ? `<span class="confirmed">✓ 已确认</span>` : `<span class="needs-label">待确认</span>`}</div><h2>${escapeHtml(person.display_name || person.name)}</h2><p>${escapeHtml(person.status)} · 置信度 ${Math.round((person.confidence || 0) * 100)}%</p><div class="person-stats"><span><strong>${person.mention_count || 0}</strong> 次出现</span><span><strong>${person.cluster_count || 0}</strong> 个人物簇</span></div><div class="person-actions"><button class="button small ghost" data-action="open-person" data-person-id="${escapeHtml(person.id)}">查看证据</button>${person.confirmed ? "" : `<button class="button small primary" data-action="confirm-person" data-person-id="${escapeHtml(person.id)}">确认</button><button class="button small ghost" data-action="split-person" data-person-id="${escapeHtml(person.id)}">驳回候选</button>`}</div></article>`).join("") : emptyState("还没有人物候选", "导入包含人脸的图片后，InsightFace 会生成待确认候选；不会凭空创建家庭成员。", `<button class="button small primary" data-view="imports">${icon("＋")}导入图片</button>`)}</section>`;
  }

  function knowledgeView() {
    const pendingClusters = state.clusters.filter((cluster) => cluster.status === "pending");
    const confirmed = state.entities.filter((entity) => entity.status === "confirmed");
    const pending = state.entities.filter((entity) => entity.status === "pending");
    const entityCard = (entity) => `<button class="entity-card" data-action="open-entity" data-entity-id="${escapeHtml(entity.id)}"><div class="entity-card-head">${faceAvatar(entity.avatar_face_instance_id, entity.canonical_name, entity.status === "confirmed" ? "green" : "gray")}<span class="needs-label">${escapeHtml(entity.status)}</span></div><h2>${escapeHtml(entity.canonical_name)}</h2><p>${escapeHtml(entity.family_role || "未确认家庭角色")}</p><div class="entity-stats"><span><strong>${entity.mention_count || 0}</strong> 次出现</span><span><strong>${entity.cluster_count || 0}</strong> 个人物簇</span><span><strong>${entity.relationship_count || 0}</strong> 条关系</span></div></button>`;
    const clusterCard = (cluster) => `<article class="cluster-card"><div class="cluster-head"><div><span class="section-kicker">FACE CLUSTER</span><strong>${escapeHtml(cluster.id)}</strong></div><span class="needs-label">${cluster.member_count || cluster.samples?.length || 0} 张样本</span></div><div class="cluster-samples">${(cluster.samples || []).slice(0, 6).map((sample) => `<div class="cluster-sample"><button data-action="open-asset" data-asset-id="${escapeHtml(sample.asset_id)}">${faceAvatar(sample.id, "人脸样本")}</button><button class="sample-split" data-action="split-face" data-cluster-id="${escapeHtml(cluster.id)}" data-face-instance-id="${escapeHtml(sample.id)}" aria-label="从人物簇拆出样本">×</button></div>`).join("")}</div><p>聚类置信度 ${Math.round((cluster.confidence || 0) * 100)}% · ${cluster.entity_status === "confirmed" ? "已绑定人物" : "待确认身份"}</p><div class="person-actions"><button class="button small primary" data-action="confirm-cluster" data-cluster-id="${escapeHtml(cluster.id)}">确认实体</button><button class="button small ghost" data-action="merge-cluster" data-cluster-id="${escapeHtml(cluster.id)}">合并到其他簇</button><button class="button small ghost" data-action="reject-cluster" data-cluster-id="${escapeHtml(cluster.id)}">驳回簇</button></div></article>`;
    return `${pageHeader("语义记忆 / 实体治理", "看见家庭记忆里稳定存在的实体。", "人物簇先由 buffalo_l 聚类，再由你确认名称和角色；确认结果会回写观察、事件、事实和原生关系图。", `<button class="button ghost" data-action="reload">${icon("↻")}刷新状态</button>`)}<section class="knowledge-summary"><article><strong>${confirmed.length}</strong><span>已确认实体</span></article><article><strong>${pending.length + pendingClusters.length}</strong><span>待维护候选</span></article><article><strong>${state.relationships.filter((item) => item.status === "active").length}</strong><span>已确认关系</span></article><article><strong>${state.relationships.filter((item) => item.status === "pending").length}</strong><span>待确认关系</span></article></section><section class="content-section"><div class="section-head"><div><p class="section-kicker">实体总览</p><h2>跨事件维护的家庭实体</h2></div><button class="text-button" data-action="relationship-graph">查看关系图 ${icon("→")}</button></div><div class="entity-grid">${state.entities.length ? state.entities.map(entityCard).join("") : emptyState("还没有实体", "完成一轮图片导入和人脸聚类后，实体会出现在这里。")}</div></section><section class="content-section"><div class="section-head"><div><p class="section-kicker">人物聚类 / 待确认</p><h2>先确认身份，再进入长期记忆</h2></div><span class="result-count">${pendingClusters.length} 簇</span></div><div class="cluster-grid">${pendingClusters.length ? pendingClusters.map(clusterCard).join("") : emptyState("没有待确认人物簇", "新的图片出现人物后，系统会把相似人脸合并为簇并保留样本证据。")}</div></section>`;
  }

  function libraryView() {
    const assets = filteredAssets();
    return `${pageHeader("家庭资料 / 全部内容", "所有资料，都有回到记忆的路径。", "这里展示真实导入的原始 Asset。点击任意条目可查看文件、处理状态、Observation 和模型原始输出。", `<button class="button primary" data-view="imports">${icon("↓")}导入资料</button>`)}<div class="library-summary"><div><strong>${state.assets.length}</strong><span>全部资产</span></div><div><strong>${state.assets.filter((a) => a.media_type === "image").length}</strong><span>图片</span></div><div><strong>${state.assets.filter((a) => ["audio", "text"].includes(a.media_type)).length}</strong><span>音频 / 文本</span></div><div class="muted-stat"><strong>${state.assets.filter((a) => a.media_type === "video").length}</strong><span>视频 · 接口预留</span></div></div><div class="filter-row">${[["all", "全部"], ["image", "图片"], ["audio", "音频"], ["text", "文本"], ["video", "视频"]].map(([key, label]) => `<button class="filter-chip ${state.assetFilter === key ? "active" : ""}" data-asset-filter="${key}">${label}</button>`).join("")}<span class="filter-spacer"></span><button class="sort-label" data-action="toggle-sort">按${state.assetSort === "newest" ? "最近" : "最早"}导入 ↕</button></div><section class="asset-grid library-grid">${assets.length ? assets.map(assetCard).join("") : emptyState("没有匹配的资料", "调整筛选条件或导入一份新的原始文件。", `<button class="button small primary" data-view="imports">${icon("＋")}导入资料</button>`)}</section>`;
  }

  function storiesView() {
    return `${pageHeader("家庭表达 / 故事工作室", "把真实事件整理成家人愿意一起看的故事。", "故事只引用你选择的事件和证据；标题、章节和内容保存为本地草稿。", `<button class="button primary" data-action="create-story">${icon("＋")}新建故事</button>`)}<section class="story-layout">${state.stories.length ? `<div class="story-canvas"><div class="story-canvas-label">选择一个故事查看</div><div class="story-title">${escapeHtml(state.stories[0].title)}</div><div class="story-caption">${escapeHtml(state.stories[0].content || "这个故事还没有内容。")}</div></div><aside class="story-editor"><div class="panel-title"><span>STORY DRAFTS</span><span class="draft-badge">${state.stories.length} 个</span></div>${state.stories.map((story) => `<div class="chapter"><button class="chapter-open" data-action="edit-story" data-story-id="${escapeHtml(story.id)}"><span>●</span><strong>${escapeHtml(story.title)}</strong>${icon("→", "muted")}</button><button class="icon-button bordered" data-action="delete-story" data-story-id="${escapeHtml(story.id)}" aria-label="删除故事">×</button></div>`).join("")}<button class="button primary full" data-action="create-story">新建本地草稿 ${icon("→")}</button></aside></section>` : `<section class="empty-search"><div class="empty-symbol">▤</div><h2>还没有故事草稿</h2><p>先导入并形成事件，再选择真实事件生成故事草稿。</p><button class="button primary" data-action="create-story">${icon("＋")}创建空白故事</button></section>`}`;
  }

  function importsView() {
    const assets = state.assets.filter((asset) => ["queued", "processing", "failed", "video-extraction-reserved"].includes(asset.status));
    return `${pageHeader("资料入口 / 本地导入", "把资料带回家，剩下的交给本地 AI。", "上传后会创建稳定 Asset ID，并在后台生成 Observation、Event 和 Fact；视频只建立原始资产。", `<button class="button ghost" data-action="open-folder">${icon("▦")}选择资料</button>`)}<section class="import-layout"><div><label class="dropzone" for="file-input"><input id="file-input" type="file" multiple accept="image/*,audio/*,text/*,video/*" /><span class="drop-icon">↓</span><strong>拖入资料，或点击选择文件</strong><small>支持图片、音频、文本和视频 · 原始文件不会离开本机</small><span class="button primary">选择资料</span></label><div class="import-notice"><span class="notice-mark">i</span><div><strong>原始证据不会被覆盖</strong><p>每个 Asset 都可以追溯到 Observation 和模型原始 JSON。</p></div></div></div><aside class="import-status"><div class="panel-title"><span>LOCAL PIPELINE</span><span class="live-label"><i></i>真实状态</span></div><h2>当前处理</h2>${[["接收与去重", `${state.assets.length} 个 Asset`, "done"], ["图片理解", `${state.assets.filter((a) => a.media_type === "image" && a.status === "processed").length} 个已完成`, "done"], ["音频转写", `${state.assets.filter((a) => a.media_type === "audio").length} 个音频`, "active"], ["事件与事实", `${stats().events} 个事件 · ${stats().facts} 条事实`, "active"], ["视频编码", `${state.assets.filter((a) => a.media_type === "video").length} 个视频`, "reserved"]].map((row) => `<div class="pipeline-row"><span class="pipeline-state ${row[2]}">${row[2] === "done" ? "✓" : row[2] === "active" ? "•" : "—"}</span><div><strong>${row[0]}</strong><small>${row[1]}</small></div><em>${row[2] === "done" ? "完成" : row[2] === "active" ? "运行中" : "预留"}</em></div>`).join("")}</aside></section><section class="content-section"><div class="section-head"><div><p class="section-kicker">导入记录</p><h2>最近处理任务</h2></div><button class="text-button" data-action="reload">刷新状态 ${icon("↻")}</button></div><div class="queue-list">${assets.length ? assets.map((asset) => `<div class="queue-row"><span class="queue-type ${asset.media_type}">${escapeHtml(mediaLabel(asset.media_type).slice(0, 3))}</span><div><strong>${escapeHtml(asset.file_name)}</strong><small>${escapeHtml(asset.id)} · ${formatDateTime(asset.updated_at)}</small></div><span class="queue-status ${asset.status === "video-extraction-reserved" ? "reserved" : "queued"}">${escapeHtml(asset.status)}</span></div>`).join("") : emptyState("没有待处理任务", "处理中的 Asset 会显示在这里。")}</div></section>`;
  }

  function settingsView() {
    const facts = state.dashboard?.facts || [];
    const pending = facts.filter((fact) => fact.status === "pending");
    return `${pageHeader("系统 / 本地状态", "你的记忆，运行在自己的家里。", "服务、模型、存储和事实修订状态都来自当前本地后端。")}${state.health ? `<section class="health-grid"><article class="health-card dark"><div class="health-title"><span>Sentrix Home</span><span class="online-pill"><i></i>在线</span></div><strong>本地服务正常</strong><p>健康接口返回正常</p><div class="health-line"><span>数据资产</span><b>${stats().assets}</b></div><div class="health-bar"><i style="width:100%"></i></div></article><article class="health-card"><div class="health-title"><span>AI MODEL ROUTER</span><span class="ready-label">READY</span></div><div class="model-row"><span>主推理</span><strong>${escapeHtml(state.health.models?.gamma4_12B?.name || "未知")}</strong><small>${escapeHtml(state.health.models?.gamma4_12B?.endpoint || "未连接")}</small></div><div class="model-row"><span>语音转写</span><strong>FunASR</strong><small>${escapeHtml(state.health.models?.asr?.name || "未连接")}</small></div><div class="model-row"><span>人物识别</span><strong>InsightFace</strong><small>${state.health.models?.face?.ready ? "已启用" : "不可用"}</small></div></article><article class="health-card"><div class="health-title"><span>MEMORY INDEX</span><span class="ready-label">LOCAL</span></div><strong>${stats().facts} <small>条事实</small></strong><p>SQLite 事实库 · 原生语义图与向量索引</p><div class="index-list"><span>${icon("●")}事件记忆 <b>${stats().events}</b></span><span>${icon("●")}观察证据 <b>${stats().observations}</b></span><span class="dim">${icon("—")}视频编码记忆 <b>预留</b></span></div></article></section>` : emptyState("正在读取本地状态", "请稍候或刷新页面。")}<section class="content-section fact-review"><div class="section-head"><div><p class="section-kicker">语义记忆 / 版本维护</p><h2>需要确认的事实</h2></div><span class="result-count">${pending.length} 条</span></div>${pending.length ? `<div class="fact-review-list">${pending.map((fact) => `<div class="fact-review-row"><div><strong>${escapeHtml(fact.subject)} ${escapeHtml(fact.predicate)} ${escapeHtml(fact.object)}</strong><small>${escapeHtml(fact.id)} · 置信度 ${Math.round((fact.confidence || 0) * 100)}% · 证据 ${(fact.evidence_ids_json || []).join(", ")}</small></div><div class="review-actions"><button class="button small primary" data-action="confirm-fact" data-fact="${escapeHtml(fact.id)}">${icon("✓")}确认</button><button class="button small ghost" data-action="reject-fact" data-fact="${escapeHtml(fact.id)}">${icon("×")}驳回</button></div></div>`).join("")}</div>` : emptyState("没有待确认事实", "冲突事实出现后会进入这里，旧版本不会被删除。")}</section><section class="content-section two-column settings-lower"><div><div class="section-head"><div><p class="section-kicker">隐私边界</p><h2>数据只在本地流动</h2></div></div><div class="privacy-list"><div><span>原始媒体</span><b>本地存储</b></div><div><span>人物特征</span><b>本地处理</b></div><div><span>原生记忆索引</span><b>本地实体与向量检索</b></div><div><span>视频编码</span><b>接口关闭</b></div></div></div><div><div class="section-head"><div><p class="section-kicker">审计入口</p><h2>可操作的系统动作</h2></div></div><div class="audit-list"><div><button class="button small ghost" data-action="reload">刷新服务状态 ${icon("↻")}</button><small>重新读取后端、模型和数据库状态</small></div><div><button class="button small ghost" data-action="recheck">重新检查失败任务 ${icon("→")}</button><small>只重试 queued 或 failed Asset</small></div><div><button class="button small ghost" data-action="open-help">查看接口与隐私说明 ${icon("?")}</button><small>当前部署边界和证据规则</small></div></div></div></section>`;
  }

  function semanticKnowledgeView() {
    const people = state.persons.filter((person) => person.confirmed);
    const claims = state.knowledge.claims || people.flatMap((person) => person.claims || []);
    const claimCards = claims.map((claim) => "<article class=\"fact-review-row\"><div><strong>" + escapeHtml(claim.predicate) + " · " + escapeHtml(claim.value_text) + "</strong><small>" + escapeHtml(claim.dimension) + " · " + escapeHtml(claim.status) + " · 置信度 " + Math.round((claim.confidence || 0) * 100) + "% · 证据 " + escapeHtml((claim.evidence_ids_json || claim.evidence_ids || []).join(", ")) + "</small></div></article>").join("");
    const personCards = people.map((person) => "<article class=\"entity-card\"><div class=\"entity-card-head\">" + faceAvatar(person.avatar_face_instance_id, person.display_name, "green") + "<span class=\"confirmed\">已确认</span></div><h2>" + escapeHtml(person.display_name) + "</h2><p>" + escapeHtml(person.family_role || "家庭成员") + "</p><p>" + escapeHtml(person.profile?.summary_zh || "正在从事件和证据形成画像。") + "</p><div class=\"entity-stats\"><span><strong>" + (person.claims || []).length + "</strong> 条声明</span><span><strong>" + (person.mention_count || 0) + "</strong> 次出现</span></div><button class=\"button small ghost\" data-action=\"open-person-profile\" data-person-id=\"" + escapeHtml(person.id) + "\">查看画像和证据</button></article>").join("");
    return pageHeader("语义记忆 / 人物知识", "看见每个人长期形成的知识。", "活动、地点、衣物、偏好和习惯都必须回到人物或事件证据，不再把人物混在普通实体里。", "<button class=\"button ghost\" data-action=\"reload\">" + icon("↻") + "刷新知识</button>") + "<section class=\"knowledge-summary\"><article><strong>" + people.length + "</strong><span>已确认人物</span></article><article><strong>" + claims.length + "</strong><span>人物语义声明</span></article><article><strong>" + state.entities.length + "</strong><span>非人物实体</span></article><article><strong>" + (state.dashboard?.pendingFacts || 0) + "</strong><span>旧事实待维护</span></article></section><section class=\"content-section\"><div class=\"section-head\"><div><p class=\"section-kicker\">人物画像</p><h2>语义记忆的主轴</h2></div><span class=\"result-count\">" + people.length + " 人</span></div><div class=\"entity-grid\">" + (personCards || emptyState("还没有已确认人物", "先在人物页面确认人脸簇，语义知识才会有稳定的中心。", "<button class=\"button small primary\" data-view=\"people\">打开人物</button>")) + "</div></section><section class=\"content-section\"><div class=\"section-head\"><div><p class=\"section-kicker\">人物关联知识</p><h2>带时间和证据的语义声明</h2></div><span class=\"result-count\">" + claims.length + " 条</span></div><div class=\"fact-review-list\">" + (claimCards || emptyState("还没有人物语义声明", "人物确认后，活动、地点、衣物和家庭角色会从关联事件中汇总。")) + "</div></section>";
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
      body = `<div class="modal-kicker">EVENT · ${escapeHtml(detail.event.id)}</div><h2>${escapeHtml(detail.event.title)}</h2><p class="modal-lead">${escapeHtml(detail.event.summary || "暂无摘要")}</p><div class="detail-facts"><span>时间 · ${formatDateTime(detail.event.time_start)}</span><span>地点 · ${escapeHtml(detail.event.place || "未标注")}</span><span>revision · ${detail.event.revision || 1}</span></div><div class="section-head"><div><p class="section-kicker">原始证据媒体</p><h3>${detail.observations.length} 条 Observation</h3></div><button class="button small ghost" data-action="edit-event" data-event-id="${escapeHtml(detail.event.id)}">修正事件</button></div><div class="evidence-list event-evidence-list">${detail.observations.length ? detail.observations.map((observation) => evidenceCard({ kind: "observation", id: observation.id, observation_id: observation.id, asset_id: observation.asset_id, file_name: observation.asset?.file_name, media_type: observation.asset?.media_type, captured_at: observation.captured_at, caption: observation.caption, transcript: observation.transcript, raw: observation.raw_json })).join("") : emptyState("没有关联 Observation", "这是一个人工创建的事件。")}</div>${detail.facts?.length ? `<div class="section-head"><div><p class="section-kicker">语义记忆</p><h3>关联事实</h3></div></div><div class="evidence-list">${detail.facts.map((fact) => evidenceCard({ kind: "fact", id: fact.id, subject: fact.subject, predicate: fact.predicate, object: fact.object, status: fact.status, evidence_ids: fact.evidence_ids_json })).join("")}</div>` : ""}`;
    } else if (modal.type === "asset") {
      const asset = modal.asset;
      body = `<div class="modal-kicker">ASSET · ${escapeHtml(asset.id)}</div><h2>${escapeHtml(asset.file_name)}</h2><div class="asset-modal-preview">${asset.media_type === "image" ? `<img src="/api/assets/${encodeURIComponent(asset.id)}/file" alt="${escapeHtml(asset.file_name)}" />` : asset.media_type === "audio" ? `<audio controls src="/api/assets/${encodeURIComponent(asset.id)}/file"></audio>` : asset.media_type === "video" ? `<video controls src="/api/assets/${encodeURIComponent(asset.id)}/file"></video>` : `<pre>${escapeHtml(asset.file_name)}</pre>`}</div><div class="detail-facts"><span>类型 · ${escapeHtml(mediaLabel(asset.media_type))}</span><span>状态 · ${escapeHtml(asset.status)}</span><span>大小 · ${asset.size_bytes || 0} bytes</span></div><div class="section-head"><div><p class="section-kicker">Observation</p><h3>这份资料产生的证据</h3></div></div><div class="evidence-list">${modal.observations.length ? modal.observations.map((observation) => evidenceCard({ kind: "observation", id: observation.id, asset_id: asset.id, file_name: asset.file_name, media_type: asset.media_type, captured_at: observation.captured_at, caption: observation.caption, transcript: observation.transcript, raw: observation.raw_json })).join("") : emptyState("还没有 Observation", "资料正在处理，刷新后查看结果。")}</div>`;
    } else if (modal.type === "event-edit" || modal.type === "event-create") {
      const event = modal.event || {};
      body = `<form id="modal-form"><div class="modal-kicker">${modal.type === "event-create" ? "CREATE EVENT" : "EDIT EVENT"}</div><h2>${modal.type === "event-create" ? "新建人工事件" : "修正事件"}</h2><label>标题<input name="title" value="${escapeHtml(event.title || "")}" required /></label><label>摘要<textarea name="summary">${escapeHtml(event.summary || "")}</textarea></label><label>地点<input name="place" value="${escapeHtml(event.place || "")}" /></label><label>开始时间<input name="time_start" type="datetime-local" value="${event.time_start ? event.time_start.slice(0, 16) : ""}" /></label><div class="modal-actions"><button type="button" class="button ghost" data-action="close-modal">取消</button><button type="submit" class="button primary">保存事件</button></div></form>`;
    } else if (modal.type === "person") {
      const person = modal.person;
      body = `<form id="modal-form"><div class="modal-kicker">IDENTITY CONFIRMATION · ${escapeHtml(person.id)}</div><h2>确认人物身份</h2><p class="modal-lead">只在这里填写姓名和家庭角色。确认后会把该人物回写到相关事件、人物画像和语义记忆。</p><label>家庭成员名称<input name="name" value="" placeholder="确认时填写名称" required /></label><label>家庭角色<select name="family_role"><option value="">暂不确认</option><option>母亲</option><option>父亲</option><option>孩子</option><option>祖父母</option><option>其他家庭成员</option></select></label><div class="modal-actions"><button type="button" class="button ghost" data-action="close-modal">取消</button><button type="submit" class="button primary">确认并更新记忆</button></div></form>`;
    } else if (modal.type === "person-evidence") {
      const detail = modal.detail;
      const entity = detail.entity;
      const samples = detail.face_samples || [];
      const events = detail.events || [];
      body = `<div class="modal-kicker">PERSON EVIDENCE · ${escapeHtml(entity.id)}</div><div class="profile-heading">${faceAvatar(samples[0]?.id || entity.avatar_face_instance_id, entity.canonical_name, entity.status === "confirmed" ? "green" : "gray")}<div><h2>${escapeHtml(entity.canonical_name)}</h2><p class="modal-lead">${escapeHtml(entity.status === "confirmed" ? "已确认人物，下面是可回看的原始证据。" : "待确认人物候选，先检查人脸样本再命名。")}</p></div></div><div class="detail-facts"><span>状态 · ${escapeHtml(entity.status)}</span><span>人脸样本 · ${samples.length}</span><span>关联事件 · ${events.length}</span></div><div class="section-head"><div><p class="section-kicker">人脸样本</p><h3>用于判断身份的头像证据</h3></div></div><div class="face-evidence-grid">${samples.length ? samples.map((sample) => `<article class="face-evidence-item"><img src="${escapeHtml(sample.crop_url)}" alt="人脸样本" loading="lazy" /><button class="text-button" data-action="open-asset" data-asset-id="${escapeHtml(sample.asset_id)}">查看原图 ${icon("→")}</button><small>${escapeHtml(sample.asset?.file_name || sample.file_name || sample.asset_id)} · ${escapeHtml(sample.cluster_id)}</small></article>`).join("") : emptyState("没有可用人脸样本", "当前候选没有可回看的 face instance。")}</div><div class="section-head"><div><p class="section-kicker">关联事件</p><h3>该人物出现过的事件</h3></div></div><div class="evidence-list">${events.length ? events.map((event) => `<button class="evidence-main" data-action="open-event" data-event-id="${escapeHtml(event.id)}"><strong>${escapeHtml(event.title)}</strong><p>${escapeHtml(event.summary || "暂无事件摘要")}</p><small>${escapeHtml(formatDateTime(event.time_start))} · ${escapeHtml(event.place || "未标注地点")}</small></button>`).join("") : emptyState("还没有关联事件", "确认后，新的事件观察会继续维护人物画像。")}</div>${entity.status === "confirmed" ? "" : `<div class="modal-actions"><button class="button ghost" data-action="close-modal">关闭</button><button class="button primary" data-action="confirm-person" data-person-id="${escapeHtml(entity.id)}">确认姓名和关系</button></div>`}`;
    } else if (modal.type === "cluster-confirm") {
      const cluster = modal.cluster;
      body = `<form id="modal-form"><div class="modal-kicker">FACE CLUSTER · ${escapeHtml(cluster.id)}</div><h2>确认这个人物实体</h2><p class="modal-lead">这组样本由 153 上的 buffalo_l embedding 聚类得到。确认后，所有样本会统一绑定到同一个实体。</p><div class="cluster-samples modal-samples">${(cluster.samples || []).map((sample) => `<button type="button" data-action="open-asset" data-asset-id="${escapeHtml(sample.asset_id)}">${faceAvatar(sample.id, "人脸样本")}</button>`).join("")}</div><label>姓名或称呼<input name="name" placeholder="例如：妈妈" required /></label><label>家庭角色<select name="family_role"><option value="">暂不确认</option><option>母亲</option><option>父亲</option><option>孩子</option><option>祖父母</option><option>其他家庭成员</option></select></label><label>与已确认实体的关系（可选）<select name="relation_target"><option value="">暂不建立关系</option>${state.entities.filter((entity) => entity.status === "confirmed").map((entity) => `<option value="${escapeHtml(entity.id)}">${escapeHtml(entity.canonical_name)}</option>`).join("")}</select></label><label>关系类型（可选）<input name="relation_predicate" placeholder="例如：母亲、父亲、兄弟姐妹" /></label><div class="modal-actions"><button type="button" class="button ghost" data-action="close-modal">取消</button><button type="submit" class="button primary">确认并更新记忆</button></div></form>`;
    } else if (modal.type === "cluster-merge") {
      const choices = state.clusters.filter((cluster) => cluster.status === "pending" && cluster.id !== modal.cluster.id);
      body = `<form id="modal-form"><div class="modal-kicker">FACE CLUSTER MERGE</div><h2>选择目标人物簇</h2><p class="modal-lead">合并会保留原始样本，并写入人物实体 revision。两个已确认人物不会被合并。</p><label>目标簇<select name="target_cluster_id" required>${choices.map((cluster) => `<option value="${escapeHtml(cluster.id)}">${escapeHtml(cluster.id)} · ${cluster.member_count || 0} 张样本</option>`).join("")}</select></label><div class="modal-actions"><button type="button" class="button ghost" data-action="close-modal">取消</button><button type="submit" class="button primary" ${choices.length ? "" : "disabled"}>合并人物簇</button></div></form>`;
    } else if (modal.type === "cluster-split") {
      const sample = modal.sample;
      body = `<form id="modal-form"><div class="modal-kicker">FACE CLUSTER SPLIT</div><h2>确认拆出这张人脸样本</h2><p class="modal-lead">拆分会创建同一记忆空间中的新候选簇，原始图片、人脸证据和审计记录都会保留。</p><div class="split-review"><img src="${escapeHtml(`/api/face-instances/${sample.id}/crop`)}" alt="待拆分人脸样本" /><div><strong>${escapeHtml(sample.file_name || sample.asset_id)}</strong><small>${escapeHtml(modal.cluster.id)} · ${escapeHtml(state.scopeId)}</small></div></div><div class="modal-actions"><button type="button" class="button ghost" data-action="close-modal">取消</button><button type="submit" class="button primary">确认拆分</button></div></form>`;
    } else if (modal.type === "person-profile") {
      const detail = modal.detail;
      const entity = detail.entity;
      const claims = detail.claims || [];
      body = "<div class=\"modal-kicker\">PERSON PROFILE · " + escapeHtml(entity.id) + "</div><div class=\"profile-heading\">" + faceAvatar(entity.avatar_face_instance_id, entity.canonical_name, "green") + "<div><h2>" + escapeHtml(entity.canonical_name) + "</h2><p class=\"modal-lead\">" + escapeHtml(detail.profile?.summary_zh || entity.summary || "暂无人物画像") + "</p></div></div><div class=\"detail-facts\"><span>家庭角色 · " + escapeHtml(entity.family_role || "未确认") + "</span><span>语义声明 · " + claims.length + "</span><span>人物簇 · " + detail.clusters.length + "</span></div><div class=\"fact-review-list\">" + (claims.map((claim) => "<div class=\"fact-review-row\"><div><strong>" + escapeHtml(claim.predicate) + " · " + escapeHtml(claim.value_text) + "</strong><small>" + escapeHtml(claim.dimension) + " · " + escapeHtml(claim.status) + " · 证据 " + escapeHtml((claim.evidence_ids_json || []).join(", ")) + "</small></div></div>").join("") || emptyState("暂无语义声明", "确认人物后，相关事件会持续维护人物画像。")) + "</div>";
    } else if (modal.type === "entity") {
      const detail = modal.detail;
      const entity = detail.entity;
      body = `<div class="modal-kicker">ENTITY · ${escapeHtml(entity.id)}</div><div class="profile-heading">${faceAvatar(entity.avatar_face_instance_id, entity.canonical_name, entity.status === "confirmed" ? "green" : "gray")}<div><h2>${escapeHtml(entity.canonical_name)}</h2><p class="modal-lead">${escapeHtml(entity.summary || "这是跨多个事件维护的实体，不是单张图片的描述。")} · 状态 ${escapeHtml(entity.status)}</p></div></div><div class="detail-facts"><span>类型 · ${escapeHtml(entity.entity_type)}</span><span>角色 · ${escapeHtml(entity.family_role || "未确认")}</span><span>人物簇 · ${detail.clusters.length}</span></div><div class="section-head"><div><p class="section-kicker">关系与事实</p><h3>${detail.relationships.length} 条关系 · ${detail.facts.length} 条事实</h3></div></div><div class="evidence-list">${detail.relationships.concat(detail.facts.map((fact) => ({ ...fact, subject_name: fact.subject, object_name: fact.object, predicate: fact.predicate, status: fact.status }))).map((item) => `<div class="fact-review-row"><div><strong>${escapeHtml(item.subject_name || item.subject)} ${escapeHtml(item.predicate)} ${escapeHtml(item.object_name || item.object)}</strong><small>${escapeHtml(item.status)} · 证据 ${(item.evidence_ids_json || []).join(", ")}</small></div></div>`).join("") || emptyState("暂无关系和事实", "确认新的实体关系后，长期语义记忆会显示在这里。")}</div><div class="section-head"><div><p class="section-kicker">人物簇样本</p><h3>原始证据</h3></div></div><div class="cluster-grid">${detail.clusters.map((cluster) => `<div class="cluster-card"><div class="cluster-head"><strong>${escapeHtml(cluster.id)}</strong><span>${cluster.member_count} 张</span></div><div class="cluster-samples">${(cluster.samples || []).map((sample) => `<button data-action="open-asset" data-asset-id="${escapeHtml(sample.asset_id)}">${faceAvatar(sample.id, "人脸样本")}</button>`).join("")}</div></div>`).join("")}</div>`;
    } else if (modal.type === "story-create" || modal.type === "story-edit") {
      const story = modal.story || {};
      body = `<form id="modal-form"><div class="modal-kicker">STORY ${modal.type === "story-create" ? "DRAFT" : "EDITOR"}</div><h2>${modal.type === "story-create" ? "创建故事草稿" : "编辑故事"}</h2><label>标题<input name="title" value="${escapeHtml(story.title || "")}" required /></label><label>故事内容<textarea name="content" rows="5">${escapeHtml(story.content || "")}</textarea></label><div class="story-event-select"><strong>选择事件证据</strong>${state.events.map((event) => `<label><input type="checkbox" name="event_ids" value="${escapeHtml(event.id)}" ${(story.event_ids || []).includes(event.id) ? "checked" : ""} />${escapeHtml(event.title)}</label>`).join("") || `<small>当前没有事件，请先导入资料。</small>`}</div><div class="modal-actions"><button type="button" class="button ghost" data-action="close-modal">取消</button><button type="submit" class="button primary">保存故事</button></div></form>`;
    } else if (modal.type === "invite") {
      body = modal.invite ? `<div class="modal-kicker">FAMILY SPACE INVITE</div><h2>局域网邀请已生成</h2><p class="modal-lead">这是一个本地邀请 token。当前不会发送到云端或第三方服务。</p><code class="invite-code">${escapeHtml(modal.invite.invite_url)}</code><div class="modal-actions"><button class="button primary" data-action="close-modal">完成</button></div>` : `<form id="modal-form"><div class="modal-kicker">FAMILY SPACE</div><h2>生成家庭成员邀请</h2><label>邀请备注<input name="label" value="家庭成员" required /></label><div class="modal-actions"><button type="button" class="button ghost" data-action="close-modal">取消</button><button type="submit" class="button primary">生成邀请</button></div></form>`;
    } else if (modal.type === "command") {
      body = `<form id="modal-form"><div class="modal-kicker">COMMAND</div><h2>打开一个工作区</h2><label>输入页面或问题<input name="command" autofocus placeholder="例如：时间线、资料库、搜索冰箱" /></label><div class="command-links">${navItems.map((item) => `<button type="button" class="button ghost" data-view="${item.id}">${item.label}</button>`).join("")}</div></form>`;
    } else if (modal.type === "relation") {
      body = `<div class="modal-kicker">RELATIONSHIP GRAPH</div><h2>实体关系图</h2><p class="modal-lead">只展示 Sentrix 原生关系表中的候选和已确认关系，不补造家庭关系。</p><div class="relation-graph">${modal.graph?.nodes?.length ? modal.graph.nodes.map((node) => `<div class="relation-node"><span class="avatar ${node.status === "confirmed" ? "green" : "gray"}">${escapeHtml((node.label || "?").slice(0, 1))}</span><strong>${escapeHtml(node.label)}</strong><small>${escapeHtml(node.status)}</small>${modal.graph.edges.filter((edge) => edge.source === node.id).map((edge) => `<em>${escapeHtml(edge.label)} · ${escapeHtml(edge.status)}</em>`).join("")}</div>`).join("") : emptyState("没有人物关系", "先确认人物实体，再创建关系候选。")}</div><div class="modal-actions"><button class="button primary" data-action="close-modal">关闭</button></div>`;
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

  async function refreshData() {
    state.loading = true;
    renderView();
    const spaceResult = await Promise.allSettled([window.sentrixApi.memorySpaces()]);
    if (spaceResult[0].status === "fulfilled") {
      state.spaces = spaceResult[0].value.spaces || [];
      if (state.spaces.length && !state.spaces.some((space) => space.id === state.scopeId)) state.scopeId = state.spaces[0].id;
    }
    const scopeId = state.scopeId;
    const calls = await Promise.allSettled([
      window.sentrixApi.dashboard(scopeId), window.sentrixApi.events(scopeId), window.sentrixApi.assets("?limit=1000", scopeId), window.sentrixApi.people("", scopeId), window.sentrixApi.stories(), window.sentrixApi.health(), window.sentrixApi.entities("", scopeId), window.sentrixApi.faceClusters("", scopeId), window.sentrixApi.relationships(scopeId), window.sentrixApi.knowledge("", scopeId),
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
    const failed = calls.find((call) => call.status === "rejected");
    state.backendError = failed ? "本地后端暂时不可用，当前页面只显示已读取到的真实数据。" : "";
    state.loading = false;
    renderShellNavigation();
  }

  function renderShellNavigation() { shell(); }

  function bindViewEvents() {
    document.querySelectorAll("[data-view]").forEach((element) => element.addEventListener("click", () => { state.view = element.dataset.view; state.modal = null; renderShellNavigation(); }));
    document.querySelectorAll("[data-query]").forEach((element) => element.addEventListener("click", () => { state.query = element.dataset.query; state.view = "search"; renderShellNavigation(); submitSearch(); }));
    document.querySelectorAll("[data-event-filter]").forEach((element) => element.addEventListener("click", () => { state.eventFilter = element.dataset.eventFilter; renderView(); }));
    document.querySelectorAll("[data-asset-filter]").forEach((element) => element.addEventListener("click", () => { state.assetFilter = element.dataset.assetFilter; renderView(); }));
    document.querySelectorAll("[data-person-filter]").forEach((element) => element.addEventListener("click", () => { state.personFilter = element.dataset.personFilter; renderView(); }));
    const spaceSelect = document.getElementById("space-select");
    if (spaceSelect) spaceSelect.addEventListener("change", async (event) => { state.scopeId = event.target.value; window.localStorage?.setItem("sentrix.scopeId", state.scopeId); state.modal = null; await refreshData(); });
    const form = document.getElementById("search-form");
    if (form) form.addEventListener("submit", submitSearch);
    const modalForm = document.getElementById("modal-form");
    if (modalForm) modalForm.addEventListener("submit", handleModalSubmit);
    document.querySelectorAll("[data-action]").forEach((element) => element.addEventListener("click", () => handleAction(element.dataset.action, element)));
    const fileInput = document.getElementById("file-input");
    if (fileInput) fileInput.addEventListener("change", handleFiles);
  }

  async function submitSearch(event) {
    if (event?.preventDefault) event.preventDefault();
    const input = document.getElementById("search-input");
    state.query = input ? input.value.trim() : state.query.trim();
    if (!state.query) return;
    state.view = "search";
    state.searchLoading = true;
    renderShellNavigation();
    try { state.searchResult = await window.sentrixApi.assistantTurn(state.query, state.conversationId, null, state.scopeId); state.conversationId = state.searchResult.conversation_id || state.conversationId; } catch (error) { state.searchResult = { answer: "检索失败，当前没有可用的本地答案。", confidence: 0, evidence: [], retrievalTrace: [], error: error.message, insufficient_evidence: true }; }
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
    try { const detail = await window.sentrixApi.event(eventId); openModal(edit ? { type: "event-edit", event: detail.event } : { type: "event", detail }); } catch { state.toast = "无法读取事件证据"; state.modal = null; renderShellNavigation(); }
  }

  async function openAsset(assetId) {
    openModal({ type: "loading" });
    try { const [asset, result] = await Promise.all([window.sentrixApi.asset(assetId), window.sentrixApi.observations(`?assetId=${encodeURIComponent(assetId)}`)]); openModal({ type: "asset", asset, observations: result.observations || [] }); } catch { state.toast = "无法读取原始资料"; state.modal = null; renderShellNavigation(); }
  }

  async function openEntity(entityId) {
    openModal({ type: "loading" });
    try { const detail = await window.sentrixApi.entity(entityId); openModal({ type: "entity", detail }); } catch { state.toast = "无法读取实体记忆"; state.modal = null; renderShellNavigation(); }
  }

  async function handleModalSubmit(event) {
    event.preventDefault();
    const form = new FormData(event.target);
    const modal = state.modal;
    try {
      if (modal.type === "event-edit") await window.sentrixApi.updateEvent(modal.event.id, { title: form.get("title"), summary: form.get("summary"), place: form.get("place"), time_start: form.get("time_start") ? new Date(form.get("time_start")).toISOString() : modal.event.time_start });
      if (modal.type === "event-create") await window.sentrixApi.createEvent({ title: form.get("title"), summary: form.get("summary"), place: form.get("place"), time_start: form.get("time_start") ? new Date(form.get("time_start")).toISOString() : null });
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
      if (modal.type === "story-create") await window.sentrixApi.createStory({ title: form.get("title"), content: form.get("content"), event_ids: form.getAll("event_ids") });
      if (modal.type === "story-edit") await window.sentrixApi.updateStory(modal.story.id, { title: form.get("title"), content: form.get("content"), event_ids: form.getAll("event_ids") });
      if (modal.type === "command") {
        const command = String(form.get("command") || "").trim();
        const target = navItems.find((item) => command.includes(item.label) || command.toLowerCase().includes(item.id));
        if (target) { state.modal = null; state.view = target.id; renderShellNavigation(); return; }
        state.modal = null; state.query = command; state.view = "search"; renderShellNavigation(); return submitSearch();
      }
      if (modal.type === "invite") { const invite = await window.sentrixApi.createInvite(form.get("label")); openModal({ type: "invite", invite }); return; }
      state.modal = null; state.toast = state.toast || "已保存到本地记忆"; await refreshData();
    } catch (error) { state.toast = `保存失败：${error.message}`; renderShellNavigation(); }
  }

  async function handleAction(action, element) {
    if (action === "close-modal") { state.modal = null; renderShellNavigation(); return; }
    if (action === "open-event") return openEvent(element.dataset.eventId);
    if (action === "edit-event") return openEvent(element.dataset.eventId, true);
    if (action === "open-asset") return openAsset(element.dataset.assetId);
    if (action === "open-entity") return openEntity(element.dataset.entityId);
    if (action === "open-observation") { const observation = await window.sentrixApi.observation(element.dataset.observationId); return openAsset(observation.asset_id); }
    if (action === "create-event") return openModal({ type: "event-create", event: {} });
    if (action === "create-story") return openModal({ type: "story-create", story: {} });
    if (action === "edit-story") { const story = state.stories.find((item) => item.id === element.dataset.storyId); return openModal({ type: "story-edit", story }); }
    if (action === "delete-story") { await window.sentrixApi.deleteStory(element.dataset.storyId); state.toast = "故事草稿已删除"; return refreshData(); }
    if (action === "open-person") { openModal({ type: "loading" }); try { const detail = await window.sentrixApi.personEvidence(element.dataset.personId, state.scopeId); return openModal({ type: "person-evidence", detail }); } catch (error) { state.modal = null; state.toast = `无法读取人物证据：${error.message}`; return renderShellNavigation(); } }
    if (action === "open-person-profile") { openModal({ type: "loading" }); const detail = await window.sentrixApi.personProfile(element.dataset.personId); return openModal({ type: "person-profile", detail }); }
    if (action === "confirm-person") { const person = state.persons.find((item) => item.id === element.dataset.personId) || { id: element.dataset.personId, name: "待确认人物" }; return openModal({ type: "person", person }); }
    if (action === "confirm-cluster") { const cluster = state.clusters.find((item) => item.id === element.dataset.clusterId); return openModal({ type: "cluster-confirm", cluster }); }
    if (action === "merge-cluster") { const cluster = state.clusters.find((item) => item.id === element.dataset.clusterId); return openModal({ type: "cluster-merge", cluster }); }
    if (action === "split-face") { const cluster = state.clusters.find((item) => item.id === element.dataset.clusterId); const sample = cluster?.samples?.find((item) => item.id === element.dataset.faceInstanceId); return openModal({ type: "cluster-split", cluster, sample }); }
    if (action === "reject-cluster") { await window.sentrixApi.rejectFaceCluster(element.dataset.clusterId); state.toast = "人物簇已驳回，原始人脸证据仍保留"; return refreshData(); }
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
    if (action === "reload") return refreshData();
    if (action === "recheck") { await fetch("/api/maintenance/recheck", { method: "POST" }); state.toast = "已提交失败任务重试"; return refreshData(); }
    if (action === "relationship-graph") { openModal({ type: "loading" }); const graph = await window.sentrixApi.relationships(); return openModal({ type: "relation", graph }); }
  }

  shell();
  refreshData();
})();
