(function () {
  const configuredBase = window.SENTRIX_API_BASE || "";

  async function request(path, options) {
    const body = options && options.body;
    const isFormData = typeof FormData !== "undefined" && body instanceof FormData;
    const response = await fetch(`${configuredBase}${path}`, {
      headers: { ...(isFormData ? {} : { "content-type": "application/json" }), ...(options && options.headers) },
      ...options,
    });
    if (!response.ok) throw new Error(`Sentrix API ${response.status}`);
    return response.json();
  }

  window.sentrixApi = {
    health: () => request("/api/health"),
    dashboard: () => request("/api/dashboard"),
    events: () => request("/api/events"),
    event: (id) => request(`/api/events/${encodeURIComponent(id)}`),
    createEvent: (payload) => request("/api/events", { method: "POST", body: JSON.stringify(payload) }),
    updateEvent: (id, payload) => request(`/api/events/${encodeURIComponent(id)}`, { method: "PATCH", body: JSON.stringify(payload) }),
    assets: (params = "") => request(`/api/assets${params}`),
    asset: (id) => request(`/api/assets/${encodeURIComponent(id)}`),
    observation: (id) => request(`/api/observations/${encodeURIComponent(id)}`),
    observations: (params = "") => request(`/api/observations${params}`),
    persons: () => request("/api/persons"),
    people: (status = "") => request("/api/people" + (status ? "?status=" + encodeURIComponent(status) : "")),
    personProfile: (id) => request("/api/people/" + encodeURIComponent(id) + "/profile"),
    knowledge: (personId = "") => request("/api/knowledge" + (personId ? "?person_id=" + encodeURIComponent(personId) : "")),
    entities: (status = "") => request(`/api/entities${status ? `?status=${encodeURIComponent(status)}` : ""}`),
    entity: (id) => request(`/api/entities/${encodeURIComponent(id)}`),
    faceClusters: (status = "") => request(`/api/face-clusters${status ? `?status=${encodeURIComponent(status)}` : ""}`),
    confirmFaceCluster: (id, payload) => request(`/api/face-clusters/${encodeURIComponent(id)}/confirm`, { method: "POST", body: JSON.stringify(payload) }),
    rejectFaceCluster: (id) => request(`/api/face-clusters/${encodeURIComponent(id)}/reject`, { method: "POST" }),
    relationships: () => request("/api/relationships"),
    createRelationship: (payload) => request("/api/relationships", { method: "POST", body: JSON.stringify(payload) }),
    confirmRelationship: (id) => request(`/api/relationships/${encodeURIComponent(id)}/confirm`, { method: "POST" }),
    confirmPerson: (id, name) => request(`/api/persons/${encodeURIComponent(id)}/confirm`, { method: "POST", body: JSON.stringify({ name }) }),
    rejectPerson: (id) => request(`/api/persons/${encodeURIComponent(id)}/reject`, { method: "POST" }),
    stories: () => request("/api/stories"),
    createStory: (payload) => request("/api/stories", { method: "POST", body: JSON.stringify(payload) }),
    updateStory: (id, payload) => request(`/api/stories/${encodeURIComponent(id)}`, { method: "PATCH", body: JSON.stringify(payload) }),
    deleteStory: (id) => request(`/api/stories/${encodeURIComponent(id)}`, { method: "DELETE" }),
    createInvite: (label) => request("/api/invites", { method: "POST", body: JSON.stringify({ label }) }),
    confirmFact: (factId) => request(`/api/facts/${encodeURIComponent(factId)}/confirm`, { method: "POST" }),
    rejectFact: (factId) => request(`/api/facts/${encodeURIComponent(factId)}/reject`, { method: "POST" }),
    search: (query) => request("/api/search", { method: "POST", body: JSON.stringify({ query }) }),
    queryGaps: () => request("/api/query-gaps"),
    queryGapFeedback: (id, payload) => request("/api/query-gaps/" + encodeURIComponent(id) + "/feedback", { method: "POST", body: JSON.stringify(payload) }),
    importAsset: (file, mediaType) => {
      if (typeof file === "string") return request("/api/import", { method: "POST", body: JSON.stringify({ fileName: file, mediaType }) });
      const form = new FormData();
      form.append("file", file, file.name);
      return request("/api/ingest", { method: "POST", body: form });
    },
  };
})();
