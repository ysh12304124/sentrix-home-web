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
    memorySpaces: () => request("/api/memory-spaces"),
    createMemorySpace: (name) => request("/api/memory-spaces", { method: "POST", body: JSON.stringify({ name }) }),
    dashboard: (scopeId = "") => request(`/api/dashboard${scopeId ? `?scope_id=${encodeURIComponent(scopeId)}` : ""}`),
    events: (scopeId = "") => request(`/api/events${scopeId ? `?scope_id=${encodeURIComponent(scopeId)}` : ""}`),
    trips: (scopeId = "", status = "") => request(`/api/trips${new URLSearchParams({ ...(scopeId ? { scope_id: scopeId } : {}), ...(status ? { status } : {}) }).toString().replace(/^/, "?")}`),
    trip: (id) => request(`/api/trips/${encodeURIComponent(id)}`),
    confirmTrip: (id, payload) => request(`/api/trips/${encodeURIComponent(id)}/confirm`, { method: "POST", body: JSON.stringify(payload) }),
    rejectTrip: (id) => request(`/api/trips/${encodeURIComponent(id)}/reject`, { method: "POST" }),
    event: (id) => request(`/api/events/${encodeURIComponent(id)}`),
    createEvent: (payload) => request("/api/events", { method: "POST", body: JSON.stringify(payload) }),
    updateEvent: (id, payload) => request(`/api/events/${encodeURIComponent(id)}`, { method: "PATCH", body: JSON.stringify(payload) }),
    assets: (params = "", scopeId = "") => request(`/api/assets${params}${scopeId ? `${params.includes("?") ? "&" : "?"}scope_id=${encodeURIComponent(scopeId)}` : ""}`),
    asset: (id) => request(`/api/assets/${encodeURIComponent(id)}`),
    observation: (id) => request(`/api/observations/${encodeURIComponent(id)}`),
    observations: (params = "") => request(`/api/observations${params}`),
    persons: () => request("/api/persons"),
    people: (status = "", scopeId = "") => request("/api/people" + new URLSearchParams({ ...(status ? { status } : {}), ...(scopeId ? { scope_id: scopeId } : {}) }).toString().replace(/^/, "?")),
    personProfile: (id) => request("/api/people/" + encodeURIComponent(id) + "/profile"),
    personEvidence: (id, scopeId = "") => request("/api/people/" + encodeURIComponent(id) + "/evidence" + (scopeId ? `?scope_id=${encodeURIComponent(scopeId)}` : "")),
    knowledge: (personId = "", scopeId = "") => request("/api/knowledge" + new URLSearchParams({ ...(personId ? { person_id: personId } : {}), ...(scopeId ? { scope_id: scopeId } : {}) }).toString().replace(/^/, "?")),
    entities: (status = "", scopeId = "") => request(`/api/entities${new URLSearchParams({ ...(status ? { status } : {}), ...(scopeId ? { scope_id: scopeId } : {}) }).toString().replace(/^/, "?")}`),
    entityGroups: (scopeId = "") => request(`/api/entity-groups${scopeId ? `?scope_id=${encodeURIComponent(scopeId)}` : ""}`),
    entityGroup: (id, scopeId = "") => request(`/api/entity-groups/${encodeURIComponent(id)}${scopeId ? `?scope_id=${encodeURIComponent(scopeId)}` : ""}`),
    entityMergeCandidates: (scopeId = "", status = "pending") => request(`/api/entity-merge-candidates${new URLSearchParams({ ...(scopeId ? { scope_id: scopeId } : {}), ...(status ? { status } : {}) }).toString().replace(/^/, "?")}`),
    deriveEntityMergeCandidates: (scopeId = "") => request(`/api/maintenance/entity-merge-candidates${scopeId ? `?scope_id=${encodeURIComponent(scopeId)}` : ""}`, { method: "POST" }),
    confirmEntityMergeCandidate: (id, targetEntityId) => request(`/api/entity-merge-candidates/${encodeURIComponent(id)}/confirm`, { method: "POST", body: JSON.stringify({ target_entity_id: targetEntityId }) }),
    rejectEntityMergeCandidate: (id) => request(`/api/entity-merge-candidates/${encodeURIComponent(id)}/reject`, { method: "POST" }),
    entity: (id) => request(`/api/entities/${encodeURIComponent(id)}`),
    setEntityProperty: (id, propertyKey, value, evidenceIds = []) => request(`/api/entities/${encodeURIComponent(id)}/properties/${encodeURIComponent(propertyKey)}`, { method: "PUT", body: JSON.stringify({ value, evidence_ids: evidenceIds }) }),
    faceClusters: (status = "", scopeId = "") => request(`/api/face-clusters${new URLSearchParams({ ...(status ? { status } : {}), ...(scopeId ? { scope_id: scopeId } : {}) }).toString().replace(/^/, "?")}`),
    confirmFaceCluster: (id, payload) => request(`/api/face-clusters/${encodeURIComponent(id)}/confirm`, { method: "POST", body: JSON.stringify(payload) }),
    rejectFaceCluster: (id) => request(`/api/face-clusters/${encodeURIComponent(id)}/reject`, { method: "POST" }),
    mergeFaceClusters: (targetClusterId, sourceClusterId) => request("/api/face-clusters/merge", { method: "POST", body: JSON.stringify({ target_cluster_id: targetClusterId, source_cluster_id: sourceClusterId }) }),
    splitFaceCluster: (clusterId, faceInstanceId) => request(`/api/face-clusters/${encodeURIComponent(clusterId)}/split`, { method: "POST", body: JSON.stringify({ face_instance_id: faceInstanceId }) }),
    relationships: (scopeId = "", kind = "") => request(`/api/relationships${new URLSearchParams({ ...(scopeId ? { scope_id: scopeId } : {}), ...(kind ? { kind } : {}) }).toString().replace(/^/, "?")}`),
    createRelationship: (payload) => request("/api/relationships", { method: "POST", body: JSON.stringify(payload) }),
    confirmRelationship: (id) => request(`/api/relationships/${encodeURIComponent(id)}/confirm`, { method: "POST" }),
    retractRelationship: (id) => request(`/api/relationships/${encodeURIComponent(id)}/retract`, { method: "POST" }),
    confirmPerson: (id, name, familyRole = "") => request(`/api/persons/${encodeURIComponent(id)}/confirm`, { method: "POST", body: JSON.stringify({ name, family_role: familyRole }) }),
    assistantTurn: (message, conversationId = "", feedback = null, scopeId = "home-default", selectedEntityId = "", viewerId = "owner") => request("/api/assistant/turn", { method: "POST", body: JSON.stringify({ message, conversation_id: conversationId || null, feedback, scope_id: scopeId, selected_entity_id: selectedEntityId || null, viewer_id: viewerId || "owner" }) }),
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
    importAssets: (items, options = {}) => {
      const form = new FormData();
      items.forEach((item) => form.append("files", item.file, item.file.name));
      form.append("metadata", JSON.stringify(items.map((item) => item.metadata || {})));
      const fields = {
        scopeId: options.scopeId,
        batchId: options.batchId,
        sourceOwnerId: options.sourceOwnerId,
        sourceOwnerLabel: options.sourceOwnerLabel,
        sourceDeviceId: options.sourceDeviceId,
        sourceAlbumId: options.sourceAlbumId,
      };
      Object.entries(fields).forEach(([key, value]) => {
        if (value !== undefined && value !== null && String(value).trim()) form.append(key, value);
      });
      return request("/api/import", { method: "POST", body: form });
    },
    getVlmBackend: () => request("/api/vlm-backend"),
    setVlmBackend: (backend) => request("/api/vlm-backend", { method: "POST", body: JSON.stringify({ backend }) }),
    importAsset: (file, metadata = {}, options = {}) => {
      return window.sentrixApi.importAssets([{ file, metadata }], options).then((result) => result.items[0]);
    },
  };
})();
