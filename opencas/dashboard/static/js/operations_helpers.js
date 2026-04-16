(function (global) {
  const http = global.OpenCASHttp || {};
  const JSON_HEADERS = { "Content-Type": "application/json" };

  function operationsDetailPanel() {
    return document.getElementById("operations-detail");
  }

  function setOperationsDetail(html) {
    const detail = operationsDetailPanel();
    if (detail) detail.innerHTML = html;
  }

  function reloadHtmxPanel(panel) {
    if (panel) htmx.trigger(panel, "load");
  }

  function panelByPrefix(prefix) {
    return document.querySelector(`[hx-get^="${prefix}"]`);
  }

  function reloadHtmxPanelByPrefix(prefix) {
    reloadHtmxPanel(panelByPrefix(prefix));
  }

  function jsonOptions(method, payload) {
    return {
      method,
      headers: JSON_HEADERS,
      body: JSON.stringify(payload),
    };
  }

  async function requestJson(url, options, failureHtml) {
    const response = await fetch(url, options);
    if (!response.ok) {
      setOperationsDetail(failureHtml);
      return null;
    }
    return await (http.safeJson ? http.safeJson(response, {}) : response.json());
  }

  async function patchOperationsResource(resourcePath, resourceId, payload, config) {
    const decodedId = decodeURIComponent(resourceId);
    const data = await requestJson(
      `/api/operations/${resourcePath}/${encodeURIComponent(decodedId)}`,
      jsonOptions("PATCH", payload),
      config.failureHtml
    );
    if (!data) return null;
    setOperationsDetail(config.renderDetail(data));
    reloadHtmxPanelByPrefix(`/api/operations/${resourcePath}`);
    return data;
  }

  function getOperationSessionsPanel() {
    return document.querySelector('[data-panel="operations-sessions"]');
  }

  function getQualificationCardPanel() {
    return document.querySelector('[data-panel="qualification-card"]');
  }

  function getOperationsQualificationPanel() {
    return document.querySelector('[data-panel="operations-qualification"]');
  }

  function getOperationScope() {
    const panel = getOperationSessionsPanel();
    return panel?.dataset.scopeKey || "";
  }

  let qualificationRefreshTimer = null;

  function clearQualificationRefresh() {
    if (qualificationRefreshTimer !== null) {
      window.clearTimeout(qualificationRefreshTimer);
      qualificationRefreshTimer = null;
    }
  }

  function refreshQualificationPanels() {
    reloadHtmxPanel(getQualificationCardPanel());
    reloadHtmxPanel(getOperationsQualificationPanel());
  }

  function updateQualificationRefresh(active) {
    clearQualificationRefresh();
    if (!active) return;
    qualificationRefreshTimer = window.setTimeout(() => {
      refreshQualificationPanels();
      refreshOperationSessions("qualification");
    }, 5000);
  }

  function refreshOperationSessions(scopeKey = null) {
    const panel = getOperationSessionsPanel();
    if (!panel) return;
    const nextScope = scopeKey === null ? getOperationScope() : decodeURIComponent(scopeKey || "");
    panel.dataset.scopeKey = nextScope;
    panel.setAttribute(
      "hx-get",
      nextScope ? `/api/operations/sessions?scope_key=${encodeURIComponent(nextScope)}` : "/api/operations/sessions"
    );
    htmx.trigger(panel, "load");
  }

  function setOperationScope(scopeKey) {
    refreshOperationSessions(scopeKey);
  }

  function showOperationDetail(url) {
    const detail = operationsDetailPanel();
    if (!detail) return;
    detail.setAttribute("hx-get", url);
    htmx.trigger(detail, "load");
  }

  function showPtySessionDetail(sessionId, scopeKey, refresh = false) {
    const decodedSession = decodeURIComponent(sessionId);
    const decodedScope = decodeURIComponent(scopeKey);
    showOperationDetail(
      `/api/operations/sessions/pty/${encodeURIComponent(decodedSession)}?scope_key=${encodeURIComponent(decodedScope)}${refresh ? "&refresh=true" : ""}`
    );
  }

  function showBrowserSessionDetail(sessionId, scopeKey, refresh = false) {
    const decodedSession = decodeURIComponent(sessionId);
    const decodedScope = decodeURIComponent(scopeKey);
    showOperationDetail(
      `/api/operations/sessions/browser/${encodeURIComponent(decodedSession)}?scope_key=${encodeURIComponent(decodedScope)}${refresh ? "&refresh=true" : ""}`
    );
  }

  function showProcessDetail(processId, scopeKey, refresh = true) {
    const decodedProcess = decodeURIComponent(processId);
    const decodedScope = decodeURIComponent(scopeKey);
    showOperationDetail(
      `/api/operations/sessions/process/${encodeURIComponent(decodedProcess)}?scope_key=${encodeURIComponent(decodedScope)}${refresh ? "&refresh=true" : ""}`
    );
  }

  function browserControlValue(sessionId, suffix) {
    const decodedSession = decodeURIComponent(sessionId);
    const uiId = encodeURIComponent(decodedSession);
    const input = document.getElementById(`browser-${suffix}-${uiId}`);
    return input ? input.value : "";
  }

  function browserControlChecked(sessionId, suffix) {
    const decodedSession = decodeURIComponent(sessionId);
    const uiId = encodeURIComponent(decodedSession);
    const input = document.getElementById(`browser-${suffix}-${uiId}`);
    return Boolean(input?.checked);
  }

  async function mutateBrowserSession(sessionId, scopeKey, actionPath, payload, failureHtml) {
    const decodedSession = decodeURIComponent(sessionId);
    const decodedScope = decodeURIComponent(scopeKey);
    const data = await requestJson(
      `/api/operations/sessions/browser/${encodeURIComponent(decodedSession)}${actionPath}?scope_key=${encodeURIComponent(decodedScope)}`,
      jsonOptions("POST", payload),
      failureHtml
    );
    if (!data) return null;
    setOperationsDetail(global.renderBrowserSessionDetail(data));
    refreshOperationSessions();
    return data;
  }

  async function navigateBrowserSession(sessionId, scopeKey) {
    const url = browserControlValue(sessionId, "navigate-url").trim();
    if (!url) return;
    await mutateBrowserSession(
      sessionId,
      scopeKey,
      "/navigate",
      { url, wait_until: "load", timeout_ms: 30000, refresh: true },
      '<p class="muted">Failed to navigate browser session.</p>'
    );
  }

  async function captureBrowserSession(sessionId, scopeKey) {
    const fullPage = browserControlChecked(sessionId, "capture-full");
    await mutateBrowserSession(
      sessionId,
      scopeKey,
      "/capture",
      { full_page: fullPage },
      '<p class="muted">Failed to capture browser screenshot.</p>'
    );
  }

  async function closeBrowserSession(sessionId, scopeKey) {
    const decodedSession = decodeURIComponent(sessionId);
    const decodedScope = decodeURIComponent(scopeKey);
    const data = await requestJson(
      `/api/operations/sessions/browser/${encodeURIComponent(decodedSession)}?scope_key=${encodeURIComponent(decodedScope)}`,
      { method: "DELETE" },
      '<p class="muted">Failed to close browser session.</p>'
    );
    if (!data) return;
    setOperationsDetail(
      `<p class="muted">Closed browser session <code>${global.escapeHtml(decodedSession)}</code>.</p>`
    );
    refreshOperationSessions();
  }

  async function clickBrowserSession(sessionId, scopeKey) {
    const selector = browserControlValue(sessionId, "click-selector").trim();
    if (!selector) return;
    await mutateBrowserSession(
      sessionId,
      scopeKey,
      "/click",
      { selector, timeout_ms: 5000, refresh: true },
      '<p class="muted">Failed to click browser session selector.</p>'
    );
  }

  async function typeBrowserSession(sessionId, scopeKey) {
    const selector = browserControlValue(sessionId, "type-selector").trim();
    if (!selector) return;
    const text = browserControlValue(sessionId, "type-text");
    const clear = browserControlChecked(sessionId, "type-clear");
    await mutateBrowserSession(
      sessionId,
      scopeKey,
      "/type",
      { selector, text, clear, timeout_ms: 5000, refresh: true },
      '<p class="muted">Failed to type into browser session selector.</p>'
    );
  }

  async function pressBrowserSession(sessionId, scopeKey) {
    const key = browserControlValue(sessionId, "press-key").trim();
    if (!key) return;
    await mutateBrowserSession(
      sessionId,
      scopeKey,
      "/press",
      { key, refresh: true },
      '<p class="muted">Failed to press browser session key.</p>'
    );
  }

  async function waitBrowserSession(sessionId, scopeKey) {
    const selector = browserControlValue(sessionId, "wait-selector").trim();
    const loadState = browserControlValue(sessionId, "wait-load-state") || "load";
    const timeoutRaw = browserControlValue(sessionId, "wait-timeout") || "5000";
    const timeoutMs = Number.parseInt(timeoutRaw || "5000", 10);
    await mutateBrowserSession(
      sessionId,
      scopeKey,
      "/wait",
      {
        selector: selector || null,
        load_state: loadState,
        timeout_ms: Number.isFinite(timeoutMs) ? timeoutMs : 5000,
        refresh: true,
      },
      '<p class="muted">Failed to wait on browser session.</p>'
    );
  }

  async function sendPtyInput(sessionId, scopeKey) {
    const decodedSession = decodeURIComponent(sessionId);
    const decodedScope = decodeURIComponent(scopeKey);
    const uiId = encodeURIComponent(decodedSession);
    const inputEl = document.getElementById(`pty-input-${uiId}`);
    const input = inputEl ? inputEl.value : window.prompt("Send PTY input");
    if (input === null || input === "") return;
    const data = await requestJson(
      `/api/operations/sessions/pty/${encodeURIComponent(decodedSession)}/input?scope_key=${encodeURIComponent(decodedScope)}`,
      jsonOptions("POST", { input, observe: true, idle_seconds: 0.25, max_wait_seconds: 1.5 }),
      '<p class="muted">Failed to send PTY input.</p>'
    );
    if (!data) return;
    setOperationsDetail(global.renderPtySessionDetail(data));
    if (inputEl) inputEl.value = "";
    refreshOperationSessions();
  }

  async function killPtySession(sessionId, scopeKey) {
    const decodedSession = decodeURIComponent(sessionId);
    const decodedScope = decodeURIComponent(scopeKey);
    const data = await requestJson(
      `/api/operations/sessions/pty/${encodeURIComponent(decodedSession)}?scope_key=${encodeURIComponent(decodedScope)}`,
      { method: "DELETE" },
      '<p class="muted">Failed to stop PTY session.</p>'
    );
    if (!data) return;
    setOperationsDetail(
      `<p class="muted">Stopped PTY session <code>${global.escapeHtml(decodedSession)}</code>.</p>`
    );
    refreshOperationSessions();
  }

  async function clearScopedSessions(scopeKey, kind, failureHtml, successLabel) {
    const resolvedScope = decodeURIComponent(scopeKey || getOperationScope() || "default");
    const data = await requestJson(
      `/api/operations/sessions/${kind}?scope_key=${encodeURIComponent(resolvedScope)}`,
      { method: "DELETE" },
      failureHtml
    );
    if (!data) return;
    setOperationsDetail(
      `<p class="muted">Cleared ${global.escapeHtml(String(data.removed ?? 0))} ${successLabel} for scope <code>${global.escapeHtml(data.scope_key || resolvedScope)}</code>.</p>`
    );
    refreshOperationSessions();
  }

  async function clearPtySessions(scopeKey = "default") {
    await clearScopedSessions(
      scopeKey,
      "pty",
      '<p class="muted">Failed to clear PTY sessions.</p>',
      "PTY session(s)"
    );
  }

  async function clearBrowserSessions(scopeKey = "default") {
    await clearScopedSessions(
      scopeKey,
      "browser",
      '<p class="muted">Failed to clear browser sessions.</p>',
      "browser session(s)"
    );
  }

  async function clearProcessSessions(scopeKey = "default") {
    await clearScopedSessions(
      scopeKey,
      "process",
      '<p class="muted">Failed to clear background processes.</p>',
      "background process(es)"
    );
  }

  async function killProcessSession(processId, scopeKey) {
    const decodedProcess = decodeURIComponent(processId);
    const decodedScope = decodeURIComponent(scopeKey);
    const data = await requestJson(
      `/api/operations/sessions/process/${encodeURIComponent(decodedProcess)}?scope_key=${encodeURIComponent(decodedScope)}`,
      { method: "DELETE" },
      '<p class="muted">Failed to stop background process.</p>'
    );
    if (!data) return;
    setOperationsDetail(
      `<p class="muted">Removed background process <code>${global.escapeHtml(decodedProcess)}</code>.</p>`
    );
    refreshOperationSessions();
  }

  async function startQualificationRerun(label, note = "") {
    const decodedLabel = decodeURIComponent(label);
    const decodedNote = decodeURIComponent(note || "");
    const data = await requestJson(
      "/api/operations/qualification/reruns",
      jsonOptions("POST", {
        label: decodedLabel,
        iterations: 2,
        include_direct_checks: false,
        source_label: decodedLabel,
        source_note: decodedNote,
      }),
      '<p class="muted">Failed to start qualification rerun.</p>'
    );
    if (!data) return;
    setOperationsDetail(
      `<p class="muted">Started qualification rerun for <code>${global.escapeHtml(decodedLabel)}</code> as process <code>${global.escapeHtml(data.process_id || "-")}</code>.</p><pre class="json">${global.escapeHtml(JSON.stringify(data, null, 2))}</pre>`
    );
    refreshQualificationPanels();
    refreshOperationSessions("qualification");
  }

  async function updatePlanStatus(planId, status) {
    await patchOperationsResource("plans", planId, { status }, {
      failureHtml: '<p class="muted">Failed to update plan status.</p>',
      renderDetail: global.renderPlanDetail,
    });
  }

  async function savePlanForm(planId) {
    const status = document.getElementById(`plan-status-${planId}`)?.value;
    const content = document.getElementById(`plan-content-${planId}`)?.value;
    await patchOperationsResource("plans", planId, { status, content }, {
      failureHtml: '<p class="muted">Failed to save plan.</p>',
      renderDetail: global.renderPlanDetail,
    });
  }

  async function updatePlanContent(planId, currentContent) {
    const content = window.prompt("Plan content", currentContent || "");
    if (content === null) return;
    await patchOperationsResource("plans", planId, { content }, {
      failureHtml: '<p class="muted">Failed to update plan content.</p>',
      renderDetail: global.renderPlanDetail,
    });
  }

  async function updateCommitmentStatus(commitmentId, status) {
    await patchOperationsResource("commitments", commitmentId, { status }, {
      failureHtml: '<p class="muted">Failed to update commitment status.</p>',
      renderDetail: global.renderCommitmentDetail,
    });
  }

  async function saveCommitmentForm(commitmentId) {
    const status = document.getElementById(`commitment-status-${commitmentId}`)?.value;
    const content = document.getElementById(`commitment-content-${commitmentId}`)?.value;
    await patchOperationsResource("commitments", commitmentId, { status, content }, {
      failureHtml: '<p class="muted">Failed to save commitment.</p>',
      renderDetail: global.renderCommitmentDetail,
    });
  }

  async function updateCommitmentContent(commitmentId, currentContent) {
    const content = window.prompt("Commitment content", currentContent || "");
    if (content === null) return;
    await patchOperationsResource("commitments", commitmentId, { content }, {
      failureHtml: '<p class="muted">Failed to update commitment content.</p>',
      renderDetail: global.renderCommitmentDetail,
    });
  }

  async function updateWorkStage(workId) {
    const stage = window.prompt("Work stage", "note");
    if (stage === null || stage === "") return;
    await patchOperationsResource("work", workId, { stage }, {
      failureHtml: '<p class="muted">Failed to update work stage.</p>',
      renderDetail: global.renderWorkItemDetail,
    });
  }

  async function saveWorkItemForm(workId) {
    const stage = document.getElementById(`work-stage-${workId}`)?.value;
    const content = document.getElementById(`work-content-${workId}`)?.value;
    const blockersRaw = document.getElementById(`work-blockers-${workId}`)?.value || "";
    const blocked_by = blockersRaw.split(",").map((item) => item.trim()).filter(Boolean);
    await patchOperationsResource("work", workId, { stage, content, blocked_by }, {
      failureHtml: '<p class="muted">Failed to save work item.</p>',
      renderDetail: global.renderWorkItemDetail,
    });
  }

  async function updateWorkContent(workId, currentContent) {
    const content = window.prompt("Work content", currentContent || "");
    if (content === null) return;
    await patchOperationsResource("work", workId, { content }, {
      failureHtml: '<p class="muted">Failed to update work content.</p>',
      renderDetail: global.renderWorkItemDetail,
    });
  }

  async function updateWorkBlockers(workId, currentBlockedBy) {
    const raw = window.prompt(
      "Blocked-by list, comma separated",
      (currentBlockedBy || []).join(", ")
    );
    if (raw === null) return;
    const blocked_by = raw.split(",").map((item) => item.trim()).filter(Boolean);
    await patchOperationsResource("work", workId, { blocked_by }, {
      failureHtml: '<p class="muted">Failed to update work blockers.</p>',
      renderDetail: global.renderWorkItemDetail,
    });
  }

  Object.assign(global, {
    clearBrowserSessions,
    clearProcessSessions,
    clearPtySessions,
    clearQualificationRefresh,
    closeBrowserSession,
    clickBrowserSession,
    showBrowserSessionDetail,
    showProcessDetail,
    showPtySessionDetail,
    setOperationScope,
    sendPtyInput,
    captureBrowserSession,
    getOperationScope,
    navigateBrowserSession,
    pressBrowserSession,
    refreshOperationSessions,
    refreshQualificationPanels,
    showOperationDetail,
    startQualificationRerun,
    killProcessSession,
    killPtySession,
    saveCommitmentForm,
    savePlanForm,
    saveWorkItemForm,
    typeBrowserSession,
    updateCommitmentContent,
    updateCommitmentStatus,
    updatePlanContent,
    updatePlanStatus,
    updateQualificationRefresh,
    updateWorkBlockers,
    updateWorkContent,
    updateWorkStage,
    waitBrowserSession,
  });
})(window);
