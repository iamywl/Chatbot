(() => {
  "use strict";

  const DEFAULT_PARAMS = {
    temperature: 0.6,
    top_p: 0.95,
    presence_penalty: 0.2,
    max_tokens: 4096,
    reasoning_effort: "medium",
  };

  const state = {
    vault: { initialized: false, unlocked: false, kdf: null, auto_lock_minutes: 0, encrypted_sessions_count: 0 },
    model: { status: "unloaded", model_id: "mlx-community/gpt-oss-120b-MXFP4-Q4", memory: {}, available_models: [] },
    git: { is_repo: false, pending_count: 0, auto_sync: true, syncing: false },
    personas: [],
    sessions: [],
    currentSession: null,
    searchQuery: "",
    personaFilter: "",
    streaming: false,
    editingPersonaName: "default",
    initBackupKey: "",
  };

  const $ = (id) => document.getElementById(id);

  // ----- Helpers -----

  function escapeHtml(str) {
    return String(str ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function toast(msg, type = "info") {
    const container = $("toast-container");
    const el = document.createElement("div");
    el.className = `toast ${type}`;
    el.textContent = msg;
    container.appendChild(el);
    setTimeout(() => {
      el.style.opacity = "0";
      el.style.transition = "opacity 0.25s";
      setTimeout(() => el.remove(), 260);
    }, 3200);
  }

  function downloadTextFile(filename, text, mime = "text/plain;charset=utf-8") {
    const blob = new Blob([text], { type: mime });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }

  async function copyToClipboard(text, okMsg = "복사되었습니다") {
    try {
      await navigator.clipboard.writeText(text);
      toast(okMsg, "success");
    } catch {
      toast("클립보드 복사에 실패했습니다", "error");
    }
  }

  function formatRelTime(isoStr) {
    if (!isoStr) return "";
    const dt = new Date(isoStr);
    if (Number.isNaN(dt.getTime())) return isoStr.slice(0, 10);
    const diffSec = Math.floor((Date.now() - dt.getTime()) / 1000);
    if (diffSec < 60) return "방금 전";
    if (diffSec < 3600) return `${Math.floor(diffSec / 60)}분 전`;
    if (diffSec < 86400) return `${Math.floor(diffSec / 3600)}시간 전`;
    return isoStr.slice(0, 10);
  }

  async function api(path, options = {}) {
    const headers = { ...(options.headers || {}) };
    if (options.body && !(options.body instanceof FormData) && !headers["Content-Type"]) {
      headers["Content-Type"] = "application/json";
    }
    const resp = await fetch(path, { ...options, headers });
    if (resp.status === 423) {
      handleVaultLocked();
      throw new Error("Vault가 잠겼습니다. 다시 잠금을 해제해 주세요.");
    }
    const isJson = (resp.headers.get("content-type") || "").includes("application/json");
    const data = isJson ? await resp.json() : await resp.text();
    if (!resp.ok) {
      const detail = typeof data === "object" && data?.detail ? data.detail : `HTTP ${resp.status}`;
      throw new Error(detail);
    }
    return data;
  }

  // ----- Markdown & Code Rendering -----

  function renderMarkdown(mdText) {
    const text = mdText || "";
    let html;
    if (window.marked && typeof window.marked.parse === "function") {
      html = window.marked.parse(text, { gfm: true, breaks: true });
      if (window.DOMPurify) {
        html = window.DOMPurify.sanitize(html, {
          USE_PROFILES: { html: true },
          ADD_TAGS: ["details", "summary"],
        });
      }
    } else {
      html = `<pre style="white-space:pre-wrap;margin:0;font-family:inherit">${escapeHtml(text)}</pre>`;
    }

    const wrapper = document.createElement("div");
    wrapper.className = "markdown-body";
    wrapper.innerHTML = html;

    // Enhance code blocks with syntax highlighting + copy button
    wrapper.querySelectorAll("pre > code").forEach((codeEl) => {
      const pre = codeEl.parentElement;
      const cls = codeEl.className || "";
      const langMatch = cls.match(/language-([\w+-]+)/);
      const lang = langMatch ? langMatch[1] : "code";

      if (window.hljs && !codeEl.dataset.highlighted) {
        try {
          window.hljs.highlightElement(codeEl);
        } catch {
          // ignore unknown language
        }
      }

      const rawCode = codeEl.textContent || "";
      const wrap = document.createElement("div");
      wrap.className = "code-block-wrap";

      const head = document.createElement("div");
      head.className = "code-block-header";
      head.innerHTML = `<span>${escapeHtml(lang)}</span>`;

      const copyBtn = document.createElement("button");
      copyBtn.className = "code-copy-btn";
      copyBtn.textContent = "복사";
      copyBtn.addEventListener("click", () => copyToClipboard(rawCode, "코드를 복사했습니다"));
      head.appendChild(copyBtn);

      pre.parentNode.insertBefore(wrap, pre);
      wrap.appendChild(head);
      wrap.appendChild(pre);
    });

    // Render KaTeX math if loaded
    if (window.renderMathInElement) {
      try {
        window.renderMathInElement(wrapper, {
          delimiters: [
            { left: "$$", right: "$$", display: true },
            { left: "$", right: "$", display: false },
            { left: "\\(", right: "\\)", display: false },
            { left: "\\[", right: "\\]", display: true },
          ],
          throwOnError: false,
        });
      } catch {
        // ignore math render errors
      }
    }
    return wrapper;
  }

  // ----- Vault Management -----

  function handleVaultLocked() {
    state.vault.unlocked = false;
    state.sessions = [];
    state.currentSession = null;
    $("session-list").innerHTML = '<div class="empty-hint">Vault가 잠겨 있습니다</div>';
    $("messages-container").innerHTML = "";
    renderVaultModal();
  }

  function showVaultError(msg) {
    const el = $("vault-modal-error");
    if (!msg) {
      el.classList.add("hidden");
      el.textContent = "";
    } else {
      el.classList.remove("hidden");
      el.textContent = msg;
    }
  }

  function renderVaultModal() {
    const modal = $("vault-modal");
    const initView = $("vault-init-view");
    const unlockView = $("vault-unlock-view");
    const backupView = $("vault-backup-view");
    const encBadge = $("vault-enc-badge");

    showVaultError("");

    if (state.initBackupKey) {
      modal.classList.remove("hidden");
      initView.classList.add("hidden");
      unlockView.classList.add("hidden");
      backupView.classList.remove("hidden");
      $("vault-modal-title").textContent = "Vault가 생성되었습니다";
      $("init-backup-key-code").textContent = state.initBackupKey;
      return;
    }

    if (state.vault.unlocked) {
      modal.classList.add("hidden");
      return;
    }

    modal.classList.remove("hidden");
    backupView.classList.add("hidden");

    const count = state.vault.encrypted_sessions_count || 0;
    if (count > 0) {
      encBadge.textContent = `암호화된 과거 세션 ${count}개 감지됨`;
      encBadge.classList.remove("hidden");
    } else {
      encBadge.classList.add("hidden");
    }

    if (!state.vault.initialized) {
      $("vault-modal-title").textContent = "Private Vault 초기 설정";
      initView.classList.remove("hidden");
      unlockView.classList.add("hidden");
    } else {
      $("vault-modal-title").textContent = "Private Vault 잠금 해제";
      initView.classList.add("hidden");
      unlockView.classList.remove("hidden");

      // If initialized with keyfile-only, auto-select key tab
      if (state.vault.kdf === "keyfile") {
        switchUnlockTab("keyfile");
        $("tab-btn-unlock-pw").classList.add("hidden");
      } else {
        $("tab-btn-unlock-pw").classList.remove("hidden");
        switchUnlockTab("passphrase");
      }
    }
  }

  function switchInitTab(tab) {
    document.querySelectorAll("[data-init-tab]").forEach((b) => {
      b.classList.toggle("active", b.dataset.initTab === tab);
    });
    $("init-tab-passphrase").classList.toggle("hidden", tab !== "passphrase");
    $("init-tab-keyfile").classList.toggle("hidden", tab !== "keyfile");
    showVaultError("");
  }

  function switchUnlockTab(tab) {
    document.querySelectorAll("[data-unlock-tab]").forEach((b) => {
      b.classList.toggle("active", b.dataset.unlockTab === tab);
    });
    $("unlock-tab-passphrase").classList.toggle("hidden", tab !== "passphrase");
    $("unlock-tab-keyfile").classList.toggle("hidden", tab !== "keyfile");
    showVaultError("");
  }

  async function refreshVaultStatus() {
    state.vault = await api("/api/vault/status");
    $("sidebar-enc-count").textContent = `${state.vault.encrypted_sessions_count || 0}개 암호화됨`;
    $("vault-autolock-select").value = String(state.vault.auto_lock_minutes || 0);
    renderVaultModal();
  }

  async function onVaultUnlocked() {
    $("unlock-passphrase").value = "";
    $("unlock-key-text").value = "";
    await Promise.all([loadPersonas(), loadSessions(), refreshGitStatus()]);
    if (state.sessions.length > 0) {
      const firstReadable = state.sessions.find((s) => !s.decrypt_error);
      if (firstReadable) {
        await openSession(firstReadable.session_id);
      }
    } else {
      await createNewSession();
    }
  }

  // ----- Model Lifecycle & Telemetry -----

  async function refreshModelStatus() {
    try {
      const data = await api("/api/model/status");
      state.model = data;
      renderModelStatus();
    } catch {
      // ignore transient network errors
    }
  }

  function renderModelStatus() {
    const m = state.model;
    const dot = $("model-status-dot");
    const statusLbl = $("model-status-label");
    const nameLbl = $("model-name-label");
    const toggleBtn = $("btn-model-toggle");
    const banner = $("model-banner");
    const bannerText = $("model-banner-text");
    const bannerBtn = $("btn-banner-load");

    const shortName = (m.model_id || "gpt-oss-120b").split("/").pop().replace("-MXFP4-Q4", "");
    nameLbl.textContent = shortName;

    dot.className = `status-dot ${m.status}`;

    const labels = {
      unloaded: "Unloaded (메모리 미적재)",
      loading: "서버 시작 중...",
      warming: "가중치 적재 중 (60GB)...",
      ready: `Ready · ${m.memory?.model_rss_gb || 0} GB`,
      stopping: "메모리 해제 중...",
      error: "오류 발생",
    };
    statusLbl.textContent = labels[m.status] || m.status;

    if (m.status === "unloaded" || m.status === "error") {
      toggleBtn.textContent = "Load";
      toggleBtn.className = "btn btn-sm btn-primary";
      toggleBtn.disabled = false;
      banner.classList.remove("hidden");
      bannerText.textContent =
        m.status === "error"
          ? `모델 로드 오류: ${m.last_error || "콘솔 로그를 확인하세요"}`
          : "로컬 모델이 메모리에 적재되어 있지 않습니다. 대화를 시작하려면 모델을 올려주세요.";
      bannerBtn.classList.remove("hidden");
      bannerBtn.disabled = false;
    } else if (m.status === "loading" || m.status === "warming") {
      toggleBtn.textContent = "Cancel";
      toggleBtn.className = "btn btn-sm btn-danger-ghost";
      toggleBtn.disabled = false;
      banner.classList.remove("hidden");
      bannerText.textContent = m.status_detail || "모델을 통합 메모리에 적재하고 있습니다...";
      bannerBtn.classList.add("hidden");
    } else if (m.status === "stopping") {
      toggleBtn.textContent = "Unloading...";
      toggleBtn.className = "btn btn-sm btn-ghost";
      toggleBtn.disabled = true;
      banner.classList.add("hidden");
    } else {
      // ready
      toggleBtn.textContent = "Unload";
      toggleBtn.className = "btn btn-sm btn-danger-ghost";
      toggleBtn.disabled = false;
      banner.classList.add("hidden");
    }

    // RAM Gauge
    const mem = m.memory || {};
    if (mem.system_total_gb) {
      $("ram-used-text").textContent = mem.system_used_gb;
      $("ram-total-text").textContent = mem.system_total_gb;
      $("ram-mlx-text").textContent = `MLX ${mem.model_rss_gb || 0} GB`;
      const bar = $("ram-bar-fill");
      bar.style.width = `${Math.min(100, mem.system_percent || 0)}%`;
      bar.classList.toggle("warn", (mem.system_percent || 0) >= 85);
    }

    // Update model modal if open
    if (!$("model-modal").classList.contains("hidden")) {
      renderModelModalDetails();
    }
  }

  function renderModelModalDetails() {
    const m = state.model;
    const select = $("model-select");
    const models = m.available_models || [];
    if (models.length && select.options.length !== models.length) {
      select.innerHTML = models
        .map(
          (item) =>
            `<option value="${escapeHtml(item.id)}" ${item.id === m.model_id ? "selected" : ""}>${escapeHtml(item.id)} (${item.size_gb} GB)</option>`
        )
        .join("");
    }
    $("idle-unload-select").value = String(m.idle_unload_minutes ?? 30);

    const mem = m.memory || {};
    $("model-metrics-grid").innerHTML = `
      <div class="metric-tile"><span>상태</span><strong>${escapeHtml(m.status)}</strong></div>
      <div class="metric-tile"><span>MLX 프로세스 메모리</span><strong>${mem.model_rss_gb || 0} GB ${mem.pid ? `(PID ${mem.pid})` : ""}</strong></div>
      <div class="metric-tile"><span>시스템 여유 RAM</span><strong>${mem.system_available_gb ?? "--"} / ${mem.system_total_gb ?? "--"} GB</strong></div>
      <div class="metric-tile"><span>유휴 시간</span><strong>${m.status === "ready" ? `${Math.floor((m.idle_seconds || 0) / 60)}분 ${(m.idle_seconds || 0) % 60}초` : "-"}</strong></div>
    `;
  }

  async function refreshModelLogs() {
    try {
      const data = await api("/api/model/logs?limit=120");
      const box = $("model-console-output");
      box.textContent = (data.logs || []).length ? data.logs.join("\n") : "아직 로그가 없습니다.";
      box.scrollTop = box.scrollHeight;
    } catch {
      // ignore
    }
  }

  async function handleModelLoad(modelId = null) {
    try {
      toast("모델을 통합 메모리에 적재합니다 (약 60GB, 수십 초 소요될 수 있음)...");
      state.model = await api("/api/model/load", {
        method: "POST",
        body: JSON.stringify({ model_id: modelId || $("model-select")?.value || null }),
      });
      renderModelStatus();
      await refreshModelLogs();
    } catch (err) {
      toast(`모델 적재 실패: ${err.message}`, "error");
    }
  }

  async function handleModelUnload() {
    try {
      toast("모델을 메모리에서 해제합니다...");
      state.model = await api("/api/model/unload", { method: "POST" });
      renderModelStatus();
      toast("모델이 메모리에서 내려갔습니다.", "success");
    } catch (err) {
      toast(`언로드 실패: ${err.message}`, "error");
    }
  }

  // ----- Git Sync -----

  async function refreshGitStatus() {
    try {
      state.git = await api("/api/git/status");
      renderGitStatus();
    } catch {
      // ignore
    }
  }

  function renderGitStatus() {
    const g = state.git;
    const txt = $("git-status-text");
    if (!g.is_repo) {
      txt.textContent = "No Git";
    } else if (g.syncing) {
      txt.textContent = "Syncing...";
    } else if (g.last_error) {
      txt.textContent = "Git 오류";
    } else if ((g.pending_count || 0) > 0 || (g.ahead || 0) > 0) {
      txt.textContent = `${g.pending_count || 0} 대기`;
    } else {
      txt.textContent = "Synced";
    }

    if (!$("git-modal").classList.contains("hidden")) {
      renderGitModal();
    }
  }

  function renderGitModal() {
    const g = state.git;
    $("git-autosync-toggle").checked = Boolean(g.auto_sync);
    $("git-modal-info").innerHTML = `
      <div class="metric-tile"><span>브랜치</span><strong>${escapeHtml(g.branch || "-")}</strong></div>
      <div class="metric-tile"><span>원격 저장소</span><strong>${escapeHtml(g.remote_url || "미설정")}</strong></div>
      <div class="metric-tile"><span>Ahead / Behind</span><strong>↑${g.ahead || 0} / ↓${g.behind || 0}</strong></div>
      <div class="metric-tile"><span>최근 동기화</span><strong>${escapeHtml(g.last_result || "-")}</strong></div>
    `;
    $("git-pending-files").textContent =
      (g.pending_files || []).length > 0 ? g.pending_files.join("\n") : "동기화 대기 중인 변경 사항이 없습니다.";

    const errBox = $("git-modal-error");
    if (g.last_error) {
      errBox.textContent = g.last_error;
      errBox.classList.remove("hidden");
    } else {
      errBox.classList.add("hidden");
    }
  }

  async function triggerGitPush() {
    try {
      toast("암호화된 채팅 기록을 Git에 커밋 & Push 중...");
      state.git = await api("/api/git/sync", { method: "POST" });
      renderGitStatus();
      if (state.git.last_error) {
        toast(`Git Push 오류: ${state.git.last_error}`, "error");
      } else {
        toast(state.git.last_result || "Git Push 완료", "success");
      }
    } catch (err) {
      toast(`Git Push 실패: ${err.message}`, "error");
    }
  }

  async function triggerGitPull() {
    try {
      toast("원격 Git에서 암호화된 기록을 Pull 중...");
      state.git = await api("/api/git/pull", { method: "POST" });
      renderGitStatus();
      if (state.git.last_error) {
        toast(`Git Pull 오류: ${state.git.last_error}`, "error");
      } else {
        toast("Git Pull 완료", "success");
        await refreshVaultStatus();
        if (state.vault.unlocked) {
          await Promise.all([loadPersonas(), loadSessions()]);
        }
      }
    } catch (err) {
      toast(`Git Pull 실패: ${err.message}`, "error");
    }
  }

  // ----- Personas -----

  async function loadPersonas() {
    const data = await api("/api/personas");
    state.personas = data.personas || [];
    renderPersonaSelectors();
  }

  function renderPersonaSelectors() {
    const select = $("session-persona-select");
    const currentPersona = state.currentSession?.persona || "default";
    select.innerHTML = state.personas
      .map(
        (p) =>
          `<option value="${escapeHtml(p.name)}" ${p.name === currentPersona ? "selected" : ""}>🎭 ${escapeHtml(p.name)}</option>`
      )
      .join("");

    // Sidebar filter chips
    const filterBox = $("persona-filter");
    const chips = [
      `<button class="filter-chip ${state.personaFilter === "" ? "active" : ""}" data-pfilter="">전체</button>`,
      ...state.personas.map(
        (p) =>
          `<button class="filter-chip ${state.personaFilter === p.name ? "active" : ""}" data-pfilter="${escapeHtml(p.name)}">${escapeHtml(p.name)}</button>`
      ),
    ];
    filterBox.innerHTML = chips.join("");
    filterBox.querySelectorAll(".filter-chip").forEach((btn) => {
      btn.addEventListener("click", () => {
        state.personaFilter = btn.dataset.pfilter || "";
        renderPersonaSelectors();
        loadSessions();
      });
    });
  }

  function renderPersonaModal() {
    const list = $("persona-modal-list");
    list.innerHTML = state.personas
      .map(
        (p) =>
          `<button class="persona-list-item ${p.name === state.editingPersonaName ? "active" : ""}" data-pname="${escapeHtml(p.name)}">${escapeHtml(p.name)}.md</button>`
      )
      .join("");
    list.querySelectorAll(".persona-list-item").forEach((btn) => {
      btn.addEventListener("click", () => {
        state.editingPersonaName = btn.dataset.pname;
        renderPersonaModal();
      });
    });

    const active = state.personas.find((p) => p.name === state.editingPersonaName) || state.personas[0];
    if (active) {
      state.editingPersonaName = active.name;
      $("editing-persona-name").textContent = `${active.name}.md`;
      $("persona-content-input").value = active.content || "";
      $("btn-delete-persona").disabled = active.name === "default";
    }
  }

  // ----- Sessions -----

  async function loadSessions() {
    if (!state.vault.unlocked) return;
    const params = new URLSearchParams();
    if (state.searchQuery) params.set("q", state.searchQuery);
    if (state.personaFilter) params.set("persona", state.personaFilter);
    const data = await api(`/api/sessions?${params.toString()}`);
    state.sessions = data.sessions || [];
    $("sidebar-enc-count").textContent = `${state.sessions.length}개 복호화됨`;
    renderSessionList();
  }

  function renderSessionList() {
    const list = $("session-list");
    if (!state.sessions.length) {
      list.innerHTML = `<div class="empty-hint">${state.searchQuery ? "검색 결과가 없습니다" : "세션이 없습니다"}</div>`;
      return;
    }

    list.innerHTML = "";
    for (const s of state.sessions) {
      const item = document.createElement("div");
      const isActive = state.currentSession?.session_id === s.session_id;
      item.className = `session-item ${isActive ? "active" : ""}`;
      item.innerHTML = `
        <div class="session-item-title">${escapeHtml(s.title)}</div>
        <div class="session-item-meta">
          ${s.persona ? `<span class="session-item-tag">${escapeHtml(s.persona)}</span>` : ""}
          <span>${s.message_count}턴</span>
          <span>· ${escapeHtml(formatRelTime(s.updated_at))}</span>
        </div>
        ${s.preview ? `<div class="session-item-preview">${escapeHtml(s.preview)}</div>` : ""}
        <div class="session-item-actions">
          <button class="mini-action" data-act="export" title="Markdown 내보내기">⬇</button>
          <button class="mini-action danger" data-act="delete" title="삭제">✕</button>
        </div>
      `;

      item.addEventListener("click", (e) => {
        const actBtn = e.target.closest("[data-act]");
        if (actBtn) {
          e.stopPropagation();
          if (actBtn.dataset.act === "delete") {
            deleteSession(s.session_id, s.title);
          } else if (actBtn.dataset.act === "export") {
            window.open(`/api/sessions/${encodeURIComponent(s.session_id)}/export`, "_blank");
          }
          return;
        }
        if (s.decrypt_error) {
          toast("다른 암호화 키로 저장된 세션입니다.", "error");
          return;
        }
        openSession(s.session_id);
      });
      list.appendChild(item);
    }
  }

  async function createNewSession(persona = null) {
    const chosenPersona = persona || state.personaFilter || "default";
    const session = await api("/api/sessions", {
      method: "POST",
      body: JSON.stringify({
        title: "새 대화",
        persona: chosenPersona,
        params: readCurrentParamsUI(),
      }),
    });
    state.currentSession = session;
    await loadSessions();
    renderCurrentSession();
    $("chat-input").focus();
  }

  async function openSession(sessionId) {
    if (state.streaming) {
      toast("응답 생성이 끝난 뒤 세션을 전환하거나 먼저 중단해 주세요.", "error");
      return;
    }
    const session = await api(`/api/sessions/${encodeURIComponent(sessionId)}`);
    state.currentSession = session;
    renderSessionList();
    renderCurrentSession();
  }

  async function deleteSession(sessionId, title) {
    if (!confirm(`"${title}" 세션을 삭제할까요? (Git 다음 동기화 시 원격에서도 제거됩니다)`)) {
      return;
    }
    await api(`/api/sessions/${encodeURIComponent(sessionId)}`, { method: "DELETE" });
    toast("세션이 삭제되었습니다.");
    if (state.currentSession?.session_id === sessionId) {
      state.currentSession = null;
    }
    await loadSessions();
    if (!state.currentSession) {
      const next = state.sessions.find((s) => !s.decrypt_error);
      if (next) {
        await openSession(next.session_id);
      } else {
        await createNewSession();
      }
    }
  }

  async function forkSessionAt(messageId) {
    if (!state.currentSession) return;
    const forked = await api(
      `/api/sessions/${encodeURIComponent(state.currentSession.session_id)}/fork`,
      {
        method: "POST",
        body: JSON.stringify({ up_to_message_id: messageId }),
      }
    );
    state.currentSession = forked;
    await loadSessions();
    renderCurrentSession();
    toast("현재 지점에서 새 세션으로 분기했습니다!", "success");
  }

  // ----- Session Parameters UI -----

  function applyParamsToUI(params) {
    const p = { ...DEFAULT_PARAMS, ...(params || {}) };
    $("param-temp").value = p.temperature;
    $("val-temp").textContent = p.temperature;
    $("param-top-p").value = p.top_p;
    $("val-top-p").textContent = p.top_p;
    $("param-presence").value = p.presence_penalty;
    $("val-presence").textContent = p.presence_penalty;
    $("param-max-tokens").value = p.max_tokens;
    $("val-max-tokens").textContent = p.max_tokens;

    document.querySelectorAll("#reasoning-effort-group .seg-btn").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.effort === (p.reasoning_effort || "medium"));
    });
  }

  function readCurrentParamsUI() {
    const activeSeg = document.querySelector("#reasoning-effort-group .seg-btn.active");
    return {
      temperature: parseFloat($("param-temp").value) || 0.6,
      top_p: parseFloat($("param-top-p").value) || 0.95,
      presence_penalty: parseFloat($("param-presence").value) || 0.2,
      max_tokens: parseInt($("param-max-tokens").value, 10) || 4096,
      reasoning_effort: activeSeg?.dataset.effort || "medium",
    };
  }

  async function saveSessionMetaPatch(patch) {
    if (!state.currentSession) return;
    const updated = await api(
      `/api/sessions/${encodeURIComponent(state.currentSession.session_id)}`,
      {
        method: "PATCH",
        body: JSON.stringify(patch),
      }
    );
    state.currentSession = updated;
    await loadSessions();
  }

  // ----- Chat Rendering -----

  function renderCurrentSession() {
    const s = state.currentSession;
    const container = $("messages-container");
    if (!s) {
      container.innerHTML = "";
      return;
    }

    $("session-title-input").value = s.title || "새 대화";
    renderPersonaSelectors();
    applyParamsToUI(s.params);

    container.innerHTML = "";
    const msgs = s.messages || [];

    if (msgs.length === 0) {
      const welcome = document.createElement("div");
      welcome.className = "welcome-card";
      welcome.innerHTML = `
        <h1>🔐 Private MLX Encrypted Chat</h1>
        <p>
          <strong>gpt-oss-120b (Apple Silicon MLX)</strong>로 로컬에서 추론하고,<br />
          모든 대화는 <strong>AES-256-GCM 암호화</strong> 후 프라이빗 Git 저장소에 자동 동기화됩니다.
        </p>
        <div class="starter-grid">
          <div class="starter-card" data-prompt="정점 N, 간선 M 그래프에서 간선 K개를 0으로 만들 수 있을 때 최단 경로를 구하는 Java 코드를 작성하고 시간복잡도를 분석해줘.">
            <strong>⚡ 알고리즘 풀이 (Java)</strong>
            다익스트라 K개 간선 제거 최단경로
          </div>
          <div class="starter-card" data-prompt="분산 환경에서 Redis 기반 분산 락을 구현할 때 발생하는 엣지 케이스(GC Pause, Failover, Fencing Token)와 대응 전략을 정리해줘.">
            <strong>🏗 시스템 아키텍처 리뷰</strong>
            Redis 분산 락 장애 시나리오와 해결책
          </div>
          <div class="starter-card" data-prompt="AES-256-GCM과 scrypt 키 유도를 사용할 때 Nonce 재사용이 왜 치명적인지, 안전한 파일 암호화 설계 원칙을 설명해줘.">
            <strong>🛡 보안 & 암호화 설계</strong>
            AES-256-GCM과 안전한 키 관리 원칙
          </div>
        </div>
      `;
      welcome.querySelectorAll(".starter-card").forEach((card) => {
        card.addEventListener("click", () => {
          $("chat-input").value = card.dataset.prompt;
          autoResizeTextarea();
          $("chat-input").focus();
        });
      });
      container.appendChild(welcome);
      return;
    }

    msgs.forEach((msg, idx) => {
      const isLast = idx === msgs.length - 1;
      container.appendChild(createMessageNode(msg, isLast));
    });
    scrollToBottom();
  }

  function createMessageNode(msg, isLast = false) {
    const row = document.createElement("div");
    row.className = `msg-row ${msg.role}`;
    row.dataset.msgId = msg.id || "";

    const avatar = document.createElement("div");
    avatar.className = "msg-avatar";
    avatar.textContent = msg.role === "user" ? "나" : "AI";

    const body = document.createElement("div");
    body.className = "msg-body";

    const bubble = document.createElement("div");
    bubble.className = "msg-bubble";

    if (msg.role === "user") {
      bubble.textContent = msg.content || "";
    } else {
      if (msg.reasoning) {
        const details = document.createElement("details");
        details.className = "reasoning-box";
        details.innerHTML = `
          <summary class="reasoning-summary">
            <span>🧠 Reasoning Process</span>
            <span>펼치기 / 접기</span>
          </summary>
          <div class="reasoning-content"></div>
        `;
        details.querySelector(".reasoning-content").textContent = msg.reasoning;
        bubble.appendChild(details);
      }
      bubble.appendChild(renderMarkdown(msg.content || ""));
    }

    const footer = document.createElement("div");
    footer.className = "msg-footer";

    const metaSpan = document.createElement("span");
    if (msg.role === "assistant" && msg.metrics?.tok_per_sec) {
      metaSpan.textContent = `${msg.metrics.tok_per_sec} tok/s · ${msg.metrics.completion_tokens || 0} tokens · TTFT ${msg.metrics.ttft_sec || 0}s`;
    } else if (msg.ts) {
      metaSpan.textContent = msg.ts.replace("T", " ").slice(0, 19);
    }

    const actions = document.createElement("div");
    actions.className = "msg-footer-actions";

    const copyBtn = document.createElement("button");
    copyBtn.className = "btn btn-xs btn-ghost";
    copyBtn.textContent = "복사";
    copyBtn.addEventListener("click", () => copyToClipboard(msg.content || ""));
    actions.appendChild(copyBtn);

    if (msg.id) {
      const forkBtn = document.createElement("button");
      forkBtn.className = "btn btn-xs btn-ghost";
      forkBtn.title = "이 메시지까지 복제하여 새 세션으로 분기";
      forkBtn.textContent = "분기";
      forkBtn.addEventListener("click", () => forkSessionAt(msg.id));
      actions.appendChild(forkBtn);
    }

    if (msg.role === "assistant" && isLast) {
      const regenBtn = document.createElement("button");
      regenBtn.className = "btn btn-xs btn-ghost";
      regenBtn.textContent = "재생성";
      regenBtn.addEventListener("click", () => sendChatMessage("", true));
      actions.appendChild(regenBtn);
    }

    footer.appendChild(metaSpan);
    footer.appendChild(actions);

    body.appendChild(bubble);
    body.appendChild(footer);
    row.appendChild(avatar);
    row.appendChild(body);
    return row;
  }

  function scrollToBottom() {
    const vp = $("messages-viewport");
    vp.scrollTop = vp.scrollHeight;
  }

  // ----- Chat Streaming -----

  function setStreamingUI(isStreaming) {
    state.streaming = isStreaming;
    $("btn-send").classList.toggle("hidden", isStreaming);
    $("btn-stop").classList.toggle("hidden", !isStreaming);
    $("chat-input").disabled = isStreaming;
  }

  async function sendChatMessage(rawText, regenerate = false) {
    if (state.streaming) return;
    if (!state.vault.unlocked) {
      renderVaultModal();
      return;
    }
    if (state.model.status !== "ready") {
      toast("먼저 상단의 Load 버튼으로 모델을 메모리에 적재해 주세요.", "error");
      return;
    }

    if (!state.currentSession) {
      await createNewSession();
    }

    const text = rawText.trim();
    if (!regenerate && !text) return;

    const sessionId = state.currentSession.session_id;
    const container = $("messages-container");
    const welcome = container.querySelector(".welcome-card");
    if (welcome) welcome.remove();

    if (regenerate) {
      // Remove last assistant row from UI
      const lastRow = container.lastElementChild;
      if (lastRow && lastRow.classList.contains("assistant")) {
        lastRow.remove();
      }
    } else {
      $("chat-input").value = "";
      autoResizeTextarea();
      const tempUserMsg = { id: "", role: "user", content: text, ts: new Date().toISOString() };
      container.appendChild(createMessageNode(tempUserMsg, false));
    }

    // Create live streaming assistant row
    const liveRow = document.createElement("div");
    liveRow.className = "msg-row assistant";
    liveRow.innerHTML = `
      <div class="msg-avatar">AI</div>
      <div class="msg-body">
        <div class="msg-bubble">
          <details class="reasoning-box hidden" open>
            <summary class="reasoning-summary">
              <span><span class="thinking-pulse"></span><span class="reasoning-title">Thinking...</span></span>
              <span>펼치기 / 접기</span>
            </summary>
            <div class="reasoning-content"></div>
          </details>
          <div class="live-content markdown-body"><span class="muted-note">응답 준비 중...</span></div>
        </div>
        <div class="msg-footer">
          <span class="live-meta">생성 중...</span>
          <div class="msg-footer-actions"></div>
        </div>
      </div>
    `;
    container.appendChild(liveRow);
    scrollToBottom();

    const reasoningDetails = liveRow.querySelector(".reasoning-box");
    const reasoningTitle = liveRow.querySelector(".reasoning-title");
    const reasoningPulse = liveRow.querySelector(".thinking-pulse");
    const reasoningContentEl = liveRow.querySelector(".reasoning-content");
    const liveContentEl = liveRow.querySelector(".live-content");
    const liveMetaEl = liveRow.querySelector(".live-meta");

    let userToggledReasoning = false;
    reasoningDetails.querySelector("summary").addEventListener("click", () => {
      userToggledReasoning = true;
    });

    let reasoningText = "";
    let answerText = "";
    let renderScheduled = false;

    const flushRender = () => {
      renderScheduled = false;
      if (reasoningText) {
        reasoningDetails.classList.remove("hidden");
        reasoningContentEl.textContent = reasoningText;
        reasoningContentEl.scrollTop = reasoningContentEl.scrollHeight;
      }
      if (answerText) {
        if (reasoningText && !userToggledReasoning && reasoningDetails.open) {
          reasoningDetails.open = false;
          reasoningTitle.textContent = "Reasoning Process";
          reasoningPulse.classList.add("hidden");
        }
        const rendered = renderMarkdown(answerText);
        liveContentEl.replaceChildren(rendered);
      }
      scrollToBottom();
    };

    const scheduleRender = () => {
      if (!renderScheduled) {
        renderScheduled = true;
        requestAnimationFrame(flushRender);
      }
    };

    setStreamingUI(true);
    try {
      const resp = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          content: text,
          params: readCurrentParamsUI(),
          regenerate,
        }),
      });

      if (resp.status === 423) {
        handleVaultLocked();
        return;
      }
      if (!resp.ok) {
        const errData = await resp.json().catch(() => ({}));
        throw new Error(errData.detail || `HTTP ${resp.status}`);
      }

      const reader = resp.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let sseBuffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        sseBuffer += decoder.decode(value, { stream: true });

        const frames = sseBuffer.split("\n\n");
        sseBuffer = frames.pop() || "";

        for (const frame of frames) {
          let evType = "message";
          let dataStr = "";
          for (const line of frame.split("\n")) {
            if (line.startsWith("event:")) evType = line.slice(6).trim();
            else if (line.startsWith("data:")) dataStr += line.slice(5).trim();
          }
          if (!dataStr) continue;
          let payload;
          try {
            payload = JSON.parse(dataStr);
          } catch {
            continue;
          }

          if (evType === "meta") {
            if (payload.title) $("session-title-input").value = payload.title;
          } else if (evType === "keepalive") {
            liveMetaEl.textContent = `프롬프트 처리 중 (${payload.detail})...`;
          } else if (evType === "reasoning") {
            reasoningText += payload.text || "";
            liveMetaEl.textContent = "추론(CoT) 진행 중...";
            scheduleRender();
          } else if (evType === "content") {
            answerText += payload.text || "";
            liveMetaEl.textContent = "답변 생성 중...";
            scheduleRender();
          } else if (evType === "error") {
            throw new Error(payload.detail || "스트리밍 오류");
          } else if (evType === "done") {
            reasoningPulse.classList.add("hidden");
            reasoningTitle.textContent = "Reasoning Process";
            flushRender();
          }
        }
      }

      // Reload canonical encrypted session from backend (gets IDs, metrics, updated sidebar)
      await openSessionForce(sessionId);
      await loadSessions();
      refreshGitStatus();
    } catch (err) {
      toast(`채팅 오류: ${err.message}`, "error");
      liveMetaEl.textContent = `오류: ${err.message}`;
    } finally {
      setStreamingUI(false);
      $("chat-input").focus();
    }
  }

  async function openSessionForce(sessionId) {
    const session = await api(`/api/sessions/${encodeURIComponent(sessionId)}`);
    state.currentSession = session;
    renderCurrentSession();
  }

  async function stopGeneration() {
    if (!state.currentSession) return;
    await api(`/api/sessions/${encodeURIComponent(state.currentSession.session_id)}/stop`, {
      method: "POST",
    });
    toast("생성 중단을 요청했습니다.");
  }

  function autoResizeTextarea() {
    const ta = $("chat-input");
    ta.style.height = "auto";
    ta.style.height = `${Math.min(ta.scrollHeight, 220)}px`;
  }

  // ----- Event Binding -----

  function bindEvents() {
    // Sidebar toggle
    $("btn-toggle-sidebar").addEventListener("click", () => {
      $("sidebar").classList.toggle("collapsed");
    });

    // Modal close buttons
    document.querySelectorAll(".modal-close").forEach((btn) => {
      btn.addEventListener("click", () => {
        $(btn.dataset.close)?.classList.add("hidden");
      });
    });

    // Vault init tabs & unlock tabs
    document.querySelectorAll("[data-init-tab]").forEach((btn) => {
      btn.addEventListener("click", () => switchInitTab(btn.dataset.initTab));
    });
    document.querySelectorAll("[data-unlock-tab]").forEach((btn) => {
      btn.addEventListener("click", () => switchUnlockTab(btn.dataset.unlockTab));
    });

    // Vault Init (Passphrase)
    const doInitPassphrase = async () => {
      const pw = $("init-passphrase").value;
      const pw2 = $("init-passphrase-confirm").value;
      if (pw.length < 6) {
        showVaultError("패스프레이즈는 최소 6자 이상 입력하세요.");
        return;
      }
      if (pw !== pw2) {
        showVaultError("패스프레이즈 확인이 일치하지 않습니다.");
        return;
      }
      try {
        const res = await api("/api/vault/init", {
          method: "POST",
          body: JSON.stringify({ mode: "passphrase", passphrase: pw }),
        });
        state.vault = res.status;
        state.initBackupKey = res.backup_key;
        $("init-passphrase").value = "";
        $("init-passphrase-confirm").value = "";
        renderVaultModal();
      } catch (err) {
        showVaultError(err.message);
      }
    };
    $("btn-init-passphrase").addEventListener("click", doInitPassphrase);
    $("init-passphrase-confirm").addEventListener("keydown", (e) => {
      if (e.key === "Enter") doInitPassphrase();
    });

    // Vault Init (Random Key File)
    $("btn-init-keyfile").addEventListener("click", async () => {
      try {
        const res = await api("/api/vault/init", {
          method: "POST",
          body: JSON.stringify({ mode: "keyfile" }),
        });
        state.vault = res.status;
        state.initBackupKey = res.backup_key;
        renderVaultModal();
      } catch (err) {
        showVaultError(err.message);
      }
    });

    $("btn-download-init-key").addEventListener("click", () => {
      downloadTextFile("vault-master.key", state.initBackupKey + "\n");
      toast("vault-master.key 파일을 다운로드했습니다.", "success");
    });
    $("btn-copy-init-key").addEventListener("click", () => {
      copyToClipboard(state.initBackupKey, "마스터 키를 복사했습니다");
    });
    $("btn-enter-app").addEventListener("click", async () => {
      state.initBackupKey = "";
      renderVaultModal();
      await onVaultUnlocked();
    });

    // Vault Unlock (Passphrase)
    const doUnlockPassphrase = async () => {
      try {
        state.vault = await api("/api/vault/unlock", {
          method: "POST",
          body: JSON.stringify({ passphrase: $("unlock-passphrase").value }),
        });
        renderVaultModal();
        toast("Vault 잠금이 해제되었습니다.", "success");
        await onVaultUnlocked();
      } catch (err) {
        showVaultError(err.message);
      }
    };
    $("btn-unlock-passphrase").addEventListener("click", doUnlockPassphrase);
    $("unlock-passphrase").addEventListener("keydown", (e) => {
      if (e.key === "Enter") doUnlockPassphrase();
    });

    // Vault Unlock (Key File / Text)
    const doUnlockKey = async (keyTextOverride = null) => {
      const keyText = keyTextOverride ?? $("unlock-key-text").value;
      try {
        state.vault = await api("/api/vault/unlock", {
          method: "POST",
          body: JSON.stringify({ key_text: keyText }),
        });
        renderVaultModal();
        toast("개인 키로 Vault 잠금이 해제되었습니다.", "success");
        await onVaultUnlocked();
      } catch (err) {
        showVaultError(err.message);
      }
    };
    $("btn-unlock-key").addEventListener("click", () => doUnlockKey());
    $("unlock-key-text").addEventListener("keydown", (e) => {
      if (e.key === "Enter") doUnlockKey();
    });

    const dropzone = $("key-dropzone");
    const fileInput = $("key-file-input");
    dropzone.addEventListener("click", () => fileInput.click());
    dropzone.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropzone.classList.add("dragover");
    });
    dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
    dropzone.addEventListener("drop", async (e) => {
      e.preventDefault();
      dropzone.classList.remove("dragover");
      const file = e.dataTransfer.files[0];
      if (file) {
        const content = await file.text();
        $("unlock-key-text").value = content.trim();
        await doUnlockKey(content.trim());
      }
    });
    fileInput.addEventListener("change", async () => {
      const file = fileInput.files[0];
      if (file) {
        const content = await file.text();
        $("unlock-key-text").value = content.trim();
        await doUnlockKey(content.trim());
      }
      fileInput.value = "";
    });

    $("btn-vault-pull").addEventListener("click", triggerGitPull);

    // Vault Lock & Key Settings
    $("btn-vault-lock").addEventListener("click", async () => {
      await api("/api/vault/lock", { method: "POST" });
      handleVaultLocked();
      toast("Vault가 잠겼습니다. 메모리에서 키가 파기되었습니다.");
    });

    $("btn-vault-settings").addEventListener("click", async () => {
      try {
        const res = await api("/api/vault/export-key");
        $("backup-key-input").value = res.key_text;
        $("backup-key-input").type = "password";
        $("btn-toggle-key-vis").textContent = "보기";
        $("vault-autolock-select").value = String(state.vault.auto_lock_minutes || 0);
        $("key-modal").classList.remove("hidden");
      } catch (err) {
        toast(err.message, "error");
      }
    });

    $("btn-toggle-key-vis").addEventListener("click", () => {
      const inp = $("backup-key-input");
      const hidden = inp.type === "password";
      inp.type = hidden ? "text" : "password";
      $("btn-toggle-key-vis").textContent = hidden ? "숨기기" : "보기";
    });
    $("btn-copy-backup-key").addEventListener("click", () => {
      copyToClipboard($("backup-key-input").value, "마스터 키를 복사했습니다");
    });
    $("btn-download-backup-key").addEventListener("click", () => {
      downloadTextFile("vault-master.key", $("backup-key-input").value + "\n");
      toast("vault-master.key 다운로드 완료", "success");
    });
    $("vault-autolock-select").addEventListener("change", async (e) => {
      state.vault = await api("/api/vault/settings", {
        method: "POST",
        body: JSON.stringify({ auto_lock_minutes: parseInt(e.target.value, 10) }),
      });
      toast("자동 잠금 설정이 저장되었습니다.", "success");
    });

    // Model Controls
    $("btn-model-toggle").addEventListener("click", () => {
      if (state.model.status === "unloaded" || state.model.status === "error") {
        handleModelLoad();
      } else {
        handleModelUnload();
      }
    });
    $("btn-banner-load").addEventListener("click", () => handleModelLoad());
    $("btn-model-console").addEventListener("click", async () => {
      $("model-modal").classList.remove("hidden");
      renderModelModalDetails();
      await refreshModelLogs();
    });
    $("btn-refresh-logs").addEventListener("click", refreshModelLogs);
    $("btn-modal-load").addEventListener("click", () => handleModelLoad($("model-select").value));
    $("btn-modal-unload").addEventListener("click", handleModelUnload);
    $("idle-unload-select").addEventListener("change", async (e) => {
      state.model = await api("/api/model/settings", {
        method: "POST",
        body: JSON.stringify({ idle_unload_minutes: parseInt(e.target.value, 10) }),
      });
      toast("미사용 자동 언로드 타이머가 업데이트되었습니다.", "success");
    });

    // Git Controls
    $("btn-git-push").addEventListener("click", triggerGitPush);
    $("btn-git-pull").addEventListener("click", triggerGitPull);
    $("btn-git-details").addEventListener("click", async () => {
      await refreshGitStatus();
      renderGitModal();
      $("git-modal").classList.remove("hidden");
    });
    $("btn-git-modal-push").addEventListener("click", triggerGitPush);
    $("btn-git-modal-pull").addEventListener("click", triggerGitPull);
    $("git-autosync-toggle").addEventListener("change", async (e) => {
      state.git = await api("/api/git/settings", {
        method: "POST",
        body: JSON.stringify({ auto_sync: e.target.checked }),
      });
      renderGitStatus();
      toast(e.target.checked ? "Git 자동 동기화 켜짐" : "Git 자동 동기화 꺼짐");
    });

    // Sessions & Search
    $("btn-new-session").addEventListener("click", () => createNewSession());
    let searchTimer = null;
    $("session-search").addEventListener("input", (e) => {
      state.searchQuery = e.target.value;
      clearTimeout(searchTimer);
      searchTimer = setTimeout(() => loadSessions(), 180);
    });

    // Inline session title & persona & export
    const commitTitle = () => {
      if (!state.currentSession) return;
      const val = $("session-title-input").value.trim();
      if (val && val !== state.currentSession.title) {
        saveSessionMetaPatch({ title: val });
      }
    };
    $("session-title-input").addEventListener("blur", commitTitle);
    $("session-title-input").addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        $("session-title-input").blur();
      }
    });

    $("session-persona-select").addEventListener("change", (e) => {
      saveSessionMetaPatch({ persona: e.target.value });
      toast(`페르소나가 '${e.target.value}'로 변경되었습니다.`);
    });

    $("btn-export-md").addEventListener("click", () => {
      if (!state.currentSession) return;
      window.open(
        `/api/sessions/${encodeURIComponent(state.currentSession.session_id)}/export`,
        "_blank"
      );
    });

    // Reasoning effort segmented control
    document.querySelectorAll("#reasoning-effort-group .seg-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.querySelectorAll("#reasoning-effort-group .seg-btn").forEach((b) =>
          b.classList.toggle("active", b === btn)
        );
        saveSessionMetaPatch({ params: readCurrentParamsUI() });
      });
    });

    // Generation params popover
    $("btn-gen-params").addEventListener("click", (e) => {
      e.stopPropagation();
      $("params-popover").classList.toggle("hidden");
    });
    document.addEventListener("click", (e) => {
      const pop = $("params-popover");
      if (!pop.classList.contains("hidden") && !pop.contains(e.target) && e.target !== $("btn-gen-params")) {
        pop.classList.add("hidden");
      }
    });

    const syncSlider = (inputId, valId) => {
      $(inputId).addEventListener("input", () => {
        $(valId).textContent = $(inputId).value;
      });
      $(inputId).addEventListener("change", () => {
        saveSessionMetaPatch({ params: readCurrentParamsUI() });
      });
    };
    syncSlider("param-temp", "val-temp");
    syncSlider("param-top-p", "val-top-p");
    syncSlider("param-presence", "val-presence");
    syncSlider("param-max-tokens", "val-max-tokens");

    $("btn-reset-params").addEventListener("click", () => {
      applyParamsToUI(DEFAULT_PARAMS);
      saveSessionMetaPatch({ params: DEFAULT_PARAMS });
      toast("기본 파라미터로 복원했습니다.");
    });

    // Persona Modal
    $("btn-open-personas").addEventListener("click", async () => {
      await loadPersonas();
      renderPersonaModal();
      $("persona-modal").classList.remove("hidden");
    });
    $("btn-create-persona").addEventListener("click", async () => {
      const name = $("new-persona-name").value.trim();
      if (!name) return;
      try {
        await api(`/api/personas/${encodeURIComponent(name)}`, {
          method: "PUT",
          body: JSON.stringify({ content: `# ${name}\n\n여기에 시스템 프롬프트를 작성하세요.` }),
        });
        $("new-persona-name").value = "";
        state.editingPersonaName = name;
        await loadPersonas();
        renderPersonaModal();
        toast(`페르소나 '${name}'이(가) 생성되었습니다.`, "success");
      } catch (err) {
        toast(err.message, "error");
      }
    });
    $("btn-save-persona").addEventListener("click", async () => {
      try {
        await api(`/api/personas/${encodeURIComponent(state.editingPersonaName)}`, {
          method: "PUT",
          body: JSON.stringify({ content: $("persona-content-input").value }),
        });
        await loadPersonas();
        renderPersonaModal();
        toast("페르소나가 저장되었습니다 (Git 동기화 예약됨).", "success");
      } catch (err) {
        toast(err.message, "error");
      }
    });
    $("btn-delete-persona").addEventListener("click", async () => {
      if (state.editingPersonaName === "default") return;
      if (!confirm(`'${state.editingPersonaName}' 페르소나를 삭제할까요?`)) return;
      try {
        await api(`/api/personas/${encodeURIComponent(state.editingPersonaName)}`, {
          method: "DELETE",
        });
        state.editingPersonaName = "default";
        await loadPersonas();
        renderPersonaModal();
        toast("페르소나가 삭제되었습니다.");
      } catch (err) {
        toast(err.message, "error");
      }
    });

    // Chat Composer
    const chatInput = $("chat-input");
    chatInput.addEventListener("input", autoResizeTextarea);
    chatInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
        e.preventDefault();
        sendChatMessage(chatInput.value);
      }
    });
    $("btn-send").addEventListener("click", () => sendChatMessage(chatInput.value));
    $("btn-stop").addEventListener("click", stopGeneration);
  }

  // ----- Boot & Polling -----

  async function pollLoop() {
    await refreshModelStatus();
    if (!$("model-modal").classList.contains("hidden")) {
      await refreshModelLogs();
    }
    if (state.vault.unlocked) {
      await refreshGitStatus();
    }
    const interval = ["loading", "warming", "stopping"].includes(state.model.status) ? 1500 : 4000;
    setTimeout(pollLoop, interval);
  }

  async function init() {
    bindEvents();
    await Promise.all([refreshVaultStatus(), refreshModelStatus(), refreshGitStatus()]);
    if (state.vault.unlocked) {
      await onVaultUnlocked();
    }
    setTimeout(pollLoop, 3000);
  }

  document.addEventListener("DOMContentLoaded", init);
})();
