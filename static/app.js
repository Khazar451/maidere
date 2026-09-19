/**
 * Maidere IDE Agent Interface — Application Logic
 * Full-screen 3-pane IDE Agent Workspace
 */

(function () {
  'use strict';

  // Configure Highlight.js with Marked
  if (window.marked && typeof window.marked.setOptions === 'function') {
    window.marked.setOptions({
      highlight: function (code, lang) {
        if (window.hljs) {
          const language = window.hljs.getLanguage(lang) ? lang : 'plaintext';
          return window.hljs.highlight(code, { language }).value;
        }
        return code;
      },
      langPrefix: 'hljs language-',
      breaks: true,
      gfm: true,
    });
  }

  // --- State Management ---
  const state = {
    activeThreadId: localStorage.getItem('maidere_active_thread') || generateUUID(),
    historyStack: [],
    historyIndex: -1,
    threads: [],
    models: [],
    selectedModel: localStorage.getItem('maidere_selected_model') || 'auto',
    selectedNumCtx: parseInt(localStorage.getItem('maidere_selected_num_ctx') || '16384', 10),
    thinkingMode: localStorage.getItem('maidere_thinking_mode') === 'true',
    deepReasoning: localStorage.getItem('maidere_deep_reasoning') === 'true',
    isGenerating: false,
    activeTasks: [],
    sidebarCollapsed: localStorage.getItem('maidere_sidebar_collapsed') === 'true',
    terminalCollapsed: localStorage.getItem('maidere_terminal_collapsed') === 'true',
    activeTerminalTab: 'agent-shell',
    terminalLogs: {
      'agent-shell': [],
      'uvicorn': [],
      'obsidian': [],
      'metrics': [],
    },
    autoScrollTerminal: true,
    activeWs: null,
    attachedFiles: [],
    username: (function () {
      const stored = localStorage.getItem('maidere_username');
      if (!stored || stored.toLowerCase() === 'khazar') {
        localStorage.setItem('maidere_username', 'User');
        return 'User';
      }
      return stored;
    })(),
  };

  // --- DOM Elements Cache ---
  const el = {};

  function initElements() {
    el.appContainer = document.getElementById('app-container');
    el.sidebar = document.getElementById('sidebar');
    el.terminalPanel = document.getElementById('terminal-panel');
    el.chatTimeline = document.getElementById('chat-timeline');
    el.chatInput = document.getElementById('chat-input');
    el.sendBtn = document.getElementById('send-btn');
    el.stopBtn = document.getElementById('stop-btn');
    el.btnModeAuto = document.getElementById('btn-mode-auto');
    el.btnModeThinking = document.getElementById('btn-mode-thinking');
    el.btnModeDeepReason = document.getElementById('btn-mode-deep-reason');
    el.deckBtnThinking = document.getElementById('deck-btn-thinking');
    el.deckBtnDeepReason = document.getElementById('deck-btn-deep-reason');
    el.contextSelect = document.getElementById('context-select');
    el.modelSelect = document.getElementById('model-select');
    el.threadsList = document.getElementById('threads-list');
    el.threadSearchInput = document.getElementById('thread-search-input');
    el.breadcrumbTitle = document.getElementById('active-thread-title') || document.getElementById('breadcrumb-thread-title');
    el.breadcrumbProject = document.getElementById('breadcrumb-project');
    el.taskBanner = document.getElementById('task-banner-tray');
    el.taskBannerText = document.getElementById('task-banner-text');
    el.terminalCanvas = document.getElementById('terminal-canvas');
    el.btnToggleSidebar = document.getElementById('btn-toggle-sidebar');
    el.btnToggleSidebarMain = document.getElementById('btn-toggle-sidebar-main');
    el.btnSidebarEdge = document.getElementById('btn-sidebar-edge');
    el.btnToggleTerminal = document.getElementById('btn-toggle-terminal');
    el.btnNewChat = document.getElementById('btn-new-chat');
    el.btnNavBack = document.getElementById('btn-nav-back');
    el.btnNavForward = document.getElementById('btn-nav-forward');
    el.btnExportThread = document.getElementById('btn-export-thread');
    el.btnClearTerminal = document.getElementById('btn-clear-terminal');
    el.btnCopyTerminal = document.getElementById('btn-copy-terminal');
    el.btnSettings = document.getElementById('btn-settings');
    el.modalOverlay = document.getElementById('modal-overlay');
    el.modalTitle = document.getElementById('modal-title');
    el.modalBody = document.getElementById('modal-body');
    el.modalClose = document.getElementById('modal-close');
    el.obsidianVaultPill = document.getElementById('obsidian-vault-pill');
    el.memoriesPill = document.getElementById('memories-pill');
    el.skillsPill = document.getElementById('skills-pill');
    el.scheduledTasksPill = document.getElementById('scheduled-tasks-pill');
    el.commandPaletteOverlay = document.getElementById('command-palette-overlay');
    el.paletteSearchInput = document.getElementById('palette-search-input');
    el.paletteResults = document.getElementById('palette-results');
    el.btnSwitchUser = document.getElementById('btn-switch-user');
    el.sidebarUserName = document.getElementById('sidebar-user-name');
  }

  function updateSidebarUser() {
    if (el.sidebarUserName) {
      el.sidebarUserName.textContent = state.username || 'Set Nickname';
    }
  }

  function updateModeUI() {
    const isDeep = Boolean(state.deepReasoning);
    const isThink = !isDeep && Boolean(state.thinkingMode);
    const isAuto = !isDeep && !isThink;

    if (el.btnModeAuto) {
      el.btnModeAuto.classList.toggle('active', isAuto);
    }
    if (el.btnModeThinking) {
      el.btnModeThinking.classList.toggle('active', isThink);
    }
    if (el.btnModeDeepReason) {
      el.btnModeDeepReason.classList.toggle('active', isDeep);
    }
    if (el.deckBtnThinking) {
      el.deckBtnThinking.classList.toggle('active', isThink);
    }
    if (el.deckBtnDeepReason) {
      el.deckBtnDeepReason.classList.toggle('active', isDeep);
    }
  }

  function setCognitiveMode(mode) {
    if (mode === 'deep-reason') {
      state.deepReasoning = true;
      state.thinkingMode = true;
      appendTerminal('agent-shell', '[CONFIG] Deep Reason Mode enabled (enforcing Socratic deliberation and System 2 cognitive synthesis)', 'info');
    } else if (mode === 'thinking') {
      state.deepReasoning = false;
      state.thinkingMode = true;
      appendTerminal('agent-shell', '[CONFIG] Thinking Mode enabled (forcing step-by-step reasoning scratchpad)', 'info');
    } else {
      state.deepReasoning = false;
      state.thinkingMode = false;
      appendTerminal('agent-shell', '[CONFIG] Switched to Auto Mode (dynamic task routing)', 'info');
    }

    localStorage.setItem('maidere_deep_reasoning', state.deepReasoning ? 'true' : 'false');
    localStorage.setItem('maidere_thinking_mode', state.thinkingMode ? 'true' : 'false');
    updateModeUI();

    if (state.thinkingMode || state.deepReasoning) {
      if (el.modelSelect) {
        const options = Array.from(el.modelSelect.options);
        const r1Opt = options.find((o) => o.value.includes('deepseek-r1') || o.value.includes('r1'));
        if (r1Opt) {
          state.selectedModel = r1Opt.value;
          el.modelSelect.value = r1Opt.value;
          localStorage.setItem('maidere_selected_model', state.selectedModel);
          appendTerminal('agent-shell', `[ROUTER] Switched model to ${r1Opt.value} for reasoning`, 'info');
        }
      }
    } else {
      if (el.modelSelect) {
        const autoOpt = Array.from(el.modelSelect.options).find((o) => o.value === 'auto');
        if (autoOpt && state.selectedModel.includes('deepseek-r1')) {
          state.selectedModel = 'auto';
          el.modelSelect.value = 'auto';
          localStorage.setItem('maidere_selected_model', 'auto');
          appendTerminal('agent-shell', '[ROUTER] Reset model selection to Auto-route', 'info');
        }
      }
    }
  }

  function setThinkingMode(enabled) {
    setCognitiveMode(enabled ? 'thinking' : 'auto');
  }

  // --- Utilities ---
  function generateUUID() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function (c) {
      const r = (Math.random() * 16) | 0,
        v = c === 'x' ? r : (r & 0x3) | 0x8;
      return v.toString(16);
    });
  }

  function escapeHTML(str) {
    if (typeof str !== 'string') return '';
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  function formatRelativeTime(timestamp) {
    if (!timestamp) return 'Just now';
    const date = new Date(timestamp);
    if (isNaN(date.getTime())) return 'Just now';
    const now = new Date();
    const diffSec = Math.floor((now - date) / 1000);
    if (diffSec < 60) return 'Just now';
    if (diffSec < 3600) return `${Math.floor(diffSec / 60)}m`;
    if (diffSec < 86400) return `${Math.floor(diffSec / 3600)}h`;
    return `${Math.floor(diffSec / 86400)}d`;
  }

  function renderMarkdownChunk(chunk) {
    if (!chunk) return '';
    if (window.marked && typeof window.marked.parse === 'function') {
      try {
        return window.marked.parse(chunk);
      } catch (e) {
        console.warn('Markdown parse failed:', e);
      }
    }
    return escapeHTML(chunk).replace(/\n/g, '<br>');
  }

  function safeMarkdownParse(text) {
    if (!text) return '';

    let processed = text;

    // Check for thinking blocks (<think>...</think>) from reasoning models (DeepSeek-R1, etc.)
    const hasCompleteThink = /<think>([\s\S]*?)<\/think>/i.test(processed);
    const hasActiveThink = /<think>([\s\S]*)$/i.test(processed);

    if (hasCompleteThink) {
      processed = processed.replace(/<think>([\s\S]*?)<\/think>/gi, (match, thought) => {
        const renderedThought = renderMarkdownChunk(thought.trim());
        return `\n\n<details class="thought-box">
  <summary class="thought-summary">
    <span class="thought-badge">[THINKING]</span>
    <span class="thought-title">Thought Process</span>
  </summary>
  <div class="thought-content">${renderedThought}</div>
</details>\n\n`;
      });
    } else if (hasActiveThink) {
      processed = processed.replace(/<think>([\s\S]*)$/gi, (match, activeThought) => {
        const renderedThought = renderMarkdownChunk(activeThought.trim());
        return `\n\n<details class="thought-box thinking-active" open>
  <summary class="thought-summary">
    <span class="thought-badge thought-pulse">[THINKING...]</span>
    <span class="thought-title">Reasoning in progress</span>
  </summary>
  <div class="thought-content">${renderedThought}</div>
</details>\n\n`;
      });
    }

    return renderMarkdownChunk(processed);
  }


  // --- Terminal Logging ---
  function appendTerminal(tab, text, type = 'info') {
    const timeStr = new Date().toTimeString().split(' ')[0];
    const logItem = { time: timeStr, text, type };
    if (!state.terminalLogs[tab]) state.terminalLogs[tab] = [];
    state.terminalLogs[tab].push(logItem);

    if (state.activeTerminalTab === tab && el.terminalCanvas) {
      const line = document.createElement('div');
      line.className = 'term-line';
      line.innerHTML = `<span class="term-ts">[${timeStr}]</span> <span class="term-text ${type}">${escapeHTML(text)}</span>`;
      el.terminalCanvas.appendChild(line);
      if (state.autoScrollTerminal) {
        el.terminalCanvas.scrollTop = el.terminalCanvas.scrollHeight;
      }
    }
  }

  function renderActiveTerminal() {
    if (!el.terminalCanvas) return;
    el.terminalCanvas.innerHTML = '';
    const logs = state.terminalLogs[state.activeTerminalTab] || [];
    logs.forEach((log) => {
      const line = document.createElement('div');
      line.className = 'term-line';
      line.innerHTML = `<span class="term-ts">[${log.time}]</span> <span class="term-text ${log.type}">${escapeHTML(log.text)}</span>`;
      el.terminalCanvas.appendChild(line);
    });
    if (state.autoScrollTerminal) {
      el.terminalCanvas.scrollTop = el.terminalCanvas.scrollHeight;
    }
  }

  // --- Layout State Persistence ---
  function applyLayout() {
    if (state.sidebarCollapsed) {
      el.appContainer.classList.add('sidebar-collapsed');
      if (el.btnToggleSidebarMain) {
        el.btnToggleSidebarMain.setAttribute('title', 'Open Sidebar (Ctrl+B)');
        el.btnToggleSidebarMain.setAttribute('aria-label', 'Open Sidebar');
      }
      if (el.btnSidebarEdge) {
        el.btnSidebarEdge.setAttribute('title', 'Open Sidebar (Ctrl+B)');
        el.btnSidebarEdge.setAttribute('aria-label', 'Open Sidebar');
      }
    } else {
      el.appContainer.classList.remove('sidebar-collapsed');
      if (el.btnToggleSidebarMain) {
        el.btnToggleSidebarMain.setAttribute('title', 'Close Sidebar (Ctrl+B)');
        el.btnToggleSidebarMain.setAttribute('aria-label', 'Close Sidebar');
      }
    }

    if (state.terminalCollapsed) {
      el.appContainer.classList.add('terminal-collapsed');
      if (el.btnToggleTerminal) {
        el.btnToggleTerminal.setAttribute('title', 'Open Terminal Panel (Ctrl+J)');
      }
    } else {
      el.appContainer.classList.remove('terminal-collapsed');
      if (el.btnToggleTerminal) {
        el.btnToggleTerminal.setAttribute('title', 'Close Terminal Panel (Ctrl+J)');
      }
    }
  }

  function toggleSidebar() {
    state.sidebarCollapsed = !state.sidebarCollapsed;
    localStorage.setItem('maidere_sidebar_collapsed', state.sidebarCollapsed);
    applyLayout();
  }

  function toggleTerminal() {
    state.terminalCollapsed = !state.terminalCollapsed;
    localStorage.setItem('maidere_terminal_collapsed', state.terminalCollapsed);
    applyLayout();
  }

  // --- Task Banner Management ---
  function setRunningTask(taskName, command) {
    if (!el.taskBanner) return;
    if (taskName) {
      el.taskBanner.classList.add('active');
      el.taskBannerText.innerHTML = `<span class="spinner-ring" style="display:inline-block; vertical-align:middle; width:10px; height:10px; border:2px solid #3b82f6; border-top-color:transparent; border-radius:50%; animation:spin 0.8s linear infinite;"></span> <strong>Task Running:</strong> ${escapeHTML(command || taskName)}`;
    } else {
      el.taskBanner.classList.remove('active');
    }
  }

  // --- Step Pill Generators (IDE Structured Execution) ---
  function createExecutionPill(toolName, args, result, duration_ms, success) {
    const pill = document.createElement('div');
    pill.className = 'step-pill';

    let icon = '[TOOL]';
    let title = `Ran ${toolName}`;
    let target = '';
    let extraClass = '';
    let openObsidianBtn = '';

    const argsObj = args || {};

    if (toolName === 'obsidian' || toolName.startsWith('obsidian_')) {
      icon = '[NOTE]';
      extraClass = 'obsidian-pill';
      const action = argsObj.action || 'write_note';
      const fileTitle = argsObj.title || argsObj.file || 'Note';
      title = action === 'write_note' ? `Saved Obsidian Note` : `Obsidian ${action}`;
      target = fileTitle;
      openObsidianBtn = `<button class="open-obsidian-btn" onclick="event.stopPropagation(); window.maidereOpenObsidian('${escapeHTML(fileTitle)}');">Open in Obsidian</button>`;
    } else if (toolName === 'write_file' || toolName === 'edit_file') {
      icon = '[DIFF]';
      extraClass = 'diff-pill';
      const path = argsObj.path || 'file';
      const lines = (argsObj.content || '').split('\n').length;
      title = `Edited ${path}`;
      target = `<span class="diff-counter"><span class="diff-add">+${lines}</span></span>`;
    } else if (toolName === 'read_file' || toolName === 'list_dir') {
      icon = '[FILE]';
      title = toolName === 'read_file' ? `Analyzed file` : `Explored directory`;
      target = argsObj.path || '';
    } else if (toolName === 'web_search') {
      icon = '[SEARCH]';
      title = `Searched Web`;
      target = `"${argsObj.query || ''}"`;
    } else if (toolName === 'browser') {
      icon = '[BROWSE]';
      title = `Browsed Page`;
      target = argsObj.url || '';
    } else if (toolName === 'shell' || toolName === 'code_runner') {
      icon = '[SHELL]';
      extraClass = 'terminal-pill';
      title = `Ran command`;
      target = argsObj.command || argsObj.code || '';
      if (target.length > 32) target = target.substring(0, 32) + '...';
    } else if (toolName === 'delegate_task' || toolName === 'Agent') {
      icon = '[SUBAGENT]';
      extraClass = 'subagent-pill';
      const agentType = argsObj.subagent_type || 'researcher';
      title = `Sub-Agent (${agentType})`;
      target = argsObj.prompt || argsObj.task || '';
      if (target.length > 40) target = target.substring(0, 40) + '...';
    }

    if (extraClass) pill.classList.add(extraClass);

    const dur = duration_ms ? ` • ${duration_ms}ms` : '';
    const status = success ? '' : ' • <span style="color:#ef4444;">Failed</span>';

    pill.innerHTML = `
      <span class="pill-icon">${icon}</span>
      <span class="pill-title">${escapeHTML(title)}</span>
      <span class="pill-target">${typeof target === 'string' ? escapeHTML(target) : target}</span>
      ${openObsidianBtn}
      <span class="pill-meta">${dur}${status} [details]</span>
    `;

    const drawer = document.createElement('div');
    drawer.className = 'step-drawer';
    drawer.innerHTML = `<strong>Args:</strong>\n${escapeHTML(JSON.stringify(argsObj, null, 2))}\n\n<strong>Output:</strong>\n${escapeHTML(typeof result === 'string' ? result : JSON.stringify(result, null, 2))}`;

    pill.addEventListener('click', () => {
      drawer.style.display = drawer.style.display === 'block' ? 'none' : 'block';
    });

    const wrapper = document.createElement('div');
    wrapper.style.display = 'flex';
    wrapper.style.flexDirection = 'column';
    wrapper.style.gap = '2px';
    wrapper.appendChild(pill);
    wrapper.appendChild(drawer);

    return wrapper;
  }

  function createThoughtPill(duration_s) {
    const pill = document.createElement('div');
    pill.className = 'step-pill thought-pill';
    const durText = duration_s ? `for ${duration_s}s` : '';
    pill.innerHTML = `
      <span class="pill-icon">[THINK]</span>
      <span class="pill-title">Thought ${durText}</span>
      <span class="pill-meta">[reasoning]</span>
    `;
    return pill;
  }

  // --- Welcome Screen & Passwordless User Nickname Login ---
  function renderWelcomeScreen() {
    if (!el.chatTimeline) return;
    el.chatTimeline.innerHTML = '';

    const welcomeDiv = document.createElement('div');
    welcomeDiv.className = 'welcome-empty-state';
    welcomeDiv.id = 'welcome-empty-state';

    if (!state.username) {
      // First-time or switched-out user: prompt for nickname
      welcomeDiv.innerHTML = `
        <div class="login-badge-tag">[IDENTITY]</div>
        <h1 class="welcome-heading">Welcome to Maidere</h1>
        <p class="welcome-subheading">Who are you? Enter a nickname so Maidere knows who she is talking to.</p>

        <form class="nickname-form" onsubmit="event.preventDefault(); window.maidereSetNickname();">
          <div class="nickname-input-wrap">
            <input
              type="text"
              id="nickname-input"
              class="nickname-input"
              placeholder="Enter your nickname (e.g. user)..."
              maxlength="32"
              autofocus
            />
            <button type="submit" id="btn-set-nickname" class="btn-primary nickname-btn">Continue →</button>
          </div>
        </form>
      `;
    } else {
      // Recognized user: personalize greeting with their nickname
      const greeting = `Welcome, ${state.username}`;
      welcomeDiv.innerHTML = `
        <h1 class="welcome-heading">${escapeHTML(greeting)}</h1>
        <p class="welcome-subheading">Maidere Autonomous Agent • Ready for deep research, coding, and Obsidian notes.</p>

        <div class="welcome-user-bar">
          <span class="welcome-user-tag">User: <strong>${escapeHTML(state.username)}</strong></span>
          <button type="button" class="welcome-switch-btn" onclick="window.maidereSwitchUser()" title="Change nickname or switch user">[Switch User]</button>
        </div>

        <div class="welcome-action-cards">
          <div class="welcome-card" onclick="window.maidereInsertPrompt('Conduct a deep research on ')">
            <div class="card-icon">[RESEARCH]</div>
            <div class="card-title">Deep Research</div>
            <div class="card-desc">Multi-query search, doc scraping & structured Obsidian note synthesis</div>
          </div>

          <div class="welcome-card" onclick="window.maidereInsertPrompt('Write a structured note in my Obsidian vault about ')">
            <div class="card-icon">[NOTE]</div>
            <div class="card-title">Obsidian Notes</div>
            <div class="card-desc">Create, format, and organize notes with frontmatter in your local vault</div>
          </div>

          <div class="welcome-card" onclick="window.maidereInsertPrompt('Write and run Python code to ')">
            <div class="card-icon">[EXEC]</div>
            <div class="card-title">Run Code & Shell</div>
            <div class="card-desc">Execute scripts, shell commands, and workspace tasks locally</div>
          </div>
        </div>
      `;
    }

    el.chatTimeline.appendChild(welcomeDiv);
  }

  window.maidereSetNickname = async function (explicitName) {
    const input = document.getElementById('nickname-input');
    const val = (explicitName !== undefined ? explicitName : (input ? input.value : '')).trim();
    if (!val) {
      if (input) input.focus();
      return;
    }
    state.username = val;
    localStorage.setItem('maidere_username', val);
    updateSidebarUser();
    try {
      await fetch('/user/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: val }),
      });
    } catch (_) {}
    renderWelcomeScreen();
  };

  window.maidereSwitchUser = function () {
    showModal(
      'Switch User / Nickname',
      `<div style="display:flex; flex-direction:column; gap:14px;">
        <p style="color:var(--text-secondary); font-size:13px; margin:0;">
          Enter a nickname. Maidere will converse with you according to this name. No passwords needed.
        </p>
        <div>
          <label style="display:block; margin-bottom:6px; font-weight:600; font-size:12px;">Nickname:</label>
          <input
            type="text"
            id="modal-nickname-input"
            value="${escapeHTML(state.username || '')}"
            placeholder="e.g. user"
            maxlength="32"
            style="width:100%; padding:9px 12px; background:#18181b; border:1px solid #27272a; border-radius:6px; color:#fff; font-size:13px;"
          />
        </div>
        <div style="display:flex; gap:8px; justify-content:flex-end;">
          <button class="btn-secondary" onclick="document.getElementById('modal-overlay').classList.remove('active')">Cancel</button>
          <button class="btn-primary" id="modal-save-nickname-btn">Save Nickname</button>
        </div>
      </div>`
    );

    const saveBtn = document.getElementById('modal-save-nickname-btn');
    const nameInput = document.getElementById('modal-nickname-input');
    if (nameInput) {
      nameInput.focus();
      nameInput.select();
      nameInput.onkeydown = (e) => {
        if (e.key === 'Enter') {
          saveBtn.click();
        }
      };
    }
    if (saveBtn) {
      saveBtn.onclick = async () => {
        const newName = nameInput ? nameInput.value.trim() : '';
        if (!newName) return;
        state.username = newName;
        localStorage.setItem('maidere_username', newName);
        updateSidebarUser();
        try {
          await fetch('/user/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username: newName }),
          });
        } catch (_) {}
        closeModal();
        renderWelcomeScreen();
      };
    }
  };

  // --- Rendering Chat Timeline ---
  function appendUserMessage(text) {
    const welcome = document.getElementById('welcome-empty-state');
    if (welcome) welcome.remove();

    const row = document.createElement('div');
    row.className = 'message-row user-row';
    row.innerHTML = `<div class="user-bubble">${escapeHTML(text)}</div>`;
    el.chatTimeline.appendChild(row);
    el.chatTimeline.scrollTop = el.chatTimeline.scrollHeight;
  }


  function createAgentMessageGroup() {
    const row = document.createElement('div');
    row.className = 'message-row agent-row';

    const container = document.createElement('div');
    container.className = 'agent-container';

    const execStream = document.createElement('div');
    execStream.className = 'execution-stream';

    const working = document.createElement('div');
    working.className = 'working-indicator';
    working.innerHTML = `<span class="spinner-ring"></span> <span>Working on your request...</span>`;

    const body = document.createElement('div');
    body.className = 'agent-response-body';

    container.appendChild(execStream);
    container.appendChild(working);
    container.appendChild(body);
    row.appendChild(container);

    el.chatTimeline.appendChild(row);
    el.chatTimeline.scrollTop = el.chatTimeline.scrollHeight;

    return { row, container, execStream, working, body, rawContent: '' };
  }

  // --- API & WebSocket Client ---
  async function loadModels() {
    try {
      const res = await fetch('/models');
      const data = await res.json();
      state.models = data.models || [];
      el.modelSelect.innerHTML = '';

      const autoOpt = document.createElement('option');
      autoOpt.value = 'auto';
      autoOpt.textContent = '[AUTO] Auto-route (Qwen & DeepSeek-R1)';
      if (state.selectedModel === 'auto') autoOpt.selected = true;
      el.modelSelect.appendChild(autoOpt);

      state.models.forEach((m) => {
        if (!m || m === 'auto') return;
        const opt = document.createElement('option');
        opt.value = m;
        if (m.includes('deepseek-r1') || m.includes('r1')) {
          opt.textContent = `[THINK] ${m} (Reasoning)`;
        } else if (m.includes('7b-instruct') || m.includes('7b')) {
          opt.textContent = `[PRIMARY] ${m}`;
        } else if (m.includes('3b')) {
          opt.textContent = `[FAST] ${m}`;
        } else {
          opt.textContent = m;
        }
        if (m === state.selectedModel) opt.selected = true;
        el.modelSelect.appendChild(opt);
      });
    } catch (e) {
      appendTerminal('uvicorn', `Failed to load models: ${e}`, 'warn');
    }
  }

  async function loadThreads() {
    try {
      const res = await fetch('/threads');
      const data = await res.json();
      state.threads = data.threads || [];
      renderThreads();
    } catch (e) {
      appendTerminal('uvicorn', `Failed to load threads: ${e}`, 'warn');
    }
  }

  function renderThreads() {
    if (!el.threadsList) return;
    el.threadsList.innerHTML = '';

    const filter = (state.threadSearchFilter || '').toLowerCase();
    const filtered = state.threads.filter((t) => {
      if (!filter) return true;
      const title = (t.title || '').toLowerCase();
      const summary = (t.summary || '').toLowerCase();
      return title.includes(filter) || summary.includes(filter);
    });

    if (filtered.length === 0) {
      el.threadsList.innerHTML = `<div style="padding: 12px; color: var(--text-dim); font-size: 11px;">No matching conversations.</div>`;
      return;
    }

    filtered.forEach((t) => {
      const item = document.createElement('div');
      item.className = `thread-item ${t.thread_id === state.activeThreadId ? 'active' : ''}`;
      item.onclick = () => switchThread(t.thread_id);

      const relTime = formatRelativeTime(t.last_updated);
      const tooltip = t.summary ? `${t.title}\n\nSummary: ${t.summary}` : (t.title || 'Conversation');

      item.innerHTML = `
        <span class="thread-dot"></span>
        <span class="thread-title" title="${escapeHTML(tooltip)}">${escapeHTML(t.title || 'Conversation')}</span>
        <span class="thread-time">${relTime}</span>
        <div class="thread-actions">
          <button class="thread-action-btn" title="Delete" onclick="event.stopPropagation(); window.maidereDeleteThread('${t.thread_id}')">✕</button>
        </div>
      `;
      el.threadsList.appendChild(item);
    });
  }

  async function switchThread(threadId) {
    if (state.isGenerating) return;
    state.activeThreadId = threadId;
    localStorage.setItem('maidere_active_thread', threadId);

    // Update history stack
    if (state.historyStack[state.historyIndex] !== threadId) {
      state.historyStack = state.historyStack.slice(0, state.historyIndex + 1);
      state.historyStack.push(threadId);
      state.historyIndex = state.historyStack.length - 1;
    }

    renderThreads();
    await loadThreadHistory(threadId);
  }

  async function loadThreadHistory(threadId) {
    el.chatTimeline.innerHTML = '';
    const activeThread = state.threads.find((t) => t.thread_id === threadId);
    if (el.breadcrumbTitle) {
      el.breadcrumbTitle.textContent = activeThread ? activeThread.title : 'Active Conversation';
    }

    try {
      const res = await fetch(`/threads/${threadId}/history`);
      const data = await res.json();
      const messages = data.messages || [];

      let currentGroup = null;

      messages.forEach((msg) => {
        if (msg.role === 'user') {
          appendUserMessage(msg.content);
          currentGroup = null;
        } else if (msg.role === 'assistant') {
          if (!currentGroup) {
            currentGroup = createAgentMessageGroup();
            currentGroup.working.style.display = 'none';
          }
          currentGroup.rawContent = msg.content || '';
          currentGroup.body.innerHTML = safeMarkdownParse(msg.content);
        } else if (msg.role === 'tool') {
          if (!currentGroup) {
            currentGroup = createAgentMessageGroup();
            currentGroup.working.style.display = 'none';
          }
          const pill = createExecutionPill('tool', {}, msg.content, 0, true);
          currentGroup.execStream.appendChild(pill);
        }
      });

      if (messages.length === 0) {
        renderWelcomeScreen();
      }
    } catch (e) {
      appendTerminal('uvicorn', `Failed to load thread history: ${e}`, 'warn');
    }
  }

  function startNewConversation() {
    if (state.isGenerating) return;
    const newId = generateUUID();
    state.activeThreadId = newId;
    localStorage.setItem('maidere_active_thread', newId);
    if (el.breadcrumbTitle) {
      el.breadcrumbTitle.textContent = 'New Conversation';
    }
    renderWelcomeScreen();
    renderThreads();
    if (el.chatInput) el.chatInput.focus();
  }

  window.maidereInsertPrompt = function (prefix) {
    if (!el.chatInput) return;
    el.chatInput.value = prefix;
    el.chatInput.focus();
    el.chatInput.selectionStart = el.chatInput.selectionEnd = el.chatInput.value.length;
    el.chatInput.style.height = 'auto';
    el.chatInput.style.height = `${Math.min(el.chatInput.scrollHeight, 160)}px`;
  };

  // --- Chat Message Dispatch with Token Streaming ---
  async function sendMessage() {
    let text = el.chatInput.value.trim();
    if (!text) return;
    if (state.isGenerating) return;

    el.chatInput.value = '';
    el.chatInput.style.height = '24px';
    state.isGenerating = true;
    el.sendBtn.style.display = 'none';
    el.stopBtn.style.display = 'inline-flex';

    appendUserMessage(text);
    const agentGroup = createAgentMessageGroup();

    setRunningTask('Agent Reasoning', text.substring(0, 40));
    appendTerminal('agent-shell', `User query: "${text}"`, 'info');

    const selectedModelParam = state.selectedModel === 'auto' ? null : state.selectedModel;

    // WebSocket Token Streaming
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${wsProtocol}//${window.location.host}/ws/chat`;

    let wsSuccess = false;

    try {
      const socket = new WebSocket(wsUrl);
      state.activeWs = socket;
      let startTime = Date.now();

      socket.onopen = () => {
        wsSuccess = true;
        socket.send(
          JSON.stringify({
            message: text,
            thread_id: state.activeThreadId,
            model: selectedModelParam,
            num_ctx: state.selectedNumCtx,
            thinking_mode: state.thinkingMode,
            deep_reasoning: state.deepReasoning,
            username: state.username || 'User',
          })
        );
      };

      socket.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);

          if (data.type === 'agent_log') {
            appendTerminal('agent-shell', data.text, data.level || 'info');
          } else if (data.type === 'stage') {
            const stageLabel = data.stage ? data.stage.charAt(0).toUpperCase() + data.stage.slice(1) : 'Working';
            const statusSpan = agentGroup.working.querySelector('span:last-child');
            if (statusSpan) statusSpan.textContent = `Stage: ${stageLabel}...`;
            appendTerminal('agent-shell', `Agent stage: ${data.stage}`, 'info');
          } else if (data.type === 'tool_started') {
            const tName = data.tool_name || 'tool';
            if (tName === 'delegate_task' || tName === 'Agent') {
              const p = data.args?.prompt || data.args?.task || '';
              const st = data.args?.subagent_type || 'researcher';
              appendTerminal('agent-shell', `[DELEGATE] Sub-Agent [${st}]: "${p}"`, 'tool');
            } else {
              const argsSummary = data.args && Object.keys(data.args).length > 0 ? ` (${JSON.stringify(data.args).substring(0, 70)})` : '';
              appendTerminal('agent-shell', `[TOOL] Executing tool: ${tName}${argsSummary}`, 'tool');
            }
          } else if (data.type === 'tool_executed') {
            const callId = data.call_id || `${data.tool_name}_${data.duration_ms}`;
            if (!agentGroup.renderedTools) agentGroup.renderedTools = new Set();
            if (!agentGroup.renderedTools.has(callId)) {
              agentGroup.renderedTools.add(callId);
              const pill = createExecutionPill(
                data.tool_name,
                data.args,
                data.result,
                data.duration_ms,
                data.success
              );
              agentGroup.execStream.appendChild(pill);
            }
            const dur = data.duration_ms !== undefined ? ` (${data.duration_ms}ms)` : '';
            appendTerminal(
              'agent-shell',
              `[OK] Completed ${data.tool_name}${dur}`,
              data.success ? 'success' : 'error'
            );
          } else if (data.type === 'subagent_started') {
            const idxStr = data.total > 1 ? `[${data.index}/${data.total}] ` : '';
            const typeStr = data.subagent_type ? `[${data.subagent_type}] ` : '';
            const statusSpan = agentGroup.working.querySelector('span:last-child');
            if (statusSpan) statusSpan.textContent = `Sub-Agent ${typeStr}${idxStr}Working...`;
            setRunningTask(`Sub-Agent:${data.subagent_type || 'worker'}`, `${data.task || ''}`);
            appendTerminal('agent-shell', `[START] Sub-Agent ${typeStr}${idxStr}Started: "${data.task}"`, 'info');
          } else if (data.type === 'subagent_tool_started') {
            const typeStr = data.subagent_type ? `[${data.subagent_type}] ` : '';
            if (data.tool_name === 'web_search') {
              const q = data.args?.query ? ` "${data.args.query}"` : '';
              appendTerminal('agent-shell', `  ↳ Sub-agent ${typeStr}searching web:${q}`, 'tool');
            } else if (data.tool_name === 'browser') {
              const u = data.args?.url ? ` ${data.args.url}` : '';
              appendTerminal('agent-shell', `  ↳ Sub-agent ${typeStr}browsing:${u}`, 'tool');
            } else if (data.tool_name === 'read_file') {
              const p = data.args?.path ? ` ${data.args.path}` : '';
              appendTerminal('agent-shell', `  ↳ Sub-agent ${typeStr}reading file:${p}`, 'tool');
            } else {
              appendTerminal('agent-shell', `  ↳ Sub-agent ${typeStr}running tool: ${data.tool_name}`, 'tool');
            }
          } else if (data.type === 'subagent_tool') {
            const typeStr = data.subagent_type ? `[${data.subagent_type}] ` : '';
            const status = data.success ? '[OK]' : '[FAIL]';
            const dur = data.duration_ms !== undefined ? ` in ${data.duration_ms}ms` : '';
            const preview = data.result_preview ? ` — ${data.result_preview.substring(0, 90)}` : '';
            appendTerminal(
              'agent-shell',
              `  ↳ Sub-agent ${typeStr}${data.tool_name} ${status}${dur}${preview}`,
              data.success ? 'success' : 'warn'
            );
          } else if (data.type === 'subagent_completed') {
            const idxStr = data.total > 1 ? `[${data.index}/${data.total}] ` : '';
            const typeStr = data.subagent_type ? `[${data.subagent_type}] ` : '';
            const toolsInfo = data.tools_count !== undefined ? ` (${data.tools_count} tools used)` : '';
            appendTerminal(
              'agent-shell',
              `[DONE] Sub-Agent ${typeStr}${idxStr}Finished in ${data.duration_ms}ms${toolsInfo}`,
              'success'
            );
            setRunningTask(null);
          } else if (data.type === 'token') {
            // Real-time incremental token streaming
            agentGroup.working.style.display = 'none';
            agentGroup.rawContent = (agentGroup.rawContent || '') + data.text;
            agentGroup.body.innerHTML = safeMarkdownParse(agentGroup.rawContent);
            el.chatTimeline.scrollTop = el.chatTimeline.scrollHeight;
          } else if (data.type === 'response') {
            agentGroup.working.style.display = 'none';
            const responseText = data.content || data.response || agentGroup.rawContent || '';
            agentGroup.rawContent = responseText;
            agentGroup.body.innerHTML = safeMarkdownParse(responseText);
            const duration_s = ((Date.now() - startTime) / 1000).toFixed(1);
            agentGroup.execStream.prepend(createThoughtPill(duration_s));
            setRunningTask(null);
            finishGeneration();
            loadThreads();
          } else if (data.type === 'cancelled') {
            agentGroup.working.style.display = 'none';
            agentGroup.body.innerHTML += `<div style="color:var(--text-dim); margin-top:8px; font-style:italic;">[Generation stopped by user]</div>`;
            setRunningTask(null);
            finishGeneration();
          } else if (data.type === 'error') {
            agentGroup.working.style.display = 'none';
            agentGroup.body.innerHTML = `<span style="color:#ef4444;">Error: ${escapeHTML(data.message)}</span>`;
            setRunningTask(null);
            finishGeneration();
          }
        } catch (err) {
          console.error('WS Parse Error:', err);
        }
      };

      socket.onerror = () => {
        if (!wsSuccess) fallbackRestChat(text, agentGroup, selectedModelParam);
      };
    } catch (e) {
      fallbackRestChat(text, agentGroup, selectedModelParam);
    }
  }

  async function fallbackRestChat(text, agentGroup, model) {
    appendTerminal('agent-shell', `Executing via REST fallback...`, 'warn');
    const startTime = Date.now();

    try {
      const res = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: text,
          thread_id: state.activeThreadId,
          model: model,
          num_ctx: state.selectedNumCtx,
          thinking_mode: state.thinkingMode,
          deep_reasoning: state.deepReasoning,
          username: state.username || 'User',
        }),
      });

      if (!res.ok) {
        const errText = await res.text();
        throw new Error(errText || res.statusText);
      }

      const data = await res.json();
      agentGroup.working.style.display = 'none';

      if (data.tools_executed) {
        data.tools_executed.forEach((t) => {
          const pill = createExecutionPill(t.tool_name, t.args, t.result, t.duration_ms, t.success);
          agentGroup.execStream.appendChild(pill);
          appendTerminal(
            'agent-shell',
            `Executed ${t.tool_name} (${t.duration_ms}ms)`,
            t.success ? 'success' : 'error'
          );
        });
      }

      const duration_s = ((Date.now() - startTime) / 1000).toFixed(1);
      agentGroup.execStream.prepend(createThoughtPill(duration_s));
      const responseText = data.response || data.content || '';
      agentGroup.body.innerHTML = safeMarkdownParse(responseText);

      setRunningTask(null);
      finishGeneration();
      loadThreads();
    } catch (e) {
      agentGroup.working.style.display = 'none';
      agentGroup.body.innerHTML = `<span style="color:#ef4444;">Error: ${escapeHTML(e.message)}</span>`;
      setRunningTask(null);
      finishGeneration();
    }
  }

  function stopGeneration() {
    if (!state.isGenerating) return;
    if (state.activeWs && state.activeWs.readyState === WebSocket.OPEN) {
      state.activeWs.send(JSON.stringify({ type: 'cancel', thread_id: state.activeThreadId }));
    }
    fetch('/chat/cancel', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ thread_id: state.activeThreadId }),
    }).catch(() => {});
    finishGeneration();
    appendTerminal('agent-shell', 'Generation cancel signal sent.', 'warn');
  }

  function finishGeneration() {
    state.isGenerating = false;
    el.sendBtn.style.display = 'inline-flex';
    el.stopBtn.style.display = 'none';
  }

  // --- Global Handlers ---
  window.maidereDeleteThread = async function (threadId) {
    if (!confirm('Are you sure you want to delete this conversation?')) return;
    try {
      await fetch(`/threads/${threadId}`, { method: 'DELETE' });
      state.threads = state.threads.filter((t) => t.thread_id !== threadId);
      if (state.activeThreadId === threadId) {
        startNewConversation();
      } else {
        renderThreads();
      }
    } catch (e) {
      alert(`Failed to delete thread: ${e}`);
    }
  };

  window.maidereOpenObsidian = async function (title) {
    try {
      await fetch('/obsidian/open', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ file: title }),
      });
      appendTerminal('obsidian', `Opened note "${title}" in desktop Obsidian`, 'success');
    } catch (e) {
      appendTerminal('obsidian', `Failed to open note: ${e}`, 'error');
    }
  };

  window.maidereDeleteMemory = async function (memId) {
    try {
      await fetch(`/memories/${memId}`, { method: 'DELETE' });
      const row = document.getElementById(`mem-row-${memId}`);
      if (row) row.remove();
      appendTerminal('agent-shell', `Deleted memory ID ${memId}`, 'info');
    } catch (e) {
      alert(`Failed to delete memory: ${e}`);
    }
  };

  // --- Modal Helpers ---
  function showModal(title, contentHtml) {
    el.modalTitle.textContent = title;
    el.modalBody.innerHTML = contentHtml;
    el.modalOverlay.classList.add('active');
  }

  function closeModal() {
    el.modalOverlay.classList.remove('active');
    if (el.commandPaletteOverlay) el.commandPaletteOverlay.classList.remove('active');
  }

  // --- Command Palette (Ctrl+K) ---
  const commandPaletteItems = [
    { title: 'New Conversation', icon: '[CHAT]', shortcut: 'Ctrl+N', action: () => startNewConversation() },
    { title: 'Filter Conversations', icon: '[SEARCH]', shortcut: '', action: () => el.threadSearchInput.focus() },
    { title: 'Toggle Sidebar', icon: '[PANEL]', shortcut: 'Ctrl+B', action: () => toggleSidebar() },
    { title: 'Toggle Terminal', icon: '[SHELL]', shortcut: 'Ctrl+J', action: () => toggleTerminal() },
    { title: 'Export Conversation (Markdown)', icon: '[EXPORT]', shortcut: '', action: () => exportActiveThread('md') },
    { title: 'Export Conversation (JSON)', icon: '[EXPORT]', shortcut: '', action: () => exportActiveThread('json') },
    { title: 'Semantic Memories Explorer', icon: '[MEMORY]', shortcut: '', action: () => openMemoriesModal() },
    { title: 'Agent Skills Explorer', icon: '[SKILLS]', shortcut: '', action: () => openSkillsModal() },
    { title: 'Obsidian Vault Explorer', icon: '[NOTE]', shortcut: '', action: () => openObsidianModal() },
    { title: 'Switch User / Nickname', icon: '[USER]', shortcut: '', action: () => window.maidereSwitchUser() },
    { title: 'Settings & System Prompt', icon: '[CONFIG]', shortcut: '', action: () => openSettingsModal() },
  ];

  function openCommandPalette() {
    el.commandPaletteOverlay.classList.add('active');
    el.paletteSearchInput.value = '';
    renderPaletteResults('');
    el.paletteSearchInput.focus();
  }

  function renderPaletteResults(query) {
    const q = (query || '').toLowerCase();
    const filtered = commandPaletteItems.filter((i) => !q || i.title.toLowerCase().includes(q));
    el.paletteResults.innerHTML = '';

    filtered.forEach((item) => {
      const row = document.createElement('div');
      row.className = 'palette-item';
      row.innerHTML = `
        <span style="display:flex; align-items:center; gap:8px;">
          <span style="font-size:10px; font-family:var(--font-mono); color:var(--text-muted);">${item.icon}</span>
          <span>${escapeHTML(item.title)}</span>
        </span>
        ${item.shortcut ? `<span class="palette-item-shortcut">${item.shortcut}</span>` : ''}
      `;
      row.onclick = () => {
        closeModal();
        item.action();
      };
      el.paletteResults.appendChild(row);
    });
  }

  function exportActiveThread(format = 'md') {
    window.location.href = `/threads/${state.activeThreadId}/export?format=${format}`;
  }

  async function openMemoriesModal() {
    try {
      const res = await fetch('/memories?top_k=30');
      const data = await res.json();
      const memories = data.memories || [];

      let listHtml = memories
        .map(
          (m) => `
        <div class="memory-row" id="mem-row-${m.id}">
          <div class="memory-row-header">
            <span>Saved: ${m.timestamp ? m.timestamp.substring(0, 16) : 'N/A'}</span>
            <button class="memory-delete-btn" onclick="window.maidereDeleteMemory(${m.id})">Delete</button>
          </div>
          <div>${escapeHTML(m.content)}</div>
        </div>
      `
        )
        .join('');

      showModal(
        'Semantic Memory Management',
        `<div style="display:flex; flex-direction:column; gap:10px;">
          <input type="text" id="memory-search-input" placeholder="Search memories..." style="width:100%; padding:8px; background:#151518; border:1px solid #27272a; border-radius:4px; color:#fff;" />
          <div id="memory-search-results" style="display:flex; flex-direction:column; gap:8px; max-height:350px; overflow-y:auto;">
            ${listHtml || '<p style="color:var(--text-dim);">No memories stored yet.</p>'}
          </div>
        </div>`
      );

      document.getElementById('memory-search-input').addEventListener('input', async (e) => {
        const q = e.target.value.trim();
        const resEl = document.getElementById('memory-search-results');
        try {
          const url = q ? `/memories?query=${encodeURIComponent(q)}&top_k=20` : '/memories?top_k=30';
          const sres = await fetch(url);
          const sdata = await sres.json();
          const items = sdata.memories || [];
          if (items.length === 0) {
            resEl.innerHTML = '<p style="color:var(--text-dim);">No matching memories found.</p>';
          } else {
            resEl.innerHTML = items
              .map(
                (m) => `
              <div class="memory-row" id="mem-row-${m.id || ''}">
                <div class="memory-row-header">
                  <span>Score Distance: ${m.distance !== undefined ? m.distance.toFixed(3) : 'N/A'}</span>
                  ${m.id ? `<button class="memory-delete-btn" onclick="window.maidereDeleteMemory(${m.id})">Delete</button>` : ''}
                </div>
                <div>${escapeHTML(m.content)}</div>
              </div>
            `
              )
              .join('');
          }
        } catch (err) {}
      });
    } catch (e) {
      alert(`Failed to load memories: ${e}`);
    }
  }

  async function openSkillsModal() {
    try {
      const res = await fetch('/skills');
      const data = await res.json();
      const skills = data.skills || [];

      let skillsHtml = skills
        .map(
          (s) => `
        <div class="skill-card">
          <div class="skill-card-header">
            <strong>${escapeHTML(s.name)}</strong>
            <span>Triggers: ${(s.triggers || []).join(', ')}</span>
          </div>
          <div style="font-size:12px; color:var(--text-secondary);">${escapeHTML(s.description)}</div>
        </div>
      `
        )
        .join('');

      showModal(
        'Active Agent Skills',
        `<div style="display:flex; flex-direction:column; gap:10px; max-height:400px; overflow-y:auto;">
          ${skillsHtml}
        </div>`
      );
    } catch (e) {
      alert(`Failed to load skills: ${e}`);
    }
  }

  async function openObsidianModal() {
    try {
      const res = await fetch('/obsidian/vault');
      const vdata = await res.json();
      showModal(
        'Obsidian Vault Explorer',
        `<div>
          <p style="margin-bottom: 8px; color: var(--text-muted);">Active Vault:</p>
          <code style="display:block; padding: 8px; background:#18181b; border:1px solid #27272a; border-radius:4px; margin-bottom:12px;">${escapeHTML(vdata.vault_path)} (${escapeHTML(vdata.vault_name)})</code>
          <button class="btn-primary" onclick="window.maidereOpenObsidian('');">Open Vault in Desktop Obsidian App</button>
        </div>`
      );
    } catch (e) {
      alert(`Failed to fetch Obsidian vault: ${e}`);
    }
  }

  async function openSettingsModal() {
    try {
      const res = await fetch('/settings/system-prompt');
      const data = await res.json();
      const currentPrompt = data.system_prompt || '';

      showModal(
        'Settings & Custom System Prompt',
        `<div style="display:flex; flex-direction:column; gap:12px;">
          <div>
            <label style="display:block; margin-bottom:6px; font-weight:600;">Custom System Prompt Instructions:</label>
            <textarea id="system-prompt-input" style="width:100%; height:180px; background:#151518; border:1px solid #27272a; border-radius:4px; color:#fff; padding:8px; font-family:var(--font-mono); font-size:11px;">${escapeHTML(currentPrompt)}</textarea>
          </div>
          <div style="display:flex; gap:8px;">
            <button class="btn-primary" id="save-prompt-btn">Save System Prompt</button>
            <button class="btn-secondary" id="reset-prompt-btn">Reset to Default</button>
          </div>
        </div>`
      );

      document.getElementById('save-prompt-btn').onclick = async () => {
        const val = document.getElementById('system-prompt-input').value;
        await fetch('/settings/system-prompt', {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ system_prompt: val }),
        });
        alert('System prompt updated!');
        closeModal();
      };

      document.getElementById('reset-prompt-btn').onclick = async () => {
        await fetch('/settings/system-prompt', {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ system_prompt: null }),
        });
        alert('System prompt reset to default!');
        closeModal();
      };
    } catch (e) {
      alert(`Failed to load settings: ${e}`);
    }
  }

  // --- Event Listeners Setup ---
  function setupEventListeners() {
    el.btnToggleSidebar?.addEventListener('click', toggleSidebar);
    el.btnToggleSidebarMain?.addEventListener('click', toggleSidebar);
    el.btnSidebarEdge?.addEventListener('click', toggleSidebar);
    el.btnToggleTerminal?.addEventListener('click', toggleTerminal);
    el.btnNewChat.addEventListener('click', startNewConversation);
    el.btnExportThread.addEventListener('click', () => exportActiveThread('md'));

    el.threadSearchInput.addEventListener('input', (e) => {
      state.threadSearchFilter = e.target.value;
      renderThreads();
    });

    el.btnNavBack.addEventListener('click', () => {
      if (state.historyIndex > 0) {
        state.historyIndex--;
        switchThread(state.historyStack[state.historyIndex]);
      }
    });

    el.btnNavForward.addEventListener('click', () => {
      if (state.historyIndex < state.historyStack.length - 1) {
        state.historyIndex++;
        switchThread(state.historyStack[state.historyIndex]);
      }
    });

    el.chatInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    });

    el.chatInput.addEventListener('input', () => {
      el.chatInput.style.height = 'auto';
      el.chatInput.style.height = `${Math.min(el.chatInput.scrollHeight, 160)}px`;
    });

    el.sendBtn.addEventListener('click', sendMessage);
    el.stopBtn.addEventListener('click', stopGeneration);

    el.modelSelect.addEventListener('change', (e) => {
      state.selectedModel = e.target.value;
      localStorage.setItem('maidere_selected_model', state.selectedModel);
      appendTerminal('agent-shell', `Model switched to ${state.selectedModel}`, 'info');
    });

    if (el.contextSelect) {
      el.contextSelect.value = String(state.selectedNumCtx || 16384);
      el.contextSelect.addEventListener('change', (e) => {
        const val = parseInt(e.target.value, 10);
        state.selectedNumCtx = val;
        localStorage.setItem('maidere_selected_num_ctx', String(val));
        const labels = { 8192: '8K - Medium', 16384: '16K - High', 32768: '32K - Ultra' };
        appendTerminal('agent-shell', `[CONFIG] Context window switched to ${labels[val] || val}`, 'info');
      });
    }

    el.btnModeAuto?.addEventListener('click', () => setCognitiveMode('auto'));
    el.btnModeThinking?.addEventListener('click', () => setCognitiveMode(state.thinkingMode && !state.deepReasoning ? 'auto' : 'thinking'));
    el.btnModeDeepReason?.addEventListener('click', () => setCognitiveMode(state.deepReasoning ? 'auto' : 'deep-reason'));
    el.deckBtnThinking?.addEventListener('click', () => setCognitiveMode(state.thinkingMode && !state.deepReasoning ? 'auto' : 'thinking'));
    el.deckBtnDeepReason?.addEventListener('click', () => setCognitiveMode(state.deepReasoning ? 'auto' : 'deep-reason'));

    document.querySelectorAll('.term-tab').forEach((tabBtn) => {
      tabBtn.addEventListener('click', () => {
        document.querySelectorAll('.term-tab').forEach((b) => b.classList.remove('active'));
        tabBtn.classList.add('active');
        state.activeTerminalTab = tabBtn.dataset.tab;
        renderActiveTerminal();
      });
    });

    el.btnClearTerminal.addEventListener('click', () => {
      state.terminalLogs[state.activeTerminalTab] = [];
      renderActiveTerminal();
    });

    el.btnCopyTerminal.addEventListener('click', () => {
      const logs = (state.terminalLogs[state.activeTerminalTab] || [])
        .map((l) => `[${l.time}] ${l.text}`)
        .join('\n');
      navigator.clipboard.writeText(logs);
      appendTerminal(state.activeTerminalTab, 'Terminal buffer copied to clipboard', 'info');
    });

    el.obsidianVaultPill?.addEventListener('click', openObsidianModal);
    el.memoriesPill?.addEventListener('click', openMemoriesModal);
    el.skillsPill?.addEventListener('click', openSkillsModal);
    el.btnSettings?.addEventListener('click', openSettingsModal);
    el.btnSwitchUser?.addEventListener('click', () => window.maidereSwitchUser());

    el.scheduledTasksPill?.addEventListener('click', async () => {
      try {
        const res = await fetch('/scheduler/tasks');
        const data = await res.json();
        const tasks = data.tasks || [];
        let html = '<div style="display:flex; flex-direction:column; gap:8px;">';
        if (tasks.length === 0) {
          html += '<p style="color:var(--text-dim);">No scheduled cron jobs active.</p>';
        } else {
          tasks.forEach((t) => {
            html += `<div style="padding:8px; background:#18181b; border:1px solid #27272a; border-radius:6px;">
              <strong>${escapeHTML(t.name || t.id)}</strong> (${escapeHTML(t.cron_expression || 'interval')})<br>
              <span style="color:var(--text-muted); font-size:11px;">Prompt: ${escapeHTML(t.prompt)}</span>
            </div>`;
          });
        }
        html += '</div>';
        showModal('Scheduled Tasks & Reminders', html);
      } catch (e) {
        alert(`Failed to fetch scheduler tasks: ${e}`);
      }
    });

    el.modalClose?.addEventListener('click', closeModal);
    el.modalOverlay?.addEventListener('click', (e) => {
      if (e.target === el.modalOverlay) closeModal();
    });

    el.commandPaletteOverlay?.addEventListener('click', (e) => {
      if (e.target === el.commandPaletteOverlay) closeModal();
    });

    el.paletteSearchInput?.addEventListener('input', (e) => {
      renderPaletteResults(e.target.value);
    });

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
      if (e.ctrlKey && e.key === 'k') {
        e.preventDefault();
        openCommandPalette();
      } else if (e.ctrlKey && e.key === 'b') {
        e.preventDefault();
        toggleSidebar();
      } else if (e.ctrlKey && e.key === 'j') {
        e.preventDefault();
        toggleTerminal();
      } else if (e.ctrlKey && e.key === 'n') {
        e.preventDefault();
        startNewConversation();
      } else if (e.key === 'Escape') {
        closeModal();
      }
    });
  }

  // --- Bootstrap ---
  async function init() {
    initElements();
    applyLayout();
    setupEventListeners();
    updateModeUI();
    if (!state.username || state.username.toLowerCase() === 'khazar') {
      state.username = 'User';
      localStorage.setItem('maidere_username', 'User');
    }
    updateSidebarUser();

    appendTerminal('agent-shell', 'Maidere IDE Agent environment initialized.', 'success');
    appendTerminal('uvicorn', 'Connected to Maidere backend on http://localhost:8000', 'info');

    await loadModels();
    await loadThreads();

    const isSessionActive = sessionStorage.getItem('maidere_session_active');
    if (!isSessionActive) {
      sessionStorage.setItem('maidere_session_active', 'true');
      startNewConversation();
    } else if (state.activeThreadId) {
      const exists = state.threads.some((t) => t.thread_id === state.activeThreadId);
      if (exists) {
        await loadThreadHistory(state.activeThreadId);
      } else {
        startNewConversation();
      }
    } else {
      startNewConversation();
    }
  }

  window.addEventListener('DOMContentLoaded', init);
})();


