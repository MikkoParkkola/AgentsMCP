/* --------------------------------------------------------------
   AgentsMCP Dashboard – Core JavaScript
   --------------------------------------------------------------
   Features:
   1️⃣  JWT token handling with transparent refresh
   2️⃣  Generic, retry‑aware API client for /auth, /agents,
       /tasks, /system endpoints
   3️⃣  Server‑Sent Events (SSE) listener for real‑time updates
   4️⃣  Chart.js visualisation of agent status
   5️⃣  Dark / Light theme toggling (persisted)
   6️⃣  Form handling (login, create‑agent, create‑task)
   7️⃣  Live table updates for agents & tasks
   8️⃣  Bootstrap 5 toast notifications
   9️⃣  Centralised error handling & exponential back‑off
   🔟  Mobile‑friendly event handlers
   1️⃣1️⃣ Bootstrap 5 modal integration
   1️⃣2️⃣ Accessibility‑first markup & ARIA usage
   -------------------------------------------------------------- */

/* -----------------------------------------------------------------
   0️⃣  GLOBAL CONSTANTS & STATE
   ----------------------------------------------------------------- */
const API_BASE = ""; // FastAPI serves from root
const TOKEN_KEY = "agentsmcp_jwt";
const REFRESH_TIMEOUT = 5 * 60 * 1000; // 5 min before expiry → refresh
const MAX_RETRY = 3;
const RETRY_BASE_DELAY = 500; // ms, exponential back‑off

/* DOM references (cached for performance) */
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

/* -----------------------------------------------------------------
   0️⃣.1  READINESS BANNER HELPERS
   ----------------------------------------------------------------- */
function showBanner(msg, level = "warning") {
  const el = $("#globalBanner");
  if (!el) return;
  el.classList.remove("d-none", "alert-warning", "alert-danger", "alert-info", "alert-success");
  el.classList.add(`alert-${level}`);
  el.textContent = msg;
}

function hideBanner() {
  const el = $("#globalBanner");
  if (!el) return;
  el.classList.add("d-none");
}

async function waitForReadiness(maxWaitMs = 10000) {
  const start = Date.now();
  const msg = "System not ready yet. Retrying connection...";
  showBanner(msg, "warning");
  while (Date.now() - start < maxWaitMs) {
    try {
      const resp = await fetch(`${API_BASE}/health/ready`, { cache: "no-store" });
      if (resp.ok) {
        hideBanner();
        return true;
      }
    } catch (e) {
      // ignore and retry with backoff
    }
    await new Promise((r) => setTimeout(r, 500));
  }
  showBanner("Service unavailable. Some features may be disabled.", "danger");
  return false;
}

/* -----------------------------------------------------------------
   1️⃣  JWT TOKEN MANAGEMENT
   ----------------------------------------------------------------- */
const tokenStore = {
  get() {
    const raw = localStorage.getItem(TOKEN_KEY);
    if (!raw) return null;
    try {
      const parsed = JSON.parse(atob(raw.split(".")[1])); // decode payload
      return { raw, exp: parsed.exp };
    } catch (e) {
      console.warn("Invalid JWT in storage – clearing");
      this.clear();
      return null;
    }
  },

  set(jwt) {
    localStorage.setItem(TOKEN_KEY, jwt);
    scheduleRefresh(jwt);
  },

  clear() {
    localStorage.removeItem(TOKEN_KEY);
    if (window.__refreshTimer) clearTimeout(window.__refreshTimer);
  },

  /** Returns the raw token string (or null). */
  raw() {
    const t = this.get();
    return t ? t.raw : null;
  },
};

/** Schedule a refresh a few minutes before the token expires. */
function scheduleRefresh(jwt) {
  const payload = JSON.parse(atob(jwt.split(".")[1]));
  const expiresInMs = payload.exp * 1000 - Date.now();
  const refreshInMs = Math.max(0, expiresInMs - REFRESH_TIMEOUT);
  if (window.__refreshTimer) clearTimeout(window.__refreshTimer);
  window.__refreshTimer = setTimeout(refreshToken, refreshInMs);
}

/** Calls /auth/refresh, updates storage and resolves with new token. */
async function refreshToken() {
  try {
    const resp = await fetch(`${API_BASE}/auth/refresh`, {
      method: "POST",
      headers: { Authorization: `Bearer ${tokenStore.raw()}` },
    });
    if (!resp.ok) throw new Error("Refresh failed");
    const { access_token } = await resp.json();
    tokenStore.set(access_token);
    showToast("Session refreshed", "success");
  } catch (err) {
    console.error(err);
    logout(); // fallback – force re‑login
  }
}

/** Log out: clear token, redirect to login page (or show modal). */
function logout() {
  tokenStore.clear();
  if (agentsEventSource) {
    agentsEventSource.close();
    agentsEventSource = null;
  }
  showSection('loginSection');
  showToast("Logged out", "info");
}

/* -----------------------------------------------------------------
   2️⃣  API CLIENT (fetch wrapper with retries & auth)
   ----------------------------------------------------------------- */
async function apiRequest(
  endpoint,
  { method = "GET", body = null, auth = true, retry = 0 } = {}
) {
  const url = `${API_BASE}${endpoint}`;
  const headers = {};

  if (auth) {
    const jwt = tokenStore.raw();
    if (!jwt) throw new Error("No JWT – user not authenticated");
    headers["Authorization"] = `Bearer ${jwt}`;
  }

  if (body && !(body instanceof FormData)) {
    headers["Content-Type"] = "application/json";
    body = JSON.stringify(body);
  }

  try {
    const resp = await fetch(url, {
      method,
      headers,
      body,
    });

    // 401 → try refresh once
    if (resp.status === 401 && auth && retry === 0) {
      await refreshToken();
      return apiRequest(endpoint, { method, body, auth, retry: 1 });
    }

    if (!resp.ok) {
      const errData = await resp.json().catch(() => ({}));
      const err = new Error(errData.detail || resp.statusText);
      err.status = resp.status;
      err.data = errData;
      throw err;
    }

    // 204 No Content
    if (resp.status === 204) return null;

    return await resp.json();
  } catch (err) {
    if (retry < MAX_RETRY && err.name !== "TypeError") {
      const delay = RETRY_BASE_DELAY * 2 ** retry;
      await new Promise((r) => setTimeout(r, delay));
      return apiRequest(endpoint, { method, body, auth, retry: retry + 1 });
    }
    // bubble up after max retries
    console.error(`API ${method} ${endpoint} failed:`, err);
    showToast(`Error: ${err.message}`, "danger");
    throw err;
  }
}

/* -----------------------------------------------------------------
   3️⃣  SERVER‑SENT EVENTS (real‑time agent updates)
   ----------------------------------------------------------------- */
let agentsEventSource = null;
function startAgentsSSE() {
  if (agentsEventSource) agentsEventSource.close();

  const jwt = tokenStore.raw();
  if (!jwt) return;

  agentsEventSource = new EventSource(`${API_BASE}/events/agents?token=${jwt}`);

  agentsEventSource.onmessage = (e) => {
    try {
      const data = JSON.parse(e.data);
      handleAgentEvent(data);
      // Show pulse animation on live events indicator
      const indicator = $("#liveEventsIndicator");
      if (indicator) {
        indicator.classList.add("pulse");
        setTimeout(() => indicator.classList.remove("pulse"), 1000);
      }
    } catch (err) {
      console.warn("Invalid SSE data:", e.data);
    }
  };

  agentsEventSource.onerror = (e) => {
    console.warn("SSE error, attempting reconnect in 5 s", e);
    agentsEventSource.close();
    setTimeout(startAgentsSSE, 5000);
  };
}

/* -----------------------------------------------------------------
   4️⃣  CHART.JS – Agent status visualisation
   ----------------------------------------------------------------- */
let agentChart = null;
function initAgentChart() {
  const ctx = $("#agentsChart")?.getContext("2d");
  if (!ctx) return;

  const data = {
    labels: ["Online", "Idle", "Busy", "Offline"],
    datasets: [
      {
        label: "Agents",
        data: [0, 0, 0, 0],
        backgroundColor: ["#198754", "#ffc107", "#fd7e14", "#6c757d"],
        borderWidth: 0,
      },
    ],
  };

  agentChart = new Chart(ctx, {
    type: "doughnut",
    data,
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: "bottom" },
        tooltip: { enabled: true },
      },
    },
  });
}

/* -----------------------------------------------------------------
   9️⃣  BOOTSTRAP – on DOM ready, gate by readiness, then init SSE/UI
   ----------------------------------------------------------------- */
document.addEventListener("DOMContentLoaded", async () => {
  const ready = await waitForReadiness(8000);
  // Initialize charts/UI regardless, but only start SSE if ready
  initAgentChart();
  if (ready) {
    startAgentsSSE();
  }
});

/** Update the chart with a fresh count object {online, idle, busy, offline}. */
function updateAgentChart(counts) {
  if (!agentChart) return;
  agentChart.data.datasets[0].data = [
    counts.online || 0,
    counts.idle || 0,
    counts.busy || 0,
    counts.offline || 0,
  ];
  agentChart.update();
}

/* -----------------------------------------------------------------
   5️⃣  THEME SWITCHING (dark / light)
   ----------------------------------------------------------------- */
const THEME_KEY = "agentsmcp_theme";

function applyTheme(theme) {
  const root = document.documentElement;
  if (theme === "dark") {
    root.setAttribute("data-theme", "dark");
  } else {
    root.setAttribute("data-theme", "light");
  }
  localStorage.setItem(THEME_KEY, theme);
  
  // Update theme toggle icon
  const themeIcon = $("#themeToggle i");
  if (themeIcon) {
    themeIcon.className = theme === "dark" ? "bi bi-sun" : "bi bi-moon";
  }
}

function initThemeToggle() {
  const saved = localStorage.getItem(THEME_KEY) || 
    (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  applyTheme(saved);
  
  const toggle = $("#themeToggle");
  if (!toggle) return;

  toggle.addEventListener("click", (e) => {
    e.preventDefault();
    const currentTheme = localStorage.getItem(THEME_KEY) || "light";
    applyTheme(currentTheme === "dark" ? "light" : "dark");
  });
}

/* -----------------------------------------------------------------
   6️⃣  FORM HANDLING
   ----------------------------------------------------------------- */
function initLoginForm() {
  const form = $("#loginForm");
  if (!form) return;

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const formData = new FormData(form);
    
    try {
      showLoading($("#loginError"), true);
      const { access_token } = await apiRequest("/auth/login", {
        method: "POST",
        body: formData,
        auth: false,
      });
      
      tokenStore.set(access_token);
      showSection('dashboardSection');
      await loadDashboardData();
      showToast("Login successful", "success");
    } catch (err) {
      console.error(err);
      showError($("#loginError"), err.message);
    } finally {
      showLoading($("#loginError"), false);
    }
  });
}

/* Agent creation form */
function initAgentForm() {
  const form = $("#agentForm");
  if (!form) return;

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const formData = new FormData(form);
    const payload = Object.fromEntries(formData.entries());

    try {
      await apiRequest("/agents", { method: "POST", body: payload });
      form.reset();
      const modal = bootstrap.Modal.getInstance($("#agentModal"));
      if (modal) modal.hide();
      showToast("Agent created successfully", "success");
    } catch (err) {
      showError($("#agentFormError"), err.message);
    }
  });
}

/* Task creation form */
function initTaskForm() {
  const form = $("#taskCreateForm");
  if (!form) return;

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const formData = new FormData(form);
    const payload = {
      agent_id: formData.get("taskAgentSelect"),
      payload: JSON.parse(formData.get("taskPayload")),
      priority: parseInt(formData.get("taskPriority")) || 0,
    };

    try {
      await apiRequest("/tasks", { method: "POST", body: payload });
      form.reset();
      showToast("Task created successfully", "success");
      showSuccess($("#taskCreateSuccess"), "Task created successfully");
    } catch (err) {
      showError($("#taskCreateError"), err.message);
    }
  });
}

/* -----------------------------------------------------------------
   7️⃣  REAL‑TIME TABLE UPDATES
   ----------------------------------------------------------------- */
function renderAgentsTable(agents) {
  const tbody = $("#agentsTable tbody");
  if (!tbody) return;

  tbody.innerHTML = ""; // clear
  
  if (agents.length === 0) {
    $("#agentsEmpty").classList.remove("d-none");
    return;
  }
  $("#agentsEmpty").classList.add("d-none");
  
  agents.forEach((a) => {
    const tr = document.createElement("tr");
    tr.dataset.agentId = a.id;
    tr.innerHTML = `
      <td>${escapeHtml(a.id)}</td>
      <td>${escapeHtml(a.name)}</td>
      <td><span class="status-badge ${a.status}">${a.status}</span></td>
      <td>${a.last_heartbeat ? new Date(a.last_heartbeat).toLocaleString() : 'Never'}</td>
      <td>
        <button class="btn btn-sm btn-primary me-1" data-action="edit">
          <i class="bi bi-pencil"></i>
        </button>
        <button class="btn btn-sm btn-danger" data-action="delete">
          <i class="bi bi-trash"></i>
        </button>
      </td>
    `;
    tbody.appendChild(tr);
  });
}

function renderTasksTable(tasks) {
  const tbody = $("#tasksTable tbody");
  if (!tbody) return;
  
  tbody.innerHTML = "";
  
  if (tasks.length === 0) {
    $("#tasksEmpty").classList.remove("d-none");
    return;
  }
  $("#tasksEmpty").classList.add("d-none");
  
  tasks.slice(0, 10).forEach((t) => { // Show only recent 10
    const tr = document.createElement("tr");
    tr.dataset.taskId = t.id;
    tr.innerHTML = `
      <td>${escapeHtml(t.id.substring(0, 8))}</td>
      <td>${escapeHtml(t.agent_id || 'N/A')}</td>
      <td><span class="status-badge ${t.status}">${t.status}</span></td>
      <td>${new Date(t.created_at).toLocaleString()}</td>
      <td>
        <button class="btn btn-sm btn-info" data-action="view">
          <i class="bi bi-eye"></i>
        </button>
      </td>
    `;
    tbody.appendChild(tr);
  });
}

function renderAllTasksTable(tasks) {
  const tbody = $("#allTasksTable tbody");
  if (!tbody) return;
  
  tbody.innerHTML = "";
  
  if (tasks.length === 0) {
    $("#allTasksEmpty").classList.remove("d-none");
    return;
  }
  $("#allTasksEmpty").classList.add("d-none");
  
  tasks.forEach((t) => {
    const tr = document.createElement("tr");
    tr.dataset.taskId = t.id;
    tr.innerHTML = `
      <td>${escapeHtml(t.id.substring(0, 8))}</td>
      <td>${escapeHtml(t.agent_id || 'N/A')}</td>
      <td><span class="status-badge ${t.status}">${t.status}</span></td>
      <td>${t.priority || 0}</td>
      <td>${new Date(t.created_at).toLocaleString()}</td>
      <td>
        <button class="btn btn-sm btn-info me-1" data-action="view">
          <i class="bi bi-eye"></i>
        </button>
        <button class="btn btn-sm btn-danger" data-action="cancel" ${t.status === 'completed' ? 'disabled' : ''}>
          <i class="bi bi-x-circle"></i>
        </button>
      </td>
    `;
    tbody.appendChild(tr);
  });
}

/** Update counts for the chart (online/idle/busy/offline) */
function computeAgentStatusCounts(agents) {
  const counts = { online: 0, idle: 0, busy: 0, offline: 0 };
  agents.forEach((a) => {
    counts[a.status] = (counts[a.status] || 0) + 1;
  });
  return counts;
}

/** Handles a single SSE event for agents */
function handleAgentEvent({ type, agent, timestamp }) {
  // pull current data from DOM → easier to keep source-of-truth in memory
  window.__agents = window.__agents || [];

  switch (type) {
    case "agent_registered":
    case "agent_created":
      window.__agents.push(agent);
      addLiveEvent(`Agent ${agent.name} registered`, "success");
      break;
    case "agent_updated":
    case "agent_heartbeat":
      window.__agents = window.__agents.map((a) => (a.id === agent.id ? agent : a));
      addLiveEvent(`Agent ${agent.name} updated`, "info");
      break;
    case "agent_deleted":
    case "agent_unregistered":
      window.__agents = window.__agents.filter((a) => a.id !== agent.id);
      addLiveEvent(`Agent ${agent.name} unregistered`, "warning");
      break;
    default:
      console.warn("Unknown SSE type", type);
      addLiveEvent(`Unknown event: ${type}`, "secondary");
  }

  renderAgentsTable(window.__agents);
  updateAgentChart(computeAgentStatusCounts(window.__agents));
}

/* -----------------------------------------------------------------
   8️⃣  TOAST NOTIFICATIONS (Bootstrap 5)
   ----------------------------------------------------------------- */
function initToastContainer() {
  if (!$("#toastContainer")) {
    const div = document.createElement("div");
    div.id = "toastContainer";
    div.className = "toast-container position-fixed bottom-0 end-0 p-3";
    div.setAttribute("aria-live", "polite");
    div.setAttribute("aria-atomic", "true");
    document.body.appendChild(div);
  }
}

function showToast(message, type = "info", delay = 5000) {
  initToastContainer();
  const container = $("#toastContainer");

  const toastEl = document.createElement("div");
  toastEl.className = `toast align-items-center text-bg-${type} border-0`;
  toastEl.role = "alert";
  toastEl.ariaLive = "assertive";
  toastEl.ariaAtomic = "true";

  toastEl.innerHTML = `
    <div class="d-flex">
      <div class="toast-body flex-grow-1">${escapeHtml(message)}</div>
      <button type="button" class="btn-close btn-close-white me-2 m-auto"
              data-bs-dismiss="toast" aria-label="Close"></button>
    </div>
  `;

  container.appendChild(toastEl);
  const bsToast = new bootstrap.Toast(toastEl, { delay });
  bsToast.show();

  toastEl.addEventListener("hidden.bs.toast", () => toastEl.remove());
}

/* -----------------------------------------------------------------
   9️⃣  ERROR HANDLING & UI HELPERS
   ----------------------------------------------------------------- */
function showError(element, message) {
  if (!element) return;
  element.textContent = message;
  element.classList.remove("d-none");
}

function showSuccess(element, message) {
  if (!element) return;
  element.textContent = message;
  element.classList.remove("d-none");
}

function showLoading(element, loading) {
  if (!element) return;
  if (loading) {
    element.innerHTML = '<div class="spinner-border spinner-border-sm me-2"></div>Loading...';
    element.classList.remove("d-none");
  } else {
    element.classList.add("d-none");
  }
}

window.addEventListener("unhandledrejection", (e) => {
  console.error("Unhandled rejection:", e.reason);
  showToast(`Unexpected error: ${e.reason?.message || e.reason}`, "danger");
});

/* -----------------------------------------------------------------
   🔟  NAVIGATION & SECTION MANAGEMENT
   ----------------------------------------------------------------- */
function showSection(sectionId) {
  // Hide all sections
  $$("section[id$='Section']").forEach(section => {
    section.classList.add("d-none");
  });
  
  // Show target section
  const targetSection = $(`#${sectionId}`);
  if (targetSection) {
    targetSection.classList.remove("d-none");
  }
  
  // Update navbar
  $$(".nav-link").forEach(link => link.classList.remove("active"));
  const activeNavLink = $(`.nav-link[data-section="${sectionId.replace('Section', '')}"]`);
  if (activeNavLink) {
    activeNavLink.classList.add("active");
  }
  
  // Initialize section-specific functionality
  if (sectionId === 'metricsSection') {
    onMetricsSectionShow();
  } else if (sectionId === 'tasksSection' && typeof onTasksSectionShow === 'function') {
    onTasksSectionShow();
  } else if (sectionId === 'configurationSection') {
    onConfigurationSectionShow();
  }
  
  // Update username display
  const usernameDisplay = $("#usernameDisplay");
  if (usernameDisplay && tokenStore.raw()) {
    try {
      const payload = JSON.parse(atob(tokenStore.raw().split(".")[1]));
      usernameDisplay.textContent = payload.sub || "User";
    } catch (e) {
      usernameDisplay.textContent = "User";
    }
  }
}

/* -----------------------------------------------------------------
   1️⃣1️⃣  LIVE EVENTS DISPLAY
   ----------------------------------------------------------------- */
function addLiveEvent(message, type = "info") {
  const eventsList = $("#eventsList");
  if (!eventsList) return;
  
  // Remove "waiting" message if present
  const waitingMsg = eventsList.querySelector(".text-muted");
  if (waitingMsg && waitingMsg.textContent.includes("Waiting")) {
    waitingMsg.remove();
  }
  
  const li = document.createElement("li");
  li.className = `list-group-item d-flex justify-content-between align-items-center`;
  
  const iconClass = {
    success: "bi-check-circle text-success",
    info: "bi-info-circle text-info", 
    warning: "bi-exclamation-triangle text-warning",
    danger: "bi-exclamation-circle text-danger",
    secondary: "bi-clock text-secondary"
  }[type] || "bi-info-circle text-info";
  
  li.innerHTML = `
    <span><i class="bi ${iconClass} me-2"></i>${escapeHtml(message)}</span>
    <small class="text-muted">${new Date().toLocaleTimeString()}</small>
  `;
  
  eventsList.insertBefore(li, eventsList.firstChild);
  
  // Keep only last 20 events
  while (eventsList.children.length > 20) {
    eventsList.removeChild(eventsList.lastChild);
  }
}

/* -----------------------------------------------------------------
   1️⃣2️⃣  ACCESSIBILITY HELPERS
   ----------------------------------------------------------------- */
function escapeHtml(str) {
  if (str == null) return '';
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

/* -----------------------------------------------------------------
   📦  Dashboard bootstrap – load data, attach listeners, etc.
   ----------------------------------------------------------------- */
async function loadDashboardData() {
  if (!tokenStore.raw()) {
    showSection('loginSection');
    return;
  }

  try {
    showSection('dashboardSection');
    
    // Show loading states
    const systemHealthBody = $("#systemHealthBody");
    if (systemHealthBody) {
      systemHealthBody.innerHTML = `
        <div class="text-center">
          <div class="spinner-border spinner-border-sm text-primary me-2"></div>
          Loading system health...
        </div>
      `;
    }
    
    const [agents, tasks, health] = await Promise.all([
      apiRequest("/agents").catch(() => []),
      apiRequest("/tasks").catch(() => []),
      apiRequest("/system/health").catch(() => ({ status: "unknown" })),
    ]);

    window.__agents = agents;
    renderAgentsTable(agents);
    renderTasksTable(tasks);
    renderAllTasksTable(tasks);
    updateAgentChart(computeAgentStatusCounts(agents));

    // Update system health display
    if (systemHealthBody) {
      const statusClass = health.status === "healthy" ? "text-success" : "text-danger";
      systemHealthBody.innerHTML = `
        <div class="text-center">
          <h3 class="${statusClass}">
            <i class="bi bi-${health.status === "healthy" ? "check-circle" : "exclamation-triangle"}"></i>
          </h3>
          <p class="mb-0">${health.status}</p>
          ${health.uptime ? `<small class="text-muted">Uptime: ${Math.round(health.uptime / 60)} min</small>` : ''}
        </div>
      `;
    }
    
    // Populate agent select in task form
    const agentSelect = $("#taskAgentSelect");
    if (agentSelect) {
      agentSelect.innerHTML = '<option value="">Select an agent...</option>';
      agents.filter(a => a.status === 'online').forEach(agent => {
        const option = document.createElement("option");
        option.value = agent.id;
        option.textContent = `${agent.name} (${agent.status})`;
        agentSelect.appendChild(option);
      });
    }

    // Start SSE for live updates
    startAgentsSSE();
    
    addLiveEvent("Dashboard loaded successfully", "success");
    
  } catch (err) {
    console.error("Failed to load initial data:", err);
    showToast("Failed to load dashboard data", "danger");
  }
}

async function refreshData() {
  try {
    const [agents, tasks] = await Promise.all([
      apiRequest("/agents").catch(() => []),
      apiRequest("/tasks").catch(() => []),
    ]);
    
    window.__agents = agents;
    renderAgentsTable(agents);
    renderTasksTable(tasks);
    renderAllTasksTable(tasks);
    updateAgentChart(computeAgentStatusCounts(agents));
    
    showToast("Data refreshed", "success");
  } catch (err) {
    showToast("Failed to refresh data", "danger");
  }
}

/* -----------------------------------------------------------------
   INITIALIZE EVERYTHING ON DOMContentLoaded
   ----------------------------------------------------------------- */
document.addEventListener("DOMContentLoaded", () => {
  initThemeToggle();
  initLoginForm();
  initAgentForm();
  initTaskForm();
  initAgentChart();

  // Navigation handlers
  $$(".nav-link[data-section]").forEach(link => {
    link.addEventListener("click", (e) => {
      e.preventDefault();
      const section = e.target.dataset.section;
      showSection(section + 'Section');
    });
  });

  // Logout button
  $("#logoutBtn")?.addEventListener("click", (e) => {
    e.preventDefault();
    logout();
  });

  // Refresh buttons
  $("#refreshAgentsBtn")?.addEventListener("click", refreshData);
  $("#refreshTasksBtn")?.addEventListener("click", refreshData);

  // Modal triggers
  $("#openAddAgentModal")?.addEventListener("click", () => {
    const modal = new bootstrap.Modal($("#agentModal"));
    modal.show();
  });

  // Table action handlers
  $("#agentsTable")?.addEventListener("click", async (e) => {
    const btn = e.target.closest("button[data-action]");
    if (!btn) return;
    
    const action = btn.dataset.action;
    const tr = btn.closest("tr");
    const agentId = tr.dataset.agentId;

    if (action === "delete") {
      if (!confirm("Are you sure you want to delete this agent?")) return;
      try {
        await apiRequest(`/agents/${agentId}`, { method: "DELETE" });
        showToast("Agent deleted successfully", "success");
      } catch (err) {
        // Error already shown by apiRequest
      }
    } else if (action === "edit") {
      const agent = window.__agents.find((a) => a.id === agentId);
      if (!agent) return;
      
      // Populate edit form (reuse the same modal)
      $("#agentNameInput").value = agent.name;
      $("#agentStatusSelect").value = agent.status;
      $("#agentIdInput").value = agent.id;
      $("#agentModalLabel").textContent = "Edit Agent";
      
      const modal = new bootstrap.Modal($("#agentModal"));
      modal.show();
    }
  });

  // Check authentication and load dashboard
  if (tokenStore.raw()) {
    loadDashboardData();
  } else {
    showSection('loginSection');
  }
});

/* --------------------------------------------------------------
   WUI3: Agent Status Monitoring Interface
   -------------------------------------------------------------- */

// Global monitoring state
const monitoringState = {
  agents: [],
  filters: {
    search: '',
    status: '',
    capabilities: new Set(),
  },
  alerts: [],
  detailChart: null,
};

// Utility functions
function formatBytes(bytes) {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

function formatMs(ms) {
  return `${(ms/1000).toFixed(2)}s`;
}

// Render agent cards with filtering
function renderAgentCards() {
  const container = $("#agent-cards-container");
  if (!container) return;
  
  container.innerHTML = '';

  // Apply filters
  const filtered = monitoringState.agents.filter(agent => {
    const matchesSearch = monitoringState.filters.search === '' ||
      agent.name.toLowerCase().includes(monitoringState.filters.search.toLowerCase()) ||
      agent.id.toString().includes(monitoringState.filters.search);
    const matchesStatus = monitoringState.filters.status === '' ||
      agent.status === monitoringState.filters.status;
    const matchesCap = monitoringState.filters.capabilities.size === 0 ||
      (agent.capabilities && agent.capabilities.some(c => monitoringState.filters.capabilities.has(c)));
    return matchesSearch && matchesStatus && matchesCap;
  });

  if (filtered.length === 0) {
    container.innerHTML = `
      <div class="col-12 text-center py-5">
        <p class="text-muted">No agents match the current filter criteria.</p>
      </div>`;
    return;
  }

  filtered.forEach(agent => {
    const card = document.createElement('div');
    card.className = 'col';
    
    // Generate mock metrics if not available
    const metrics = agent.metrics || {
      cpu: Math.random() * 100,
      memory: Math.random() * 1024 * 1024 * 1024,
      tasks_completed: Math.floor(Math.random() * 100),
    };
    
    card.innerHTML = `
      <div class="card h-100 agent-card" tabindex="0" role="button"
           data-agent-id="${agent.id}"
           aria-label="Agent ${agent.name} – ${agent.status}"
           aria-describedby="agent-${agent.id}-status">
        <div class="card-body">
          <h5 class="card-title text-truncate mb-0">${escapeHtml(agent.name)}</h5>
          <h6 class="card-subtitle mb-2 text-muted">#${agent.id}</h6>

          <p class="mt-2 mb-0">
            <span class="status-indicator ${agent.status}"></span>
            <span id="agent-${agent.id}-status"
                  class="badge badge-status-${agent.status}"
                  aria-live="polite">
              ${agent.status.toUpperCase()}
            </span>
          </p>

          <div class="metrics-summary mt-auto">
            <span title="CPU Usage"><i class="bi bi-cpu"></i> ${metrics.cpu.toFixed(1)}%</span>
            <span title="Memory Usage"><i class="bi bi-memory"></i> ${formatBytes(metrics.memory)}</span>
            <span title="Tasks Completed"><i class="bi bi-check2-square"></i> ${metrics.tasks_completed}</span>
          </div>
        </div>
      </div>`;
    
    const cardEl = card.firstElementChild;
    cardEl.addEventListener('click', () => openAgentDetail(agent.id));
    cardEl.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        openAgentDetail(agent.id);
      }
    });
    
    container.appendChild(card);
  });
}

// Render capability filter options
function renderCapabilityFilterOptions() {
  const select = $("#capability-filter");
  if (!select) return;
  
  const allCaps = new Set();
  monitoringState.agents.forEach(a => {
    if (a.capabilities) {
      a.capabilities.forEach(c => allCaps.add(c));
    }
  });
  
  const sorted = Array.from(allCaps).sort();
  select.innerHTML = sorted.map(cap => `<option value="${cap}">${cap}</option>`).join('');

  // Preserve previously selected capabilities
  Array.from(select.options).forEach(opt => {
    if (monitoringState.filters.capabilities.has(opt.value)) {
      opt.selected = true;
    }
  });
}

// Render alerts
function renderAlerts() {
  const list = $("#alert-list");
  if (!list) return;
  
  list.innerHTML = '';
  
  if (monitoringState.alerts.length === 0) {
    list.innerHTML = `
      <div class="list-group-item text-muted">
        <i class="bi bi-info-circle me-2"></i>No alerts at this time.
      </div>`;
  } else {
    monitoringState.alerts.forEach(alert => {
      const li = document.createElement('div');
      li.className = `list-group-item alert-${alert.severity}`;
      li.setAttribute('role', 'alert');
      li.innerHTML = `
        <i class="bi ${alert.icon} text-${alert.severity}"></i>
        <div class="flex-grow-1">
          <strong>${escapeHtml(alert.title)}</strong>
          <small class="text-muted d-block">${alert.time}</small>
        </div>`;
      list.appendChild(li);
    });
  }
  
  // Update alert count
  const alertCount = $("#alert-count");
  if (alertCount) {
    alertCount.textContent = monitoringState.alerts.length;
  }
}

// Open agent detail modal
async function openAgentDetail(agentId) {
  const agent = monitoringState.agents.find(a => a.id === agentId);
  if (!agent) return;

  // Update modal header
  $("#modal-agent-id").textContent = agent.id;
  $("#modal-agent-name").textContent = agent.name;
  $("#modal-connectivity-status").textContent = agent.status.toUpperCase();
  $("#modal-connectivity-status").className = `badge bg-${getStatusColor(agent.status)}`;
  
  // Update heartbeat
  const heartbeatEl = $("#modal-heartbeat-ts");
  if (heartbeatEl) {
    heartbeatEl.textContent = agent.last_heartbeat ? 
      new Date(agent.last_heartbeat).toLocaleString() : 
      'Never';
  }

  // Load agent details
  try {
    await loadAgentMetrics(agentId);
    await loadAgentCapabilities(agentId);
    await loadAgentHistory(agentId);
  } catch (err) {
    console.error('Failed to load agent details:', err);
  }

  // Show modal
  const modal = new bootstrap.Modal($("#agentDetailModal"));
  modal.show();
}

// Load agent metrics and render chart
async function loadAgentMetrics(agentId) {
  const loadList = $("#modal-load-list");
  if (loadList) {
    loadList.innerHTML = '<li>Loading metrics...</li>';
  }

  try {
    // Generate mock metrics data for demonstration
    const mockMetrics = generateMockMetrics();
    
    // Update load list
    if (loadList) {
      const latest = mockMetrics.data[mockMetrics.data.length - 1];
      loadList.innerHTML = `
        <li><strong>CPU:</strong> ${latest.cpu.toFixed(1)}%</li>
        <li><strong>Memory:</strong> ${formatBytes(latest.memory)}</li>
        <li><strong>Tasks:</strong> ${latest.tasks}</li>
        <li><strong>Latency:</strong> ${formatMs(latest.latency)}</li>
      `;
    }

    // Render chart
    renderMetricsChart(mockMetrics);
    
  } catch (err) {
    console.error('Failed to load metrics:', err);
    if (loadList) {
      loadList.innerHTML = '<li class="text-danger">Failed to load metrics.</li>';
    }
  }
}

// Generate mock metrics for demonstration
function generateMockMetrics() {
  const now = new Date();
  const data = [];
  const labels = [];
  
  for (let i = 29; i >= 0; i--) {
    const time = new Date(now.getTime() - i * 60000); // 1 minute intervals
    labels.push(time.toLocaleTimeString());
    data.push({
      cpu: Math.random() * 100,
      memory: Math.random() * 1024 * 1024 * 1024,
      tasks: Math.floor(Math.random() * 20),
      latency: Math.random() * 1000,
    });
  }
  
  return { labels, data };
}

// Render metrics chart using Chart.js
function renderMetricsChart(metrics) {
  const ctx = $("#modal-metrics-chart");
  if (!ctx) return;

  // Destroy existing chart
  if (monitoringState.detailChart) {
    monitoringState.detailChart.destroy();
  }

  monitoringState.detailChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: metrics.labels,
      datasets: [
        {
          label: 'CPU %',
          data: metrics.data.map(d => d.cpu),
          borderColor: '#28a745',
          backgroundColor: 'rgba(40, 167, 69, 0.1)',
          tension: 0.3,
          fill: false,
          yAxisID: 'y',
        },
        {
          label: 'Memory (MB)',
          data: metrics.data.map(d => d.memory / (1024 * 1024)),
          borderColor: '#17a2b8',
          backgroundColor: 'rgba(23, 162, 184, 0.1)',
          tension: 0.3,
          fill: false,
          yAxisID: 'y1',
        },
        {
          label: 'Latency (ms)',
          data: metrics.data.map(d => d.latency),
          borderColor: '#fd7e14',
          backgroundColor: 'rgba(253, 126, 20, 0.1)',
          tension: 0.3,
          fill: false,
          yAxisID: 'y2',
        },
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      scales: {
        y: {
          type: 'linear',
          position: 'left',
          title: { display: true, text: 'CPU %' },
          min: 0,
          max: 100,
        },
        y1: {
          type: 'linear',
          position: 'right',
          title: { display: true, text: 'Memory (MB)' },
          grid: { drawOnChartArea: false },
        },
        y2: {
          type: 'linear',
          position: 'right',
          title: { display: true, text: 'Latency (ms)' },
          grid: { drawOnChartArea: false },
          offset: true,
        },
      },
      plugins: {
        legend: { position: 'bottom' },
        tooltip: {
          callbacks: {
            label: (context) => {
              const label = context.dataset.label;
              const value = context.parsed.y;
              if (label === 'Memory (MB)') return `${label}: ${value.toFixed(1)} MB`;
              if (label === 'Latency (ms)') return `${label}: ${value.toFixed(0)} ms`;
              return `${label}: ${value.toFixed(1)}%`;
            }
          }
        }
      }
    }
  });
}

// Load agent capabilities
async function loadAgentCapabilities(agentId) {
  const list = $("#modal-capability-list");
  if (!list) return;
  
  list.innerHTML = '<li class="list-group-item">Loading capabilities...</li>';
  
  try {
    const agent = monitoringState.agents.find(a => a.id === agentId);
    const capabilities = agent?.capabilities || ['data-processing', 'file-operations', 'api-calls'];
    
    if (capabilities.length > 0) {
      list.innerHTML = capabilities.map(cap => `
        <li class="list-group-item">
          <span class="capability-name">${escapeHtml(cap)}</span>
          <i class="bi bi-check-circle-fill text-success"></i>
        </li>
      `).join('');
    } else {
      list.innerHTML = '<li class="list-group-item text-muted">No capabilities reported.</li>';
    }
  } catch (err) {
    console.error('Failed to load capabilities:', err);
    list.innerHTML = '<li class="list-group-item text-danger">Failed to load capabilities.</li>';
  }
}

// Load agent history
async function loadAgentHistory(agentId) {
  const list = $("#modal-history-list");
  if (!list) return;
  
  list.innerHTML = '<li class="list-group-item">Loading history...</li>';
  
  try {
    // Generate mock history data
    const history = generateMockHistory();
    
    if (history.length > 0) {
      list.innerHTML = history.map(event => {
        const time = new Date(event.timestamp).toLocaleString();
        const badgeClass = getStatusBadgeClass(event.status);
        return `
          <li class="list-group-item">
            <div>
              <span class="badge ${badgeClass} me-2">${event.status.toUpperCase()}</span>
              ${escapeHtml(event.message)}
            </div>
            <time class="event-time">${time}</time>
          </li>
        `;
      }).join('');
    } else {
      list.innerHTML = '<li class="list-group-item text-muted">No history available.</li>';
    }
  } catch (err) {
    console.error('Failed to load history:', err);
    list.innerHTML = '<li class="list-group-item text-danger">Failed to load history.</li>';
  }
}

// Generate mock history data
function generateMockHistory() {
  const events = [];
  const now = new Date();
  const statuses = ['online', 'idle', 'busy', 'offline'];
  const messages = [
    'Agent started successfully',
    'Processing task queue',
    'Completed batch operation',
    'Entering idle state',
    'Connection established',
    'Health check passed'
  ];
  
  for (let i = 0; i < 10; i++) {
    events.push({
      timestamp: new Date(now.getTime() - i * 3600000).toISOString(), // hourly intervals
      status: statuses[Math.floor(Math.random() * statuses.length)],
      message: messages[Math.floor(Math.random() * messages.length)]
    });
  }
  
  return events.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
}

// Helper functions
function getStatusColor(status) {
  const colors = {
    online: 'success',
    idle: 'warning',
    busy: 'info',
    offline: 'secondary',
    error: 'danger'
  };
  return colors[status] || 'secondary';
}

function getStatusBadgeClass(status) {
  const classes = {
    online: 'bg-success',
    idle: 'bg-warning',
    busy: 'bg-info',
    offline: 'bg-secondary',
    error: 'bg-danger'
  };
  return classes[status] || 'bg-secondary';
}

// Alert system
function addAlert(title, severity = 'info', icon = 'bi-info-circle') {
  const alert = {
    id: Date.now().toString(),
    title,
    severity,
    icon,
    time: new Date().toLocaleTimeString()
  };
  
  monitoringState.alerts.unshift(alert);
  
  // Keep only last 20 alerts
  if (monitoringState.alerts.length > 20) {
    monitoringState.alerts = monitoringState.alerts.slice(0, 20);
  }
  
  renderAlerts();
}

// Fetch agents for monitoring
async function fetchAgentsForMonitoring() {
  try {
    const agents = await apiRequest("/agents");
    
    // Add mock data for demonstration
    const enhancedAgents = agents.map(agent => ({
      ...agent,
      capabilities: agent.capabilities || ['data-processing', 'file-operations'],
      metrics: {
        cpu: Math.random() * 100,
        memory: Math.random() * 1024 * 1024 * 1024,
        tasks_completed: Math.floor(Math.random() * 100),
        latency: Math.random() * 1000,
      }
    }));
    
    monitoringState.agents = enhancedAgents;
    renderCapabilityFilterOptions();
    renderAgentCards();
    renderAlerts();
    
  } catch (err) {
    console.error('Failed to fetch agents for monitoring:', err);
    const container = $("#agent-cards-container");
    if (container) {
      container.innerHTML = `
        <div class="col-12 text-center py-5">
          <p class="text-danger">Unable to load agents. Please try again later.</p>
          <button class="btn btn-primary" onclick="fetchAgentsForMonitoring()">Retry</button>
        </div>`;
    }
  }
}

// Initialize monitoring interface
function initMonitoring() {
  // Filter event handlers
  const searchInput = $("#agent-search");
  if (searchInput) {
    searchInput.addEventListener('input', (e) => {
      monitoringState.filters.search = e.target.value.trim();
      renderAgentCards();
    });
  }

  const statusFilter = $("#status-filter");
  if (statusFilter) {
    statusFilter.addEventListener('change', (e) => {
      monitoringState.filters.status = e.target.value;
      renderAgentCards();
    });
  }

  const capabilityFilter = $("#capability-filter");
  if (capabilityFilter) {
    capabilityFilter.addEventListener('change', (e) => {
      const selected = Array.from(e.target.selectedOptions).map(o => o.value);
      monitoringState.filters.capabilities = new Set(selected);
      renderAgentCards();
    });
  }

  // Load initial data
  fetchAgentsForMonitoring();
  
  // Set up periodic refresh
  setInterval(fetchAgentsForMonitoring, 30000); // 30 seconds
  
  // Add some demo alerts
  setTimeout(() => {
    addAlert('Agent performance normal', 'success', 'bi-check-circle-fill');
  }, 2000);
}

// Extend the main dashboard initialization
document.addEventListener('DOMContentLoaded', () => {
  // Wait for the main dashboard to initialize, then add monitoring
  setTimeout(() => {
    initMonitoring();
  }, 1000);
});

/* ==============================================================
   TASK EXECUTION CONTROLS & LOGS (WUI4)
   ============================================================== */

// Task management state
const taskState = {
  tasks: [],
  selectedIds: new Set(),
  currentPage: 1,
  pageSize: 20,
  totalPages: 1,
  searchQuery: '',
  logBuffer: new Map(), // taskId -> log content
  isLogPaused: false,
  logSSE: null,
  metricsChart: null
};

// Task execution controls helpers
const taskHelpers = {
  showError(msg) {
    const alert = document.createElement('div');
    alert.className = 'alert alert-danger alert-dismissible fade show';
    alert.setAttribute('role', 'alert');
    alert.innerHTML = `${msg}
      <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>`;
    const container = $('#tasksSection .container-fluid');
    if (container) container.insertBefore(alert, container.firstChild);
  },

  showSuccess(msg) {
    const alert = document.createElement('div');
    alert.className = 'alert alert-success alert-dismissible fade show';
    alert.setAttribute('role', 'alert');
    alert.innerHTML = `${msg}
      <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>`;
    const container = $('#tasksSection .container-fluid');
    if (container) container.insertBefore(alert, container.firstChild);
  },

  getJwt() {
    const token = tokenStore.get();
    return token ? token.raw : '';
  },

  async apiFetch(url, opts = {}) {
    const token = this.getJwt();
    const headers = opts.headers || {};
    headers.Authorization = `Bearer ${token}`;
    headers['Content-Type'] = 'application/json';

    try {
      const response = await fetch(API_BASE + url, { ...opts, headers });
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText || response.statusText);
      }
      return await response.json();
    } catch (err) {
      console.error('API fetch error:', err);
      throw err;
    }
  }
};

// Task rendering functions
function renderTaskRow(task) {
  const tr = document.createElement('tr');

  // Checkbox column
  const tdSelect = document.createElement('td');
  const checkbox = document.createElement('input');
  checkbox.type = 'checkbox';
  checkbox.className = 'form-check-input';
  checkbox.setAttribute('aria-label', `Select task ${task.id}`);
  checkbox.dataset.id = task.id;
  checkbox.checked = taskState.selectedIds.has(task.id);
  checkbox.addEventListener('change', handleTaskSelection);
  tdSelect.appendChild(checkbox);
  tr.appendChild(tdSelect);

  // Task ID
  const tdId = document.createElement('td');
  tdId.textContent = task.id;
  tr.appendChild(tdId);

  // Task Name
  const tdName = document.createElement('td');
  tdName.textContent = task.name || `Task ${task.id}`;
  tr.appendChild(tdName);

  // Status badge
  const tdStatus = document.createElement('td');
  const statusBadge = document.createElement('span');
  statusBadge.className = `task-status ${task.status.toLowerCase()}`;
  statusBadge.textContent = task.status;
  statusBadge.setAttribute('role', 'status');
  tdStatus.appendChild(statusBadge);
  tr.appendChild(tdStatus);

  // Priority
  const tdPriority = document.createElement('td');
  tdPriority.textContent = task.priority || 'Normal';
  tr.appendChild(tdPriority);

  // Queue Position
  const tdQueue = document.createElement('td');
  tdQueue.textContent = task.queuePosition || '-';
  tr.appendChild(tdQueue);

  // Started At
  const tdStarted = document.createElement('td');
  tdStarted.textContent = task.startedAt || '-';
  tr.appendChild(tdStarted);

  // Duration
  const tdDuration = document.createElement('td');
  tdDuration.textContent = task.duration || '-';
  tr.appendChild(tdDuration);

  // Actions
  const tdActions = document.createElement('td');
  tdActions.className = 'text-end';
  
  const actions = [
    { icon: 'bi-play-fill', action: 'start', disabled: task.status !== 'Pending' },
    { icon: 'bi-pause-fill', action: 'pause', disabled: task.status !== 'Running' },
    { icon: 'bi-play-circle', action: 'resume', disabled: task.status !== 'Paused' },
    { icon: 'bi-stop-fill', action: 'stop', disabled: ['Failed', 'Completed'].includes(task.status) },
    { icon: 'bi-trash', action: 'delete', disabled: false, danger: true }
  ];

  actions.forEach(({ icon, action, disabled, danger }) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = `btn btn-sm btn-link action-icon ${danger ? 'text-danger' : ''}`;
    btn.innerHTML = `<i class="bi ${icon}"></i>`;
    btn.dataset.action = action;
    btn.dataset.id = task.id;
    btn.title = `${action} task`;
    
    if (disabled) {
      btn.setAttribute('aria-disabled', 'true');
      btn.disabled = true;
    } else {
      btn.addEventListener('click', handleSingleTaskAction);
    }
    
    tdActions.appendChild(btn);
  });

  tr.appendChild(tdActions);
  return tr;
}

// Event handlers
function handleTaskSelection(e) {
  const taskId = e.target.dataset.id;
  if (e.target.checked) {
    taskState.selectedIds.add(taskId);
  } else {
    taskState.selectedIds.delete(taskId);
  }
  updateBulkActionButton();
  updateSelectAllCheckbox();
}

function handleSelectAll(e) {
  const checkboxes = $$('#tasks-table tbody input[type="checkbox"]');
  const isChecked = e.target.checked;
  
  checkboxes.forEach(cb => {
    cb.checked = isChecked;
    const taskId = cb.dataset.id;
    if (isChecked) {
      taskState.selectedIds.add(taskId);
    } else {
      taskState.selectedIds.delete(taskId);
    }
  });
  
  updateBulkActionButton();
}

function updateBulkActionButton() {
  const bulkBtn = $('#bulk-action-btn');
  if (bulkBtn) {
    bulkBtn.disabled = taskState.selectedIds.size === 0;
  }
}

function updateSelectAllCheckbox() {
  const selectAll = $('#select-all');
  const checkboxes = $$('#tasks-table tbody input[type="checkbox"]');
  if (selectAll && checkboxes.length > 0) {
    selectAll.checked = checkboxes.length === taskState.selectedIds.size;
  }
}

async function handleSingleTaskAction(e) {
  const action = e.target.dataset.action;
  const taskId = e.target.dataset.id;
  
  try {
    await taskHelpers.apiFetch(`/api/tasks/${action}`, {
      method: 'POST',
      body: JSON.stringify({ ids: [taskId] })
    });
    
    taskHelpers.showSuccess(`Task ${taskId} ${action} successful`);
    await fetchTasks();
  } catch (err) {
    taskHelpers.showError(`Failed to ${action} task: ${err.message}`);
  }
}

async function handleBulkAction(action) {
  if (taskState.selectedIds.size === 0) return;
  
  try {
    await taskHelpers.apiFetch(`/api/tasks/${action}`, {
      method: 'POST',
      body: JSON.stringify({ ids: Array.from(taskState.selectedIds) })
    });
    
    taskHelpers.showSuccess(`Bulk ${action} successful for ${taskState.selectedIds.size} tasks`);
    taskState.selectedIds.clear();
    updateBulkActionButton();
    updateSelectAllCheckbox();
    await fetchTasks();
  } catch (err) {
    taskHelpers.showError(`Bulk ${action} failed: ${err.message}`);
  }
}

// Task data fetching
async function fetchTasks() {
  const offset = (taskState.currentPage - 1) * taskState.pageSize;
  const params = new URLSearchParams({
    offset: offset.toString(),
    limit: taskState.pageSize.toString(),
    search: taskState.searchQuery
  });

  try {
    showTasksLoading(true);
    const data = await taskHelpers.apiFetch(`/api/tasks?${params.toString()}`);
    
    taskState.tasks = data.tasks || [];
    taskState.totalPages = data.totalPages || 1;
    
    renderTasksTable();
    renderTasksPagination();
    
  } catch (err) {
    taskHelpers.showError(`Failed to fetch tasks: ${err.message}`);
    // Show empty state on error
    const tbody = $('#tasks-table tbody');
    if (tbody) {
      tbody.innerHTML = '<tr><td colspan="9" class="text-center text-muted">Failed to load tasks</td></tr>';
    }
  } finally {
    showTasksLoading(false);
  }
}

function renderTasksTable() {
  const tbody = $('#tasks-table tbody');
  if (!tbody) return;

  tbody.innerHTML = '';
  taskState.selectedIds.clear();
  updateBulkActionButton();
  updateSelectAllCheckbox();

  if (taskState.tasks.length === 0) {
    tbody.innerHTML = '<tr><td colspan="9" class="text-center text-muted">No tasks found</td></tr>';
    return;
  }

  taskState.tasks.forEach(task => {
    tbody.appendChild(renderTaskRow(task));
  });
}

function renderTasksPagination() {
  const pagination = $('#task-pagination');
  if (!pagination) return;

  pagination.innerHTML = '';

  if (taskState.totalPages <= 1) return;

  // Previous button
  const prevItem = document.createElement('li');
  prevItem.className = `page-item ${taskState.currentPage <= 1 ? 'disabled' : ''}`;
  const prevLink = document.createElement('a');
  prevLink.className = 'page-link';
  prevLink.href = '#';
  prevLink.textContent = '‹';
  prevLink.addEventListener('click', (e) => {
    e.preventDefault();
    if (taskState.currentPage > 1) {
      taskState.currentPage--;
      fetchTasks();
    }
  });
  prevItem.appendChild(prevLink);
  pagination.appendChild(prevItem);

  // Page numbers
  const start = Math.max(1, taskState.currentPage - 2);
  const end = Math.min(taskState.totalPages, taskState.currentPage + 2);
  
  for (let i = start; i <= end; i++) {
    const pageItem = document.createElement('li');
    pageItem.className = `page-item ${i === taskState.currentPage ? 'active' : ''}`;
    const pageLink = document.createElement('a');
    pageLink.className = 'page-link';
    pageLink.href = '#';
    pageLink.textContent = i;
    pageLink.addEventListener('click', (e) => {
      e.preventDefault();
      taskState.currentPage = i;
      fetchTasks();
    });
    pageItem.appendChild(pageLink);
    pagination.appendChild(pageItem);
  }

  // Next button
  const nextItem = document.createElement('li');
  nextItem.className = `page-item ${taskState.currentPage >= taskState.totalPages ? 'disabled' : ''}`;
  const nextLink = document.createElement('a');
  nextLink.className = 'page-link';
  nextLink.href = '#';
  nextLink.textContent = '›';
  nextLink.addEventListener('click', (e) => {
    e.preventDefault();
    if (taskState.currentPage < taskState.totalPages) {
      taskState.currentPage++;
      fetchTasks();
    }
  });
  nextItem.appendChild(nextLink);
  pagination.appendChild(nextItem);
}

function showTasksLoading(show) {
  const tbody = $('#tasks-table tbody');
  if (!tbody) return;

  if (show) {
    tbody.innerHTML = '<tr><td colspan="9" class="text-center"><div class="spinner-border spinner-border-sm me-2"></div>Loading tasks...</td></tr>';
  }
}

// Task metrics and monitoring
async function initTaskMetrics() {
  const canvas = $('#metrics-chart');
  if (!canvas) return;

  const ctx = canvas.getContext('2d');
  taskState.metricsChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [{
        label: 'Queue Length',
        data: [],
        borderColor: 'rgba(13, 110, 253, 1)',
        backgroundColor: 'rgba(13, 110, 253, 0.1)',
        tension: 0.4,
        fill: true
      }, {
        label: 'Running Tasks',
        data: [],
        borderColor: 'rgba(25, 135, 84, 1)',
        backgroundColor: 'rgba(25, 135, 84, 0.1)',
        tension: 0.4,
        fill: true
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          title: {
            display: true,
            text: 'Time'
          }
        },
        y: {
          beginAtZero: true,
          title: {
            display: true,
            text: 'Count'
          }
        }
      },
      plugins: {
        legend: {
          display: true,
          position: 'top'
        }
      }
    }
  });

  // Start metrics collection
  setInterval(fetchTaskMetrics, 5000); // every 5 seconds
}

async function fetchTaskMetrics() {
  try {
    const data = await taskHelpers.apiFetch('/api/tasks/metrics');
    const now = new Date().toLocaleTimeString();
    
    if (taskState.metricsChart) {
      const chart = taskState.metricsChart;
      
      // Keep only last 20 data points
      if (chart.data.labels.length >= 20) {
        chart.data.labels.shift();
        chart.data.datasets[0].data.shift();
        chart.data.datasets[1].data.shift();
      }
      
      chart.data.labels.push(now);
      chart.data.datasets[0].data.push(data.queueLength || 0);
      chart.data.datasets[1].data.push(data.runningTasks || 0);
      
      chart.update('quiet');
    }
  } catch (err) {
    console.warn('Failed to fetch task metrics:', err);
  }
}

// Live log streaming
function initTaskLogSSE() {
  if (taskState.logSSE) return; // already connected

  try {
    taskState.logSSE = new EventSource(`${API_BASE}/api/tasks/logs/stream`);
    
    taskState.logSSE.addEventListener('message', (event) => {
      if (taskState.isLogPaused) return;
      
      try {
        const data = JSON.parse(event.data);
        const { taskId, timestamp, level, message } = data;
        
        // Update log buffer
        const logEntry = `[${timestamp}] [${level.toUpperCase()}] Task ${taskId}: ${message}\n`;
        const current = taskState.logBuffer.get(taskId) || '';
        taskState.logBuffer.set(taskId, current + logEntry);
        
        // Update live log viewer
        appendToLogViewer(logEntry, level);
        
        // Update task status if needed
        updateTaskRowFromLog(data);
        
      } catch (err) {
        console.warn('Failed to parse log message:', err);
      }
    });

    taskState.logSSE.addEventListener('error', () => {
      console.warn('Task log SSE connection lost, reconnecting...');
      taskState.logSSE = null;
      setTimeout(initTaskLogSSE, 5000);
    });

    console.log('Task log SSE connected');
  } catch (err) {
    console.error('Failed to initialize task log SSE:', err);
  }
}

function appendToLogViewer(message, level = 'info') {
  const logOutput = $('#log-output');
  if (!logOutput) return;

  const logEntry = document.createElement('span');
  logEntry.className = `log-entry ${level}`;
  logEntry.textContent = message;
  
  logOutput.appendChild(logEntry);
  
  // Auto-scroll to bottom
  const container = logOutput.closest('.log-container');
  if (container) {
    container.scrollTop = container.scrollHeight;
  }
  
  // Limit log entries to prevent memory issues
  const entries = logOutput.querySelectorAll('.log-entry');
  if (entries.length > 1000) {
    entries[0].remove();
  }
}

function updateTaskRowFromLog(logData) {
  if (!logData.status) return;
  
  const checkbox = $(`#tasks-table input[data-id="${logData.taskId}"]`);
  if (!checkbox) return;
  
  const row = checkbox.closest('tr');
  const statusBadge = row?.querySelector('.task-status');
  if (statusBadge) {
    statusBadge.textContent = logData.status;
    statusBadge.className = `task-status ${logData.status.toLowerCase()}`;
  }
}

// Quick task form
async function handleQuickTaskSubmit(e) {
  e.preventDefault();
  
  const form = e.target;
  const formData = new FormData(form);
  
  const taskData = {
    name: $('#quicktask-name').value.trim(),
    command: $('#quicktask-command').value.trim(),
    priority: parseInt($('#quicktask-priority').value),
    async: $('#quicktask-async').checked
  };
  
  try {
    await taskHelpers.apiFetch('/api/tasks/quick', {
      method: 'POST',
      body: JSON.stringify(taskData)
    });
    
    taskHelpers.showSuccess('Quick task created and started successfully');
    form.reset();
    
    // Close modal
    const modal = bootstrap.Modal.getInstance($('#quicktemplate-modal'));
    if (modal) modal.hide();
    
    // Refresh task list
    await fetchTasks();
    
  } catch (err) {
    taskHelpers.showError(`Failed to create quick task: ${err.message}`);
  }
}

// Task search
function handleTaskSearch(e) {
  e.preventDefault();
  taskState.searchQuery = $('#task-search').value.trim();
  taskState.currentPage = 1;
  fetchTasks();
}

// Log controls
function handleLogControls() {
  const clearBtn = $('#clear-logs-btn');
  const pauseBtn = $('#pause-logs-btn');
  const downloadBtn = $('#download-logs-btn');

  if (clearBtn) {
    clearBtn.addEventListener('click', () => {
      const logOutput = $('#log-output');
      if (logOutput) logOutput.innerHTML = '';
      taskState.logBuffer.clear();
    });
  }

  if (pauseBtn) {
    pauseBtn.addEventListener('click', () => {
      taskState.isLogPaused = !taskState.isLogPaused;
      pauseBtn.innerHTML = taskState.isLogPaused ? 
        '<i class="bi bi-play-fill me-1"></i>Resume' :
        '<i class="bi bi-pause-fill me-1"></i>Pause';
    });
  }

  if (downloadBtn) {
    downloadBtn.addEventListener('click', () => {
      const logOutput = $('#log-output');
      if (!logOutput) return;
      
      const content = logOutput.textContent;
      const blob = new Blob([content], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);
      
      const a = document.createElement('a');
      a.href = url;
      a.download = `task_logs_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.log`;
      a.click();
      
      URL.revokeObjectURL(url);
    });
  }
}

// Initialize task management
function initTaskManagement() {
  // Set up event listeners
  const selectAll = $('#select-all');
  if (selectAll) {
    selectAll.addEventListener('change', handleSelectAll);
  }

  // Bulk action dropdown
  const bulkActionItems = $$('#bulk-action-btn + .dropdown-menu .dropdown-item');
  bulkActionItems.forEach(item => {
    item.addEventListener('click', (e) => {
      e.preventDefault();
      const action = item.dataset.action;
      if (action) handleBulkAction(action);
    });
  });

  // Search form
  const searchForm = $('#task-search-form');
  if (searchForm) {
    searchForm.addEventListener('submit', handleTaskSearch);
  }

  // Quick task form
  const quickTaskForm = $('#quicktask-form');
  if (quickTaskForm) {
    quickTaskForm.addEventListener('submit', handleQuickTaskSubmit);
  }

  // Initialize components
  handleLogControls();
  initTaskMetrics();
  initTaskLogSSE();

  console.log('Task management initialized');
}

// Initialize when Tasks section is shown
function initTasksSection() {
  const tasksNav = $('[data-section="tasks"]');
  if (tasksNav) {
    tasksNav.addEventListener('click', async () => {
      // Initialize task management on first access
      if (!taskState.initialized) {
        initTaskManagement();
        taskState.initialized = true;
      }
      
      // Fetch initial task data
      await fetchTasks();
    });
  }
}

// Clean up resources when leaving tasks section
function cleanupTasksSection() {
  if (taskState.logSSE) {
    taskState.logSSE.close();
    taskState.logSSE = null;
  }
  
  if (taskState.metricsChart) {
    taskState.metricsChart.destroy();
    taskState.metricsChart = null;
  }
}

// Initialize task section handlers when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  initTasksSection();
  
  // Clean up when switching away from tasks
  $$('[data-section]').forEach(nav => {
    nav.addEventListener('click', (e) => {
      if (e.target.dataset.section !== 'tasks') {
        cleanupTasksSection();
      }
    });
  });
});

/*=================================================================
  System Metrics & Performance Charts (WUI5)
=================================================================*/

// Metrics management state
const metricsState = {
  charts: new Map(),
  sseConnection: null,
  currentRange: '1h',
  updateInterval: null,
  isConnected: false
};

// Utility functions for metrics
const formatMetricValue = (value, unit = '') => {
  if (value === null || value === undefined || isNaN(value)) return '-';
  
  // Format large numbers
  if (Math.abs(value) >= 1000000000) {
    return (value / 1000000000).toFixed(1) + 'B' + unit;
  } else if (Math.abs(value) >= 1000000) {
    return (value / 1000000).toFixed(1) + 'M' + unit;
  } else if (Math.abs(value) >= 1000) {
    return (value / 1000).toFixed(1) + 'K' + unit;
  } else if (value % 1 !== 0) {
    return parseFloat(value.toFixed(2)) + unit;
  } else {
    return value + unit;
  }
};

// MetricChart class for managing individual charts
class MetricChart {
  constructor(canvasId, metricId, config = {}) {
    this.canvasId = canvasId;
    this.metricId = metricId;
    this.config = {
      type: 'line',
      title: metricId,
      unit: '',
      color: '#0d6efd',
      backgroundColor: 'rgba(13, 110, 253, 0.1)',
      maxPoints: 100,
      ...config
    };
    
    this.chart = null;
    this.dataStack = [];
    this.currentValue = null;
    
    this.initChart();
  }
  
  initChart() {
    const canvas = document.getElementById(this.canvasId);
    if (!canvas) {
      console.error(`Canvas element ${this.canvasId} not found`);
      return;
    }
    
    const ctx = canvas.getContext('2d');
    
    // Chart.js configuration
    const chartConfig = {
      type: this.config.type,
      data: {
        datasets: [{
          label: this.config.title,
          data: [],
          borderColor: this.config.color,
          backgroundColor: this.config.backgroundColor,
          borderWidth: 2,
          pointRadius: 0,
          pointHoverRadius: 4,
          tension: 0.4,
          fill: this.config.type === 'line'
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 300 },
        interaction: {
          intersect: false,
          mode: 'index'
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: 'rgba(0,0,0,0.8)',
            titleColor: '#fff',
            bodyColor: '#fff',
            cornerRadius: 4,
            callbacks: {
              label: (context) => {
                const value = formatMetricValue(context.parsed.y, this.config.unit);
                return `${this.config.title}: ${value}`;
              },
              labelColor: () => ({
                borderColor: this.config.color,
                backgroundColor: this.config.color
              })
            }
          }
        },
        scales: this.getScalesConfig()
      }
    };
    
    // Apply theme-specific colors
    this.applyThemeColors(chartConfig);
    
    this.chart = new Chart(ctx, chartConfig);
    
    // Store chart reference globally for theme updates
    if (!window.chartInstances) window.chartInstances = [];
    window.chartInstances.push(this.chart);
  }
  
  getScalesConfig() {
    const isDarkTheme = document.documentElement.getAttribute('data-theme') === 'dark';
    const gridColor = isDarkTheme ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)';
    const tickColor = isDarkTheme ? 'rgba(255,255,255,0.6)' : 'rgba(0,0,0,0.6)';
    
    if (this.config.type === 'doughnut') {
      return {}; // No axes for doughnut charts
    }
    
    return {
      x: {
        type: 'time',
        time: {
          displayFormats: {
            minute: 'HH:mm',
            hour: 'HH:mm',
            day: 'MMM DD'
          }
        },
        grid: { color: gridColor },
        ticks: { color: tickColor, maxTicksLimit: 8 }
      },
      y: {
        beginAtZero: true,
        grid: { color: gridColor },
        ticks: { 
          color: tickColor,
          callback: (value) => formatMetricValue(value, this.config.unit)
        }
      }
    };
  }
  
  applyThemeColors(chartConfig) {
    const isDarkTheme = document.documentElement.getAttribute('data-theme') === 'dark';
    if (isDarkTheme) {
      // Theme colors are already applied via getScalesConfig
    }
  }
  
  addDataPoint(timestamp, value) {
    const point = { 
      x: new Date(timestamp), 
      y: parseFloat(value) || 0 
    };
    
    this.chart.data.datasets[0].data.push(point);
    this.dataStack.push({ time: timestamp, value: parseFloat(value) || 0 });
    
    // Keep only recent data points
    if (this.dataStack.length > this.config.maxPoints) {
      this.dataStack.shift();
      this.chart.data.datasets[0].data.shift();
    }
    
    this.currentValue = parseFloat(value) || 0;
    this.updateCurrentValueDisplay();
    this.chart.update('none');
  }
  
  setData(dataArray) {
    const formattedData = dataArray.map(item => ({
      x: new Date(item.time || item.timestamp),
      y: parseFloat(item.value) || 0
    }));
    
    this.chart.data.datasets[0].data = formattedData;
    this.dataStack = dataArray.map(item => ({
      time: item.time || item.timestamp,
      value: parseFloat(item.value) || 0
    }));
    
    if (this.dataStack.length > 0) {
      this.currentValue = this.dataStack[this.dataStack.length - 1].value;
      this.updateCurrentValueDisplay();
    }
    
    this.chart.update();
  }
  
  updateCurrentValueDisplay() {
    const card = document.getElementById(this.canvasId).closest('.metric-card');
    let valueDisplay = card.querySelector('.metric-current-value');
    
    if (!valueDisplay) {
      valueDisplay = document.createElement('div');
      valueDisplay.className = 'metric-current-value';
      card.querySelector('.metric-card-body').appendChild(valueDisplay);
    }
    
    const formattedValue = formatMetricValue(this.currentValue, this.config.unit);
    valueDisplay.textContent = formattedValue;
    
    // Animate value update
    card.classList.add('updated');
    setTimeout(() => card.classList.remove('updated'), 300);
  }
  
  switchChartType(newType) {
    if (newType === this.config.type) return;
    
    this.config.type = newType;
    const currentData = [...this.dataStack];
    
    this.chart.destroy();
    this.initChart();
    this.setData(currentData);
  }
  
  exportCSV() {
    if (this.dataStack.length === 0) {
      showToast('No data available for export', 'warning');
      return;
    }
    
    const headers = ['Timestamp', 'Value'];
    const csvContent = [
      headers.join(','),
      ...this.dataStack.map(row => [
        new Date(row.time).toISOString(),
        row.value
      ].join(','))
    ].join('\n');
    
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${this.metricId}_${Date.now()}.csv`;
    document.body.appendChild(a);
    a.click();
    
    setTimeout(() => {
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }, 1000);
    
    showToast(`Exported ${this.dataStack.length} data points`, 'success');
  }
}

// Initialize metrics section
async function initMetricsSection() {
  console.log('Initializing WUI5: System Metrics & Performance Charts');
  
  // Create default metrics
  const metricsConfig = [
    { 
      id: 'cpu_usage', 
      title: 'CPU Usage', 
      type: 'line', 
      unit: '%',
      color: '#dc3545',
      backgroundColor: 'rgba(220, 53, 69, 0.1)'
    },
    { 
      id: 'memory_usage', 
      title: 'Memory Usage', 
      type: 'line', 
      unit: ' GB',
      color: '#0dcaf0',
      backgroundColor: 'rgba(13, 202, 240, 0.1)'
    },
    { 
      id: 'disk_io', 
      title: 'Disk I/O', 
      type: 'bar', 
      unit: ' MB/s',
      color: '#198754',
      backgroundColor: 'rgba(25, 135, 84, 0.1)'
    },
    { 
      id: 'network_traffic', 
      title: 'Network Traffic', 
      type: 'line', 
      unit: ' MB/s',
      color: '#fd7e14',
      backgroundColor: 'rgba(253, 126, 20, 0.1)'
    },
    { 
      id: 'task_throughput', 
      title: 'Task Throughput', 
      type: 'bar', 
      unit: '/min',
      color: '#6610f2',
      backgroundColor: 'rgba(102, 16, 242, 0.1)'
    },
    { 
      id: 'error_rate', 
      title: 'Error Rate', 
      type: 'line', 
      unit: '%',
      color: '#d63384',
      backgroundColor: 'rgba(214, 51, 132, 0.1)'
    }
  ];
  
  // Render metric cards
  const grid = document.getElementById('metrics-widget-grid');
  if (!grid) return;
  
  grid.innerHTML = ''; // Clear existing content
  
  metricsConfig.forEach(config => {
    renderMetricCard(config);
  });
  
  // Initialize time range buttons
  initTimeRangeButtons();
  
  // Setup SSE connection
  await setupMetricsSSE();
  
  // Load initial data
  await loadInitialMetricsData();
  
  // Start performance alerts monitoring
  initPerformanceAlerts();
  
  console.log('WUI5: System metrics initialized successfully');
}

// Render individual metric card
function renderMetricCard(config) {
  const grid = document.getElementById('metrics-widget-grid');
  
  const cardHtml = `
    <div class="metric-card" data-metric-id="${config.id}">
      <div class="metric-card-header">
        <h5 class="metric-card-title">${config.title}</h5>
        <div class="metric-card-actions">
          <button class="btn btn-sm export-btn" data-metric-id="${config.id}" title="Export CSV">
            📊
          </button>
          <select class="chart-type-selector" data-metric-id="${config.id}">
            <option value="line" ${config.type === 'line' ? 'selected' : ''}>Line</option>
            <option value="bar" ${config.type === 'bar' ? 'selected' : ''}>Bar</option>
            <option value="doughnut" ${config.type === 'doughnut' ? 'selected' : ''}>Doughnut</option>
          </select>
        </div>
      </div>
      <div class="metric-card-body">
        <canvas id="chart-${config.id}" class="chart-container"></canvas>
        <div class="chart-loading d-none">
          <div class="spinner-border" role="status">
            <span class="visually-hidden">Loading...</span>
          </div>
        </div>
      </div>
    </div>
  `;
  
  grid.insertAdjacentHTML('beforeend', cardHtml);
  
  // Create chart instance after DOM is updated
  setTimeout(() => {
    const chart = new MetricChart(`chart-${config.id}`, config.id, config);
    metricsState.charts.set(config.id, chart);
  }, 100);
}

// Initialize time range buttons
function initTimeRangeButtons() {
  const buttons = document.querySelectorAll('.time-range-buttons .btn');
  
  buttons.forEach(btn => {
    btn.addEventListener('click', async () => {
      // Update active state
      buttons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      
      // Update current range
      const range = btn.dataset.range;
      metricsState.currentRange = range;
      
      // Reload data for all charts
      await loadHistoricalData(range);
    });
  });
  
  // Set default active button
  const defaultBtn = document.querySelector('.time-range-buttons .btn[data-range="1h"]');
  if (defaultBtn) defaultBtn.classList.add('active');
}

// Setup Server-Sent Events for real-time metrics
async function setupMetricsSSE() {
  if (metricsState.sseConnection) {
    metricsState.sseConnection.close();
  }
  
  try {
    const eventSource = new EventSource('/api/metrics/stream');
    
    eventSource.onopen = () => {
      console.log('Metrics SSE connection established');
      updateConnectionStatus(true);
      metricsState.isConnected = true;
    };
    
    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        handleMetricUpdate(data);
      } catch (error) {
        console.error('Error parsing metrics SSE data:', error);
      }
    };
    
    eventSource.onerror = (error) => {
      console.error('Metrics SSE error:', error);
      updateConnectionStatus(false);
      metricsState.isConnected = false;
      
      // Attempt to reconnect after 5 seconds
      setTimeout(() => {
        if (eventSource.readyState === EventSource.CLOSED) {
          setupMetricsSSE();
        }
      }, 5000);
    };
    
    metricsState.sseConnection = eventSource;
    
  } catch (error) {
    console.error('Failed to setup metrics SSE:', error);
    updateConnectionStatus(false);
  }
}

// Handle individual metric updates from SSE
function handleMetricUpdate(data) {
  const { metricId, value, timestamp, unit } = data;
  const chart = metricsState.charts.get(metricId);
  
  if (chart) {
    chart.addDataPoint(timestamp || Date.now(), value);
  }
}

// Update connection status indicator
function updateConnectionStatus(isConnected) {
  const indicator = document.querySelector('.connection-indicator');
  const status = document.querySelector('.connection-status span');
  
  if (indicator && status) {
    if (isConnected) {
      indicator.classList.remove('disconnected');
      status.textContent = 'Connected';
    } else {
      indicator.classList.add('disconnected');
      status.textContent = 'Disconnected';
    }
  }
}

// Load historical data for specified time range
async function loadHistoricalData(range) {
  const promises = Array.from(metricsState.charts.entries()).map(async ([metricId, chart]) => {
    try {
      const response = await fetch(`/api/metrics/${metricId}?range=${range}`);
      if (response.ok) {
        const data = await response.json();
        chart.setData(data.points || data || []);
      } else {
        console.warn(`Failed to load data for ${metricId}:`, response.statusText);
        // Show sample data for demo
        generateSampleData(chart, range);
      }
    } catch (error) {
      console.error(`Error loading ${metricId} data:`, error);
      // Show sample data for demo
      generateSampleData(chart, range);
    }
  });
  
  await Promise.all(promises);
}

// Load initial metrics data
async function loadInitialMetricsData() {
  await loadHistoricalData(metricsState.currentRange);
}

// Generate sample data for demonstration
function generateSampleData(chart, range) {
  const now = new Date();
  const points = [];
  let intervalMs, numPoints;
  
  switch (range) {
    case '5min':
      intervalMs = 5 * 1000; // 5 seconds
      numPoints = 60;
      break;
    case '1h':
      intervalMs = 60 * 1000; // 1 minute
      numPoints = 60;
      break;
    case '24h':
      intervalMs = 24 * 60 * 1000; // 24 minutes
      numPoints = 60;
      break;
    case '7d':
      intervalMs = 7 * 24 * 60 * 1000; // 7 days in minutes, sampled every 2.8 hours
      numPoints = 60;
      break;
    default:
      intervalMs = 60 * 1000;
      numPoints = 60;
  }
  
  let baseValue = Math.random() * 100;
  
  for (let i = numPoints - 1; i >= 0; i--) {
    const timestamp = new Date(now.getTime() - (i * intervalMs));
    // Generate realistic sample data with some variation
    const variation = (Math.random() - 0.5) * 20;
    const value = Math.max(0, Math.min(100, baseValue + variation));
    
    points.push({ time: timestamp.toISOString(), value: value });
    baseValue = value; // Gradual drift
  }
  
  chart.setData(points);
}

// Initialize performance alerts
function initPerformanceAlerts() {
  loadPerformanceAlerts();
  
  // Refresh alerts every 30 seconds
  setInterval(loadPerformanceAlerts, 30000);
}

// Load and display performance alerts
async function loadPerformanceAlerts() {
  const alertsContainer = document.getElementById('alertsSection');
  if (!alertsContainer) return;
  
  try {
    // Try to load from API first
    const response = await fetch('/api/alerts?since=' + (Date.now() - 5 * 60 * 1000));
    if (response.ok) {
      const alerts = await response.json();
      renderAlerts(alerts);
    } else {
      // Generate sample alerts for demo
      const sampleAlerts = generateSampleAlerts();
      renderAlerts(sampleAlerts);
    }
  } catch (error) {
    console.error('Error loading alerts:', error);
    // Generate sample alerts for demo
    const sampleAlerts = generateSampleAlerts();
    renderAlerts(sampleAlerts);
  }
}

// Render alerts in the alerts section
function renderAlerts(alerts) {
  const alertsContainer = document.getElementById('alertsSection');
  
  // Clear existing alerts
  const existingAlerts = alertsContainer.querySelectorAll('.alert-card');
  existingAlerts.forEach(alert => alert.remove());
  
  if (alerts.length === 0) {
    alertsContainer.insertAdjacentHTML('beforeend', `
      <div class="alert-card low">
        <div class="alert-icon low">
          <span>✓</span>
        </div>
        <div class="alert-content">
          <div class="alert-message">All systems operating normally</div>
          <div class="alert-timestamp">${new Date().toLocaleString()}</div>
        </div>
      </div>
    `);
    return;
  }
  
  alerts.forEach(alert => {
    const alertHtml = `
      <div class="alert-card ${alert.severity}" role="alert">
        <div class="alert-icon ${alert.severity}">
          <span>${getAlertIcon(alert.severity)}</span>
        </div>
        <div class="alert-content">
          <div class="alert-message">${alert.message}</div>
          <div class="alert-timestamp">${new Date(alert.timestamp).toLocaleString()}</div>
        </div>
      </div>
    `;
    alertsContainer.insertAdjacentHTML('beforeend', alertHtml);
  });
}

// Get appropriate icon for alert severity
function getAlertIcon(severity) {
  switch (severity) {
    case 'high': return '⚠️';
    case 'medium': return '⚡';
    case 'low': return 'ℹ️';
    default: return 'ℹ️';
  }
}

// Generate sample alerts for demonstration
function generateSampleAlerts() {
  const alerts = [
    {
      severity: 'medium',
      message: 'CPU usage above 80% for the last 5 minutes',
      timestamp: Date.now() - 2 * 60 * 1000
    },
    {
      severity: 'low',
      message: 'Memory usage is at 65% capacity',
      timestamp: Date.now() - 10 * 60 * 1000
    }
  ];
  
  // Randomly show/hide alerts
  return Math.random() > 0.3 ? alerts : [];
}

// Event handlers for metrics
document.addEventListener('click', (e) => {
  if (e.target.matches('.export-btn, .export-btn *')) {
    const btn = e.target.closest('.export-btn');
    const metricId = btn.dataset.metricId;
    const chart = metricsState.charts.get(metricId);
    
    if (chart) {
      chart.exportCSV();
    }
  }
});

document.addEventListener('change', (e) => {
  if (e.target.matches('.chart-type-selector')) {
    const metricId = e.target.dataset.metricId;
    const newType = e.target.value;
    const chart = metricsState.charts.get(metricId);
    
    if (chart) {
      chart.switchChartType(newType);
    }
  }
});

// Initialize when metrics section is shown
function onMetricsSectionShow() {
  if (metricsState.charts.size === 0) {
    initMetricsSection();
  }
}

// Cleanup function for metrics
function cleanupMetrics() {
  if (metricsState.sseConnection) {
    metricsState.sseConnection.close();
    metricsState.sseConnection = null;
  }
  
  if (metricsState.updateInterval) {
    clearInterval(metricsState.updateInterval);
    metricsState.updateInterval = null;
  }
  
  metricsState.charts.forEach(chart => {
    if (chart.chart) {
      chart.chart.destroy();
    }
  });
  metricsState.charts.clear();
}

/* End of System Metrics & Performance Charts (WUI5) */

/*=================================================================
  🔥 WUI6: Configuration Management Interface JavaScript
=================================================================*/

/* Configuration Management State */
const configState = {
  currentConfig: {},
  sseConnection: null,
  timezonePupolated: false,
  formStates: {
    system: {},
    agents: [],
    discovery: {},
    web: {},
    security: {}
  }
};

/* Utility helpers for configuration management */
const configUtils = {
  // Show an alert in the alert area
  showAlert(msg, type = 'info', timeout = 5000) {
    const alertArea = $('#alert-area');
    if (!alertArea) return;
    
    const alert = document.createElement('div');
    alert.className = `alert alert-${type} alert-dismissible fade show`;
    alert.role = 'alert';
    alert.innerHTML = `
      ${msg}
      <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    
    alertArea.appendChild(alert);
    
    if (timeout > 0) {
      setTimeout(() => {
        if (alert.parentNode) {
          alert.remove();
        }
      }, timeout);
    }
    
    return alert;
  },

  // Clear all alerts
  clearAlerts() {
    const alertArea = $('#alert-area');
    if (alertArea) {
      alertArea.innerHTML = '';
    }
  },

  // Populate timezone dropdown
  populateTimezones() {
    if (configState.timezonePupolated) return;
    
    const timezoneSelect = $('#timezone');
    if (!timezoneSelect) return;
    
    const timezones = [
      'UTC', 'America/New_York', 'America/Chicago', 'America/Denver', 
      'America/Los_Angeles', 'Europe/London', 'Europe/Paris', 'Europe/Berlin',
      'Europe/Madrid', 'Asia/Tokyo', 'Asia/Shanghai', 'Asia/Kolkata',
      'Australia/Sydney', 'Pacific/Auckland'
    ];
    
    timezoneSelect.innerHTML = '';
    timezones.forEach(tz => {
      const option = document.createElement('option');
      option.value = tz;
      option.textContent = tz;
      timezoneSelect.appendChild(option);
    });
    
    configState.timezonePupolated = true;
  },

  // Form validation
  validateForm(formElement) {
    const requiredInputs = formElement.querySelectorAll('[required]');
    let isValid = true;
    
    requiredInputs.forEach(input => {
      input.classList.remove('is-valid', 'is-invalid');
      
      if (!input.value.trim()) {
        input.classList.add('is-invalid');
        isValid = false;
      } else {
        input.classList.add('is-valid');
      }
    });
    
    return isValid;
  }
};

/* Configuration API methods */
const configAPI = {
  // Load current configuration
  async loadConfig() {
    try {
      const response = await apiClient.get('/api/config/current');
      if (response.ok) {
        const config = await response.json();
        configState.currentConfig = config;
        return config;
      }
      throw new Error('Failed to load configuration');
    } catch (error) {
      console.error('Error loading configuration:', error);
      configUtils.showAlert(`Cannot fetch configuration: ${error.message}`, 'danger');
      return {};
    }
  },

  // Save configuration for a specific section
  async saveConfig(endpoint, payload) {
    try {
      const response = await apiClient.post(`/api/config${endpoint}`, payload);
      if (response.ok) {
        configUtils.showAlert('Configuration saved successfully.', 'success');
        return true;
      }
      
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.message || 'Unknown server error');
    } catch (error) {
      console.error('Error saving configuration:', error);
      configUtils.showAlert(`Failed to save: ${error.message}`, 'danger');
      return false;
    }
  },

  // Import configuration
  async importConfig(file) {
    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch('/api/config/import', {
        method: 'POST',
        body: formData,
        credentials: 'include',
        headers: {
          'Authorization': `Bearer ${tokenStore.get()?.raw}`
        }
      });

      if (response.ok) {
        configUtils.showAlert('Import successful. Configuration will reload automatically.', 'success');
        return true;
      }
      
      throw new Error('Import failed');
    } catch (error) {
      console.error('Error importing configuration:', error);
      configUtils.showAlert(`Import failed: ${error.message}`, 'danger');
      return false;
    }
  },

  // Export configuration
  async exportConfig() {
    try {
      const response = await fetch('/api/config/export', {
        method: 'GET',
        credentials: 'include',
        headers: {
          'Authorization': `Bearer ${tokenStore.get()?.raw}`
        }
      });

      if (response.ok) {
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = 'agentsmcp_config.json';
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
        
        configUtils.showAlert('Export completed.', 'success', 3000);
        return true;
      }
      
      throw new Error('Export failed');
    } catch (error) {
      console.error('Error exporting configuration:', error);
      configUtils.showAlert(`Export failed: ${error.message}`, 'danger');
      return false;
    }
  },

  // Get configuration history
  async getHistory() {
    try {
      const response = await apiClient.get('/api/config/history');
      if (response.ok) {
        return await response.json();
      }
      throw new Error('Failed to fetch history');
    } catch (error) {
      console.error('Error fetching history:', error);
      configUtils.showAlert('Cannot fetch history.', 'danger');
      return [];
    }
  },

  // Rollback to a specific configuration
  async rollback(historyId) {
    try {
      const response = await apiClient.post(`/api/config/rollback/${historyId}`);
      if (response.ok) {
        configUtils.showAlert('Rollback succeeded.', 'success');
        return true;
      }
      throw new Error('Rollback failed');
    } catch (error) {
      console.error('Error during rollback:', error);
      configUtils.showAlert(`Rollback error: ${error.message}`, 'danger');
      return false;
    }
  }
};

/* Configuration form handling */
const configForms = {
  // Populate all forms with current configuration
  populateAllForms(config) {
    this.populateSystemForm(config);
    this.populateAgentsForm(config);
    this.populateDiscoveryForm(config);
    this.populateWebForm(config);
    this.populateSecurityForm(config);
  },

  // Populate system configuration form
  populateSystemForm(config) {
    const timezoneSelect = $('#timezone');
    const languageSelect = $('#language');
    const logLevelSelect = $('#logLevel');

    if (timezoneSelect) timezoneSelect.value = config.timezone || 'UTC';
    if (languageSelect) languageSelect.value = config.language || 'en';
    if (logLevelSelect) logLevelSelect.value = config.logLevel || 'info';
  },

  // Populate agents configuration form
  populateAgentsForm(config) {
    const agentList = $('#agent-list');
    if (!agentList) return;

    agentList.innerHTML = '';

    if (Array.isArray(config.agents)) {
      config.agents.forEach((agent, idx) => {
        const li = document.createElement('li');
        li.className = 'list-group-item agent-config';
        li.innerHTML = `
          <div class="agent-header">
            <h6>${agent.name || `Agent #${idx + 1}`}</h6>
            <div>
              <button class="btn btn-sm btn-outline-secondary me-1" 
                      data-bs-toggle="collapse" data-bs-target="#agent-form-${idx}"
                      aria-expanded="false" aria-controls="agent-form-${idx}">
                <i class="bi bi-gear"></i> Configure
              </button>
              <button class="btn btn-sm btn-outline-info me-1" 
                      data-bs-toggle="collapse" data-bs-target="#agent-form-${idx}"
                      title="Agent Info">
                <i class="bi bi-info-circle"></i>
              </button>
            </div>
          </div>
          <div class="collapse" id="agent-form-${idx}">
            <form class="row g-3 agent-inner-form" data-agent-index="${idx}">
              <div class="col-md-6">
                <label class="form-label" for="agent-name-${idx}">Name</label>
                <input type="text" class="form-control" id="agent-name-${idx}" name="name"
                       value="${agent.name || ''}" required>
              </div>
              <div class="col-md-6">
                <div class="form-check">
                  <input type="checkbox" class="form-check-input" id="agent-enabled-${idx}"
                         ${agent.enabled ? 'checked' : ''}>
                  <label class="form-check-label" for="agent-enabled-${idx}">Enabled</label>
                </div>
              </div>
              <div class="col-md-6">
                <label class="form-label" for="agent-interval-${idx}">Collect Interval (s)</label>
                <input type="number" class="form-control" id="agent-interval-${idx}" name="interval"
                      min="5" value="${agent.interval || 60}" required>
              </div>
              <div class="col-12 d-flex justify-content-end">
                <button type="button" class="btn btn-secondary me-2 reset-agent">Reset</button>
                <button type="submit" class="btn btn-primary">Save Agent</button>
              </div>
            </form>
          </div>
        `;
        agentList.appendChild(li);
      });

      // Attach event handlers to agent forms
      agentList.querySelectorAll('form.agent-inner-form').forEach(form => {
        form.addEventListener('submit', this.handleAgentSubmit.bind(this));
        form.querySelector('.reset-agent').addEventListener('click', this.handleAgentReset.bind(this));
      });
    }
  },

  // Populate discovery configuration form
  populateDiscoveryForm(config) {
    const enabledCheckbox = $('#discovery-enabled');
    const intervalInput = $('#discovery-interval');

    if (enabledCheckbox) enabledCheckbox.checked = config.discovery?.enabled ?? false;
    if (intervalInput) intervalInput.value = config.discovery?.interval ?? 60;
  },

  // Populate web interface configuration form
  populateWebForm(config) {
    const themeSelect = $('#theme');
    const refreshIntervalInput = $('#refresh-interval');
    const layoutSelect = $('#layout');

    if (themeSelect) themeSelect.value = config.web?.theme ?? 'light';
    if (refreshIntervalInput) refreshIntervalInput.value = config.web?.refreshInterval ?? 30;
    if (layoutSelect) layoutSelect.value = config.web?.layout ?? 'grid';
  },

  // Populate security configuration form
  populateSecurityForm(config) {
    const authMethodSelect = $('#auth-method');
    const tokenTtlInput = $('#token-ttl');
    const auditLogSelect = $('#audit-log');

    if (authMethodSelect) authMethodSelect.value = config.security?.authMethod ?? 'basic';
    if (tokenTtlInput) tokenTtlInput.value = config.security?.tokenTTL ?? 30;
    if (auditLogSelect) auditLogSelect.value = config.security?.auditLog ?? 'enabled';
  },

  // Handle system form submission
  async handleSystemSubmit(e) {
    e.preventDefault();
    
    if (!configUtils.validateForm(e.target)) {
      configUtils.showAlert('Please fill in all required fields.', 'warning');
      return;
    }

    const payload = {
      timezone: $('#timezone').value,
      language: $('#language').value,
      logLevel: $('#logLevel').value
    };

    await configAPI.saveConfig('/system', payload);
  },

  // Handle agent form submission
  async handleAgentSubmit(e) {
    e.preventDefault();
    const form = e.target;
    const idx = form.dataset.agentIndex;

    if (!configUtils.validateForm(form)) {
      configUtils.showAlert('Please fill in all required fields.', 'warning');
      return;
    }

    const payload = {
      name: form.querySelector(`#agent-name-${idx}`).value,
      enabled: form.querySelector(`#agent-enabled-${idx}`).checked,
      interval: Number(form.querySelector(`#agent-interval-${idx}`).value)
    };

    await configAPI.saveConfig(`/agents/${idx}`, payload);
  },

  // Handle discovery form submission
  async handleDiscoverySubmit(e) {
    e.preventDefault();

    if (!configUtils.validateForm(e.target)) {
      configUtils.showAlert('Please fill in all required fields.', 'warning');
      return;
    }

    const payload = {
      enabled: $('#discovery-enabled').checked,
      interval: Number($('#discovery-interval').value)
    };

    await configAPI.saveConfig('/discovery', payload);
  },

  // Handle web interface form submission
  async handleWebSubmit(e) {
    e.preventDefault();

    if (!configUtils.validateForm(e.target)) {
      configUtils.showAlert('Please fill in all required fields.', 'warning');
      return;
    }

    const payload = {
      theme: $('#theme').value,
      refreshInterval: Number($('#refresh-interval').value),
      layout: $('#layout').value
    };

    await configAPI.saveConfig('/web', payload);
  },

  // Handle security form submission
  async handleSecuritySubmit(e) {
    e.preventDefault();

    if (!configUtils.validateForm(e.target)) {
      configUtils.showAlert('Please fill in all required fields.', 'warning');
      return;
    }

    const payload = {
      authMethod: $('#auth-method').value,
      tokenTTL: Number($('#token-ttl').value),
      auditLog: $('#audit-log').value
    };

    await configAPI.saveConfig('/security', payload);
  },

  // Handle reset button clicks
  handleReset() {
    this.populateAllForms(configState.currentConfig);
    configUtils.showAlert('Form reset to last saved configuration.', 'info', 3000);
  },

  // Handle agent reset button clicks
  handleAgentReset() {
    this.populateAllForms(configState.currentConfig);
    configUtils.showAlert('Agent configuration reset to last saved state.', 'info', 3000);
  }
};

/* Configuration history and rollback functionality */
const configHistory = {
  // Show history modal with rollback functionality
  async showHistory() {
    const history = await configAPI.getHistory();
    
    const tbody = $('#history-table-body');
    if (!tbody) return;

    tbody.innerHTML = '';

    if (history.length === 0) {
      tbody.innerHTML = '<tr><td colspan="4" class="text-center text-muted">No history available</td></tr>';
    } else {
      history.forEach(entry => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td>${new Date(entry.timestamp).toLocaleString()}</td>
          <td>${entry.user || 'System'}</td>
          <td>${entry.comment || 'Configuration update'}</td>
          <td>
            <button class="btn btn-sm btn-outline-success rollback-btn" 
                    data-id="${entry.id}" title="Rollback to this configuration">
              <i class="bi bi-arrow-counterclockwise me-1"></i>Rollback
            </button>
          </td>
        `;
        tbody.appendChild(tr);
      });

      // Attach rollback event handlers
      tbody.querySelectorAll('.rollback-btn').forEach(btn => {
        btn.addEventListener('click', async () => {
          const id = btn.dataset.id;
          if (confirm('Are you sure you want to rollback to this configuration? This action cannot be undone.')) {
            const success = await configAPI.rollback(id);
            if (success) {
              // Close modal and reload configuration
              const modal = bootstrap.Modal.getInstance('#modalHistory');
              if (modal) modal.hide();
              
              // Reload configuration after rollback
              const newConfig = await configAPI.loadConfig();
              configForms.populateAllForms(newConfig);
            }
          }
        });
      });
    }

    // Show the modal
    const modal = new bootstrap.Modal('#modalHistory');
    modal.show();
  }
};

/* SSE connection for real-time configuration updates */
const configSSE = {
  connection: null,

  // Initialize SSE connection
  connect() {
    if (this.connection) {
      this.connection.close();
    }

    try {
      this.connection = new EventSource('/api/config/stream', {
        withCredentials: true
      });

      this.connection.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          configState.currentConfig = data;
          configForms.populateAllForms(data);
          configUtils.showAlert('Configuration updated in real-time.', 'info', 2000);
        } catch (error) {
          console.error('SSE parse error:', error);
        }
      };

      this.connection.onerror = () => {
        console.warn('Configuration SSE connection lost, attempting to reconnect...');
        setTimeout(() => this.connect(), 5000);
      };

      this.connection.onopen = () => {
        console.log('Configuration SSE connection established');
      };
    } catch (error) {
      console.error('Failed to establish SSE connection:', error);
    }
  },

  // Close SSE connection
  disconnect() {
    if (this.connection) {
      this.connection.close();
      this.connection = null;
    }
  }
};

/* Configuration section initialization and event handlers */
function initConfigurationSection() {
  // Populate timezone dropdown
  configUtils.populateTimezones();

  // Load and populate initial configuration
  configAPI.loadConfig().then(config => {
    configForms.populateAllForms(config);
  });

  // Initialize SSE connection for real-time updates
  configSSE.connect();

  // Attach form event handlers
  const systemForm = $('#form-system');
  const discoveryForm = $('#form-discovery');
  const webForm = $('#form-web');
  const securityForm = $('#form-security');

  if (systemForm) {
    systemForm.addEventListener('submit', configForms.handleSystemSubmit.bind(configForms));
  }
  if (discoveryForm) {
    discoveryForm.addEventListener('submit', configForms.handleDiscoverySubmit.bind(configForms));
  }
  if (webForm) {
    webForm.addEventListener('submit', configForms.handleWebSubmit.bind(configForms));
  }
  if (securityForm) {
    securityForm.addEventListener('submit', configForms.handleSecuritySubmit.bind(configForms));
  }

  // Reset button handlers
  $('#reset-system')?.addEventListener('click', configForms.handleReset.bind(configForms));
  $('#reset-discovery')?.addEventListener('click', configForms.handleReset.bind(configForms));
  $('#reset-web')?.addEventListener('click', configForms.handleReset.bind(configForms));
  $('#reset-security')?.addEventListener('click', configForms.handleReset.bind(configForms));

  // Export/Import/History button handlers
  $('#btn-export')?.addEventListener('click', configAPI.exportConfig);
  $('#btn-history')?.addEventListener('click', configHistory.showHistory);
  
  $('#file-import')?.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
      configAPI.importConfig(file);
      e.target.value = ''; // Reset file input
    }
  });
}

// Function to call when configuration section is shown
function onConfigurationSectionShow() {
  if (!configState.timezonePupolated) {
    initConfigurationSection();
  }
}

// Cleanup function for configuration section
function cleanupConfiguration() {
  configSSE.disconnect();
  configUtils.clearAlerts();
  configState.timezonePupolated = false;
}

/* End of Configuration Management Interface (WUI6) */

/* --------------------------------------------------------------
   End of /static/js/dashboard.js
   -------------------------------------------------------------- */
