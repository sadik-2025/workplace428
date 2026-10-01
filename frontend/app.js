const API_BASE = "http://127.0.0.1:8000";
const TOKEN_KEY = "wex428_token";

// ---------- Simple view router ----------
const views = document.querySelectorAll(".view");
function showView(name) {
  views.forEach((v) => v.classList.toggle("active", v.dataset.view === name));
  window.scrollTo(0, 0);
  if (name === "dashboard") loadDashboard();
  if (name === "profile") loadProfile();
}
document.body.addEventListener("click", (e) => {
  const target = e.target.closest("[data-nav]");
  if (target) {
    e.preventDefault();
    const view = target.dataset.nav;
    if (["dashboard", "recognize", "profile", "result", "robustness", "formula"].includes(view) && !getToken()) {
      showView("login");
      return;
    }
    showView(view);
  }
});

// ---------- Auth helpers ----------
function getToken() { return localStorage.getItem(TOKEN_KEY); }
function setToken(t) { localStorage.setItem(TOKEN_KEY, t); }
function clearToken() { localStorage.removeItem(TOKEN_KEY); }

function authHeaders() {
  const t = getToken();
  return t ? { Authorization: `Bearer ${t}` } : {};
}

function renderNav() {
  const navLinks = document.getElementById("navLinks");
  if (getToken()) {
    navLinks.innerHTML = `
      <a data-nav="dashboard">Dashboard</a>
      <a data-nav="recognize">Recognize</a>
      <a data-nav="formula">Formula</a>
      <a data-nav="profile">Profile</a>
      <a id="navLogout">Log Out</a>
    `;
    document.getElementById("navLogout").addEventListener("click", () => {
      clearToken();
      renderNav();
      showView("landing");
    });
  } else {
    navLinks.innerHTML = `
      <a data-nav="login">Log In</a>
      <a data-nav="register">Register</a>
    `;
  }
}

// ---------- Register ----------
document.getElementById("registerForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const username = document.getElementById("regUsername").value;
  const email = document.getElementById("regEmail").value;
  const password = document.getElementById("regPassword").value;
  const errEl = document.getElementById("registerError");
  errEl.textContent = "";

  try {
    const res = await fetch(`${API_BASE}/auth/register`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, email, password }),
    });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.detail || "Registration failed");
    }
    showView("login");
  } catch (err) {
    errEl.textContent = err.message;
  }
});

// ---------- Login ----------
document.getElementById("loginForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const username = document.getElementById("loginUsername").value;
  const password = document.getElementById("loginPassword").value;
  const errEl = document.getElementById("loginError");
  errEl.textContent = "";

  try {
    const body = new URLSearchParams();
    body.append("username", username);
    body.append("password", password);

    const res = await fetch(`${API_BASE}/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body,
    });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.detail || "Login failed");
    }
    const data = await res.json();
    setToken(data.access_token);
    renderNav();
    showView("dashboard");
  } catch (err) {
    errEl.textContent = err.message;
  }
});

document.getElementById("logoutBtn").addEventListener("click", () => {
  clearToken();
  renderNav();
  showView("landing");
});

// ---------- Profile ----------
async function loadProfile() {
  try {
    const res = await fetch(`${API_BASE}/auth/me`, { headers: authHeaders() });
    if (!res.ok) throw new Error();
    const data = await res.json();
    document.getElementById("profileUsername").textContent = data.username;
    document.getElementById("profileEmail").textContent = data.email;
    document.getElementById("profileCreated").textContent = new Date(data.created_at).toLocaleString();
  } catch {
    clearToken();
    renderNav();
    showView("login");
  }
}

// ---------- Dashboard ----------
let distChartInstance, confChartInstance, timeChartInstance, modelCompareChartInstance;

async function loadDashboard() {
  try {
    const statsRes = await fetch(`${API_BASE}/stats`, { headers: authHeaders() });
    const stats = await statsRes.json();

    document.getElementById("statTotal").textContent = stats.total_predictions;
    document.getElementById("statMostFrequent").textContent = stats.most_frequent_symbol ?? "-";
    document.getElementById("statAvgConf").textContent = stats.average_confidence
      ? (stats.average_confidence * 100).toFixed(1) + "%"
      : "-";

    drawBarChart("distChart", distChartInstance, Object.keys(stats.prediction_distribution),
      Object.values(stats.prediction_distribution), "Predictions")
      .then((c) => (distChartInstance = c));

    drawBarChart("confChart", confChartInstance, Object.keys(stats.confidence_distribution),
      Object.values(stats.confidence_distribution), "Count")
      .then((c) => (confChartInstance = c));

    drawLineChart("timeChart", timeChartInstance, Object.keys(stats.predictions_over_time),
      Object.values(stats.predictions_over_time))
      .then((c) => (timeChartInstance = c));

    const modelsRes = await fetch(`${API_BASE}/models/compare`, { headers: authHeaders() });
    const models = await modelsRes.json();
    renderModelTable(models);
    renderModelCompareChart(models);
  } catch (err) {
    console.error(err);
  }
}

function renderModelTable(models) {
  const wrap = document.getElementById("modelTableWrap");
  const rows = Object.entries(models)
    .filter(([, m]) => m)
    .map(
      ([key, m]) => `
      <tr>
        <td>${m.model_name}</td>
        <td>${(m.test_accuracy * 100).toFixed(1)}%</td>
        <td>${m.unseen_domain_accuracy != null ? (m.unseen_domain_accuracy * 100).toFixed(1) + "%" : "-"}</td>
        <td>${(m.f1_macro * 100).toFixed(1)}%</td>
        <td>${m.inference_time_ms_per_batch.toFixed(1)} ms</td>
        <td>${m.num_parameters.toLocaleString()}</td>
        <td>${m.model_size_mb.toFixed(2)} MB</td>
      </tr>`
    )
    .join("");
  wrap.innerHTML = `
    <table>
      <thead><tr>
        <th>Model</th><th>Test Acc</th><th>Unseen-Domain Acc</th><th>F1</th>
        <th>Inference</th><th>Params</th><th>Size</th>
      </tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
}

function renderModelCompareChart(models) {
  const ctx = document.getElementById("modelCompareChart").getContext("2d");
  const entries = Object.entries(models).filter(([, m]) => m);
  if (modelCompareChartInstance) modelCompareChartInstance.destroy();
  modelCompareChartInstance = new Chart(ctx, {
    type: "bar",
    data: {
      labels: entries.map(([, m]) => m.model_name),
      datasets: [
        { label: "Standard Test", data: entries.map(([, m]) => m.test_accuracy), backgroundColor: "#6c8cff" },
        {
          label: "Unseen Domain",
          data: entries.map(([, m]) => m.unseen_domain_accuracy ?? 0),
          backgroundColor: "#e29a4c",
        },
      ],
    },
    options: { scales: { y: { beginAtZero: true, max: 1 } }, responsive: true },
  });
}

async function drawBarChart(canvasId, existing, labels, data, label) {
  const ctx = document.getElementById(canvasId).getContext("2d");
  if (existing) existing.destroy();
  return new Chart(ctx, {
    type: "bar",
    data: { labels, datasets: [{ label, data, backgroundColor: "#6c8cff" }] },
    options: { scales: { y: { beginAtZero: true } }, responsive: true, plugins: { legend: { display: false } } },
  });
}

async function drawLineChart(canvasId, existing, labels, data) {
  const ctx = document.getElementById(canvasId).getContext("2d");
  if (existing) existing.destroy();
  return new Chart(ctx, {
    type: "line",
    data: { labels, datasets: [{ label: "Predictions", data, borderColor: "#6c8cff", tension: 0.3 }] },
    options: { scales: { y: { beginAtZero: true } }, responsive: true, plugins: { legend: { display: false } } },
  });
}

// ---------- Symbol recognition: upload + draw ----------
let currentImageBlob = null;

document.getElementById("fileInput").addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (!file) return;
  currentImageBlob = file;
  document.getElementById("uploadPreview").src = URL.createObjectURL(file);
});

function setupDrawCanvas(canvas, lineWidth) {
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "white";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.lineWidth = lineWidth;
  ctx.lineCap = "round";
  ctx.strokeStyle = "black";

  let drawing = false;
  function getPos(e) {
    const rect = canvas.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    return { x: clientX - rect.left, y: clientY - rect.top };
  }
  function startDraw(e) { drawing = true; const p = getPos(e); ctx.beginPath(); ctx.moveTo(p.x, p.y); }
  function draw(e) {
    if (!drawing) return;
    const p = getPos(e);
    ctx.lineTo(p.x, p.y);
    ctx.stroke();
    e.preventDefault();
  }
  function endDraw() { drawing = false; }

  canvas.addEventListener("mousedown", startDraw);
  canvas.addEventListener("mousemove", draw);
  canvas.addEventListener("mouseup", endDraw);
  canvas.addEventListener("mouseleave", endDraw);
  canvas.addEventListener("touchstart", startDraw);
  canvas.addEventListener("touchmove", draw);
  canvas.addEventListener("touchend", endDraw);

  return ctx;
}

const canvas = document.getElementById("drawCanvas");
setupDrawCanvas(canvas, 10);

document.getElementById("clearCanvas").addEventListener("click", () => {
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "white";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
});

document.getElementById("useDrawing").addEventListener("click", () => {
  canvas.toBlob((blob) => {
    currentImageBlob = blob;
    document.getElementById("uploadPreview").src = URL.createObjectURL(blob);
  }, "image/png");
});

let lastResult = null;
let lastModelUsed = null;

document.getElementById("predictBtn").addEventListener("click", async () => {
  const errEl = document.getElementById("predictError");
  errEl.textContent = "";
  if (!currentImageBlob) {
    errEl.textContent = "Upload an image or draw a symbol first.";
    return;
  }
  const modelName = document.getElementById("modelSelect").value;

  const formData = new FormData();
  formData.append("file", currentImageBlob, "input.png");
  formData.append("model_name", modelName);

  try {
    const res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: authHeaders(),
      body: formData,
    });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.detail || "Prediction failed");
    }
    const data = await res.json();
    lastResult = data;
    lastModelUsed = modelName;
    renderResult(data);
    showView("result");
  } catch (err) {
    errEl.textContent = err.message;
  }
});

function renderResult(data) {
  document.getElementById("resultImage").src = document.getElementById("uploadPreview").src;
  document.getElementById("resultSymbol").textContent = data.predicted_class;
  document.getElementById("resultConfidence").textContent = (data.confidence * 100).toFixed(1) + "%";
  document.getElementById("resultModel").textContent = data.model_used;
  document.getElementById("resultDate").textContent = new Date(data.created_at).toLocaleString();

  const ctx = document.getElementById("topKChart").getContext("2d");
  if (window._topKChart) window._topKChart.destroy();
  window._topKChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: data.top_k.map((t) => t.class),
      datasets: [{ label: "Probability", data: data.top_k.map((t) => t.probability), backgroundColor: "#6c8cff" }],
    },
    options: { indexAxis: "y", scales: { x: { beginAtZero: true, max: 1 } }, plugins: { legend: { display: false } } },
  });
}

// ---------- Robustness analysis ----------
document.getElementById("runRobustness").addEventListener("click", async () => {
  if (!currentImageBlob) return;
  const formData = new FormData();
  formData.append("file", currentImageBlob, "input.png");
  formData.append("model_name", lastModelUsed || "ssl_domain_gen");

  try {
    const res = await fetch(`${API_BASE}/predict/robustness`, {
      method: "POST",
      headers: authHeaders(),
      body: formData,
    });
    const data = await res.json();
    renderRobustness(data);
    showView("robustness");
  } catch (err) {
    console.error(err);
  }
});

function renderRobustness(data) {
  const grid = document.getElementById("robustnessGrid");
  grid.innerHTML = Object.entries(data.transformations)
    .map(
      ([name, result]) => `
      <div class="robustness-item">
        <h4>${name.replace(/_/g, " ")}</h4>
        <div class="symbol">${result.predicted_class}</div>
        <div>${(result.confidence * 100).toFixed(1)}% confidence</div>
      </div>`
    )
    .join("");
}

// ---------- Formula recognition: upload + draw ----------
let currentFormulaBlob = null;

document.getElementById("formulaFileInput").addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (!file) return;
  currentFormulaBlob = file;
  document.getElementById("formulaPreview").src = URL.createObjectURL(file);
});

const formulaCanvas = document.getElementById("formulaDrawCanvas");
setupDrawCanvas(formulaCanvas, 8);

document.getElementById("clearFormulaCanvas").addEventListener("click", () => {
  const ctx = formulaCanvas.getContext("2d");
  ctx.fillStyle = "white";
  ctx.fillRect(0, 0, formulaCanvas.width, formulaCanvas.height);
});

document.getElementById("useFormulaDrawing").addEventListener("click", () => {
  formulaCanvas.toBlob((blob) => {
    currentFormulaBlob = blob;
    document.getElementById("formulaPreview").src = URL.createObjectURL(blob);
  }, "image/png");
});

document.getElementById("formulaPredictBtn").addEventListener("click", async () => {
  const errEl = document.getElementById("formulaError");
  errEl.textContent = "";
  document.getElementById("formulaResultCard").style.display = "none";

  if (!currentFormulaBlob) {
    errEl.textContent = "Upload a photo or draw a formula first.";
    return;
  }
  const modelName = document.getElementById("formulaModelSelect").value;

  const formData = new FormData();
  formData.append("file", currentFormulaBlob, "formula.png");
  formData.append("model_name", modelName);

  try {
    const res = await fetch(`${API_BASE}/predict/formula`, {
      method: "POST",
      headers: authHeaders(),
      body: formData,
    });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.detail || "Formula recognition failed");
    }
    const data = await res.json();
    renderFormulaResult(data);
  } catch (err) {
    errEl.textContent = err.message;
  }
});

function renderFormulaResult(data) {
  document.getElementById("formulaExpression").textContent = data.reconstructed_expression || "(none)";
  const grid = document.getElementById("formulaSegments");
  grid.innerHTML = data.segments
    .map(
      (s, i) => `
      <div class="robustness-item">
        <h4>Symbol ${i + 1}</h4>
        <div class="symbol">${s.display_symbol}</div>
        <div>${(s.confidence * 100).toFixed(1)}% confidence</div>
      </div>`
    )
    .join("");
  document.getElementById("formulaResultCard").style.display = "block";
}

// ---------- Init ----------
renderNav();
showView(getToken() ? "dashboard" : "landing");
