/* Traffic Control Room frontend.
 * Plain JavaScript dashboard served by the local FastAPI server (traffic_monitoring/server.py),
 * so the API is on the same origin. Opened as a file, it falls back to http://localhost:8000.
 */
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];

const store = {
  get(k) { try { return localStorage.getItem(k); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch { /* private mode */ } },
};

const API = location.protocol.startsWith("http") ? "" : "http://localhost:8000";

const MODES = {
  detect: { step: "Module 01", title: "Vehicle Detection", btn: "Start detection",
    sub: "Detect and classify cars, motorcycles, buses and trucks using YOLO.", label: "Vehicles detected" },
  track: { step: "Module 02", title: "Vehicle Tracking", btn: "Start tracking",
    sub: "Follow every vehicle with a unique tracking ID using ByteTrack.", label: "Currently tracked" },
  count: { step: "Module 03", title: "Traffic Analytics", btn: "Start analysis",
    sub: "Count vehicles crossing the line and estimate traffic density.", label: null },
};
const CRUMBS = { overview: "Overview", reports: "Reports", ...Object.fromEntries(Object.entries(MODES).map(([k, m]) => [k, m.title])) };
const TYPES = [["Car", "Cars"], ["Motorcycle", "Motorcycles"], ["Bus", "Buses"], ["Truck", "Trucks"]];
const TONE = { Low: "go", Medium: "slow", High: "stop" };

let online = false;
let videosLoaded = false;
let uploading = false;
let view = "overview";
let pollTimer = null;

/* ---------- helpers ---------- */
const esc = (v) => String(v ?? "—").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

function signal(tone) {
  const lit = { stop: "r", slow: "a", go: "g" }[tone];
  return `<span class="signal">${["r", "a", "g"].map((c) => `<i class="${c === lit ? c : ""}"></i>`).join("")}</span>`;
}

function readout(label, value, tone = "") {
  return `<div class="ro ${tone}"><div class="lbl">${esc(label)}</div><div class="v">${tone ? signal(tone) : ""}${esc(value)}</div></div>`;
}

async function api(path, opts = {}) {
  const res = await fetch(API + path, { ...opts, signal: AbortSignal.timeout(8000) });
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch { /* not json */ }
    throw new Error(detail);
  }
  return res.json();
}

function summaryRow(r) {
  return `<div class="row c3">${readout("Total vehicles crossed", r.total)}${readout("Currently visible", r.visible)}${readout("Traffic density", r.density, TONE[r.density])}</div>`;
}

function reportRows(r) {
  return [...(r.video ? [["Video", r.video]] : []), ["Total Vehicles Crossed", r.total], ["Currently Visible", r.visible], ["Traffic Density", r.density],
    ...TYPES.map(([k, name]) => [name, r[k]])];
}

function resultsBlock(r) {
  const max = Math.max(1, ...TYPES.map(([k]) => r[k] || 0));
  const bars = TYPES.map(([k, name]) =>
    `<div><span>${name}</span><i style="width:${((r[k] || 0) / max) * 100}%"></i><b>${esc(r[k])}</b></div>`).join("");
  const rows = reportRows(r).map(([m, v]) => `<tr><td>${m}</td><td>${esc(v)}</td></tr>`).join("");
  return `
    <div class="row c4">${TYPES.map(([k, name]) => readout(name, r[k])).join("")}</div>
    <div class="two">
      <div class="panel"><div class="ptitle"><span class="lbl"><b>Vehicles by type</b></span><span class="lbl">line crossings</span></div><div class="bars">${bars}</div></div>
      <div class="panel"><div class="ptitle"><span class="lbl"><b>Traffic report</b></span><span class="lbl">${esc(r.finished_at || "")}</span></div>
        <table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>${rows}</tbody></table></div>
    </div>
    <button class="btn ghostbtn dl" data-csv>Download CSV report</button>`;
}

function downloadCsv(r) {
  const cell = (v) => `"${String(v ?? "").replace(/"/g, '""')}"`;
  const csv = "Metric,Value\n" + reportRows(r).map(([m, v]) => `${cell(m)},${cell(v)}`).join("\n") + "\n";
  const a = Object.assign(document.createElement("a"), {
    href: URL.createObjectURL(new Blob([csv], { type: "text/csv" })), download: "traffic_report.csv",
  });
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

document.addEventListener("click", (e) => {
  if (e.target.closest("[data-csv]")) {
    const r = lastReport();
    if (r) downloadCsv(r);
  }
});

/* ---------- report cache (survives backend restarts) ---------- */
function lastReport() {
  try { return JSON.parse(store.get("report") || "null"); } catch { return null; }
}
function saveReport(r) { if (r) store.set("report", JSON.stringify(r)); }

async function refreshReport() {
  if (online) {
    try { const { report } = await api("/report"); if (report) saveReport(report); } catch { /* keep cache */ }
  }
  return lastReport();
}

/* ---------- backend status ---------- */
async function checkHealth() {
  const led = $("#led");
  try {
    const h = await api("/health");
    online = true;
    led.className = "led";
    led.textContent = "ENGINE READY";
    const s = { conf: h.conf.toFixed(2), imgsz: `${h.imgsz}px`, classes: h.classes.length, tracker: h.tracker };
    $$("#settings b").forEach((b) => { b.textContent = s[b.dataset.k]; });
    if (!videosLoaded) await loadVideos();
  } catch {
    online = false;
    led.className = "led off";
    led.textContent = "BACKEND OFFLINE";
  }
  updateButtons();
}

/* ---------- live pages ---------- */
let running = false;

function updateButtons() {
  if (!MODES[view]) return;
  $("#startBtn").disabled = !online || running;
  $("#stopBtn").disabled = !online || !running;
  const busy = !online || running || uploading;
  $("#videoSel").disabled = busy || !videosLoaded;
  $("#upBtn").classList.toggle("off", busy);
  $("#videoFile").disabled = busy;
  if (!online && !running) setMsg("Backend offline: start the server with  uvicorn server:app --port 8000  in traffic_monitoring/.", "err");
  else if ($("#liveMsg").classList.contains("err") && !running) setMsg("");
}

function setMsg(text, cls = "") {
  const m = $("#liveMsg");
  m.textContent = text;
  m.className = "msg " + cls;
}

function renderLiveReadouts(mode, stats = {}) {
  const box = $("#readouts");
  if (mode === "count") {
    box.innerHTML = readout("Total vehicles crossed", stats.total ?? 0) + readout("Currently visible", stats.visible ?? 0)
      + readout("Traffic density", stats.density ?? "—", TONE[stats.density]);
  } else {
    box.innerHTML = readout(MODES[mode].label, stats.count ?? 0);
  }
}

function resetFeed() {
  const img = $("#feedImg");
  img.removeAttribute("src");
  img.hidden = true;
  $("#nosig").hidden = false;
  $("#fps").textContent = "CAM 01";
}

function setupLive(mode) {
  const m = MODES[mode];
  $("#liveStep").textContent = m.step;
  $("#liveTitle").textContent = m.title;
  $("#liveSub").textContent = m.sub;
  $("#startBtn").textContent = m.btn;
  $("#feedTitle").textContent = `Live feed · ${m.title}`;
  $("#results").innerHTML = "";
  setMsg("");
  resetFeed();
  renderLiveReadouts(mode);
  updateButtons();
}

/* ---------- videos ---------- */

async function loadVideos(select) {
  const { videos, default: def } = await api("/videos");
  const want = select || store.get("video");
  const current = videos.includes(want) ? want : def;
  $("#videoSel").innerHTML = videos.map((v) =>
    `<option value="${esc(v)}"${v === current ? " selected" : ""}>${esc(v)}${v === def ? " (demo)" : ""}</option>`).join("");
  store.set("video", current);
  $("#liveVideo").textContent = current;
  videosLoaded = true;
  updateButtons();
}

function upload(file) {
  uploading = true;
  updateButtons();
  const msg = $("#upMsg");
  msg.className = "msg";
  const xhr = new XMLHttpRequest();  // XHR, not fetch, so we get upload progress
  xhr.open("POST", `${API}/videos?name=${encodeURIComponent(file.name)}`);
  xhr.upload.onprogress = (e) => {
    if (e.lengthComputable) msg.textContent = `Uploading ${file.name}… ${Math.round((e.loaded / e.total) * 100)}%`;
  };
  const done = (text, cls) => {
    uploading = false;
    $("#videoFile").value = "";
    msg.textContent = text;
    msg.className = "msg " + cls;
    updateButtons();
  };
  xhr.onload = async () => {
    let body = {};
    try { body = JSON.parse(xhr.responseText); } catch { /* not json */ }
    if (xhr.status !== 200) return done(`Upload failed: ${body.detail || xhr.statusText}`, "err");
    await loadVideos(body.name);
    done(`Uploaded ${body.name}. It is selected; press start to run it.`, "ok");
  };
  xhr.onerror = () => done("Upload failed: could not reach the server.", "err");
  msg.textContent = `Uploading ${file.name}…`;
  xhr.send(file);
}

async function start() {
  const mode = view;
  try {
    running = true;
    updateButtons();
    $("#results").innerHTML = "";
    setMsg("Loading model…");
    const video = $("#videoSel").value;
    await api(`/start/${mode}?video=${encodeURIComponent(video)}`, { method: "POST" });
    const img = $("#feedImg");
    img.src = `${API}/stream?t=${Date.now()}`;
    img.hidden = false;
    $("#nosig").hidden = true;
    poll(mode);
  } catch (e) {
    running = false;
    setMsg(`Could not start: ${e.message}`, "err");
    updateButtons();
  }
}

async function stop() {
  try { await api("/stop", { method: "POST" }); } catch { /* poll will notice */ }
}

function poll(mode) {
  clearTimeout(pollTimer);
  pollTimer = setTimeout(async () => {
    if (view !== mode) return;
    let s;
    try { s = await api("/stats"); } catch { pollTimer = null; poll(mode); return; }
    renderLiveReadouts(mode, s.stats);
    if (s.frames) {
      $("#fps").textContent = `${s.frames} frames · ${s.fps} fps`;
      setMsg(`Running ${MODES[mode].title.toLowerCase()} on ${s.video}…`);
    }
    if (s.running) return poll(mode);

    running = false;
    updateButtons();
    if (s.error) return setMsg(`${MODES[mode].title} failed: ${s.error}`, "err");
    if (s.stopped) return setMsg(`${MODES[mode].title} stopped after ${s.frames} frames.`);
    if (!s.frames) return setMsg("No video frames were processed.", "err");
    setMsg(`${MODES[mode].title} completed.`, "ok");
    if (mode === "count") {
      const r = await refreshReport();
      if (r) $("#results").innerHTML = resultsBlock(r);
    }
  }, 300);
}

/* ---------- routing ---------- */
async function route() {
  const next = (location.hash.slice(1) || "overview");
  view = CRUMBS[next] ? next : "overview";
  clearTimeout(pollTimer);
  $$(".nav a").forEach((a) => a.classList.toggle("on", a.dataset.view === view));
  $$(".view").forEach((v) => v.classList.toggle("on", v.dataset.view === (MODES[view] ? "live" : view)));
  $("#crumb").textContent = CRUMBS[view];
  window.scrollTo(0, 0);

  if (MODES[view]) {
    setupLive(view);
    // re-attach to a job already running on the backend (e.g. after a page refresh)
    if (online) {
      try {
        const s = await api("/stats");
        if (s.running && s.mode === view) {
          running = true;
          updateButtons();
          $("#feedImg").src = `${API}/stream?t=${Date.now()}`;
          $("#feedImg").hidden = false;
          $("#nosig").hidden = true;
          poll(view);
        }
      } catch { /* offline */ }
    }
  } else {
    running = false;
    const r = await refreshReport();
    const empty = (txt) => `<div class="note">${txt}</div>`;
    if (view === "overview") {
      $("#overviewReport").innerHTML = r ? summaryRow(r) : empty("No report yet. Run Traffic Analytics to generate your first traffic report.");
    } else if (view === "reports") {
      $("#reportBody").innerHTML = r ? summaryRow(r) + `<div style="margin-top:16px">${resultsBlock(r)}</div>`
        : empty("No report available yet. Run Traffic Analytics first.");
    }
  }
}

/* ---------- hero art ---------- */
function heroBoxes() {
  const boxes = [[188, 96, 30, 20, "CAR 0.91"], [246, 134, 58, 44, "TRUCK 0.87"], [116, 182, 30, 50, "MOTO 0.74"], [300, 190, 66, 50, "CAR 0.95"]];
  $("#heroBoxes").innerHTML = boxes.map(([x, y, w, h, lab]) => `
    <rect x="${x}" y="${y}" width="${w}" height="${h}" rx="4" fill="#262b31"/>
    <rect x="${x + w * 0.15}" y="${y + h * 0.15}" width="${w * 0.7}" height="${h * 0.25}" rx="2" fill="#4a525b"/>
    <rect x="${x - 4}" y="${y - 4}" width="${w + 8}" height="${h + 8}" fill="none" stroke="#f5a524" stroke-width="1.5"/>
    <rect x="${x - 4}" y="${y - 16}" width="${lab.length * 6 + 8}" height="12" fill="#f5a524"/>
    <text x="${x}" y="${y - 7}" fill="#17130a" font-size="8.5" font-family="monospace" font-weight="700">${lab}</text>`).join("");
}

function tick() {
  const now = new Date();
  $("#clock").textContent = now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", hour12: false });
  $("#date").textContent = now.toLocaleDateString([], { weekday: "short", day: "2-digit", month: "short", year: "numeric" });
}

/* ---------- init ---------- */
$("#startBtn").addEventListener("click", start);
$("#videoSel").addEventListener("change", (e) => {
  store.set("video", e.target.value);
  $("#liveVideo").textContent = e.target.value;
});
const source = $("#source");
["dragenter", "dragover"].forEach((t) => source.addEventListener(t, (e) => {
  e.preventDefault();
  if (!$("#videoFile").disabled) source.classList.add("drag");
}));
["dragleave", "drop"].forEach((t) => source.addEventListener(t, (e) => {
  e.preventDefault();
  source.classList.remove("drag");
  if (t === "drop" && !$("#videoFile").disabled && e.dataTransfer.files[0]) upload(e.dataTransfer.files[0]);
}));
$("#videoFile").addEventListener("change", (e) => { if (e.target.files[0]) upload(e.target.files[0]); });
$("#stopBtn").addEventListener("click", stop);
window.addEventListener("hashchange", route);

heroBoxes();
tick();
setInterval(tick, 10000);
checkHealth().then(route);
setInterval(checkHealth, 5000);
