// Full-Duplex-Bench-JA leaderboard: reads results.json (raw score-offline
// output) + meta.json, aggregates per model, and renders the tables.
"use strict";

const TASK_ORDER = ["pause", "smooth_tt"];  // bc は実験的タスクとして掲載しない (2026-09-29)

Promise.all([
  fetch("results.json").then((r) => r.json()),
  fetch("meta.json").then((r) => r.json()),
]).then(([results, meta]) => render(results, meta))
  .catch((e) => { document.getElementById("load-err").textContent =
    "結果の読み込みに失敗しました: " + e.message; });

function fmt(v, nd) {
  return (v === null || v === undefined || Number.isNaN(v))
    ? "—" : Number(v).toFixed(nd);
}
function pct(v) {
  return (v === null || v === undefined || Number.isNaN(v))
    ? "—" : (v * 100).toFixed(1) + "%";
}
// desired-behavior agreement rate from mean TO and the task's desired TO
function accOf(row) {
  if (row.TOR === null || row.TOR === undefined) return null;
  return row.desired_TO === 1 ? row.TOR : 1 - row.TOR;
}

function aggregate(summary) {
  const models = {};
  for (const r of summary) {
    const m = (models[r.model] ||= { name: r.model, domain: r.domain, tasks: {} });
    m.tasks[r.task] = {
      tor: r.TOR, acc: accOf(r), dkl: r.event_mean_dkl,
      score: r.score, n: r.n,
    };
  }
  for (const m of Object.values(models)) {
    const ts = Object.values(m.tasks);
    m.acc = mean(ts.map((t) => t.acc));
    m.dkl = mean(ts.map((t) => t.dkl));
    m.score = mean(ts.map((t) => t.score));
  }
  return models;
}
function mean(xs) {
  const v = xs.filter((x) => x !== null && x !== undefined && !Number.isNaN(x));
  return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null;
}

function render(results, meta) {
  const label = (m) => meta.model_labels[m] || m;
  document.getElementById("meta").textContent =
    `データ: ${meta.dataset} ・ 生成日: ${meta.generated}`;
  const repo = document.getElementById("repo");
  repo.href = meta.repo_url; repo.textContent = meta.repo_url.replace(/^https?:\/\//, "");

  const models = aggregate(results.summary || []);
  // primary ranking key = TOR agreement (higher = better). VAP-distance is experimental
  // and shown for reference only (2026-10-12).
  const ranked = Object.values(models)
    .sort((a, b) => (b.acc ?? -Infinity) - (a.acc ?? -Infinity));

  // ---- overall ranking ----
  const best = { acc: Math.max(...ranked.map((m) => m.acc ?? -Infinity)),
                 dkl: Math.min(...ranked.map((m) => m.dkl ?? Infinity)),
                 score: Math.max(...ranked.map((m) => m.score ?? -Infinity)) };
  const ov = document.getElementById("overall");
  ov.innerHTML =
    "<thead><tr><th class='rank'>#</th><th class='model'>モデル</th>" +
    "<th>適合率 ↑</th><th>VAP-distance ↓ <small>(実験的)</small></th>" +
    "<th>スコア ↑ <small>(実験的)</small></th></tr></thead>";
  const tb = document.createElement("tbody");
  ranked.forEach((m, i) => {
    tb.appendChild(rowEl([
      cell(String(i + 1), "rank"),
      modelCell(label(m.name), m.domain, meta, m.name),
      cell(pct(m.acc), m.acc === best.acc ? "best" : ""),
      cell(fmt(m.dkl, 3), m.dkl === best.dkl ? "best" : ""),
      cell(fmt(m.score, 3), m.score === best.score ? "best" : ""),
    ]));
  });
  ov.appendChild(tb);

  // ---- per task ----
  const host = document.getElementById("per-task");
  for (const task of TASK_ORDER) {
    const rows = (results.summary || []).filter((r) => r.task === task);
    if (!rows.length) continue;
    const h = document.createElement("h3");
    h.textContent = meta.task_labels[task] || task;
    host.appendChild(h);
    const wrap = document.createElement("div"); wrap.className = "table-wrap";
    const t = document.createElement("table");
    t.innerHTML = "<thead><tr><th class='model'>モデル</th><th>TOR</th>" +
      "<th>適合率 ↑</th><th>VAP-distance ↓ <small>(実験的)</small></th>" +
      "<th>スコア ↑ <small>(実験的)</small></th><th>AUC</th><th>n</th></tr></thead>";
    const body = document.createElement("tbody");
    const bDkl = Math.min(...rows.map((r) => r.event_mean_dkl ?? Infinity));
    const bAcc = Math.max(...rows.map((r) => accOf(r) ?? -Infinity));
    rows.sort((a, b) => (accOf(b) ?? -Infinity) - (accOf(a) ?? -Infinity))
      .forEach((r) => {
        body.appendChild(rowEl([
          modelCell(label(r.model), r.domain, meta, r.model),
          cell(fmt(r.TOR, 3)),
          cell(pct(accOf(r)), accOf(r) === bAcc ? "best" : ""),
          cell(fmt(r.event_mean_dkl, 3), r.event_mean_dkl === bDkl ? "best" : ""),
          cell(fmt(r.score, 3)),
          cell(fmt(r.AUC, 3)),
          cell(String(r.n)),
        ]));
      });
    t.appendChild(body); wrap.appendChild(t); host.appendChild(wrap);
  }

  // ---- audio samples ----
  renderSamples(meta);

  // ---- pending ----
  const pl = document.getElementById("pending-list");
  if (!(meta.pending_models || []).length) {
    document.getElementById("pending").style.display = "none";
  }
  (meta.pending_models || []).forEach((p) => {
    const li = document.createElement("li");
    const name = typeof p === "string" ? p : p.name;
    const url = typeof p === "string" ? null : p.url;
    if (url) {
      const a = document.createElement("a");
      a.href = url; a.textContent = name; a.target = "_blank"; a.rel = "noopener";
      li.appendChild(a);
    } else {
      li.textContent = name;
    }
    pl.appendChild(li);
  });
}

// ---- sample-audio waveforms -------------------------------------------
const CLIP_SEC = 20;           // all sample files share a 0-20 s time axis
let audioCtx = null;
let nowPlaying = null;         // {state, canvas, raf, co} currently playing
const waveRegistry = [];       // [{canvas, state}] for resize redraw
const stimByTask = {};         // task -> {canvas, state} of the stimulus row

function renderSamples(meta) {
  const host = document.getElementById("samples");
  const samples = meta.samples || {};
  if (!Object.keys(samples).length) {
    document.getElementById("samples-sec").style.display = "none";
    return;
  }
  for (const task of TASK_ORDER) {
    const s = samples[task];
    if (!s) continue;
    const h = document.createElement("h3");
    h.textContent = meta.task_labels[task] || task;
    host.appendChild(h);
    const p = document.createElement("p");
    p.className = "hint"; p.textContent = s.note || "";
    host.appendChild(p);

    const box = document.createElement("div");
    box.className = "wave-box";
    host.appendChild(box);

    const rows = [
      { key: "stimulus", label: "刺激音声 (話者A)", cls: "stim" },
      { key: "human", label: "人間B (正解)", cls: "human" },
      ...Object.entries(s.models || {})
        .sort((a, b) => (a[1].dkl ?? Infinity) - (b[1].dkl ?? Infinity))
        .map(([m, v]) => ({
          key: m, label: (meta.model_labels || {})[m] || m, cls: "model",
          badge: v,
        })),
    ];
    for (const r of rows) box.appendChild(sampleRow(task, r, s));
  }
}

function sampleRow(task, r, s) {
  const row = document.createElement("div");
  row.className = "wave-row " + r.cls;

  const lab = document.createElement("div");
  lab.className = "wave-label";
  lab.textContent = r.label;
  if (r.badge) {
    const ok = r.badge.TO === s.desired_TO;
    const b = document.createElement("span");
    b.className = "wave-badge " + (ok ? "ok" : "ng");
    b.textContent = `TO=${r.badge.TO}${ok ? " ✓" : " ✗"} D=${r.badge.dkl.toFixed(2)}`;
    b.title = "TO: ターンを取ったか(✓=望ましい挙動) / D: このイベントの VAP-distance";
    lab.appendChild(document.createElement("br"));
    lab.appendChild(b);
  }
  row.appendChild(lab);

  const cwrap = document.createElement("div");
  cwrap.className = "wave-canvas-wrap";
  const canvas = document.createElement("canvas");
  canvas.className = "wave-canvas";
  cwrap.appendChild(canvas);
  row.appendChild(cwrap);

  const url = `samples/${task}_${r.key}.mp3`;
  const state = { peaks: null, audio: null, audios: {}, window: s.window,
                  marker: s.marker, silent: false, row };
  if (r.key === "stimulus") stimByTask[task] = { canvas, state };

  loadPeaks(url).then(({ peaks, silent }) => {
    state.peaks = peaks;
    state.silent = silent;
    waveRegistry.push({ canvas, state });
    // a model that returned no audio at all is a failed generation, not a
    // deliberate silence: say so instead of drawing a flat line
    if (silent && r.cls === "model") {
      const b = document.createElement("span");
      b.className = "wave-badge none";
      b.textContent = "無出力";
      b.title = "応答音声が全区間ゼロ(モデルが何も出力しなかった)";
      lab.appendChild(document.createElement("br"));
      lab.appendChild(b);
    }
    drawWave(canvas, state, 0);
  }).catch(() => { canvas.classList.add("wave-missing"); });

  canvas.addEventListener("click", () => {
    if (!state.peaks) return;
    if (nowPlaying && nowPlaying.state === state) { stopPlayback(); return; }
    stopPlayback();
    // stimulus row always plays solo; others play the stereo mix
    // (left = speaker A, right = response) unless the toggle is off
    const mixOn = document.getElementById("mix-toggle")?.checked;
    const playUrl = (r.key === "stimulus" || !mixOn)
      ? url : `samples/${task}_${r.key}_mix.mp3`;
    if (!state.audios[playUrl]) {
      const a = new Audio(playUrl);
      a.addEventListener("ended", () => stopPlayback());
      state.audios[playUrl] = a;
    }
    state.audio = state.audios[playUrl];
    state.audio.currentTime = 0;
    state.audio.play();
    row.classList.add("playing");
    // in mix mode the stimulus track is audible too: mirror the playhead
    // on the stimulus row's waveform
    const co = (playUrl.endsWith("_mix.mp3") && stimByTask[task]) || null;
    if (co && co.state.peaks) co.state.row.classList.add("playing");
    nowPlaying = { state, canvas, raf: 0, co };
    const tick = () => {
      if (!nowPlaying || nowPlaying.state !== state) return;
      const p = state.audio.currentTime / CLIP_SEC;
      drawWave(canvas, state, p);
      if (co && co.state.peaks) drawWave(co.canvas, co.state, p);
      nowPlaying.raf = requestAnimationFrame(tick);
    };
    tick();
  });
  return row;
}

function stopPlayback() {
  if (!nowPlaying) return;
  const { state, canvas, raf, co } = nowPlaying;
  cancelAnimationFrame(raf);
  if (state.audio) state.audio.pause();
  state.row.classList.remove("playing");
  nowPlaying = null;
  drawWave(canvas, state, 0);
  if (co && co.state.peaks) {
    co.state.row.classList.remove("playing");
    drawWave(co.canvas, co.state, 0);
  }
}

async function loadPeaks(url) {
  audioCtx ||= new (window.AudioContext || window.webkitAudioContext)();
  const buf = await fetch(url).then((r) => {
    if (!r.ok) throw new Error(String(r.status));
    return r.arrayBuffer();
  });
  const audio = await audioCtx.decodeAudioData(buf);
  const data = audio.getChannelData(0);
  const nBuckets = 600;
  // bucket by absolute time so a shorter file still aligns on the 20 s axis
  const perBucket = Math.floor(audio.sampleRate * CLIP_SEC / nBuckets);
  const peaks = new Float32Array(nBuckets);
  for (let i = 0; i < nBuckets; i++) {
    let mx = 0;
    const s0 = i * perBucket, s1 = Math.min(s0 + perBucket, data.length);
    for (let j = s0; j < s1; j++) { const v = Math.abs(data[j]); if (v > mx) mx = v; }
    peaks[i] = mx;
  }
  const silent = !peaks.some((v) => v > 0);
  const top = Math.max(0.08, ...peaks);
  for (let i = 0; i < nBuckets; i++) peaks[i] /= top;
  return { peaks, silent };
}

function drawWave(canvas, state, progress) {
  const cssW = canvas.parentElement.clientWidth || 600;
  const cssH = 52;
  const dpr = window.devicePixelRatio || 1;
  if (canvas.width !== cssW * dpr) {
    canvas.width = cssW * dpr; canvas.height = cssH * dpr;
    canvas.style.width = cssW + "px"; canvas.style.height = cssH + "px";
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssW, cssH);

  const css = getComputedStyle(document.documentElement);
  // event window shading
  if (state.window) {
    const x0 = state.window[0] / CLIP_SEC * cssW;
    const x1 = state.window[1] / CLIP_SEC * cssW;
    ctx.fillStyle = "rgba(37, 99, 235, 0.13)";
    ctx.fillRect(x0, 0, Math.max(x1 - x0, 2), cssH);
    ctx.fillStyle = "rgba(37, 99, 235, 0.45)";
    ctx.fillRect(x0, 0, 1, cssH);
    ctx.fillRect(x1, 0, 1, cssH);
  }
  // point-in-time marker (e.g. the human backchannel inside a bc window)
  if (state.marker) {
    const xm = state.marker[0] / CLIP_SEC * cssW;
    ctx.fillStyle = "rgba(217, 119, 6, 0.85)";
    ctx.fillRect(xm, 0, 1.5, cssH);
  }
  // waveform bars
  const peaks = state.peaks;
  const mid = cssH / 2;
  const n = peaks.length;
  const bw = cssW / n;
  const played = progress > 0 ? progress * cssW : -1;
  for (let i = 0; i < n; i++) {
    const x = i * bw;
    const h = Math.max(peaks[i] * (cssH - 6), 1);
    ctx.fillStyle = x <= played ? (css.getPropertyValue("--accent").trim() || "#2563eb")
                                : "rgba(100, 116, 139, 0.55)";
    ctx.fillRect(x, mid - h / 2, Math.max(bw - 0.6, 0.6), h);
  }
  // playhead
  if (progress > 0) {
    ctx.fillStyle = css.getPropertyValue("--accent").trim() || "#2563eb";
    ctx.fillRect(progress * cssW, 0, 1.5, cssH);
  }
  // "no audio at all" is easy to misread as a quiet response: label it
  if (state.silent) {
    ctx.fillStyle = "rgba(100, 116, 139, .85)";
    ctx.font = "11px sans-serif";
    ctx.fillText("応答音声なし(無出力)", 8, mid + 4);
  }
  // time ticks every 5 s
  ctx.fillStyle = "rgba(100,116,139,.7)";
  ctx.font = "9px sans-serif";
  for (let t = 5; t < CLIP_SEC; t += 5) {
    const x = t / CLIP_SEC * cssW;
    ctx.fillRect(x, cssH - 5, 1, 5);
    ctx.fillText(t + "s", x + 3, cssH - 2);
  }
}

window.addEventListener("resize", () => {
  for (const { canvas, state } of waveRegistry) {
    if (nowPlaying && nowPlaying.state === state) continue; // rAF handles it
    drawWave(canvas, state, 0);
  }
});

function rowEl(cells) { const tr = document.createElement("tr");
  cells.forEach((c) => tr.appendChild(c)); return tr; }
function cell(text, cls) {
  const td = document.createElement("td");
  if (cls) td.className = cls; td.textContent = text; return td;
}
function modelCell(name, domain, meta, key) {
  const td = document.createElement("td"); td.className = "model";
  const url = (meta.model_urls || {})[key];
  if (url) {
    const a = document.createElement("a");
    a.href = url; a.textContent = name;
    a.target = "_blank"; a.rel = "noopener";
    td.appendChild(a);
  } else {
    td.appendChild(document.createTextNode(name));
  }
  const b = document.createElement("span");
  b.className = "dom " + (domain === "in" ? "in" : "out");
  b.textContent = domain === "in" ? "in-domain" : "out-of-domain";
  b.title = (meta.domain_note || {})[domain] || "";
  td.appendChild(b);
  if (domain === "in") {
    const n = document.createElement("span");
    n.className = "note"; n.textContent = " ※参考値";
    td.appendChild(n);
  }
  return td;
}
