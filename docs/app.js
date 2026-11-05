// Full-Duplex-Bench-JA leaderboard: reads results.json + meta.json (+ the
// listening-test summary human_eval.json) and draws bar charts, tables and
// the audio samples. Plain JS + inline SVG, no libraries.
"use strict";

const TASK_ORDER = ["pause", "smooth_tt"];   // scored tasks
// bc (相槌) is not scored: its sample is shown as a definition example only
const SAMPLE_TASKS = ["pause", "smooth_tt", "bc"];
const UNSCORED = new Set(["bc"]);
const COMMERCIAL = new Set(["gemini", "gpt_realtime", "grok_voice"]);
const SHORT = { gpt_realtime: "GPT-Realtime", gemini: "Gemini 3.1 Flash Live",
  grok_voice: "Grok Voice 3.0" };
const SVGNS = "http://www.w3.org/2000/svg";

const params = new URLSearchParams(location.search);
const state = {
  ov: pick(params.get("ov"), ["acc", "dkl"], "acc"),
  view: pick(params.get("view"), ["pause", "197", "72"], "pause"),
  metric: pick(params.get("metric"), ["acc", "dkl"], "acc"),
  rank: pick(params.get("rank"), ["pause", "smooth_tt"], "pause"),
};
function pick(v, ok, d) { return ok.includes(v) ? v : d; }

Promise.all([
  fetch("results.json").then((r) => r.json()),
  fetch("meta.json").then((r) => r.json()),
  fetch("human_eval.json").then((r) => r.json()).catch(() => null),
]).then(([results, meta, human]) => render(results, meta, human))
  .catch((e) => { document.getElementById("load-err").textContent =
    "結果の読み込みに失敗しました: " + e.message; });

function isNum(v) { return v !== null && v !== undefined && !Number.isNaN(v); }
function fmt(v, nd) { return isNum(v) ? Number(v).toFixed(nd) : "—"; }
function pct(v) { return isNum(v) ? (v * 100).toFixed(1) + "%" : "—"; }
function ci(lo, hi, nd) {
  return isNum(lo) && isNum(hi) ? `[${fmt(lo, nd)}, ${fmt(hi, nd)}]` : "";
}
function pctCi(lo, hi) {
  return isNum(lo) && isNum(hi)
    ? `[${(lo * 100).toFixed(1)}, ${(hi * 100).toFixed(1)}]` : "";
}
// desired-behavior agreement rate from mean TO and the task's desired TO
function accOf(row) {
  if (!isNum(row.TOR)) return null;
  return row.desired_TO === 1 ? row.TOR : 1 - row.TOR;
}
function accCiOf(row) {
  const c = row.TOR_ci || [];
  if (!isNum(c[0]) || !isNum(c[1])) return [null, null];
  return row.desired_TO === 1 ? [c[0], c[1]] : [1 - c[1], 1 - c[0]];
}
const isBaseline = (d) => d === "baseline";

function aggregate(summary) {
  const models = {};
  for (const r of summary) {
    const m = (models[r.model] ||= { name: r.model, domain: r.domain, tasks: {} });
    m.tasks[r.task] = { acc: accOf(r), dkl: r.event_mean_dkl };
  }
  for (const m of Object.values(models)) {
    const ts = TASK_ORDER.map((t) => m.tasks[t] || {});
    m.acc = mean(ts.map((t) => t.acc));
    m.dkl = mean(ts.map((t) => t.dkl));
  }
  return models;
}
function mean(xs) {
  const v = xs.filter(isNum);
  return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null;
}
// systems first (best first), trivial baselines last and unranked
function order(list, key, asc) {
  const k = (x) => key(x) ?? (asc ? Infinity : -Infinity);
  const sys = list.filter((x) => !isBaseline(x.domain))
    .sort((a, b) => asc ? k(a) - k(b) : k(b) - k(a));
  const base = list.filter((x) => isBaseline(x.domain))
    .sort((a, b) => asc ? k(a) - k(b) : k(b) - k(a));
  return [...sys, ...base];
}

// ---- segmented toggles ----------------------------------------------------
const redraws = [];
function bindSeg(id, key, onChange) {
  const host = document.getElementById(id);
  const btns = [...host.querySelectorAll("button")];
  const sync = () => btns.forEach((b) => {
    const on = b.dataset.v === state[key];
    b.classList.toggle("on", on); b.setAttribute("aria-pressed", String(on));
  });
  btns.forEach((b) => b.addEventListener("click", () => {
    state[key] = b.dataset.v; sync(); onChange();
    const q = new URLSearchParams(location.search); q.set(key, state[key]);
    history.replaceState(null, "", "?" + q.toString() + location.hash);
  }));
  sync(); onChange(); redraws.push(onChange);
}
let resizeT = 0;
window.addEventListener("resize", () => {
  clearTimeout(resizeT);
  resizeT = setTimeout(() => redraws.forEach((f) => f()), 120);
});

// ---- tooltip ----------------------------------------------------------------
const tip = () => document.getElementById("tip");
function showTip(ev, html) {
  const t = tip(); t.innerHTML = html; t.hidden = false;
  const pad = 12, w = t.offsetWidth, h = t.offsetHeight;
  let x = ev.clientX + pad, y = ev.clientY + pad;
  if (x + w > window.innerWidth - 8) x = Math.max(8, ev.clientX - w - pad);
  if (y + h > window.innerHeight - 8) y = Math.max(8, ev.clientY - h - pad);
  t.style.left = x + "px"; t.style.top = y + "px";
}
function hideTip() { tip().hidden = true; }
function hoverable(el, html) {
  el.addEventListener("pointerenter", (e) => showTip(e, html));
  el.addEventListener("pointermove", (e) => showTip(e, html));
  el.addEventListener("pointerleave", hideTip);
}
document.addEventListener("scroll", hideTip, { passive: true });

// ---- svg helpers ------------------------------------------------------------
function svgEl(tag, attrs, parent) {
  const e = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs || {})) if (v !== null && v !== undefined) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}
function svgText(parent, x, y, str, attrs) {
  const t = svgEl("text", { x, y, ...attrs }, parent); t.textContent = str; return t;
}
function defs(svg) {
  const d = svgEl("defs", {}, svg);
  const p = svgEl("pattern", { id: "hatch", width: 6, height: 6,
    patternUnits: "userSpaceOnUse", patternTransform: "rotate(45)" }, d);
  svgEl("rect", { width: 6, height: 6, fill: "var(--prop-bg)" }, p);
  svgEl("line", { x1: 0, y1: 0, x2: 0, y2: 6, stroke: "var(--accent)",
    "stroke-width": 2.2 }, p);
}

// Horizontal bar chart, label line above each bar so it fits a phone.
// items: {label, tag, v, lo, hi, kind: sys|base|prop, valText, tip} or {sep}
// opt: {min, max, ref, ticks:[..], tickFmt, zero}
function innerW(host) {
  const cs = getComputedStyle(host);
  return host.clientWidth - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight);
}
function hbar(host, items, opt) {
  host.innerHTML = "";
  const W = Math.max(260, innerW(host) || 600);
  const rowH = 42, sepH = 26, top = 4, axisH = 22;
  const H = top + items.reduce((s, it) => s + (it.sep ? sepH : rowH), 0) + axisH;
  const svg = svgEl("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`,
    role: "img", "aria-label": opt.aria || "棒グラフ" }, host);
  defs(svg);
  const L = 2, R = W - 6;
  const x = (v) => L + (Math.max(opt.min, Math.min(opt.max, v)) - opt.min)
    / (opt.max - opt.min) * (R - L);
  const plotTop = top, plotBot = H - axisH;
  // grid + ticks
  const g = svgEl("g", { class: "axis" }, svg);
  for (const t of opt.ticks) {
    svgEl("line", { x1: x(t), x2: x(t), y1: plotTop, y2: plotBot, class: "grid" }, g);
    const anchor = t === opt.ticks[0] ? "start" : t === opt.ticks.at(-1) ? "end" : "middle";
    svgText(g, x(t), H - 6, opt.tickFmt(t), { "text-anchor": anchor, class: "tick" });
  }
  let y = top;
  const zeroX = x(isNum(opt.zero) ? opt.zero : opt.min);
  for (const it of items) {
    if (it.sep) {
      svgText(svg, L, y + 18, it.sep, { class: "sep" });
      y += sepH; continue;
    }
    const grp = svgEl("g", { class: "row " + (it.kind || "sys") }, svg);
    const lab = svgText(grp, L, y + 15, "", { class: "lab" });
    const ts = svgEl("tspan", {}, lab); ts.textContent = it.label;
    if (it.tag) {
      const t2 = svgEl("tspan", { class: "lab-tag " + (it.tagCls || ""), dx: 6 }, lab);
      t2.textContent = it.tag;
    }
    svgText(grp, R, y + 15, it.valText || "", { class: "val", "text-anchor": "end" });
    const by = y + 22, bh = 11;
    if (isNum(it.v)) {
      const x0 = Math.min(zeroX, x(it.v)), x1 = Math.max(zeroX, x(it.v));
      svgEl("rect", { x: x0, y: by, width: Math.max(x1 - x0, 1.5), height: bh, rx: 2,
        class: "bar" }, grp);
      if (isNum(it.lo) && isNum(it.hi)) {
        const cy = by + bh / 2;
        svgEl("line", { x1: x(it.lo), x2: x(it.hi), y1: cy, y2: cy, class: "whisk" }, grp);
        for (const e of [it.lo, it.hi]) {
          svgEl("line", { x1: x(e), x2: x(e), y1: cy - 5, y2: cy + 5, class: "whisk" }, grp);
        }
      }
    }
    const hit = svgEl("rect", { x: 0, y, width: W, height: rowH, class: "hit" }, grp);
    if (it.tip) hoverable(hit, it.tip);
    y += rowH;
  }
  if (isNum(opt.ref)) {
    svgEl("line", { x1: x(opt.ref), x2: x(opt.ref), y1: plotTop, y2: plotBot,
      class: "ref" }, svg);
  }
  if (isNum(opt.zero)) {
    svgEl("line", { x1: zeroX, x2: zeroX, y1: plotTop, y2: plotBot, class: "zero" }, svg);
  }
}

function render(results, meta, human) {
  const label = (m) => meta.model_labels[m] || m;
  const shortL = (m) => SHORT[m] || label(m);
  const sc = meta.scoring || {};
  document.getElementById("meta").textContent =
    `データ: ${meta.dataset} ・ 更新 ${meta.generated}`;
  const repo = document.getElementById("repo");
  repo.href = meta.repo_url; repo.textContent = meta.repo_url.replace(/^https?:\/\//, "");
  const tag = (d) => d === "in" ? "in-domain※" : "";

  const summary = results.summary || [];
  const models = aggregate(summary);

  // ---- overall: bar chart + table ----
  bindSeg("ov-metric", "ov", () => {
    const acc = state.ov === "acc";
    document.getElementById("ov-hint").textContent = acc
      ? "2 課題(Pause・Smooth TT)の望ましい振る舞いの割合の平均。長いほど良い。"
      : "2 課題の VAP-distance の平均(実験的指標、参考値)。短いほど人間の応答に近い。";
    const list = order(Object.values(models), (m) => acc ? m.acc : m.dkl, !acc);
    const items = []; let rank = 0, sepDone = false;
    for (const m of list) {
      const base = isBaseline(m.domain);
      if (base && !sepDone) { items.push({ sep: "自明なベースライン(順位外)" }); sepDone = true; }
      const v = acc ? m.acc : m.dkl;
      const p = m.tasks.pause || {}, s = m.tasks.smooth_tt || {};
      items.push({
        label: (base ? "" : `${++rank}. `) + shortL(m.name), tag: tag(m.domain),
        tagCls: "in", kind: base ? "base" : "sys", v,
        valText: acc ? pct(v) : fmt(v, 3),
        tip: `<b>${label(m.name)}</b><br>` + (acc
          ? `総合 ${pct(m.acc)}<br>Pause ${pct(p.acc)} / Smooth TT ${pct(s.acc)}`
          : `総合 ${fmt(m.dkl, 3)}<br>Pause ${fmt(p.dkl, 3)} / Smooth TT ${fmt(s.dkl, 3)}`),
      });
    }
    const max = acc ? 1 : Math.ceil(Math.max(...list.map((m) => m.dkl)) * 2) / 2;
    hbar(document.getElementById("overall-chart"), items, acc
      ? { min: 0, max: 1, ticks: [0, .25, .5, .75, 1], tickFmt: (t) => t * 100 + "%" }
      : { min: 0, max, ticks: range(0, max, 0.5), tickFmt: (t) => t.toFixed(1) });
  });
  overallTable(models, meta, label);

  // ---- per task: one chart, toggled ----
  const tt = (results.turn_transition_only || []).filter((r) => r.task === "smooth_tt");
  bindSeg("task-view", "view", drawTask);
  bindSeg("task-metric", "metric", drawTask);
  function drawTask() {
    const rows = state.view === "pause" ? summary.filter((r) => r.task === "pause")
      : state.view === "72" ? tt : summary.filter((r) => r.task === "smooth_tt");
    const acc = state.metric === "acc";
    const n = rows[0] ? rows[0].n : "";
    const hints = {
      pause: "間でターンを取らなかった割合(= 1 − TOR)。黙るほど高くなる定義で、常時沈黙が 100% になります。人の評価とは逆を向きます(下の検証)。",
      197: "話者 A のターンの終わりから 2 秒以内に話し始めた割合(= TOR)。",
      72: "Smooth TT のうち本当の話者交替の 72 事象に限った値(A が相槌主体の発話、A の終わりの直前に B が話している、B の応答が相槌の連なり、のいずれかを除く)。",
    };
    document.getElementById("task-hint").textContent = (acc ? hints[state.view]
      : "VAP-distance(実験的指標、参考値)。人間の音声活動予測分布との距離で、短いほど人間の応答に近い。")
      + ` ${n} 事象。`;
    const list = order(rows.map((r) => ({ ...r })), (r) => acc ? accOf(r) : r.event_mean_dkl, !acc);
    const items = []; let rank = 0, sepDone = false;
    for (const r of list) {
      const base = isBaseline(r.domain);
      if (base && !sepDone) { items.push({ sep: "自明なベースライン(順位外)" }); sepDone = true; }
      const [alo, ahi] = accCiOf(r); const dc = r.dkl_ci || [];
      const v = acc ? accOf(r) : r.event_mean_dkl;
      const lo = acc ? alo : dc[0], hi = acc ? ahi : dc[1];
      items.push({
        label: (base ? "" : `${++rank}. `) + shortL(r.model), tag: tag(r.domain), tagCls: "in",
        kind: base ? "base" : "sys", v, lo, hi,
        valText: acc ? pct(v) : fmt(v, 3),
        tip: `<b>${label(r.model)}</b><br>` + (acc
          ? `望ましい振る舞い ${pct(v)} ${pctCi(lo, hi)}<br>TOR ${fmt(r.TOR, 3)}`
          : `VAP-distance ${fmt(v, 3)} ${ci(lo, hi, 3)}`) + `<br>n = ${r.n}`,
      });
    }
    const mx = Math.ceil(Math.max(...list.map((r) => (r.dkl_ci || [])[1] ?? r.event_mean_dkl)) * 2) / 2;
    hbar(document.getElementById("task-chart"), items, acc
      ? { min: 0, max: 1, ticks: [0, .25, .5, .75, 1], tickFmt: (t) => t * 100 + "%" }
      : { min: 0, max: mx, ticks: range(0, mx, 0.5), tickFmt: (t) => t.toFixed(1) });
    const tbl = document.getElementById("task-table"); tbl.innerHTML = "";
    tbl.appendChild(taskTable(rows, meta, label));
  }

  // ---- notes ----
  setText("tb-note", meta.transcript_boundaries_note);
  setText("tt-note", meta.turn_transition_note
    ? meta.turn_transition_note.replace(/と次のとおり。?$/, "した値。上の「課題別」で切り替えて表示します。") : "");
  const vck = meta.vap_checkpoint || {};
  setText("vap-note", `VAP: ${sc.vap || "—"}。リーダーボードの重み: ${vck.leaderboard || results.vap_checkpoint || "—"}。` +
    ` 95% 区間: ${sc.ci || results.ci || "—"}。再採点: ${sc.rescored || results.rescored || "—"}。`);

  // ---- human validation ----
  if (human) renderHuman(human, label, shortL);
  else document.getElementById("validation-sec").style.display = "none";

  // ---- audio samples ----
  const sn = document.getElementById("samples-note");
  if (sn && meta.samples_note) sn.textContent = "※ " + meta.samples_note;
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
function setText(id, s) { const e = document.getElementById(id); if (e) e.textContent = s || ""; }
function range(a, b, step) { const o = []; for (let v = a; v <= b + 1e-9; v += step) o.push(+v.toFixed(6)); return o; }

// ---- human validation ---------------------------------------------------------
function renderHuman(h, label, shortL) {
  const p = h.panel;
  setText("panel-line", `クラウドソーシングで ${p.submitted} 人が、同じ事象に対する 2 システムの応答を聞き比べて` +
    `自然な方を選びました(${p.design}、${p.systems} システム、品質確認を通った ${p.analyzed} 人で解析)。`);
  const stats = document.getElementById("panel-stats");
  stats.innerHTML = [
    [p.agreement_one_vs_rest.toFixed(2), "人どうしの一致", "1 票と残りの多数派"],
    [p.fleiss_kappa.toFixed(2), "Fleiss κ", "評価者間の一致"],
    [`${p.analyzed}<small> / ${p.submitted} 人</small>`, "主解析の人数", "品質確認を通過"],
  ].map(([v, a, b]) => `<div class="stat"><div class="stat-v">${v}</div>` +
    `<div class="stat-a">${a}</div><div class="stat-b">${b}</div></div>`).join("");

  bindSeg("rank-task", "rank", () => drawSlope(h, label, shortL));

  document.getElementById("headline").innerHTML = h.headline.map((x) =>
    `<div class="big-item"><div class="big-v">${x.value}</div><div class="big-t">${x.text}</div></div>`).join("");
  setText("headline-scope", h.headline_scope);

  setText("agree-scope", h.agreement.scope + "。人の多数派が選んだ側を指標も良いとした割合。");
  const drawAgree = () => {
    const items = [];
    for (const task of ["pause", "smooth_tt"]) {
      items.push({ sep: task === "pause" ? "Pause" : "Smooth TT" });
      for (const r of h.agreement.rows.filter((r) => r.task === task)) {
        const c = r.ci || [];
        items.push({ label: r.label, tag: r.proposed ? "検証中" : "", tagCls: "prop",
          kind: r.proposed ? "prop" : "sys", v: r.v, lo: c[0], hi: c[1],
          valText: r.v.toFixed(2),
          tip: `<b>${r.label}</b><br>一致率 ${r.v.toFixed(3)}` +
            (c.length ? ` [${c[0].toFixed(2)}–${c[1].toFixed(2)}]` : "") + `<br>${r.n} 対` +
            (r.proposed ? "<br>検証中・ツールキット未実装" : "") });
      }
    }
    hbar(document.getElementById("agree-chart"), items,
      { min: 0, max: 1, ref: 0.5, ticks: [0, .25, .5, .75, 1], tickFmt: (t) => t.toFixed(2),
        aria: "指標と人の多数派の一致率" });
  };
  drawAgree(); redraws.push(drawAgree);

  setText("rho-scope", h.rho.scope + "。+1 に近いほど人と同じ順に並べる。");
  const drawRho = () => {
    hbar(document.getElementById("rho-chart"), h.rho.rows.map((r) => ({
      label: r.label, tag: r.proposed ? "検証中" : "", tagCls: "prop",
      kind: r.proposed ? "prop" : "sys", v: r.v,
      valText: (r.v > 0 ? "+" : r.v < 0 ? "−" : "") + Math.abs(r.v).toFixed(2),
      tip: `<b>${r.label}</b><br>ρ = ${r.v.toFixed(2)}` })),
      { min: -1, max: 1, zero: 0, ticks: [-1, -0.5, 0, 0.5, 1],
        tickFmt: (t) => (t > 0 ? "+" : t < 0 ? "−" : "") + Math.abs(t), aria: "順位相関" });
  };
  drawRho(); redraws.push(drawRho);
  setText("proposed-note", "※ " + h.proposed_note);
  document.getElementById("caveats").innerHTML =
    h.caveats.map((c) => `<li>${c}</li>`).join("");
}

// two rank columns joined by lines: same system, left = people, right = TOR
function drawSlope(h, label, shortL) {
  const d = h.rank[state.rank];
  setText("rank-hint", state.rank === "pause"
    ? `左が人の評価の順位、右が TOR(望ましい振る舞いの割合)の順位。線が交差するほど食い違う。Pause では、間で短く反応したモデルを人は上に、黙り続けた商用 API を常時沈黙より下に置き、TOR はほぼ逆順になった(9 システムの順位相関 ρ = ${d.rho9.toFixed(2)})。`
    : `Smooth TT では人の順位と TOR がおおむね揃う(9 システムの順位相関 ρ = +${d.rho9.toFixed(2)})。`);
  const host = document.getElementById("rank-chart"); host.innerHTML = "";
  const W = Math.max(280, innerW(host) || 600);
  const n = d.human.length, rowH = 36, head = 40, H = head + n * rowH + 4;
  const lw = Math.min(190, Math.floor(W * 0.36));
  const x1 = lw + 4, x2 = W - lw - 4;
  const svg = svgEl("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img",
    "aria-label": "人の順位と TOR の順位" }, host);
  svgText(svg, x1 - 10, 15, "人の順位", { class: "col-h", "text-anchor": "end" });
  svgText(svg, x1 - 10, 30, d.human_label.replace(/^人の評価/, ""), { class: "col-s", "text-anchor": "end" });
  svgText(svg, x2 + 10, 15, "TOR の順位", { class: "col-h" });
  svgText(svg, x2 + 10, 30, d.metric_label.replace(/^望ましい振る舞いの割合/, ""), { class: "col-s" });
  const yOf = (i) => head + i * rowH + rowH / 2;
  const posR = Object.fromEntries(d.metric.map((r, i) => [r.model, i]));
  const valR = Object.fromEntries(d.metric.map((r) => [r.model, r.v]));
  const groups = [];
  d.human.forEach((r, i) => {
    const j = posR[r.model];
    const kind = r.model.startsWith("baseline") ? "base" : COMMERCIAL.has(r.model) ? "com" : "sys";
    const g = svgEl("g", { class: "slope " + kind }, svg);
    svgEl("line", { x1, y1: yOf(i), x2, y2: yOf(j), class: "sl" }, g);
    svgEl("circle", { cx: x1, cy: yOf(i), r: 4.5, class: "dot" }, g);
    svgEl("circle", { cx: x2, cy: yOf(j), r: 4.5, class: "dot" }, g);
    const nm = shortL(r.model).replace(" 3.1 Flash Live", "").replace(" 3.0", "");
    svgText(g, x1 - 10, yOf(i) - 1, `${i + 1}. ${nm}`, { class: "sn", "text-anchor": "end" });
    svgText(g, x1 - 10, yOf(i) + 12, (r.v > 0 ? "+" : "") + r.v.toFixed(2), { class: "sv", "text-anchor": "end" });
    svgText(g, x2 + 10, yOf(j) - 1, `${j + 1}. ${nm}`, { class: "sn" });
    svgText(g, x2 + 10, yOf(j) + 12, valR[r.model].toFixed(2), { class: "sv" });
    const hit = svgEl("rect", { x: 0, y: yOf(i) - rowH / 2, width: x1 + 6, height: rowH, class: "hit" }, g);
    const hit2 = svgEl("rect", { x: x2 - 6, y: yOf(j) - rowH / 2, width: W - x2 + 6, height: rowH, class: "hit" }, g);
    const html = `<b>${label(r.model)}</b><br>人の評価 ${i + 1} 位(${r.v.toFixed(2)} [${r.lo.toFixed(2)}, ${r.hi.toFixed(2)}])` +
      `<br>TOR の順位 ${j + 1} 位(${valR[r.model].toFixed(3)})`;
    for (const t of [hit, hit2]) {
      hoverable(t, html);
      t.addEventListener("pointerenter", () => { svg.classList.add("focus"); g.classList.add("hl"); });
      t.addEventListener("pointerleave", () => { svg.classList.remove("focus"); g.classList.remove("hl"); });
    }
    groups.push(g);
  });
}

function overallTable(models, meta, label) {
  const ranked = order(Object.values(models), (m) => m.acc);
  const sysOnly = ranked.filter((m) => !isBaseline(m.domain));
  const best = { acc: Math.max(...sysOnly.map((m) => m.acc ?? -Infinity)),
                 dkl: Math.min(...sysOnly.map((m) => m.dkl ?? Infinity)) };
  const ov = document.getElementById("overall");
  ov.innerHTML =
    "<thead><tr><th class='model'>モデル</th>" +
    "<th>望ましい振る舞い ↑</th>" +
    TASK_ORDER.map((t) => `<th>${shortTask(t)}</th>`).join("") +
    "<th>VAP-d ↓</th></tr></thead>";
  const tb = document.createElement("tbody");
  ranked.forEach((m) => {
    const base = isBaseline(m.domain);
    const tr = rowEl([
      modelCell(label(m.name), m.domain, meta, m.name),
      cell(pct(m.acc), !base && m.acc === best.acc ? "best" : ""),
      ...TASK_ORDER.map((t) => cell(pct((m.tasks[t] || {}).acc))),
      cell(fmt(m.dkl, 3), !base && m.dkl === best.dkl ? "best" : ""),
    ]);
    if (base) tr.className = "baseline";
    tb.appendChild(tr);
  });
  ov.appendChild(tb);
}

function shortTask(t) { return t === "pause" ? "Pause" : t === "smooth_tt" ? "Smooth TT" : t; }

function taskTable(rows, meta, label) {
  const wrap = document.createElement("div"); wrap.className = "table-wrap";
  const t = document.createElement("table");
  t.innerHTML = "<thead><tr><th class='model'>モデル</th><th>TOR</th>" +
    "<th>望ましい振る舞い ↑</th><th>VAP-d ↓</th></tr></thead>";
  const body = document.createElement("tbody");
  const sys = rows.filter((r) => !isBaseline(r.domain));
  const bDkl = Math.min(...sys.map((r) => r.event_mean_dkl ?? Infinity));
  const bAcc = Math.max(...sys.map((r) => accOf(r) ?? -Infinity));
  order(rows, accOf).forEach((r) => {
    const base = isBaseline(r.domain);
    const [alo, ahi] = accCiOf(r);
    const tc = r.TOR_ci || [], dc = r.dkl_ci || [];
    const tr = rowEl([
      modelCell(label(r.model), r.domain, meta, r.model),
      cellCi(fmt(r.TOR, 3), ci(tc[0], tc[1], 3)),
      cellCi(pct(accOf(r)), pctCi(alo, ahi), !base && accOf(r) === bAcc ? "best" : ""),
      cellCi(fmt(r.event_mean_dkl, 3), ci(dc[0], dc[1], 3),
             !base && r.event_mean_dkl === bDkl ? "best" : ""),
    ]);
    if (base) tr.className = "baseline";
    body.appendChild(tr);
  });
  t.appendChild(body); wrap.appendChild(t);
  const cap = document.createElement("p"); cap.className = "hint";
  cap.textContent = `n = ${rows[0] ? rows[0].n : "—"} 事象。括弧は 95% 区間。`;
  wrap.appendChild(cap);
  return wrap;
}

function cellCi(main, sub, cls) {
  const td = document.createElement("td");
  if (cls) td.className = cls;
  td.appendChild(document.createTextNode(main));
  if (sub) {
    const s = document.createElement("span");
    s.className = "ci"; s.textContent = " " + sub;
    td.appendChild(s);
  }
  return td;
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
  for (const task of SAMPLE_TASKS) {
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
          badge: UNSCORED.has(task) ? null : v,
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
  b.className = "dom " + (domain === "in" ? "in" : domain === "baseline" ? "base" : "out");
  b.textContent = domain === "in" ? "in-domain" : domain === "baseline" ? "自明なベースライン" : "out-of-domain";
  b.title = (meta.domain_note || {})[domain] || "";
  td.appendChild(b);
  if (domain === "in") {
    const n = document.createElement("span");
    n.className = "note"; n.textContent = " ※参考値";
    td.appendChild(n);
  }
  return td;
}
