// Full-Duplex-Bench-JA human-evaluation web UI.
// Loads manifest.json (next to this file), presents items one at a time,
// gates answering on full playback, and exports responses as JSON.

"use strict";

// NOTE: the task category (pause / smooth_tt / bc) is intentionally NOT shown
// to the annotator — revealing it would leak the expected answer (esp. Study A).
// It is kept in the manifest/responses for analysis only.

let manifest = null;
let items = [];          // ordered work list
let idx = 0;             // current item index
let responses = [];      // collected answers
let workerId = "";
let startedAt = "";
let cur = { listened: false, shownAt: 0, answer: {} };

const $ = (id) => document.getElementById(id);

// ---- seeded shuffle (so a worker's order is stable on reload) ----
function hashSeed(s) {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); }
  return h >>> 0;
}
function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function shuffle(arr, rnd) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rnd() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

// ---- load manifest ----
fetch("manifest.json")
  .then((r) => { if (!r.ok) throw new Error("manifest.json が読み込めません"); return r.json(); })
  .then((m) => {
    manifest = m;
    const nA = (m.study_a?.items || []).length;
    const nB = (m.study_b?.items || []).length;
    const cap = m.max_per_worker || (nA + nB);
    const shown = Math.min(cap, nA + nB);
    $("est-time").textContent = `約 ${Math.ceil(shown * 0.5)} 分(最大 ${shown} 項目)`;
    const notes = m.intro_notes || [];
    if (notes.length) {
      const ul = document.createElement("ul");
      notes.forEach((n) => {
        const li = document.createElement("li"); li.textContent = n;
        ul.appendChild(li);
      });
      $("intro-notes").appendChild(ul);
    }
  })
  .catch((e) => { $("load-err").textContent = e.message; });

$("start-btn").addEventListener("click", () => {
  workerId = $("worker-id").value.trim();
  if (!workerId) { $("load-err").textContent = "作業者 ID を入力してください"; return; }
  if (!manifest) { $("load-err").textContent = "データ未読み込みです"; return; }

  const which = $("study-select").value;
  let a = (which === "both" || which === "A") ? (manifest.study_a.items || []).map(x => ({ ...x, study: "A" })) : [];
  let b = (which === "both" || which === "B") ? (manifest.study_b.items || []).map(x => ({ ...x, study: "B" })) : [];
  const rnd = mulberry32(hashSeed(workerId));
  a = shuffle(a, rnd); b = shuffle(b, rnd);
  // cap items per worker (seeded subset); split proportionally when doing both
  const cap = manifest.max_per_worker || (a.length + b.length);
  if (a.length && b.length) {
    const nA = Math.max(1, Math.round(cap * a.length / (a.length + b.length)));
    a = a.slice(0, nA); b = b.slice(0, Math.max(0, cap - a.length));
  } else if (a.length) { a = a.slice(0, cap); }
  else { b = b.slice(0, cap); }
  items = a.concat(b);            // keep A before B
  if (!items.length) { $("load-err").textContent = "対象項目がありません"; return; }

  startedAt = new Date().toISOString();
  $("start").classList.add("hidden");
  $("task").classList.remove("hidden");
  idx = 0; responses = [];
  renderItem();
});

function renderItem() {
  const it = items[idx];
  cur = { listened: false, shownAt: Date.now(), answer: {} };

  const pct = Math.round((idx / items.length) * 100);
  $("progress-bar").style.width = pct + "%";
  $("progress-text").textContent = `${idx + 1} / ${items.length}`;

  const player = $("player");
  player.src = it.audio;
  player.load();
  $("play-hint").style.display = "block";
  $("next-btn").disabled = true;

  const q = $("question");
  q.innerHTML = "";
  if (it.study === "A") {
    setInstruction(manifest.study_a.question, manifest.study_a.breakdown);
    manifest.study_a.choices.forEach((c, i) => {
      q.appendChild(optionRow("a", i, c, (v) => { cur.answer.choice = v; checkReady(); }));
    });
  } else {
    setInstruction(manifest.study_b.q_main, manifest.study_b.breakdown);
    q.appendChild(likert(manifest.study_b.main_scale, (v) => { cur.answer.main = v; checkReady(); },
                         manifest.study_b.main_scale_labels));
    const sub = document.createElement("div");
    sub.className = "subq";
    const lab = document.createElement("p");
    lab.textContent = manifest.study_b.q_cat;
    lab.style.fontWeight = "600";
    sub.appendChild(lab);
    manifest.study_b.categories.forEach((c, i) => {
      sub.appendChild(optionRow("cat", i, c, (v) => { cur.answer.category = v; checkReady(); }));
    });
    q.appendChild(sub);
  }
}

function optionRow(group, value, text, onPick) {
  const lab = document.createElement("label");
  lab.className = "opt";
  const inp = document.createElement("input");
  inp.type = "radio"; inp.name = group; inp.value = value;
  inp.addEventListener("change", () => {
    [...document.getElementsByName(group)].forEach(r => r.parentElement.classList.remove("sel"));
    lab.classList.add("sel");
    onPick(value);
  });
  lab.appendChild(inp);
  lab.appendChild(document.createTextNode(text));
  return lab;
}

// question text + optional gray breakdown paragraph above it
function setInstruction(question, breakdown) {
  const el = $("instruction");
  el.innerHTML = "";
  if (breakdown) {
    const bd = document.createElement("span");
    bd.className = "breakdown";
    bd.textContent = breakdown;
    el.appendChild(bd);
  }
  const qs = document.createElement("span");
  qs.className = "q-main";
  qs.textContent = question;
  el.appendChild(qs);
}

function likert(endsText, onPick, labels) {
  const wrap = document.createElement("div");
  const row = document.createElement("div");
  row.className = "likert";
  for (let v = 1; v <= 5; v++) {
    const b = document.createElement("button");
    b.type = "button";
    // full verbal anchor on every point when labels are provided
    b.textContent = (labels && labels[v - 1]) ? labels[v - 1] : String(v);
    if (labels) b.classList.add("labeled");
    b.addEventListener("click", () => {
      [...row.children].forEach(x => x.classList.remove("sel"));
      b.classList.add("sel");
      onPick(v);
    });
    row.appendChild(b);
  }
  wrap.appendChild(row);
  if (!labels) {
    const ends = document.createElement("div");
    ends.className = "likert-ends";
    const parts = endsText.split("〜");
    ends.innerHTML = `<span>${(parts[0] || "").trim()}</span><span>${(parts[1] || "").trim()}</span>`;
    wrap.appendChild(ends);
  }
  return wrap;
}

$("player").addEventListener("ended", () => {
  cur.listened = true;
  $("play-hint").style.display = "none";
  checkReady();
});

function answered() {
  const it = items[idx];
  if (it.study === "A") return cur.answer.choice !== undefined;
  return cur.answer.main !== undefined && cur.answer.category !== undefined;
}
function checkReady() {
  $("next-btn").disabled = !(cur.listened && answered());
}

$("next-btn").addEventListener("click", () => {
  const it = items[idx];
  responses.push({
    item_id: it.item_id, study: it.study, task: it.task,
    model: it.model || null,
    answer: cur.answer, listened: cur.listened,
    response_time_ms: Date.now() - cur.shownAt,
    ts: new Date().toISOString(),
  });
  idx++;
  if (idx < items.length) { renderItem(); }
  else { finish(); }
});

function finish() {
  $("task").classList.add("hidden");
  $("done").classList.remove("hidden");
  const payload = {
    worker_id: workerId, started_at: startedAt,
    finished_at: new Date().toISOString(),
    n_items: responses.length,
    responses,
  };
  // completion code = hash of the payload content: any post-hoc edit of the
  // submitted token invalidates the code (verified at analysis time)
  const canonical = JSON.stringify(payload);
  const b64 = btoa(unescape(encodeURIComponent(canonical)));
  const code = (hashSeed(b64) >>> 0).toString(36).toUpperCase();
  $("code").textContent = code;
  $("submit-token").value = `FDBJA1.${b64}.${code}`;
  payload.completion_code = code;
  window._results = payload;
}

$("copy-btn").addEventListener("click", () => {
  const ta = $("submit-token");
  ta.select();
  navigator.clipboard.writeText(ta.value).then(
    () => { $("copy-btn").textContent = "コピーしました ✓"; },
    () => { document.execCommand("copy"); $("copy-btn").textContent = "コピーしました ✓"; });
});

$("download-btn").addEventListener("click", () => {
  const blob = new Blob([JSON.stringify(window._results, null, 1)],
    { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `responses_${workerId}.json`;
  a.click();
});
