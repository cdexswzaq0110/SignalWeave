/* SignalWeave console.
   Plain DOM, no framework, no build step. Everything on screen comes from the API;
   this file formats it and never recomputes a metric on its own. */

"use strict";

const state = {
  users: [],
  evaluation: null,
  system: null,
  validation: null,
  payload: null,
  userId: "",
  policy: "accuracy",
  operations: null,
  limit: 8,
  openRow: null,
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

const f2 = (v) => Number(v).toFixed(2);
const f3 = (v) => Number(v).toFixed(3);
const f4 = (v) => Number(v).toFixed(4);
const pct = (v, d = 1) => `${(v * 100).toFixed(d)}%`;
const signed = (v, d = 4) => `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(d)}`;
const esc = (v) => String(v).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

const TERMS = ["relevance", "diversity", "novelty", "freshness"];

const METRICS = [
  { key: "ndcg_at_10", label: "NDCG@10", better: "high", fmt: f4 },
  { key: "recall_at_10", label: "Recall@10", better: "high", fmt: f4 },
  { key: "mrr_at_10", label: "MRR@10", better: "high", fmt: f4 },
  { key: "catalog_coverage", label: "Catalog coverage", better: "high", fmt: (v) => pct(v, 2) },
  { key: "intra_list_diversity", label: "Intra-list diversity", better: "high", fmt: f4 },
  { key: "novelty_bits", label: "Novelty (bits)", better: "high", fmt: f2 },
  { key: "creator_hhi", label: "Creator HHI (lower is better)", better: "low", fmt: f4 },
];

const DRIFT_LABEL = {
  stable: "STABLE",
  watch: "WATCH",
  drifted: "DRIFTED",
  insufficient_data: "NO DATA",
};

const POLICY_ROLE = {
  popularity: "baseline",
  content: "baseline",
  accuracy: "served",
  balanced: "served, challenger",
  discovery: "served",
};

async function getJSON(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `${path} failed with ${response.status}`);
  }
  return response.json();
}

function toast(message) {
  const element = $("#toast");
  element.textContent = message;
  element.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { element.hidden = true; }, 2400);
}

/* ── slate ──────────────────────────────────────────────────────────── */

function signedBar(value, scale) {
  const half = scale > 0 ? (Math.abs(value) / scale) * 50 : 0;
  const left = value >= 0 ? 50 : 50 - half;
  return `<span class="bar"><i class="${value < 0 ? "negative" : ""}" style="left:${left}%;width:${half}%"></i></span>`;
}

function utilityDetail(item) {
  const stack = item.utility_terms
    .map((t) => `<span class="s-${t.term}" style="width:${Math.max(0, t.share) * 100}%" title="${t.term}"></span>`)
    .join("");
  const rows = item.utility_terms.map((t) => `
    <tr>
      <td><span class="swatch s-${t.term}" style="background:var(--${t.term})"></span>${t.term}</td>
      <td class="num">${f2(t.weight)}</td>
      <td class="num">${f3(t.value)}</td>
      <td class="num">${f4(t.contribution)}</td>
      <td class="num">${pct(t.share, 0)}</td>
    </tr>`).join("");
  return `
    <h3>Why this rank: slate utility ${f4(item.score)}</h3>
    <div class="stack">${stack}</div>
    <table>
      <thead><tr><th>Term</th><th class="num">Weight</th><th class="num">Value</th><th class="num">Contribution</th><th class="num">Share</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
}

function rankerDetail(item) {
  const scale = Math.max(...item.contributions.map((c) => Math.abs(c.value)), 1e-9);
  const rows = item.contributions.map((c) => `
    <tr>
      <td>${esc(c.feature)}</td>
      <td class="num">${f3(c.raw)}</td>
      <td>${signedBar(c.value, scale)}</td>
      <td class="num">${signed(c.value, 3)}</td>
    </tr>`).join("");
  return `
    <h3>Ranker score ${f4(item.relevance)} — feature contributions to the logit</h3>
    <table>
      <thead><tr><th>Feature</th><th class="num">Value</th><th>Effect</th><th class="num">Logit</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
    <ul class="provenance">
      <li>Retrieved by: ${item.sources.map(esc).join(", ")}.</li>
      ${item.runner_up ? `<li>Beat <em>${esc(item.runner_up.title)}</em> by ${f4(item.runner_up.margin)} of utility at this position.</li>` : ""}
      <li>${item.blocked_by_creator_cap} candidate${item.blocked_by_creator_cap === 1 ? " was" : "s were"} skipped here by the creator cap.</li>
    </ul>`;
}

function renderSlate(payload) {
  state.payload = payload;
  const config = payload.policy_config;

  $("#funnel-line").innerHTML =
    `<b>${payload.catalog_size}</b> items in catalog · <b>${payload.already_seen}</b> already seen and excluded · ` +
    `<b>${payload.candidate_count}</b> unique candidates retrieved · <b>${payload.recommendations.length}</b> placed in the slate.`;

  const shadow = payload.shadow;
  $("#shadow-line").innerHTML = shadow
    ? `Shadow policy <b>${esc(shadow.policy)}</b> ran behind this request and was not shown: it would keep ` +
      `<b>${Math.round(shadow.overlap * shadow.k)}</b> of <b>${shadow.k}</b> items, ` +
      `${shadow.top1_agree ? "same top pick" : "a different top pick"}, mean rank shift <b>${f2(shadow.mean_rank_shift)}</b> ` +
      `(${f2(shadow.champion_ms)} ms served + ${f2(shadow.shadow_ms)} ms shadow).`
    : `Served under <b>${esc(payload.policy)}</b>. Shadow comparison runs behind the champion policy only, ` +
      `so operator experiments like this one are not logged.`;

  $("#slate-body").innerHTML = payload.recommendations.map((item) => `
    <tr class="item" data-rank="${item.rank}">
      <td class="rank num">${item.rank}</td>
      <td>
        <div class="title">${esc(item.title)}</div>
        <div class="meta">${esc(item.format)} · ${esc(item.difficulty_name)} · ${item.duration_min} min · ${esc(item.creator)}</div>
        <div class="why">${esc(item.why)}</div>
      </td>
      <td class="sources">${item.sources.map(esc).join("<br>")}</td>
      <td class="num utility">${f4(item.score)}<span class="decided">on ${esc(item.decided_by)}</span></td>
      <td>
        <span class="feedback">
          <button type="button" data-action="save" data-item="${esc(item.item_id)}">Save</button>
          <button type="button" data-action="complete" data-item="${esc(item.item_id)}">Done</button>
          <button type="button" data-action="dismiss" data-item="${esc(item.item_id)}">Hide</button>
        </span>
      </td>
    </tr>
    <tr class="detail" data-detail="${item.rank}" hidden>
      <td colspan="5">
        <div class="detail-grid">
          <div>${utilityDetail(item)}</div>
          <div>${rankerDetail(item)}</div>
        </div>
      </td>
    </tr>`).join("");

  $("#learner-facts").innerHTML = [
    ["Name", payload.user.name],
    ["Role", payload.user.role],
    ["Primary topic", payload.user.primary_topic],
    ["Secondary topic", payload.user.secondary_topic],
    ["Preferred format", payload.user.preferred_format],
    ["Level", payload.user.level],
    ["Time budget", `${payload.user.time_budget_min} min`],
  ].map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("");

  const routes = Object.entries(payload.retrieval_sources);
  const total = routes.reduce((sum, [, n]) => sum + n, 0);
  $("#routes-table").innerHTML = routes
    .map(([name, count]) => `<tr><td>${esc(name)}</td><td>${count}</td></tr>`)
    .join("") + `<tr><td>candidates (deduplicated)</td><td>${payload.candidate_count}</td></tr>`;
  $("#routes-note").textContent =
    `Routes propose ${total} slots for ${payload.candidate_count} distinct items, so most candidates are ` +
    `found by more than one route. Each route offers its top 22 unseen items.`;

  $("#constraints-table").innerHTML = TERMS
    .map((term) => `<tr><td>${term} weight</td><td>${f2(config[term])}</td></tr>`)
    .join("") + `<tr><td>max items per creator</td><td>${config.creator_cap}</td></tr>`;

  restoreOpenRow();
}

function restoreOpenRow() {
  if (state.openRow === null) return;
  const row = $(`.slate tr.item[data-rank="${state.openRow}"]`);
  const detail = $(`.slate tr[data-detail="${state.openRow}"]`);
  if (row && detail) { row.classList.add("open"); detail.hidden = false; }
}

async function loadSlate() {
  $("#slate-body").innerHTML = `<tr><td colspan="5" class="loading">Recomputing slate…</td></tr>`;
  try {
    const query = `user_id=${encodeURIComponent(state.userId)}&policy=${state.policy}&limit=${state.limit}`;
    renderSlate(await getJSON(`/api/recommendations?${query}`));
  } catch (error) {
    $("#slate-body").innerHTML = `<tr><td colspan="5" class="error">${esc(error.message)}</td></tr>`;
  }
}

async function sendFeedback(button) {
  button.disabled = true;
  try {
    await getJSON("/api/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: state.userId, item_id: button.dataset.item, action: button.dataset.action }),
    });
    toast(`Recorded "${button.dataset.action}" — the profile and next slate now include it.`);
    state.openRow = null;
    await loadSlate();
  } catch (error) {
    toast(error.message);
    button.disabled = false;
  }
}

/* ── policies ───────────────────────────────────────────────────────── */

function renderPolicies() {
  const report = state.evaluation;
  const names = Object.keys(report.policies);

  const header = names.map((name) =>
    `<th scope="col">${esc(name)}<br><span class="rule-note">${POLICY_ROLE[name] || ""}</span></th>`).join("");
  const body = METRICS.map((metric) => {
    const values = names.map((name) => report.policies[name][metric.key]);
    const best = metric.better === "high" ? Math.max(...values) : Math.min(...values);
    return `<tr><td>${esc(metric.label)}</td>` + values
      .map((value) => `<td class="${value === best ? "best" : ""}">${metric.fmt(value)}</td>`)
      .join("") + "</tr>";
  }).join("");
  $("#matrix-table").innerHTML =
    `<colgroup>${names.map((n) => `<col class="${n === "balanced" ? "challenger" : ""}">`).join("")}</colgroup>` +
    `<thead><tr><th scope="col">Metric</th>${header}</tr></thead><tbody>${body}</tbody>`;

  const gate = report.release_gate;
  const failing = gate.guardrails.filter((rule) => !rule.passed).length;
  $("#gate-summary").innerHTML =
    `Challenger <b>${esc(gate.challenger)}</b> against reference <b>${esc(gate.reference)}</b>. ` +
    `Status <span class="gate-status">${esc(gate.status)}</span> — ` +
    `${failing === 0 ? "every guardrail passes" : `${failing} guardrail(s) failing`}. ` +
    esc(gate.note);
  $("#gate-table").innerHTML = gate.guardrails.map((rule) => `
    <tr>
      <td class="who"><span class="verdict ${rule.passed ? "pass" : "fail"}">${rule.passed ? "PASS" : "FAIL"}</span></td>
      <td>
        <div class="claim">${esc(rule.claim)}</div>
        <div class="cid">${esc(rule.id)}</div>
        <div class="measure"><span>threshold</span><span>${rule.threshold}</span><span>observed</span><span>${rule.observed}</span></div>
      </td>
    </tr>`).join("");

  renderBootstrap(report.paired_bootstrap);

  const training = report.training;
  const scale = Math.max(...training.coefficients.map((c) => Math.abs(c.coefficient)));
  $("#coefficients-table").innerHTML =
    `<thead><tr><th>Feature</th><th>Weight</th><th class="num">Coefficient</th></tr></thead><tbody>` +
    training.coefficients.map((c) => `
      <tr><td>${esc(c.feature)}</td><td>${signedBar(c.coefficient, scale)}</td><td>${signed(c.coefficient, 4)}</td></tr>`).join("") +
    "</tbody>";

  $("#ranker-facts").innerHTML = [
    ["Fit rows", training.rows],
    ["Positive rate", f3(training.positive_rate)],
    ["CV folds", training.cv_folds],
    ["ROC AUC", f4(training.cv_roc_auc)],
    ["Brier score", f4(training.cv_brier)],
    ["Mean score", f4(training.mean_held_out_score)],
    ["Actual rate", f4(training.observed_positive_rate)],
  ].map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("");
  $("#calibration-note").textContent =
    `ROC AUC and Brier are cross-validated inside the ranker-fit window, never on the frozen window. ` +
    training.calibration_note;
}

function renderBootstrap(ci) {
  const [low, high] = ci.confidence_interval_95;
  const pad = (high - low) * 0.35 || 0.05;
  const min = Math.min(0, low) - pad;
  const max = Math.max(0, high) + pad;
  const at = (value) => ((value - min) / (max - min)) * 100;
  $("#bootstrap").innerHTML = `
    <p class="headline">${signed(ci.mean_delta)}</p>
    <p>Mean per-learner NDCG@10 difference, ${esc(ci.metric)}.</p>
    <div class="axis">
      <span class="line"></span>
      <span class="span" style="left:${at(low)}%;width:${at(high) - at(low)}%"></span>
      <span class="point" style="left:${at(ci.mean_delta)}%"></span>
      <span class="zero" style="left:${at(0)}%"><b>0</b></span>
      <span class="ticks"><span>${f3(min)}</span><span>${f3(max)}</span></span>
    </div>
    <p>95% interval ${f4(low)} to ${f4(high)} over ${ci.resamples} paired resamples of the 62 evaluable
    learners. This is sampling uncertainty within one simulated log. It is not an online treatment effect,
    and it does not correct for the exposure bias in how the log was generated.</p>`;
}

/* ── checks ─────────────────────────────────────────────────────────── */

function renderChecks() {
  const report = state.validation;
  const s = report.summary;
  $("#checks-tally").textContent =
    `${s.checks} checks · ${s.passed} pass · ${s.failed} fail · ${s.known_limitations} known`;

  $("#checks-groups").innerHTML = report.groups.map((group) => `
    <section class="check-group">
      <h3 class="rule">${esc(group.name)}</h3>
      <p>${esc(group.purpose)}</p>
      <table class="checks">
        <tbody>${group.checks.map((check) => `
          <tr>
            <td class="who"><span class="verdict ${check.status}">${check.status.toUpperCase()}</span></td>
            <td>
              <div class="claim">${esc(check.claim)}</div>
              <div class="cid">${esc(check.id)}</div>
              <div class="measure">
                <span>expected</span><span>${esc(check.expected)}</span>
                <span>observed</span><span>${esc(check.observed)}</span>
              </div>
              ${check.detail ? `<p class="detail-text">${esc(check.detail)}</p>` : ""}
            </td>
          </tr>`).join("")}
        </tbody>
      </table>
    </section>`).join("");
}

/* ── operations ─────────────────────────────────────────────────────── */

function verdictToken(status, label) {
  return `<span class="verdict ${status}">${label || status.toUpperCase()}</span>`;
}

function renderOperations(ops) {
  state.operations = ops;
  const model = ops.model;
  const manifest = model.manifest || {};

  $("#ops-version").textContent = model.loaded || "unregistered";

  const rows = [
    ["Loaded version", model.loaded || "—", model.provenance === "registry" ? "pass" : "known",
      model.provenance === "registry" ? "FROM REGISTRY" : "IN-PROCESS"],
    ["Champion policy", `${ops.serving.champion_policy} — served by default`, "info", "SERVED"],
    ["Shadow policy", `${ops.serving.shadow_policy} — scored on every champion request, never shown`, "info", "SHADOW"],
    ["Training code", `code_digest ${manifest.code_digest || "—"}`, model.code_is_current ? "pass" : "known",
      model.code_is_current ? "CURRENT" : "STALE"],
    ["Dataset", `data_digest ${manifest.data_digest || "—"}, seed ${manifest.seed ?? "—"}`, "info", "PINNED"],
    ["Registered", manifest.created_at || "—", "info", "WHEN"],
  ];
  $("#ops-serving").innerHTML = rows.map(([label, value, status, token]) => `
    <tr>
      <td class="who">${verdictToken(status, token)}</td>
      <td><div class="claim">${esc(label)}</div><div class="measure"><span></span><span>${esc(value)}</span></div></td>
    </tr>`).join("");

  const environment = manifest.environment || {};
  const baseline = manifest.serving_baseline || {};
  $("#ops-environment").innerHTML = [
    ["Python", environment.python],
    ["numpy", environment.numpy],
    ["scikit-learn", environment.scikit_learn],
    ["Checks at training", manifest.validation ? `${manifest.validation.passed}/${manifest.validation.checks}` : "—"],
    ["Gate at training", manifest.release_gate],
    ["Baseline slate score", baseline.mean_slate_score],
  ].filter(([, value]) => value !== undefined && value !== null)
    .map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("");
  $("#ops-note").textContent = model.note || manifest.note || "";

  const versions = model.versions || [];
  $("#ops-versions").innerHTML =
    `<thead><tr><th>Version</th><th>Registered</th><th class="num">AUC</th><th class="num">NDCG@10</th><th>Gate</th><th>Note</th></tr></thead><tbody>` +
    (versions.length
      ? versions.map((entry) => `
        <tr class="${entry.version === model.champion ? "served" : ""}">
          <td>${esc(entry.version)}${entry.version === model.champion ? " ◂ champion" : ""}</td>
          <td>${esc((entry.created_at || "").replace("T", " ").replace("+00:00", ""))}</td>
          <td>${f4(entry.training.cv_roc_auc)}</td>
          <td>${f4(entry.metrics.balanced.ndcg_at_10)}</td>
          <td>${esc(entry.release_gate)}</td>
          <td>${esc(entry.note || "")}</td>
        </tr>`).join("")
      : `<tr><td colspan="6" class="loading">Nothing registered yet.</td></tr>`) +
    "</tbody>";

  const shadow = ops.shadow;
  $("#shadow-hint").textContent = shadow.note;
  $("#ops-shadow").innerHTML = shadow.requests
    ? [
        ["Comparisons recorded", shadow.requests],
        ["Mean overlap with the served slate", pct(shadow.mean_overlap)],
        ["Worst single overlap", pct(shadow.min_overlap)],
        ["Top-1 agreement", pct(shadow.top1_agreement)],
        ["Mean rank shift of shared items", f2(shadow.mean_rank_shift)],
        ["p95 latency, champion", `${shadow.champion_p95_ms} ms`],
        ["p95 latency, shadow", `${shadow.shadow_p95_ms} ms`],
      ].map(([k, v]) => `<tr><td>${esc(k)}</td><td>${esc(v)}</td></tr>`).join("")
    : `<tr><td class="loading">No comparisons yet.</td></tr>`;

  $("#ops-latency").innerHTML =
    `<thead><tr><th>Route</th><th class="num">n</th><th class="num">p50</th><th class="num">p95</th></tr></thead><tbody>` +
    ops.latency.routes.map((route) => `
      <tr><td>${esc(route.route.replace("/api/", ""))}</td><td>${route.requests}</td>
      <td>${route.p50_ms}</td><td>${route.p95_ms}</td></tr>`).join("") +
    "</tbody>";

  const drift = ops.drift;
  $("#drift-hint").textContent = drift.action_mix.note;
  const maxShare = Math.max(...Object.values(drift.action_mix.trained_on), ...Object.values(drift.action_mix.live), 0.01);
  $("#ops-drift-actions").innerHTML =
    `<thead><tr><th>Action</th><th>Trained vs live</th><th class="num">Trained</th><th class="num">Live</th></tr></thead><tbody>` +
    Object.keys(drift.action_mix.trained_on).map((action) => {
      const trained = drift.action_mix.trained_on[action];
      const live = drift.action_mix.live[action];
      return `<tr>
        <td>${esc(action)}</td>
        <td><span class="pair"><i class="trained" style="width:${(trained / maxShare) * 100}%"></i><i class="live" style="width:${(live / maxShare) * 100}%"></i></span></td>
        <td>${pct(trained)}</td><td>${pct(live)}</td></tr>`;
    }).join("") + "</tbody>";

  const statusRows = [
    ["Action mix", drift.action_mix.status,
      drift.action_mix.psi === null ? `${drift.action_mix.live_events} live events` : `PSI ${drift.action_mix.psi}`],
    ["Slate score", drift.slate_score.status,
      drift.slate_score.observed === null
        ? "no served requests yet"
        : `${f4(drift.slate_score.observed)} vs baseline ${f4(drift.slate_score.baseline)} (${signed(drift.slate_score.delta)})`],
    ["Retrain", drift.retrain.recommended ? "watch" : "stable",
      `${drift.retrain.events_since_registration} of ${drift.retrain.threshold} events`],
  ];
  $("#ops-drift-status").innerHTML = statusRows.map(([label, status, measure]) => `
    <tr>
      <td class="who">${verdictToken(status, DRIFT_LABEL[status] || status.toUpperCase())}</td>
      <td><div class="claim">${esc(label)}</div><div class="measure"><span></span><span>${esc(measure)}</span></div></td>
    </tr>`).join("") + `<tr><td></td><td><p class="detail-text">${esc(drift.retrain.note)}</p></td></tr>`;
}

async function loadOperations() {
  try {
    renderOperations(await getJSON("/api/operations"));
  } catch (error) {
    $("#ops-serving").innerHTML = `<tr><td class="error">${esc(error.message)}</td></tr>`;
  }
}

/* ── pipeline ───────────────────────────────────────────────────────── */

function renderPipeline() {
  $("#stages").innerHTML = state.system.stages
    .map((stage) => `<li><b>${esc(stage.name)}</b><span>${esc(stage.detail)}</span></li>`)
    .join("");
}

/* ── shell ──────────────────────────────────────────────────────────── */

function renderMasthead() {
  const dataset = state.evaluation.dataset;
  $("#build-facts").innerHTML = [
    ["learners", dataset.users],
    ["items", dataset.items],
    ["events", dataset.events.toLocaleString()],
    ["evaluable", dataset.eligible_evaluation_users],
  ].map(([k, v]) => `<div><dd>${esc(v)}</dd><dt>${esc(k)}</dt></div>`).join("");
}

function bindEvents() {
  $$(".tabs button").forEach((button) => button.addEventListener("click", () => {
    $$(".tabs button").forEach((other) => other.removeAttribute("aria-current"));
    button.setAttribute("aria-current", "page");
    $$(".view").forEach((view) => { view.hidden = view.id !== `${button.dataset.view}-view`; });
    // Latency, shadow divergence and drift all move while the console is open.
    if (button.dataset.view === "operations") loadOperations();
  }));

  $("#user-select").addEventListener("change", (event) => {
    state.userId = event.target.value;
    state.openRow = null;
    loadSlate();
  });
  $("#policy-group").addEventListener("change", (event) => {
    state.policy = event.target.value;
    state.openRow = null;
    loadSlate();
  });
  $("#limit-input").addEventListener("change", (event) => {
    const value = Math.min(20, Math.max(1, Number(event.target.value) || 8));
    event.target.value = value;
    state.limit = value;
    state.openRow = null;
    loadSlate();
  });
  $("#refresh").addEventListener("click", loadSlate);
  $("#toolbar").addEventListener("submit", (event) => event.preventDefault());

  $("#slate-body").addEventListener("click", (event) => {
    const button = event.target.closest("button[data-action]");
    if (button) { sendFeedback(button); return; }
    const row = event.target.closest("tr.item");
    if (!row) return;
    const rank = Number(row.dataset.rank);
    const detail = $(`.slate tr[data-detail="${rank}"]`);
    const opening = detail.hidden;
    $$(".slate tr.detail").forEach((element) => { element.hidden = true; });
    $$(".slate tr.item").forEach((element) => element.classList.remove("open"));
    detail.hidden = !opening;
    row.classList.toggle("open", opening);
    state.openRow = opening ? rank : null;
  });
}

async function boot() {
  try {
    const [users, evaluation, system] = await Promise.all([
      getJSON("/api/users"), getJSON("/api/evaluation"), getJSON("/api/system"),
    ]);
    state.users = users;
    state.evaluation = evaluation;
    state.system = system;
    state.userId = users[0].user_id;

    $("#user-select").innerHTML = users
      .map((user) => `<option value="${esc(user.user_id)}">${esc(user.user_id)} · ${esc(user.name)} · ${esc(user.role)}</option>`)
      .join("");

    renderMasthead();
    renderPolicies();
    renderPipeline();
    bindEvents();
    await loadSlate();

    // Checks re-run retrieval server-side, so they load after the first slate.
    state.validation = await getJSON("/api/validation");
    renderChecks();
    await loadOperations();
    $("#colophon-line").textContent =
      `SignalWeave ${state.system.version} · model ${state.system.model_version} · ` +
      `seed ${state.evaluation.dataset.seed} · checks last run ${state.validation.generated_at}`;
  } catch (error) {
    document.querySelector("main").innerHTML =
      `<p class="error">SignalWeave could not start: ${esc(error.message)}</p>`;
  }
}

boot();
