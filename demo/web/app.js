const byId = (id) => document.getElementById(id);

const state = {
  cases: [],
  activeCaseId: "synthetic_picture_description",
  activeResult: null,
  uploadedFile: null,
  loadSequence: 0,
};

const staticUrl = (path) => new URL(`../${path}`, window.location.href).href;
const escapeHtml = (value) => String(value ?? "")
  .replaceAll("&", "&amp;")
  .replaceAll("<", "&lt;")
  .replaceAll(">", "&gt;")
  .replaceAll('"', "&quot;")
  .replaceAll("'", "&#039;");
const formatNumber = (value, digits = 2) => value === null || value === undefined || Number.isNaN(Number(value))
  ? "N/A"
  : Number(value).toFixed(digits);
const percent = (value, digits = 1) => `${(Number(value) * 100).toFixed(digits)}%`;

const evidenceRole = {
  clinical_state: "Clinical state",
  model_auxiliary: "Model auxiliary",
  quality_control: "Quality control",
};

async function fetchJsonWithFallback(apiPath, staticPath) {
  try {
    const response = await fetch(apiPath, { cache: "no-store" });
    if (response.ok) return response.json();
  } catch (_) {
    // Static hosting intentionally has no API.
  }
  const response = await fetch(staticUrl(staticPath), { cache: "no-store" });
  if (!response.ok) throw new Error(`Unable to load ${staticPath}.`);
  return response.json();
}

function isPublicCase(caseId) {
  return caseId.startsWith("synthetic_");
}

function caseAudioUrl(caseId) {
  return isPublicCase(caseId)
    ? staticUrl(`assets/${caseId}.wav`)
    : `/api/case-audio/${encodeURIComponent(caseId)}`;
}

async function fetchCaseResult(caseId) {
  if (isPublicCase(caseId)) {
    return fetchJsonWithFallback(`/api/case/${encodeURIComponent(caseId)}`, `output/${caseId}.json`);
  }
  const response = await fetch(`/api/case/${encodeURIComponent(caseId)}`);
  const result = await response.json();
  if (!response.ok) throw new Error(result.message || result.error || "Case analysis failed.");
  return result;
}

async function drawWaveform(source) {
  const buffer = source instanceof File ? await source.arrayBuffer() : await fetch(source).then((response) => response.arrayBuffer());
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  const audioContext = new AudioContextClass();
  const decoded = await audioContext.decodeAudioData(buffer.slice(0));
  const data = decoded.getChannelData(0);
  const canvas = byId("waveform");
  const context = canvas.getContext("2d");
  const { width, height } = canvas;
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#f1f4f3";
  context.fillRect(0, 0, width, height);
  context.strokeStyle = "#176b55";
  context.lineWidth = 2;
  context.beginPath();
  const step = Math.max(1, Math.floor(data.length / width));
  for (let x = 0; x < width; x += 1) {
    let min = 1;
    let max = -1;
    for (let index = x * step; index < Math.min((x + 1) * step, data.length); index += 1) {
      min = Math.min(min, data[index]);
      max = Math.max(max, data[index]);
    }
    context.moveTo(x, (1 + min) * height / 2);
    context.lineTo(x, (1 + max) * height / 2);
  }
  context.stroke();
  context.strokeStyle = "#aebbc0";
  context.lineWidth = 1;
  context.beginPath();
  context.moveTo(0, height / 2);
  context.lineTo(width, height / 2);
  context.stroke();
  await audioContext.close();
}

function renderCaseSelector() {
  byId("case-selector").innerHTML = state.cases.map((item, index) => `
    <button class="case-button ${item.case_id === state.activeCaseId ? "active" : ""}" data-case-id="${escapeHtml(item.case_id)}">
      <span>${String(index + 1).padStart(2, "0")}</span>
      <strong>${escapeHtml(item.channel_name)}</strong>
      <small>${escapeHtml(item.task_name)}</small>
    </button>
  `).join("");
  document.querySelectorAll(".case-button").forEach((button) => {
    button.addEventListener("click", () => loadCase(button.dataset.caseId));
  });
}

function renderReport(result) {
  const report = result.agent_report;
  byId("agent-report").innerHTML = `
    <article class="report-sheet">
      <header class="report-header">
        <div><span class="mono-label">EVIDENCE-CONSTRAINED OUTPUT</span><h4>${escapeHtml(report.title)}</h4></div>
        <span>OFFLINE PREVIEW</span>
      </header>
      <section class="report-section"><h4>Screening impression</h4><p>${escapeHtml(report.screening_impression)}</p></section>
      <section class="report-section"><h4>Observed cognitive-state evidence</h4><ul>${report.observations.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul><div class="evidence-citations">${report.evidence_ids.map((id) => `<code>${escapeHtml(id)}</code>`).join("")}</div></section>
      <section class="report-section"><h4>Acquisition quality and interpretation boundary</h4><p>${escapeHtml(report.quality_statement)}</p></section>
      <section class="report-section"><h4>Recommended next action</h4><p>${escapeHtml(report.next_action)}</p></section>
    </article>
  `;
}

function renderStates(result) {
  byId("states").innerHTML = result.state_cards.map((card) => `
    <article class="state-card">
      <header><h3>${escapeHtml(card.name)}</h3><span>${escapeHtml(card.id)}</span></header>
      <p>${escapeHtml(card.clinical_question)}</p>
      <div class="state-values"><strong>${formatNumber(card.score)}</strong><small>Evidence completeness ${formatNumber(card.confidence)}</small></div>
      <div class="meter"><span style="width:${Math.max(0, Math.min(100, card.confidence * 100))}%"></span></div>
    </article>
  `).join("");
}

function renderEvidence(result) {
  byId("evidence").innerHTML = result.metric_evidence.map((item) => `
    <tr>
      <td><strong>${escapeHtml(item.name || item.metric_id)}</strong><code>${escapeHtml(item.metric_id)}</code></td>
      <td><span class="role-badge role-${escapeHtml(item.evidence_role)}">${escapeHtml(evidenceRole[item.evidence_role] || item.evidence_role)}</span></td>
      <td>${formatNumber(item.value, 3)} ${escapeHtml(item.unit)}</td>
      <td>${formatNumber(item.reference_median, 3)} &plusmn; ${formatNumber(item.reference_scale, 3)}</td>
      <td>${formatNumber(item.directional_z, 3)}</td>
      <td>${formatNumber(item.reliability, 3)}</td>
      <td>${escapeHtml(item.source)}</td>
    </tr>
  `).join("");
}

function renderTrace(result) {
  const metrics = result.metric_evidence;
  const states = result.state_cards;
  const segments = result.segments.slice(0, 4);
  const height = Math.max(450, metrics.length * 43, states.length * 78);
  const x = { segment: 32, metric: 280, state: 640 };
  const yMap = (items, padding = 42) => Object.fromEntries(items.map((item, index) => [
    item.id || item.segment_id,
    padding + index * ((height - padding * 2) / Math.max(items.length - 1, 1)),
  ]));
  const segmentY = yMap(segments);
  const metricY = yMap(metrics);
  const stateY = yMap(states);
  const evidenceLines = result.trace.map((edge) => `
    <path d="M ${x.metric + 190} ${metricY[edge.from]} C 535 ${metricY[edge.from]}, 550 ${stateY[edge.to]}, ${x.state} ${stateY[edge.to]}" fill="none" stroke="#b7c2c4" stroke-width="1.4"/>
  `).join("");
  const segmentLines = metrics.flatMap((metric) => (metric.segment_ids || []).filter((id) => segmentY[id] !== undefined).map((id) => `
    <path d="M ${x.segment + 152} ${segmentY[id]} C 215 ${segmentY[id]}, 232 ${metricY[metric.id]}, ${x.metric} ${metricY[metric.id]}" fill="none" stroke="#dfc2bf" stroke-width="1.2"/>
  `)).join("");
  const segmentNodes = segments.map((segment, index) => `
    <g><rect x="${x.segment}" y="${segmentY[segment.segment_id] - 14}" width="152" height="28" rx="3" fill="#f8e9e7" stroke="#b6534d"/><text x="${x.segment + 8}" y="${segmentY[segment.segment_id] + 4}" font-size="10" fill="#172027">Segment ${index + 1} / ${formatNumber(segment.start_sec, 1)}-${formatNumber(segment.end_sec, 1)} s</text></g>
  `).join("");
  const metricNodes = metrics.map((metric) => `
    <g><rect x="${x.metric}" y="${metricY[metric.id] - 15}" width="190" height="30" rx="3" fill="#eaf0f6" stroke="#356a96"/><text x="${x.metric + 8}" y="${metricY[metric.id] + 4}" font-size="10" fill="#172027">${escapeHtml(metric.name || metric.metric_id)}</text></g>
  `).join("");
  const stateNodes = states.map((card) => `
    <g><rect x="${x.state}" y="${stateY[card.id] - 20}" width="208" height="40" rx="3" fill="#e7f3f3" stroke="#147b83"/><text x="${x.state + 9}" y="${stateY[card.id] - 3}" font-size="10" font-weight="700" fill="#172027">${escapeHtml(card.id)} / ${escapeHtml(card.name)}</text><text x="${x.state + 9}" y="${stateY[card.id] + 12}" font-size="9" fill="#617078">score ${formatNumber(card.score)} / completeness ${formatNumber(card.confidence)}</text></g>
  `).join("");
  byId("trace").innerHTML = `
    <svg viewBox="0 0 880 ${height}" role="img" aria-label="Trace map from segments to metric evidence and StateCards">
      <text x="${x.segment}" y="20" font-size="10" font-weight="700">AUDIO SEGMENTS</text>
      <text x="${x.metric}" y="20" font-size="10" font-weight="700">METRIC EVIDENCE</text>
      <text x="${x.state}" y="20" font-size="10" font-weight="700">COGNITIVE STATE CARDS</text>
      ${segmentLines}${evidenceLines}${segmentNodes}${metricNodes}${stateNodes}
    </svg>
  `;
}

function selectSegment(segment, index) {
  const player = byId("player");
  player.currentTime = segment.start_sec;
  player.play().catch(() => {});
  document.querySelectorAll(".segment-button").forEach((button) => button.classList.toggle("active", Number(button.dataset.segmentIndex) === index));
  byId("segment-detail").innerHTML = `
    <span class="mono-label">SEGMENT ${String(index + 1).padStart(2, "0")}</span>
    <h4>${formatNumber(segment.start_sec, 1)}-${formatNumber(segment.end_sec, 1)} seconds</h4>
    <p>Voiced fraction ${formatNumber(segment.voiced_fraction)} / RMS ${formatNumber(segment.rms_db, 1)} dB</p>
    <p class="segment-text">${escapeHtml(segment.text || "No timestamp-aligned transcript is available in this demo result.")}</p>
  `;
}

function renderSegments(result) {
  byId("segments").innerHTML = result.segments.map((segment, index) => `
    <button class="segment-button" data-segment-index="${index}">
      <strong>Segment ${String(index + 1).padStart(2, "0")}</strong>
      <small>${formatNumber(segment.start_sec, 1)}-${formatNumber(segment.end_sec, 1)} s / voiced ${formatNumber(segment.voiced_fraction)}</small>
    </button>
  `).join("");
  document.querySelectorAll(".segment-button").forEach((button) => {
    button.addEventListener("click", () => selectSegment(result.segments[Number(button.dataset.segmentIndex)], Number(button.dataset.segmentIndex)));
  });
  if (result.segments.length) selectSegment(result.segments[0], 0);
}

function renderResult(result) {
  state.activeResult = result;
  const duration = result.case.original_duration_sec > result.case.duration_sec + 0.1
    ? `${formatNumber(result.case.duration_sec, 1)} s participant audio`
    : `${formatNumber(result.case.duration_sec, 1)} s analyzed`;
  byId("summary").innerHTML = `
    <div><strong>${result.metric_evidence.length}</strong><span>Evidence objects</span></div>
    <div><strong>${result.state_cards.length}</strong><span>StateCards</span></div>
    <div><strong>${result.segments.length}</strong><span>Segments</span></div>
    <div><strong>${formatNumber(result.quality.audio_reliability)}</strong><span>Audio reliability</span></div>
  `;
  byId("run-status").textContent = `${duration} / no diagnosis`;
  renderReport(result);
  renderStates(result);
  renderEvidence(result);
  renderTrace(result);
  renderSegments(result);
}

async function loadCase(caseId) {
  const sequence = ++state.loadSequence;
  state.activeCaseId = caseId;
  state.uploadedFile = null;
  byId("audio").value = "";
  byId("error").textContent = "";
  renderCaseSelector();
  const metadata = state.cases.find((item) => item.case_id === caseId) || {};
  byId("case-dataset").textContent = metadata.dataset_id || "PUBLIC SYNTHETIC";
  byId("case-task").textContent = metadata.task_name || "Synthetic case";
  byId("case-description").textContent = metadata.description || "";
  byId("case-focus").innerHTML = (metadata.evidence_focus || []).map((item) => `<span>${escapeHtml(item)}</span>`).join("");
  byId("case-scope").textContent = isPublicCase(caseId) ? "PUBLIC SYNTHETIC" : "LOCAL RESTRICTED";
  const audioUrl = caseAudioUrl(caseId);
  byId("player").src = audioUrl;
  byId("audio-name").textContent = isPublicCase(caseId) ? "Packaged synthetic audio" : "Locally mounted restricted audio";
  const [, result] = await Promise.all([drawWaveform(audioUrl), fetchCaseResult(caseId)]);
  if (sequence !== state.loadSequence) return;
  byId("transcript").value = result.case.transcript || "";
  byId("task").value = result.case.task_type || byId("task").value;
  byId("language").value = result.case.language || byId("language").value;
  renderResult(result);
}

async function fileToDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

function renderEvaluation(payload) {
  byId("evaluation-status").textContent = payload.status.statement;
  const metrics = [
    ["accuracy", "Accuracy"],
    ["micro_auroc", "Micro AUROC"],
    ["micro_auprc", "Micro AUPRC"],
    ["macro_f1", "Macro F1"],
  ];
  byId("performance-chart").innerHTML = metrics.map(([key, label]) => `
    <div class="metric-group">
      <div class="metric-name">${label}</div>
      <div>${payload.models.map((model) => {
        const value = model[key];
        const modelClass = model.id === "B3" ? "ours" : model.id === "SpeechCARE" ? "speechcare" : "";
        return `<div class="bar-row ${modelClass}"><label>${escapeHtml(model.id)}</label><div class="bar-track"><div class="bar-fill" style="width:${value === null ? 0 : value * 100}%"></div></div><output>${value === null ? "N/A" : formatNumber(value, 3)}</output></div>`;
      }).join("")}</div>
    </div>
  `).join("");
  byId("validation-grid").innerHTML = payload.framework_validation.map((item) => `
    <div class="validation-item"><strong>${item.format === "delta" ? `+${formatNumber(item.value, 3)}` : percent(item.value)}</strong><span>${escapeHtml(item.label)}</span></div>
  `).join("");
  byId("rubric-chart").innerHTML = `<span class="figure-label">Paired report rubric / ${payload.report_rubric.matched_cases} cases</span>${payload.report_rubric.dimensions.map((item) => `
    <div class="rubric-row"><label>${escapeHtml(item.name)}</label><div class="rubric-track"><div class="rubric-fill" style="width:${item.ours / 5 * 100}%"></div></div><output>${formatNumber(item.ours, 2)}</output><div class="rubric-track baseline"><div class="rubric-fill baseline" style="width:${item.direct_llm / 5 * 100}%"></div></div><output>${formatNumber(item.direct_llm, 2)}</output></div>
  `).join("")}`;
  byId("evaluation-note").textContent = `Source snapshot: ${payload.provenance.snapshot}. ADvoice report rubric ${payload.report_rubric.ours_total_25}/25 versus direct LLM ${payload.report_rubric.direct_llm_total_25}/25. The report rubric is an automated structural audit, not a completed clinician study.`;
}

async function initialize() {
  const casePayload = await fetchJsonWithFallback("/api/cases", "output/public_cases.json");
  state.cases = casePayload.cases.filter((item) => isPublicCase(item.case_id));
  if (!state.cases.some((item) => item.case_id === state.activeCaseId)) state.activeCaseId = state.cases[0].case_id;
  renderCaseSelector();
  await Promise.all([
    loadCase(state.activeCaseId),
    fetchJsonWithFallback("/api/evaluation", "output/prepare_evaluation_summary.json").then(renderEvaluation),
  ]);
}

byId("audio").addEventListener("change", async (event) => {
  state.uploadedFile = event.target.files[0] || null;
  if (!state.uploadedFile) return;
  const objectUrl = URL.createObjectURL(state.uploadedFile);
  byId("player").src = objectUrl;
  byId("audio-name").textContent = state.uploadedFile.name;
  await drawWaveform(state.uploadedFile);
});

byId("analyze").addEventListener("click", async () => {
  const button = byId("analyze");
  button.disabled = true;
  byId("error").textContent = "";
  try {
    if (!state.uploadedFile) {
      await loadCase(state.activeCaseId);
      return;
    }
    const response = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        audio_base64: await fileToDataUrl(state.uploadedFile),
        transcript: byId("transcript").value,
        task_type: byId("task").value,
        language: byId("language").value,
      }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.message || result.error || "Analysis failed.");
    renderResult(result);
  } catch (error) {
    byId("error").textContent = window.location.hostname.endsWith("github.io")
      ? "Static hosting displays packaged cases. Run `make demo` locally to analyze an uploaded WAV."
      : error.message;
  } finally {
    button.disabled = false;
  }
});

byId("reset").addEventListener("click", () => loadCase(state.activeCaseId));
byId("player").addEventListener("timeupdate", () => {
  byId("playback-time").textContent = `${formatNumber(byId("player").currentTime, 1)} s`;
});
document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((item) => item.classList.toggle("active", item === tab));
    document.querySelectorAll(".view").forEach((view) => view.classList.toggle("active", view.id === `view-${tab.dataset.view}`));
  });
});

initialize().catch((error) => {
  byId("error").textContent = error.message;
  byId("run-status").textContent = "Load failed";
});
