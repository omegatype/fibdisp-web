const $ = (id) => document.getElementById(id);
const C_LIGHT = 299792458.0;
const POWER_COLOR = "#0072BD";
const PHASE_COLOR = "#D95319";
const AXIS_COLOR = "#303030";

const state = {
  worker: null,
  ready: false,
  running: false,
  lastRun: null,
  display: null,
  phaseFit: null,
  compressor: null,
  sweep: null,
  sweepCompressionCache: new Map(),
  lastSweepSettings: null,
  pendingSweepSettings: null,
  fixedRequestId: 0,
  exportPending: null,
};

const plotConfig = {
  responsive: true,
  displaylogo: false,
  scrollZoom: true,
  modeBarButtonsToRemove: [
    "lasso2d", "select2d", "zoomIn2d", "zoomOut2d", "autoScale2d",
    "hoverClosestCartesian", "hoverCompareCartesian", "toggleSpikelines"
  ],
  toImageButtonOptions: { format: "png", scale: 2 },
};

function baseLayout(title, xTitle, yTitle, { showLegend = false, legendPosition = "top-left" } = {}) {
  const legend = legendPosition === "bottom"
    ? { orientation: "h", x: 0.02, y: 0.02, xanchor: "left", yanchor: "bottom" }
    : { orientation: "v", x: 0.02, y: 0.98, xanchor: "left", yanchor: "top" };
  return {
    title: { text: title, font: { size: 13 }, x: 0.02, xanchor: "left", y: 0.91, yanchor: "top" },
    margin: { l: 72, r: 28, t: 68, b: 62 },
    paper_bgcolor: "white",
    plot_bgcolor: "white",
    xaxis: {
      title: { text: xTitle, font: { color: AXIS_COLOR } },
      tickfont: { color: AXIS_COLOR },
      linecolor: AXIS_COLOR,
      mirror: false,
      automargin: true,
      zeroline: false,
    },
    yaxis: {
      title: { text: yTitle, font: { color: AXIS_COLOR } },
      tickfont: { color: AXIS_COLOR },
      linecolor: AXIS_COLOR,
      automargin: true,
      zeroline: false,
    },
    showlegend: showLegend,
    legend: {
      ...legend,
      bgcolor: "rgba(255,255,255,0.68)",
      bordercolor: "rgba(80,80,80,0.28)",
      borderwidth: 1,
      font: { size: 11 },
    },
    hovermode: "closest",
  };
}

function dualLayout(title, xTitle, yTitle, y2Title, y2Range = null, options = {}) {
  const l = baseLayout(title, xTitle, yTitle, options);
  l.margin.l = 78;
  l.margin.r = 82;
  l.yaxis.color = POWER_COLOR;
  l.yaxis.linecolor = POWER_COLOR;
  l.yaxis.tickfont = { color: POWER_COLOR };
  l.yaxis.title = { text: yTitle, font: { color: POWER_COLOR } };
  l.yaxis2 = {
    title: { text: y2Title, font: { color: PHASE_COLOR } },
    tickfont: { color: PHASE_COLOR },
    color: PHASE_COLOR,
    linecolor: PHASE_COLOR,
    overlaying: "y",
    side: "right",
    automargin: true,
    showgrid: false,
    zeroline: false,
  };
  if (y2Range) l.yaxis2.range = y2Range;
  return l;
}

function spectralAxisLabel(unit) {
  return unit === "nm" ? "Wavelength λ (nm)" : "Physical frequency ν (THz)";
}

function cleanErrorMessage(message) {
  const text = String(message || "Unknown error").trim();
  const lines = text.split(/\r?\n/).map((x) => x.trim()).filter(Boolean);
  const typed = [...lines].reverse().find((x) => /^(RuntimeError|ValueError|OverflowError|ZeroDivisionError|TypeError|MemoryError|IndexError|PythonError):/.test(x));
  return (typed || lines.at(-1) || text).replace(/^PythonError:\s*/, "");
}

function showAlert(title, message, kind = "error", details = "") {
  const dialog = $("alert-dialog");
  $("alert-dialog-title").textContent = title;
  $("alert-dialog-content").textContent = String(message);
  const detailBox = $("alert-dialog-details");
  const detailText = $("alert-dialog-details-text");
  if (details && String(details).trim() && String(details).trim() !== String(message).trim()) {
    detailText.textContent = String(details);
    detailBox.hidden = false;
  } else {
    detailText.textContent = "";
    detailBox.hidden = true;
  }
  dialog.classList.remove("error-dialog", "warning-dialog");
  dialog.classList.add(kind === "warning" ? "warning-dialog" : "error-dialog");
  if (dialog.open) dialog.close();
  dialog.showModal();
}

function runBoundaryMessage(metrics) {
  if (!metrics) return "";
  const kinds = [];
  if (metrics.grid_warning_frequency) kinds.push("frequency-grid boundary");
  if (metrics.grid_warning_time) kinds.push("time-window boundary");
  if (!kinds.length) return "";
  const pct = Number(metrics.warning_percent);
  const where = Number.isFinite(pct) && pct > 0 ? ` near ${pct}% of the propagation` : " very early in the propagation";
  return `The signal reached the ${kinds.join(" and the ")}${where}. The simulation completed, but results beyond that point may be affected by the finite computational grid.`;
}

function setStatus(message, kind = "loading") {
  $("runtime-status").textContent = message;
  $("runtime-dot").className = `runtime-dot ${kind}`;
}

function setProgress(scope, value) {
  const id = scope === "sweep" ? "sweep-progress" : "progress";
  const label = scope === "sweep" ? "sweep-progress-label" : "progress-label";
  const v = Math.max(0, Math.min(100, Number(value) || 0));
  $(id).value = v;
  $(label).textContent = `${Math.round(v)}%`;
}

function numberValue(id) {
  const v = Number($(id).value);
  if (!Number.isFinite(v)) throw new Error(`Invalid numeric value: ${id}.`);
  return v;
}

function checkedRadio(name) {
  return document.querySelector(`input[name="${name}"]:checked`)?.value;
}

function showMessage(id, text, kind = "error") {
  const el = $(id);
  el.textContent = text;
  el.className = `message ${kind}`;
  el.hidden = false;
}

function hideMessage(id) {
  $(id).hidden = true;
}

function format(value, digits = 5) {
  const n = Number(value);
  if (!Number.isFinite(n)) return "—";
  const a = Math.abs(n);
  if ((a !== 0 && a < 1e-4) || a >= 1e6) return n.toExponential(4);
  return n.toFixed(digits).replace(/\.?0+$/, "");
}

function collectParams() {
  const gas = $("gas").value;
  const p = {
    gas,
    pressure: numberValue("pressure"),
    fiber_length_cm: numberValue("fiber_length_cm"),
    radius_um: numberValue("radius_um"),
    pulse_shape: $("pulse_shape").value,
    energy_mJ: numberValue("energy_mJ"),
    tl_fwhm_fs: numberValue("tl_fwhm_fs"),
    wavelength_nm: numberValue("wavelength_nm"),
    GDD_fs2: numberValue("GDD_fs2"),
    TOD_fs3: numberValue("TOD_fs3"),
    NZ: Math.round(numberValue("NZ")),
    n_exp: Math.round(numberValue("n_exp")),
    XN: numberValue("XN"),
    n_z_spectra: Math.round(numberValue("n_z_spectra")),
    enable_spm: $("enable_spm").checked,
    enable_self_steepening: $("enable_self_steepening").checked,
    enable_gvd: $("enable_gvd").checked,
    enable_tod: $("enable_tod").checked,
    enable_loss: $("enable_loss").checked,
    self_steepening_solver: $("self_steepening_solver").value,
  };
  if (gas === "Manual") {
    p.beta2_fs2_m = numberValue("beta2_fs2_m");
    p.beta3_fs3_m = numberValue("beta3_fs3_m");
    p.alpha_1_m = numberValue("alpha_1_m");
    p.gamma_1_W_m = numberValue("gamma_1_W_m");
  }
  if (p.NZ < 2 || p.NZ > 5000) throw new Error("Propagation points NZ must be between 2 and 5000.");
  if (p.n_exp < 9 || p.n_exp > 15) throw new Error("FFT exponent n must be between 9 and 15.");
  if (p.n_z_spectra < 2 || p.n_z_spectra > 100) throw new Error("Saved z samples must be between 2 and 100.");
  if (p.XN <= 0) throw new Error("Time-window factor must be positive.");
  return p;
}

function toggleManual() {
  const manual = $("gas").value === "Manual";
  for (const id of ["beta2_fs2_m", "beta3_fs3_m", "alpha_1_m", "gamma_1_W_m"]) {
    $(id).readOnly = !manual;
    if (manual && $(id).value === "") $(id).value = id === "gamma_1_W_m" ? "1e-9" : "0";
  }
}

function setCoeffFields(coeff) {
  if (!coeff) return;
  $("beta2_fs2_m").value = format(coeff.beta2_fs2_m, 8);
  $("beta3_fs3_m").value = format(coeff.beta3_fs3_m, 8);
  $("alpha_1_m").value = format(coeff.alpha_1_m, 8);
  $("gamma_1_W_m").value = Number(coeff.gamma_1_W_m).toExponential(6);
}

function switchTab(name) {
  document.querySelectorAll(".tab-button").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  document.querySelectorAll(".tab-panel").forEach((p) => p.classList.toggle("active", p.id === `tab-${name}`));
  requestAnimationFrame(() => {
    document.querySelectorAll(`#tab-${name} .js-plotly-plot`).forEach((plot) => Plotly.Plots.resize(plot));
  });
}

function renderInputPreview(result) {
  setCoeffFields(result.coefficients);
  const t = result.time;
  const s = result.spectrum;
  Plotly.react("plot-input-time", [
    { x: t.fs, y: t.power_mJ_fs, name: "Power", type: "scatter", mode: "lines", line: { color: POWER_COLOR } },
    { x: t.fs, y: t.phase_rad, name: "Phase", type: "scatter", mode: "lines", line: { color: PHASE_COLOR }, yaxis: "y2" },
  ], dualLayout("Pulse power & phase", "Retarded time T (fs)", "Pulse power (mJ/fs)", "Temporal phase φ(T) (rad)", [-Math.PI, Math.PI]), plotConfig);

  Plotly.react("plot-input-spectrum", [
    { x: s.THz, y: s.power_mW_THz, name: "Spectrum", type: "scatter", mode: "lines", line: { color: POWER_COLOR } },
    { x: s.THz, y: s.phase_rad, name: "Phase", type: "scatter", mode: "lines", line: { color: PHASE_COLOR }, yaxis: "y2" },
  ], dualLayout("Spectral power density & phase", "Physical frequency ν (THz)", "Spectral power density (mW/THz)", "Spectral phase φ(ν) (rad)", [-Math.PI, Math.PI]), plotConfig);
}

function requestPreview(showErrors = true) {
  if (!state.ready) return;
  try {
    if (showErrors) hideMessage("settings-error");
    const params = collectParams();
    state.worker.postMessage({ type: "preview", params });
  } catch (e) {
    if (showErrors) showMessage("settings-error", e.message);
  }
}

function resultsAxis() { return checkedRadio("results-axis") || "THz"; }
function phaseAxis() { return checkedRadio("phase-axis") || "THz"; }
function compressorAxis() { return checkedRadio("compressor-axis") || "THz"; }
function sweepAxis() { return checkedRadio("sweep-axis") || "THz"; }

function axisData(spectrum, unit) {
  return unit === "nm" ? spectrum.nm : spectrum.THz;
}

function finitePairs(x, y) {
  const xx = [], yy = [];
  for (let i = 0; i < Math.min(x.length, y.length); i++) {
    if (x[i] != null && y[i] != null && Number.isFinite(Number(x[i])) && Number.isFinite(Number(y[i]))) {
      xx.push(Number(x[i])); yy.push(Number(y[i]));
    }
  }
  return [xx, yy];
}

function renderResultSummary() {
  if (!state.lastRun) return;
  const m = state.lastRun.metrics;
  $("out-energy").textContent = format(m.out_energy_mJ, 4);
  $("out-transmission").textContent = format(m.transmission, 2);
  $("out-broadening").textContent = format(m.broadening, 3);
  $("out-tl-fwhm").textContent = format(m.tl_fwhm_fs, 3);
  const warning = runBoundaryMessage(m);
  if (warning) showMessage("results-message", warning, "warning"); else hideMessage("results-message");
}

function renderResults() {
  if (!state.lastRun || !state.display) return;
  const r = state.lastRun;
  const d = state.display;
  const unit = resultsAxis();
  const hidePhase = $("hide-phase").checked;
  const sx = axisData(d.spectrum, unit);
  const suffix = d.suffix;

  const timeTraces = [{ x: d.time.fs, y: d.time.power_mJ_fs, name: "Power", type: "scatter", mode: "lines", line: { color: POWER_COLOR } }];
  if (!hidePhase) timeTraces.push({ x: d.time.fs, y: d.time.phase_rad, name: "Phase", type: "scatter", mode: "markers", marker: { size: 2, color: PHASE_COLOR }, yaxis: "y2" });
  const timeLayout = dualLayout(`Pulse power & phase<br><span style="font-size:11px">${suffix}</span>`, "Retarded time T (fs)", "Pulse power (mJ/fs)", "Temporal phase φ(T) (rad)", [-Math.PI, Math.PI]);
  if (hidePhase) delete timeLayout.yaxis2;
  Plotly.react("plot-time", timeTraces, timeLayout, plotConfig);

  const specTraces = [{ x: sx, y: d.spectrum.power_mW_THz, name: "Spectrum", type: "scatter", mode: "lines", line: { color: POWER_COLOR } }];
  if (!hidePhase) specTraces.push({ x: sx, y: d.spectrum.phase_rad, name: "Phase", type: "scatter", mode: "markers", marker: { size: 2, color: PHASE_COLOR }, yaxis: "y2" });
  const specLayout = dualLayout(`Spectral power density & phase<br><span style="font-size:11px">${suffix}</span>`, spectralAxisLabel(unit), "Spectral power density (mW/THz)", "Spectral phase φ(ν) (rad)", [-Math.PI, Math.PI]);
  if (hidePhase) delete specLayout.yaxis2;
  Plotly.react("plot-spectrum", specTraces, specLayout, plotConfig);

  Plotly.react("plot-tl", [{ x: r.tl.fs, y: r.tl.power_mJ_fs, name: "TL pulse", type: "scatter", mode: "lines" }], baseLayout("TL pulse power", "Retarded time T (fs)", "Pulse power (mJ/fs)"), plotConfig);
  Plotly.react("plot-chirp", [{ x: d.time.fs, y: d.time.chirp_THz, name: "Δν", type: "scatter", mode: "lines" }], baseLayout(`Instantaneous frequency shift<br><span style="font-size:11px">${suffix}</span>`, "Retarded time T (fs)", "Frequency shift Δν (THz)"), plotConfig);
  Plotly.react("plot-gd", [{ x: sx, y: d.spectrum.group_delay_fs, name: "GD", type: "scatter", mode: "lines" }], baseLayout(`Group delay<br><span style="font-size:11px">${suffix}</span>`, spectralAxisLabel(unit), "Group delay (fs)"), plotConfig);

  const evo = checkedRadio("evolution-mode") || "Spectrum";
  if (evo === "Time") {
    const m = r.temporal_map;
    Plotly.react("plot-evolution", [{ x: m.fs, y: m.z_cm, z: m.power_mJ_fs, type: "heatmap", colorscale: "Viridis", colorbar: { title: "mJ/fs" } }], baseLayout(`Temporal evolution (${r.n_z_spectra_saved} z samples)<br><span style="font-size:11px">pre-compensation</span>`, "Retarded time T (fs)", "Propagation distance z (cm)"), plotConfig);
  } else {
    const m = r.spectral_map;
    let x = unit === "nm" ? m.nm : m.THz;
    let z = m.power_mW_THz;
    if (unit === "nm") {
      const order = x.map((v, i) => [v, i]).filter(([v]) => v != null && Number.isFinite(Number(v))).sort((a, b) => a[0] - b[0]).map(([, i]) => i);
      x = order.map((i) => x[i]);
      z = z.map((row) => order.map((i) => row[i]));
    }
    Plotly.react("plot-evolution", [{ x, y: m.z_cm, z, type: "heatmap", colorscale: "Viridis", colorbar: { title: "mW/THz" } }], baseLayout(`Spectral evolution (${r.n_z_spectra_saved} z samples)<br><span style="font-size:11px">pre-compensation</span>`, spectralAxisLabel(unit), "Propagation distance z (cm)"), plotConfig);
  }
}

function requestFixedDisplay() {
  if (!state.lastRun) return;
  if ((checkedRadio("post-mode") || "Raw") === "Raw") {
    state.display = state.lastRun.raw_display;
    renderResults();
    return;
  }
  const gdd = numberValue("results-fixed-gdd");
  state.fixedRequestId += 1;
  state.worker.postMessage({ type: "fixed-gdd", gdd_fs2: gdd, requestId: state.fixedRequestId });
}

function setPhaseDefaults() {
  if (!state.lastRun) return;
  const unit = phaseAxis();
  const lim = state.lastRun.phase_default_limits[unit];
  if (lim && lim.length === 2) {
    $("phase-min").value = Number(lim[0]).toFixed(2);
    $("phase-max").value = Number(lim[1]).toFixed(2);
  }
  updatePhaseLabels();
}

function updatePhaseLabels() {
  const unit = phaseAxis();
  $("phase-min-label").textContent = unit === "nm" ? "min. wavelength (nm)" : "min. freq (THz)";
  $("phase-max-label").textContent = unit === "nm" ? "max. wavelength (nm)" : "max. freq (THz)";
}

function convertPhaseRange(oldUnit, newUnit) {
  if (oldUnit === newUnit) return;
  let a = Number($("phase-min").value), b = Number($("phase-max").value);
  if (!(Number.isFinite(a) && Number.isFinite(b) && a > 0 && b > 0)) { setPhaseDefaults(); return; }
  let freqs;
  if (oldUnit === "THz") freqs = [a, b].map((v) => v * 1e12);
  else freqs = [a, b].map((v) => C_LIGHT / (v * 1e-9));
  let vals;
  if (newUnit === "THz") vals = freqs.map((v) => v * 1e-12);
  else vals = freqs.map((v) => C_LIGHT / v * 1e9);
  vals.sort((x, y) => x - y);
  $("phase-min").value = vals[0].toFixed(2);
  $("phase-max").value = vals[1].toFixed(2);
}

function visibleYRange(series, xmin, xmax, pad = 0.08) {
  const vals = [];
  for (const [x, y] of series) {
    for (let i = 0; i < Math.min(x.length, y.length); i++) {
      const xv = Number(x[i]), yv = Number(y[i]);
      if (Number.isFinite(xv) && Number.isFinite(yv) && xv >= xmin && xv <= xmax) vals.push(yv);
    }
  }
  if (!vals.length) return undefined;
  let lo = Math.min(...vals), hi = Math.max(...vals);
  let span = hi - lo;
  if (!(span > 0)) span = 0.1 * Math.max(Math.abs(lo), 1);
  return [lo - pad * span, hi + pad * span];
}

function renderPhase() {
  if (!state.lastRun) {
    Plotly.react("plot-phase", [], baseLayout("Spectral phase", spectralAxisLabel(phaseAxis()), "Spectral phase φ (rad)"), plotConfig);
    Plotly.react("plot-phase-spectrum", [], baseLayout("Spectral power density", spectralAxisLabel(phaseAxis()), "Spectral power density (mW/THz)"), plotConfig);
    return;
  }
  const p = state.lastRun.phase_display;
  const unit = phaseAxis();
  const x = unit === "nm" ? p.nm : p.THz;
  const pairs = finitePairs(x, p.phase_unwrapped_rad);
  const powerPairs = finitePairs(x, p.power_mW_THz);
  const traces = [{ x: pairs[0], y: pairs[1], type: "scatter", mode: "lines", name: "Spectral phase", line: { color: PHASE_COLOR } }];
  const series = [[pairs[0], pairs[1]]];
  if (state.phaseFit) {
    const fx = unit === "nm" ? state.phaseFit.x_nm : state.phaseFit.x_THz;
    const fp = finitePairs(fx, state.phaseFit.fitted_rad);
    traces.push({ x: fp[0], y: fp[1], type: "scatter", mode: "lines", name: "Polynomial fit", line: { color: "black" } });
    series.push([fp[0], fp[1]]);
  }
  const finiteX = pairs[0];
  const xmin = Math.min(...finiteX), xmax = Math.max(...finiteX);
  const layout = baseLayout("Spectral phase", spectralAxisLabel(unit), "Spectral phase φ (rad)", {
    showLegend: Boolean(state.phaseFit), legendPosition: "top-left"
  });
  if (Number.isFinite(xmin) && Number.isFinite(xmax)) {
    layout.xaxis.range = [xmin, xmax];
    const yr = visibleYRange(series, xmin, xmax);
    if (yr) layout.yaxis.range = yr;
  }
  Plotly.react("plot-phase", traces, layout, plotConfig);
  const pl = baseLayout("Spectral power density", spectralAxisLabel(unit), "Spectral power density (mW/THz)");
  if (Number.isFinite(xmin) && Number.isFinite(xmax)) pl.xaxis.range = [xmin, xmax];
  Plotly.react("plot-phase-spectrum", [{ x: powerPairs[0], y: powerPairs[1], type: "scatter", mode: "lines", name: "Spectrum", line: { color: POWER_COLOR } }], pl, plotConfig);
}

function requestPhaseFit() {
  if (!state.lastRun) { showMessage("phase-message", "Run a propagation first (Start!)."); return; }
  try {
    hideMessage("phase-message");
    state.worker.postMessage({ type: "phase-fit", request: { min: numberValue("phase-min"), max: numberValue("phase-max"), order: Math.round(numberValue("phase-order")), unit: phaseAxis() } });
  } catch (e) { const m=cleanErrorMessage(e.message); showMessage("phase-message", m); showAlert("Phase Analysis error",m,"error"); }
}

const metricMap = {
  "Main FWHM": "fwhm",
  "95%-bounded FWHM": "outer_fwhm95",
  "RMS duration": "rms",
  "95% energy width": "energy95",
};

function renderCompressor(c, optimized = false) {
  state.compressor = c;
  const m = c.metrics, o = c.output_metrics, tl = c.tl_metrics;
  $("compressor-gdd").value = format(c.gdd_fs2, 8);
  $("compressor-metrics").innerHTML =
    `Output: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; FWHM ${format(o.fwhm_fs,3)} fs &nbsp;&nbsp; 95%-bounded FWHM ${format(o.outer_fwhm95_fs,3)} fs &nbsp;&nbsp; RMS σ ${format(o.rms_fs,3)} fs &nbsp;&nbsp; 95% energy ${format(o.energy95_fs,3)} fs<br>` +
    `After GDD: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; FWHM ${format(m.fwhm_fs,3)} fs &nbsp;&nbsp; 95%-bounded FWHM ${format(m.outer_fwhm95_fs,3)} fs &nbsp;&nbsp; RMS σ ${format(m.rms_fs,3)} fs &nbsp;&nbsp; 95% energy ${format(m.energy95_fs,3)} fs<br>` +
    `Transform limited: &nbsp; FWHM ${format(tl.fwhm_fs,3)} fs &nbsp;&nbsp; 95%-bounded FWHM ${format(tl.outer_fwhm95_fs,3)} fs &nbsp;&nbsp; RMS σ ${format(tl.rms_fs,3)} fs &nbsp;&nbsp; 95% energy ${format(tl.energy95_fs,3)} fs &nbsp;&nbsp; | &nbsp; Applied GDD ${format(c.gdd_fs2,6)} fs²`;
  if (optimized) {
    const opt = c.optimization || {};
    $("compressor-status").textContent = `Optimized on ${$("compressor-metric").value}. Best GDD = ${format(c.gdd_fs2,6)} fs² via ${opt.source || "search"}. Weighted-fit compensation seed = ${format(opt.fit_seed_gdd_fs2,6)} fs².`;
  } else {
    $("compressor-status").textContent = `Applied pure quadratic spectral phase: GDD = ${format(c.gdd_fs2,6)} fs².`;
  }
  const t = c.time;
  Plotly.react("plot-compressor-time", [
    { x: t.output_fs, y: t.output_mJ_fs, name: "Fiber output", type: "scatter", mode: "lines", opacity: 0.65 },
    { x: t.compressed_fs, y: t.compressed_mJ_fs, name: "After GDD", type: "scatter", mode: "lines" },
    { x: t.tl_fs, y: t.tl_mJ_fs, name: "Transform limited", type: "scatter", mode: "lines", line: { dash: "dash" } },
  ], { ...baseLayout("Output vs GDD-compressed vs TL", "Time relative to pulse peak (fs)", "Pulse power (mJ/fs)", { showLegend: true, legendPosition: "top-left" }), xaxis: { title: { text: "Time relative to pulse peak (fs)" }, range: [-t.half_width_fs, t.half_width_fs], automargin: true, zeroline: false } }, plotConfig);
  renderCompressorSpectrum();
}

function renderCompressorSpectrum() {
  if (!state.compressor) return;
  const s = state.compressor.spectrum;
  const unit = compressorAxis();
  const x = unit === "nm" ? s.nm : s.THz;
  Plotly.react("plot-compressor-spectrum", [
    { x, y: s.power_mW_THz, name: "Spectrum", type: "scatter", mode: "lines", line: { color: POWER_COLOR } },
    { x, y: s.phase_rad, name: "Phase", type: "scatter", mode: "markers", marker: { size: 2, color: PHASE_COLOR }, yaxis: "y2" },
  ], dualLayout("Spectrum after GDD transfer", spectralAxisLabel(unit), "Spectral power density (mW/THz)", "Spectral phase φ(ν) (rad)", [-Math.PI, Math.PI]), plotConfig);
}

function requestCompressor(optimize) {
  if (!state.lastRun) { showMessage("compressor-error", "Run a propagation first (Start!)."); return; }
  hideMessage("compressor-error");
  if (optimize) {
    $("compressor-status").textContent = "Optimizing GDD locally…";
    state.worker.postMessage({ type: "compressor-optimize", request: { metric: metricMap[$("compressor-metric").value] || "outer_fwhm95" } });
  } else {
    try { state.worker.postMessage({ type: "compressor-apply", request: { gdd_fs2: numberValue("compressor-gdd") } }); }
    catch (e) { const m=cleanErrorMessage(e.message); showMessage("compressor-error", m); showAlert("GDD Compressor error",m,"error"); }
  }
}

const sweepParamMap = {
  "Pressure (atm)": ["pressure", "Pressure (atm)"],
  "Fiber length (cm)": ["fiber_length_cm", "Fiber length (cm)"],
  "Radius (µm)": ["radius_um", "Radius (µm)"],
  "Energy (mJ)": ["energy_mJ", "Energy (mJ)"],
  "TL FWHM (fs)": ["tl_fwhm_fs", "TL FWHM (fs)"],
  "Wavelength (nm)": ["wavelength_nm", "Wavelength (nm)"],
  "Input GDD (fs²)": ["GDD_fs2", "Input GDD (fs²)"],
  "Input TOD (fs³)": ["TOD_fs3", "Input TOD (fs³)"],
  "β2 (fs²/m)": ["beta2_fs2_m", "β2 (fs²/m)"],
  "β3 (fs³/m)": ["beta3_fs3_m", "β3 (fs³/m)"],
  "α (1/m)": ["alpha_1_m", "α (1/m)"],
  "γ (1/(W·m))": ["gamma_1_W_m", "γ (1/(W·m))"],
  "Compensating GDD (fs²)": ["output_GDD_fs2", "Compensating GDD (fs²)"],
};

function sweepInfo() { return sweepParamMap[$("sweep-parameter").value]; }

function updateSweepDefaults() {
  const [key] = sweepInfo();
  let current = 0, lo, hi;
  const coeffKeys = ["beta2_fs2_m", "beta3_fs3_m", "alpha_1_m", "gamma_1_W_m"];
  if (key === "output_GDD_fs2") {
    current = Number($("compressor-gdd").value) || 0; const span = Math.max(100, Math.abs(current)); lo = current - span; hi = current + span;
  } else if (coeffKeys.includes(key)) {
    current = Number($(key).value);
    if (!Number.isFinite(current)) return;
    if ((key === "beta2_fs2_m" || key === "beta3_fs3_m") && current === 0) { const half = key === "beta2_fs2_m" ? 1 : 10; lo = -half; hi = half; }
    else { [lo, hi] = [0.5 * current, 1.5 * current].sort((a,b)=>a-b); if (lo === hi) hi = lo + Math.max(Math.abs(lo), 1); }
  } else {
    current = Number($(key).value); if (!Number.isFinite(current)) return;
    if (key === "fiber_length_cm") { lo = Math.max(0, 0.1 * current); hi = current; }
    else if (key === "GDD_fs2" || key === "TOD_fs3") { const span = Math.max(key === "GDD_fs2" ? 100 : 1000, Math.abs(current)); lo = current - span; hi = current + span; }
    else { lo = Math.max(0, 0.5 * current); hi = 1.5 * current; }
  }
  $("sweep-min").value = format(lo, 8); $("sweep-max").value = format(hi, 8);
  updateSweepPreview(); updateSweepCompressionControls();
}

function generateSweepValues() {
  const vmin = numberValue("sweep-min"), vmax = numberValue("sweep-max"), n = Math.round(numberValue("sweep-points"));
  if (n < 2 || n > 80) throw new Error("N values must be between 2 and 80 in the browser version.");
  if (!(vmin < vmax)) throw new Error("Sweep minimum must be smaller than maximum.");
  const spacing = $("sweep-spacing").value;
  const base = numberValue("sweep-base");
  if (!(base > 0)) throw new Error("Spacing base must be positive.");
  const values = [];
  for (let i = 0; i < n; i++) {
    const u = i / (n - 1); let s = u;
    if (Math.abs(base - 1) >= 1e-12) {
      if (spacing === "Exponential") s = (Math.pow(base, u) - 1) / (base - 1);
      else if (spacing === "Logarithmic") s = Math.log(1 + (base - 1) * u) / Math.log(base);
    }
    values.push(vmin + (vmax - vmin) * s);
  }
  values[0] = vmin; values[values.length - 1] = vmax;
  return { values, spacing, base: spacing === "Linear" ? null : base };
}

function updateSweepPreview() {
  try {
    const s = generateSweepValues();
    $("sweep-base").disabled = s.spacing === "Linear";
    let shown;
    if (s.values.length <= 18) shown = s.values.map((x) => format(x, 7)).join(", ");
    else shown = `${s.values.slice(0,8).map((x)=>format(x,7)).join(", ")}, …, ${s.values.slice(-5).map((x)=>format(x,7)).join(", ")}`;
    $("sweep-preview").textContent = `${s.spacing}${s.base == null ? "" : `, base=${format(s.base,5)}`} | ${s.values.length} values: [${shown}]`;
  } catch (e) { $("sweep-preview").textContent = `Invalid sweep definition: ${e.message}`; }
}

function currentPhaseFitConfig() {
  const order = Math.round(numberValue("phase-order"));
  if (order < 2) throw new Error("Phase Analysis polynomial order must be at least 2 to extract GDD during a sweep.");
  const a = Number($("phase-min").value), b = Number($("phase-max").value);
  if (!(Number.isFinite(a) && Number.isFinite(b) && a < b)) return { mode: "auto", order };
  if (phaseAxis() === "nm") {
    if (a <= 0 || b <= 0) return { mode: "auto", order };
    const fa = C_LIGHT / (a * 1e-9) * 1e-12, fb = C_LIGHT / (b * 1e-9) * 1e-12;
    return { mode: "fixed", order, min_f_THz: Math.min(fa, fb), max_f_THz: Math.max(fa, fb) };
  }
  return { mode: "fixed", order, min_f_THz: a, max_f_THz: b };
}

function updateSweepCompressionControls() {
  const key = state.sweep?.parameter_key || sweepInfo()[0];
  const select = $("sweep-compression-mode");
  const choices = key === "output_GDD_fs2" ? ["Swept GDD", "Optimized GDD", "Fixed GDD"] : ["Optimized GDD", "Fixed GDD"];
  const current = select.value;
  select.innerHTML = choices.map((x) => `<option>${x}</option>`).join("");
  select.value = choices.includes(current) ? current : (key === "output_GDD_fs2" ? "Swept GDD" : "Optimized GDD");
  const relevant = ["Compressed duration (fs)", "Pulse power map (compressed)"].includes($("sweep-plot").value);
  select.disabled = !relevant;
  $("sweep-fixed-gdd").disabled = !relevant || select.value !== "Fixed GDD";
}

function currentSweepCompressionRequest() {
  const label = $("sweep-compression-mode").value;
  const mode = label === "Fixed GDD" ? "fixed" : label === "Swept GDD" ? "swept" : "optimized";
  return { mode, fixed_gdd_fs2: mode === "fixed" ? numberValue("sweep-fixed-gdd") : 0, metric: state.sweep?.optimize_metric || metricMap[$("sweep-metric").value] || "outer_fwhm95" };
}

function runSweep() {
  try {
    hideMessage("sweep-error");
    const base_params = collectParams();
    const [parameter_key, parameter_label] = sweepInfo();
    const generated = generateSweepValues();
    const phase_fit_config = currentPhaseFitConfig();
    const optimize_metric = metricMap[$("sweep-metric").value] || "outer_fwhm95";
    const snapshot = {
      base_params: structuredClone(base_params),
      base_coefficients: {
        beta2_fs2_m: Number($("beta2_fs2_m").value), beta3_fs3_m: Number($("beta3_fs3_m").value), alpha_1_m: Number($("alpha_1_m").value), gamma_1_W_m: Number($("gamma_1_W_m").value),
      },
      parameter_key, parameter_label, values: generated.values.slice(), sweep_spacing: generated.spacing, sweep_spacing_base: generated.base,
      phase_fit_config: structuredClone(phase_fit_config), compressor_label: $("sweep-metric").value,
      compressed_plot_mode: $("sweep-compression-mode").value, compressed_fixed_gdd_fs2: Number($("sweep-fixed-gdd").value),
    };
    state.pendingSweepSettings = snapshot;
    $("view-last-sweep-settings").disabled = true;
    $("sweep-run").disabled = true;
    setProgress("sweep", 0);
    $("sweep-status").textContent = parameter_key === "output_GDD_fs2" ? `Running one fiber propagation, then applying ${generated.values.length} values of ${parameter_label}…` : `Running ${generated.values.length} values of ${parameter_label}; compressor criterion: ${$("sweep-metric").value}…`;
    state.worker.postMessage({ type: "sweep", request: { base_params, parameter_key, values: generated.values, phase_fit_config, optimize_metric, parameter_label } });
  } catch (e) { const m=cleanErrorMessage(e.message); showMessage("sweep-error", m); showAlert("Parameter Sweep error",m,"error"); }
}

function renderSweepTable() {
  const r = state.sweep; if (!r) return;
  const label = r.parameter_label || r.parameter_key;
  $("sweep-table-param").textContent = label;
  const tbody = $("sweep-table").querySelector("tbody"); tbody.innerHTML = "";
  for (let i = 0; i < r.values.length; i++) {
    const tr = document.createElement("tr"); if (r.boundary_warning[i]) tr.classList.add("warning");
    const warning = r.boundary_warning[i] ? `${r.boundary_warning_type[i]} @ ${r.boundary_warning_percent[i]}%` : "OK";
    const vals = [r.values[i], r.phase_gdd_fs2[i], r.optimized_gdd_fs2[i], r.actual_output_fwhm_fs[i], r.tl_duration_fs[i], r.optimized_compressed_duration_fs[i], warning];
    vals.forEach((v, j) => { const td = document.createElement("td"); td.textContent = j === 6 ? v : format(v, 7); tr.appendChild(td); });
    tbody.appendChild(tr);
  }
}

function compressionCacheKey(req) { return `${req.mode}|${Number(req.fixed_gdd_fs2).toFixed(10)}|${req.metric}`; }

function ensureSweepCompression(callback) {
  const req = currentSweepCompressionRequest(); const key = compressionCacheKey(req);
  if (state.sweepCompressionCache.has(key)) { callback(state.sweepCompressionCache.get(key)); return; }
  $("sweep-status").textContent = "Building compressed sweep view locally…";
  state.worker.postMessage({ type: "sweep-compression", request: req });
}

function renderSweepPlot() {
  const r = state.sweep;
  if (!r) { Plotly.react("plot-sweep", [], baseLayout("Parameter sweep", sweepInfo()[1], $("sweep-plot").value), plotConfig); return; }
  const selected = $("sweep-plot").value;
  const xLabel = r.parameter_label || r.parameter_key;
  if (selected === "Spectral power map") {
    let x = r.spectral_map.THz.slice(), z = r.spectral_map.power_mW_THz.map((row) => row.slice()), xlabel = "Physical frequency ν (THz)";
    if (sweepAxis() === "nm") {
      const nm = x.map((v) => v > 0 ? C_LIGHT/(v*1e12)*1e9 : null);
      const order = nm.map((v,i)=>[v,i]).filter(([v])=>v!=null&&Number.isFinite(v)).sort((a,b)=>a[0]-b[0]).map(([,i])=>i);
      x = order.map((i)=>nm[i]); z = z.map((row)=>order.map((i)=>row[i])); xlabel = "Wavelength λ (nm)";
    }
    Plotly.react("plot-sweep", [{ x, y: r.values, z, type: "heatmap", colorscale: "Viridis", colorbar: { title: "mW/THz" } }], baseLayout(`Spectral power vs ${xLabel}`, xlabel, xLabel), plotConfig);
  } else if (selected === "Pulse power map (uncompressed)") {
    Plotly.react("plot-sweep", [{ x: r.temporal_map.fs, y: r.values, z: r.temporal_map.power_GW, type: "heatmap", colorscale: "Viridis", colorbar: { title: "GW" } }], baseLayout(`Uncompressed pulse power vs ${xLabel}`, "Retarded time (fs)", xLabel), plotConfig);
  } else if (selected === "Pulse power map (compressed)") {
    ensureSweepCompression((c) => Plotly.react("plot-sweep", [{ x: c.time_fs, y: r.values, z: c.power_GW, type: "heatmap", colorscale: "Viridis", colorbar: { title: "GW" } }], baseLayout(`Compressed pulse power (${ $("sweep-compression-mode").value }) vs ${xLabel}`, "Retarded time (fs)", xLabel), plotConfig));
  } else if (selected === "Compressed duration (fs)") {
    const req = currentSweepCompressionRequest();
    if (req.mode === "optimized") {
      Plotly.react("plot-sweep", [{ x: r.values, y: r.optimized_compressed_duration_fs, type: "scatter", mode: "lines+markers", name: selected }], baseLayout(`${selected} vs ${xLabel}`, xLabel, selected), plotConfig);
    } else {
      ensureSweepCompression((c) => Plotly.react("plot-sweep", [{ x: r.values, y: c.duration_fs, type: "scatter", mode: "lines+markers", name: selected }], baseLayout(`${selected} (${ $("sweep-compression-mode").value }) vs ${xLabel}`, xLabel, selected), plotConfig));
    }
  } else {
    const mapping = {
      "Phase-fit GDD (fs²)": r.phase_gdd_fs2,
      "Optimized GDD (fs²)": r.optimized_gdd_fs2,
      "Actual output FWHM (fs)": r.actual_output_fwhm_fs,
      "TL duration (fs)": r.tl_duration_fs,
    };
    Plotly.react("plot-sweep", [{ x: r.values, y: mapping[selected], type: "scatter", mode: "lines+markers", name: selected }], baseLayout(`${selected} vs ${xLabel}`, xLabel, selected), plotConfig);
  }
}

function currentSettingsSnapshot() {
  let base;
  try { base = collectParams(); } catch { base = {}; }
  return {
    settings: base,
    displayed_coefficients: {
      beta2_fs2_m: Number($("beta2_fs2_m").value), beta3_fs3_m: Number($("beta3_fs3_m").value), alpha_1_m: Number($("alpha_1_m").value), gamma_1_W_m: Number($("gamma_1_W_m").value),
    },
    phase_analysis: { order: Number($("phase-order").value), min: Number($("phase-min").value), max: Number($("phase-max").value), unit: phaseAxis() },
    sweep: { parameter: $("sweep-parameter").value, min: Number($("sweep-min").value), max: Number($("sweep-max").value), points: Number($("sweep-points").value), spacing: $("sweep-spacing").value, base: Number($("sweep-base").value), compressor_criterion: $("sweep-metric").value, compressed_plot_mode: $("sweep-compression-mode").value, fixed_gdd_fs2: Number($("sweep-fixed-gdd").value) },
  };
}

function showSettingsDialog(title, data) {
  $("settings-dialog-title").textContent = title;
  $("settings-dialog-content").textContent = JSON.stringify(data, null, 2);
  $("settings-dialog").showModal();
}

function downloadText(filename, text, mime = "text/plain") {
  const blob = new Blob([text], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = filename; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function downloadBytes(filename, bytes, mime = "application/octet-stream") {
  const data = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  const blob = new Blob([data], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = filename; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}

function exportTimestamp() {
  const d = new Date();
  const z = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}${z(d.getMonth()+1)}${z(d.getDate())}_${z(d.getHours())}${z(d.getMinutes())}${z(d.getSeconds())}`;
}

function currentSingleExportRequest() {
  const postMode = checkedRadio("post-mode") || "Raw";
  const phaseRequest = state.phaseFit ? {
    min: Number($("phase-min").value), max: Number($("phase-max").value),
    order: Math.round(Number($("phase-order").value)), unit: phaseAxis()
  } : null;
  return {
    post_display: { mode: postMode, gdd_fs2: postMode === "Fixed" ? Number($("results-fixed-gdd").value) : 0 },
    phase_analysis: phaseRequest,
    phase_analysis_summary: state.phaseFit,
    compressor: state.compressor ? {
      gdd_fs2: Number(state.compressor.gdd_fs2),
      selected_metric: $("compressor-metric").value,
      optimization: state.compressor.optimization || null,
      metrics: state.compressor.metrics || null,
    } : null,
  };
}

function downloadSingleH5() {
  if (!state.lastRun || state.exportPending) return;
  state.exportPending = "single";
  $("single-download").disabled = true;
  $("single-download").textContent = "Preparing HDF5…";
  setStatus("Preparing full-resolution HDF5 export locally…", "loading");
  state.worker.postMessage({ type: "export-h5", scope: "single", request: currentSingleExportRequest() });
}

function downloadSweepH5() {
  if (!state.sweep || state.exportPending) return;
  state.exportPending = "sweep";
  $("sweep-download").disabled = true;
  $("sweep-download").textContent = "Preparing HDF5…";
  let compression = null;
  try { compression = currentSweepCompressionRequest(); } catch { compression = null; }
  setStatus("Preparing full-resolution sweep HDF5 export locally…", "loading");
  state.worker.postMessage({
    type: "export-h5", scope: "sweep",
    request: { settings_snapshot: state.lastSweepSettings, compressed_view: compression }
  });
}

function downloadSweepCsv() {
  const r = state.sweep; if (!r) return;
  const rows = [["parameter","phase_fit_GDD_fs2","optimized_GDD_fs2","actual_output_FWHM_fs","TL_duration_fs","optimized_compressed_duration_fs","grid_warning"]];
  for (let i=0;i<r.values.length;i++) rows.push([r.values[i],r.phase_gdd_fs2[i],r.optimized_gdd_fs2[i],r.actual_output_fwhm_fs[i],r.tl_duration_fs[i],r.optimized_compressed_duration_fs[i],r.boundary_warning[i] ? `${r.boundary_warning_type[i]} @ ${r.boundary_warning_percent[i]}%` : "OK"]);
  downloadText("FibDisp_sweep_table.csv", rows.map((row)=>row.map((v)=>`"${String(v).replaceAll('"','""')}"`).join(",")).join("\n"), "text/csv");
}

function resetCompressor() {
  state.compressor = null; $("compressor-gdd").value = "0"; $("compressor-metrics").innerHTML = "Output: -<br>After GDD: -<br>Transform limited: -"; $("compressor-status").textContent = "GDD = 0 fs². Enter a value manually or use Optimize GDD.";
  if (state.lastRun) state.worker.postMessage({ type: "compressor-apply", request: { gdd_fs2: 0 } });
}

function startWorker() {
  state.worker = new Worker("./worker.mjs", { type: "module" });
  state.worker.onmessage = (event) => {
    const msg = event.data || {};
    if (msg.type === "status") setStatus(msg.message, "loading");
    else if (msg.type === "ready") {
      state.ready = true; $("run-button").disabled = false; $("preview-button").disabled = false; $("sweep-run").disabled = false;
      setStatus(`Ready — Pyodide ${msg.pyodideVersion}. Computation runs on this machine.`, "ready");
      requestPreview(false);
    } else if (msg.type === "progress") setProgress(msg.scope, msg.value);
    else if (msg.type === "preview-result") { renderInputPreview(msg.result); setCoeffFields(msg.result.coefficients); }
    else if (msg.type === "run-result") {
      state.lastRun = msg.result; state.display = msg.result.raw_display; state.phaseFit = null;
      $("run-button").disabled = false; setProgress("run", 100); setStatus(`Ready — last propagation completed locally in ${format(msg.result.runtime.browser_compute_seconds,2)} s.`, "ready");
      setCoeffFields(msg.result.coefficients); renderResultSummary(); renderResults(); setPhaseDefaults(); renderPhase(); resetCompressor(); $("single-download").disabled = false; hideMessage("results-error");
      const boundaryWarning = runBoundaryMessage(msg.result.metrics);
      if (boundaryWarning) showAlert("Computational boundary warning", boundaryWarning, "warning");
    } else if (msg.type === "fixed-gdd-result") {
      if (msg.requestId === state.fixedRequestId && checkedRadio("post-mode") === "Fixed") { state.display = msg.result; renderResults(); }
    } else if (msg.type === "phase-fit-result") {
      state.phaseFit = msg.result; const names = ["p0 (rad)","p1 (fs)","p2 (fs²)","p3 (fs³)","p4 (fs⁴)"];
      $("phase-coefficients").textContent = "coefficients:  " + msg.result.coefficients.map((c,i)=>`${names[i]||`p${i}`} = ${format(c,4)}`).join("   "); renderPhase();
    } else if (msg.type === "compressor-result") {
      renderCompressor(msg.result, msg.optimized); hideMessage("compressor-error"); setStatus("Ready — local computation complete.", "ready");
    } else if (msg.type === "sweep-result") {
      state.sweep = msg.result; state.sweepCompressionCache.clear(); state.lastSweepSettings = state.pendingSweepSettings; state.pendingSweepSettings = null; $("sweep-run").disabled = false; $("sweep-download").disabled = false; $("sweep-download-csv").disabled = false; $("view-last-sweep-settings").disabled = !state.lastSweepSettings; setProgress("sweep",100); updateSweepCompressionControls(); renderSweepTable(); renderSweepPlot();
      const mode = msg.result.single_pass_length_sweep ? "Fiber-length sweep completed with one propagation to Lmax." : msg.result.single_pass_output_gdd_sweep ? "Compensating-GDD sweep completed with one fiber propagation." : "Sweep completed with one independent propagation per value.";
      $("sweep-status").textContent = `${mode} Browser compute: ${format(msg.result.browser_compute_seconds,2)} s.`; setStatus("Ready — parameter sweep complete.", "ready");
      const warningRows = msg.result.boundary_warning.map((flag, i) => flag ? i : -1).filter((i) => i >= 0);
      if (warningRows.length) {
        const preview = warningRows.slice(0, 8).map((i) => `${format(msg.result.values[i],6)}: ${msg.result.boundary_warning_type[i]} boundary @ ${msg.result.boundary_warning_percent[i]}%`).join("\n");
        const more = warningRows.length > 8 ? `\n… and ${warningRows.length - 8} additional sweep point(s).` : "";
        showAlert("Sweep boundary warning", `${warningRows.length} of ${msg.result.values.length} sweep point(s) reached a computational boundary.\n\n${preview}${more}`, "warning");
      }
    } else if (msg.type === "sweep-compression-result") {
      const key = compressionCacheKey(msg.request); state.sweepCompressionCache.set(key,msg.result); $("sweep-status").textContent = "Compressed sweep view ready."; renderSweepPlot();
    } else if (msg.type === "export-h5-result") {
      const filename = msg.scope === "sweep" ? `FibDisp_sweep_${exportTimestamp()}.h5` : `FibDisp_run_${exportTimestamp()}.h5`;
      downloadBytes(filename, msg.bytes, "application/x-hdf5");
      state.exportPending = null;
      $("single-download").disabled = !state.lastRun; $("single-download").textContent = "Save run HDF5";
      $("sweep-download").disabled = !state.sweep; $("sweep-download").textContent = "Save sweep HDF5";
      setStatus(`Ready — ${msg.scope === "sweep" ? "sweep" : "run"} HDF5 exported locally.`, "ready");
    } else if (msg.type === "error") {
      setStatus("A local computation failed.", "error");
      const friendly = cleanErrorMessage(msg.message);
      if (msg.scope === "export-h5") {
        state.exportPending = null;
        $("single-download").disabled = !state.lastRun; $("single-download").textContent = "Save run HDF5";
        $("sweep-download").disabled = !state.sweep; $("sweep-download").textContent = "Save sweep HDF5";
        showAlert("HDF5 export error", friendly, "error", msg.details || msg.message);
      } else if (msg.scope === "preview") {
        showMessage("settings-error", friendly);
      } else if (["run","fixed-gdd"].includes(msg.scope)) {
        showMessage("results-error", friendly); $("run-button").disabled = false;
        showAlert(msg.scope === "run" ? "Simulation error" : "Output GDD error", friendly, "error", msg.details || msg.message);
      } else if (msg.scope === "phase-fit") {
        showMessage("phase-message", friendly); showAlert("Phase Analysis error", friendly, "error", msg.details || msg.message);
      } else if (["compressor-apply","compressor-optimize"].includes(msg.scope)) {
        showMessage("compressor-error", friendly); showAlert("GDD Compressor error", friendly, "error", msg.details || msg.message);
      } else if (["sweep","sweep-compression"].includes(msg.scope)) {
        showMessage("sweep-error", friendly); $("sweep-run").disabled = false; state.pendingSweepSettings = null; $("view-last-sweep-settings").disabled = !state.lastSweepSettings;
        showAlert("Parameter Sweep error", friendly, "error", msg.details || msg.message);
      } else {
        showMessage("settings-error", friendly); showAlert("FibDisp error", friendly, "error", msg.details || msg.message);
      }
    }
  };
  state.worker.onerror = (error) => {
    const friendly = cleanErrorMessage(error.message || String(error));
    setStatus("WebAssembly worker failed to start.", "error"); showMessage("settings-error", friendly);
    showAlert("WebAssembly worker error", friendly, "error", error.error?.stack || "");
  };
}

// Tabs
document.querySelectorAll(".tab-button").forEach((b)=>b.addEventListener("click",()=>switchTab(b.dataset.tab)));

// Settings
$("gas").addEventListener("change",()=>{ toggleManual(); if ($("gas").value!=="Manual") requestPreview(false); updateSweepDefaults(); });
for (const id of ["pressure","radius_um","wavelength_nm"]) $(id).addEventListener("change",()=>{ if ($("gas").value!=="Manual") requestPreview(false); });
$("preview-button").addEventListener("click",()=>requestPreview(true));

// Results
$("run-button").addEventListener("click",()=>{
  try { hideMessage("results-error"); const params=collectParams(); $("run-button").disabled=true; setProgress("run",0); setStatus("Sending parameters to the local WebAssembly worker…","loading"); state.worker.postMessage({type:"run",params}); }
  catch(e){ const m=cleanErrorMessage(e.message); showMessage("results-error",m); showAlert("Simulation error",m,"error"); }
});
$("hide-phase").addEventListener("change",renderResults);
document.querySelectorAll('input[name="results-axis"]').forEach((x)=>x.addEventListener("change",renderResults));
document.querySelectorAll('input[name="evolution-mode"]').forEach((x)=>x.addEventListener("change",renderResults));
document.querySelectorAll('input[name="post-mode"]').forEach((x)=>x.addEventListener("change",requestFixedDisplay));
$("results-fixed-gdd").addEventListener("change",()=>{ if (checkedRadio("post-mode")==="Fixed") requestFixedDisplay(); });
$("single-download").addEventListener("click",downloadSingleH5);

// Phase
let previousPhaseAxis="THz";
document.querySelectorAll('input[name="phase-axis"]').forEach((x)=>x.addEventListener("change",()=>{ const next=phaseAxis(); convertPhaseRange(previousPhaseAxis,next); previousPhaseAxis=next; updatePhaseLabels(); renderPhase(); }));
$("phase-fit-button").addEventListener("click",requestPhaseFit);

// Compressor
$("compressor-apply").addEventListener("click",()=>requestCompressor(false));
$("compressor-optimize").addEventListener("click",()=>requestCompressor(true));
document.querySelectorAll('input[name="compressor-axis"]').forEach((x)=>x.addEventListener("change",renderCompressorSpectrum));

// Sweep
$("sweep-parameter").addEventListener("change",updateSweepDefaults);
for (const id of ["sweep-min","sweep-max","sweep-points","sweep-base"]) $(id).addEventListener("input",updateSweepPreview);
$("sweep-spacing").addEventListener("change",updateSweepPreview);
$("sweep-plot").addEventListener("change",()=>{ updateSweepCompressionControls(); renderSweepPlot(); });
$("sweep-compression-mode").addEventListener("change",()=>{ updateSweepCompressionControls(); renderSweepPlot(); });
$("sweep-fixed-gdd").addEventListener("change",renderSweepPlot);
document.querySelectorAll('input[name="sweep-axis"]').forEach((x)=>x.addEventListener("change",renderSweepPlot));
$("sweep-run").addEventListener("click",runSweep);
$("sweep-download").addEventListener("click",downloadSweepH5);
$("sweep-download-csv").addEventListener("click",downloadSweepCsv);
$("view-current-settings").addEventListener("click",()=>showSettingsDialog("View current settings",currentSettingsSnapshot()));
$("view-last-sweep-settings").addEventListener("click",()=>showSettingsDialog("View last run sweep settings",state.lastSweepSettings||{}));
$("settings-dialog-close").addEventListener("click",()=>$("settings-dialog").close());
$("alert-dialog-close").addEventListener("click",()=>$("alert-dialog").close());

// Initial empty plots and controls.
toggleManual(); updateSweepPreview(); updateSweepCompressionControls();
renderPhase(); renderSweepPlot();
Plotly.react("plot-compressor-time",[],baseLayout("Pulse comparison","Time relative to pulse peak (fs)","Pulse power (mJ/fs)"),plotConfig);
Plotly.react("plot-compressor-spectrum",[],dualLayout("Spectrum after GDD transfer","Physical frequency ν (THz)","Spectral power density (mW/THz)","Spectral phase φ(ν) (rad)",[-Math.PI,Math.PI]),plotConfig);
startWorker();
