/* Data Center Price Impact dashboard. Static: reads precomputed JSON from data/. No keys, no backend. */
(function () {
  "use strict";

  const DATA = {};
  const charts = {};
  const rendered = {};
  const state = { size: 300, year: 2024, ppa: false, variant: "central", scatterYear: 2024, cfYear: 2024 };
  const VARIANTS = ["central", "binned", "structural"];
  const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const MIX_LABELS = { wind: "Wind", solar: "Solar", nuclear: "Nuclear", hydro: "Hydro", other: "Other (CHP, biomass, industry)", net_import: "Net imports" };
  const REGIME_LABELS = {
    coupled: "Coupled with SE1", coupled_se3: "Coupled with SE3 only", import_constrained: "Import-constrained",
    separated_other: "Other separated", export_constrained: "Export-constrained",
  };

  // ------------------------------------------------------------ helpers
  const $ = (id) => document.getElementById(id);
  const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const palette = () => ({
    s: [1, 2, 3, 4, 5, 6].map((i) => cssVar(`--series-${i}`)),
    text: cssVar("--text-primary"), text2: cssVar("--text-secondary"), muted: cssVar("--text-muted"),
    grid: cssVar("--grid"), context: cssVar("--context"), band: cssVar("--band"), surface: cssVar("--surface-1"),
  });
  const fmt = (v, d = 1) => (v === null || v === undefined || Number.isNaN(v) ? "–" : Number(v).toFixed(d));
  const signed = (v, d = 1) => (v > 0 ? "+" : "") + fmt(v, d);
  const int = (v) => (v === null || v === undefined ? "–" : Math.round(v).toLocaleString("en-US"));
  const withAlpha = (hex, a) => {
    const h = hex.replace("#", "");
    const n = parseInt(h.length === 3 ? h.replace(/./g, "$&$&") : h, 16);
    return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${a})`;
  };

  function applyChartDefaults() {
    const p = palette();
    Chart.defaults.color = p.text2;
    Chart.defaults.borderColor = p.grid;
    Chart.defaults.font.family = getComputedStyle(document.body).fontFamily;
    Chart.defaults.font.size = 12;
    Chart.defaults.animation = { duration: 250 };
    Chart.defaults.maintainAspectRatio = false;
    Chart.defaults.plugins.legend.labels.boxWidth = 12;
    Chart.defaults.plugins.legend.labels.boxHeight = 12;
    Chart.defaults.plugins.legend.labels.usePointStyle = true;
    Chart.defaults.elements.line.pointStyle = "line";  // line datasets: line marker in the legend
    Chart.defaults.datasets.line.pointStyle = "line";
    Chart.defaults.datasets.bar.pointStyle = "rect";
    Chart.defaults.datasets.scatter.pointStyle = "circle";
    Chart.defaults.plugins.tooltip.backgroundColor = p.surface;
    Chart.defaults.plugins.tooltip.titleColor = p.text;
    Chart.defaults.plugins.tooltip.bodyColor = p.text;
    Chart.defaults.plugins.tooltip.borderColor = p.grid;
    Chart.defaults.plugins.tooltip.borderWidth = 1;
  }

  function makeChart(id, config) {
    if (charts[id]) charts[id].destroy();
    charts[id] = new Chart($(id), config);
    return charts[id];
  }

  // Draw one chart in isolation: an error is logged and shown in that chart's box only,
  // so it can never stop the other charts on the page from rendering.
  function draw(id, fn) {
    const box = $(id).parentElement;
    box.querySelectorAll(".chart-error").forEach((el) => el.remove());
    try {
      fn();
    } catch (e) {
      console.error(`Chart "${id}" failed to render:`, e);
      if (charts[id]) { charts[id].destroy(); delete charts[id]; }
      box.insertAdjacentHTML("beforeend", `<p class="chart-error">This chart could not be drawn (${e.message}).</p>`);
    }
  }
  // Same isolation for non-chart pieces (stat cards, tables).
  function guard(name, fn) {
    try { fn(); } catch (e) { console.error(`"${name}" failed to render:`, e); }
  }

  function axis(title, extra = {}) {
    return Object.assign({ title: { display: !!title, text: title }, grid: { color: palette().grid } }, extra);
  }

  function table(containerId, headers, rows) {
    const el = $(containerId);
    const head = headers.map((h) => `<th scope="col">${h}</th>`).join("");
    const body = rows.map((r) => `<tr>${r.map((c, i) => (i === 0 ? `<th scope="row">${c}</th>` : `<td>${c}</td>`)).join("")}</tr>`).join("");
    el.innerHTML = `<div class="scroll"><table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
  }

  // whiskers for bar charts: options.plugins.errorBars.ranges = [[lo, hi], ...] for dataset 0
  const errorBarsPlugin = {
    id: "errorBars",
    afterDatasetsDraw(chart, _args, opts) {
      if (!opts || !opts.ranges) return;
      const meta = chart.getDatasetMeta(0);
      const y = chart.scales.y;
      const ctx = chart.ctx;
      ctx.save();
      ctx.strokeStyle = opts.color;
      ctx.lineWidth = 1.5;
      meta.data.forEach((bar, i) => {
        const r = opts.ranges[i];
        if (!r) return;
        const x = bar.x, cap = Math.min(6, bar.width / 4);
        const y0 = y.getPixelForValue(r[0]), y1 = y.getPixelForValue(r[1]);
        ctx.beginPath();
        ctx.moveTo(x, y0); ctx.lineTo(x, y1);
        ctx.moveTo(x - cap, y0); ctx.lineTo(x + cap, y0);
        ctx.moveTo(x - cap, y1); ctx.lineTo(x + cap, y1);
        ctx.stroke();
      });
      ctx.restore();
    },
  };
  Chart.register(errorBarsPlugin);

  // band = two line datasets, the upper one filled down to the lower one
  function band(lo, hi, color, label) {
    return [
      { label: `${label} (low)`, data: lo, borderWidth: 0, pointRadius: 0, fill: false, isBand: true },
      { label, data: hi, borderWidth: 0, pointRadius: 0, fill: "-1", backgroundColor: color, isBand: true, pointStyle: "rect" },
    ];
  }
  const hideBandLegend = { labels: { filter: (item, data) => !(data.datasets[item.datasetIndex].isBand && item.text.endsWith("(low)")) } };

  // ------------------------------------------------------------ theme
  const THEMES = ["auto", "light", "dark"];
  function currentTheme() {
    return document.documentElement.getAttribute("data-theme") || "auto";
  }
  function setTheme(t) {
    if (t === "auto") document.documentElement.removeAttribute("data-theme");
    else document.documentElement.setAttribute("data-theme", t);
    try { localStorage.setItem("theme", t); } catch (e) { /* storage unavailable */ }
    $("theme-toggle").textContent = `Theme: ${t}`;
    rerenderAll();
  }
  function initTheme() {
    $("theme-toggle").textContent = `Theme: ${currentTheme()}`;
    $("theme-toggle").addEventListener("click", () => {
      setTheme(THEMES[(THEMES.indexOf(currentTheme()) + 1) % THEMES.length]);
    });
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
      if (currentTheme() === "auto") rerenderAll();
    });
  }
  function rerenderAll() {
    applyChartDefaults();
    Object.keys(rendered).forEach((tab) => { if (rendered[tab]) renderTab(tab); });
  }

  // ------------------------------------------------------------ tabs
  const TABS = ["impact", "system", "renewables", "method"];
  function showTab() {
    const tab = TABS.includes(location.hash.slice(1)) ? location.hash.slice(1) : "impact";
    TABS.forEach((t) => {
      $(`tab-${t}`).hidden = t !== tab;
      document.querySelector(`[data-tab="${t}"]`).setAttribute("aria-selected", String(t === tab));
    });
    renderTab(tab);
  }
  function renderTab(tab) {
    rendered[tab] = true;
    if (tab === "impact") renderImpact();
    if (tab === "system") renderSystem();
    if (tab === "renewables") renderRenewables();
  }
  function renderImpact() {
    guard("impact cards", impactCards);
    draw("c-pdc", impactPdc);
    draw("c-month", impactMonth);
    draw("c-hour", impactHour);
    draw("c-size", impactSize);
  }
  function renderSystem() {
    draw("c-mix", systemMix);
    draw("c-scatter", systemScatter);
    draw("c-regime", systemRegime);
  }
  function renderRenewables() {
    draw("c-wind", renewablesWind);
    draw("c-cf", renewablesCf);
  }

  // ------------------------------------------------------------ 1. impact
  function runFor(variant) {
    return DATA.impact.runs[`${state.year}|${state.ppa ? 1 : 0}|${variant}`];
  }

  function initImpactControls() {
    const imp = DATA.impact;
    // optional shareable state: ?size=800&year=2025&ppa=1&variant=binned
    const q = new URLSearchParams(location.search);
    // q.has(): Number(null) is 0, a valid size, so a missing parameter must not be read as one
    if (q.has("size") && imp.sizes.includes(Number(q.get("size")))) state.size = Number(q.get("size"));
    if (q.has("year") && imp.years.includes(Number(q.get("year")))) state.year = Number(q.get("year"));
    if (VARIANTS.includes(q.get("variant"))) state.variant = q.get("variant");
    if (q.get("ppa") === "1") state.ppa = true;
    $("size").value = state.size; $("variant").value = state.variant; $("ppa").checked = state.ppa;
    $("year").innerHTML = imp.years.map((y) => `<option value="${y}">${y}</option>`).join("");
    $("year").value = state.year;
    $("size").min = imp.sizes[0];
    $("size").max = imp.sizes[imp.sizes.length - 1];
    $("size").step = imp.sizes[1] - imp.sizes[0];
    const on = () => {
      state.size = Number($("size").value);
      state.year = Number($("year").value);
      state.variant = $("variant").value;
      state.ppa = $("ppa").checked;
      renderImpact();
    };
    ["size", "year", "variant", "ppa"].forEach((id) => $(id).addEventListener("input", on));
  }

  // what every impact chart needs for the current selection
  function impactCtx() {
    const imp = DATA.impact;
    return {
      imp, p: palette(), i: imp.sizes.indexOf(state.size), run: runFor(state.variant),
      all: VARIANTS.map(runFor), base: imp.baseline[String(state.year)],
    };
  }

  function impactCards() {
    const { imp, i, run, all, base } = impactCtx();
    $("size-label").textContent = `${state.size} MW`;
    $("upper-note").hidden = state.size <= imp.upper_bound_note_mw;

    // cards
    const means = all.map((r) => r.mean[i]);
    const dp = run.mean[i];
    $("stat-dprice").textContent = `${signed(dp, 2)} €/MWh`;
    $("stat-dprice-sub").textContent =
      `${signed((dp / base.mean_price) * 100, 1)}% of the ${state.year} average (${fmt(base.mean_price, 1)} €/MWh) · range ${signed(Math.min(...means), 2)} to ${signed(Math.max(...means), 2)}`;
    const costs = all.map((r) => r.cost[i]);
    $("stat-cost").textContent = `€${int(run.cost[i])} million`;
    $("stat-ppa").textContent = state.ppa && state.size > 0 ? `${int(run.ppa_mw[i])} MW` : "–";
    $("stat-ppa-sub").textContent = state.ppa
      ? "New wind sized to the data center's annual energy"
      : `PPA off · cost range €${int(Math.min(...costs))}–${int(Math.max(...costs))} million`;
  }

  function impactPdc() {
    const { imp, p, i, run, base } = impactCtx();
    makeChart("c-pdc", {
      type: "line",
      data: {
        labels: imp.pdc_pct,
        datasets: [
          { label: "Without data center", data: base.pdc_without, borderColor: p.text2, borderDash: [5, 4], borderWidth: 2, pointRadius: 0 },
          { label: `With ${state.size} MW`, data: run.pdc[i], borderColor: p.s[0], borderWidth: 2, pointRadius: 0 },
        ],
      },
      options: {
        interaction: { mode: "index", intersect: false },
        scales: {
          x: axis("Share of hours (%)", { type: "linear", min: 0, max: 100, ticks: { stepSize: 20 } }),
          y: axis("€/MWh", { min: -30, max: Math.min(400, Math.ceil(Math.max(...run.pdc[i].slice(1)) / 50) * 50) }),
        },
        plugins: { tooltip: { callbacks: { title: (it) => `${it[0].label}% of hours`, label: (c) => `${c.dataset.label}: ${fmt(c.parsed.y, 1)} €/MWh` } } },
      },
    });
  }

  // monthly bars + whiskers
  function impactMonth() {
    const { p, i, run, all } = impactCtx();
    const mRanges = MONTHS.map((_, m) => [Math.min(...all.map((r) => r.monthly[i][m])), Math.max(...all.map((r) => r.monthly[i][m]))]);
    makeChart("c-month", {
      type: "bar",
      data: { labels: MONTHS, datasets: [{ label: `Price change (${state.variant})`, data: run.monthly[i], backgroundColor: p.s[0], borderRadius: 4, maxBarThickness: 28 }] },
      options: {
        scales: { x: axis(""), y: axis("€/MWh", { suggestedMin: 0, suggestedMax: Math.max(1, ...mRanges.map((r) => r[1])) }) },
        plugins: {
          legend: { display: false },
          errorBars: { ranges: mRanges, color: p.text2 },
          tooltip: { callbacks: { label: (c) => `${signed(c.parsed.y, 2)} €/MWh (range ${signed(mRanges[c.dataIndex][0], 2)} to ${signed(mRanges[c.dataIndex][1], 2)})` } },
        },
      },
    });
    table("t-month", ["Month", `${state.variant}`, "Low", "High"],
      MONTHS.map((m, k) => [m, signed(run.monthly[i][k], 2), signed(mRanges[k][0], 2), signed(mRanges[k][1], 2)]));
  }

  function impactHour() {
    const { p, i, run, all } = impactCtx();
    const hours = [...Array(24).keys()];
    const hLo = hours.map((h) => Math.min(...all.map((r) => r.hour[i][h])));
    const hHi = hours.map((h) => Math.max(...all.map((r) => r.hour[i][h])));
    makeChart("c-hour", {
      type: "line",
      data: {
        labels: hours.map((h) => `${String(h).padStart(2, "0")}`),
        datasets: [
          ...band(hLo, hHi, p.band, "Range across variants"),
          { label: `Price change (${state.variant})`, data: run.hour[i], borderColor: p.s[0], borderWidth: 2, pointRadius: 0 },
        ],
      },
      options: {
        interaction: { mode: "index", intersect: false },
        scales: { x: axis("Hour (local time)"), y: axis("€/MWh", { suggestedMin: 0 }) },
        plugins: { legend: hideBandLegend, tooltip: { filter: (it) => !it.dataset.isBand, callbacks: { label: (c) => `${c.dataset.label}: ${signed(c.parsed.y, 2)} €/MWh` } } },
      },
    });
  }

  function impactSize() {
    const { imp, p, run, all } = impactCtx();
    const sLo = imp.sizes.map((_, k) => Math.min(...all.map((r) => r.mean[k])));
    const sHi = imp.sizes.map((_, k) => Math.max(...all.map((r) => r.mean[k])));
    makeChart("c-size", {
      type: "line",
      data: {
        labels: imp.sizes,
        datasets: [
          ...band(sLo, sHi, p.band, "Range across variants"),
          { label: `Price change (${state.variant})`, data: run.mean, borderColor: p.s[0], borderWidth: 2,
            pointStyle: "circle", pointRadius: imp.sizes.map((s) => (s === state.size ? 7 : 0)), pointBackgroundColor: p.s[0],
            pointBorderColor: p.surface, pointBorderWidth: 2 },
        ],
      },
      options: {
        interaction: { mode: "index", intersect: false },
        scales: { x: axis("Data center size (MW)"), y: axis("€/MWh", { suggestedMin: 0 }) },
        plugins: { legend: hideBandLegend, tooltip: { filter: (it) => !it.dataset.isBand, callbacks: { title: (it) => `${it[0].label} MW`, label: (c) => `${c.dataset.label}: ${signed(c.parsed.y, 2)} €/MWh` } } },
      },
    });
    table("t-size", ["Size (MW)", "Price change", "Low", "High", "Cost (€ m)", "Switching hours"],
      imp.sizes.map((s, k) => [s, signed(run.mean[k], 2), signed(sLo[k], 2), signed(sHi[k], 2), int(run.cost[k]), run.switching[k]]));
  }

  // ------------------------------------------------------------ 2. Finnish power system
  function initSystemControls() {
    const years = Object.keys(DATA.history.scatter_rl_gw_price);
    $("scatter-year").innerHTML = years.map((y) => `<option value="${y}">${y}${y === "2022" ? " (energy crisis)" : ""}</option>`).join("");
    $("scatter-year").value = String(state.scatterYear);
    // the year only affects the scatter: redraw that chart alone
    $("scatter-year").addEventListener("input", () => { state.scatterYear = Number($("scatter-year").value); draw("c-scatter", systemScatter); });
  }

  function systemMix() {
    const h = DATA.history;
    const p = palette();
    const mixSets = h.mix_order.map((k, j) => ({
      type: "bar", label: MIX_LABELS[k], data: h.mix_gwh[k], backgroundColor: p.s[j], stack: "supply", order: 2,
      borderWidth: 0, barPercentage: 1, categoryPercentage: 0.9,
    }));
    makeChart("c-mix", {
      data: {
        labels: h.months,
        datasets: [...mixSets, { type: "line", label: "Consumption", data: h.mix_gwh.consumption, borderColor: p.text, borderWidth: 1.5, pointRadius: 0, order: 1 }],
      },
      options: {
        interaction: { mode: "index", intersect: false },
        scales: {
          x: axis("", { stacked: true, grid: { display: false }, ticks: { autoSkip: false, maxRotation: 0, callback: (v, k) => (h.months[k].endsWith("-01") ? h.months[k].slice(0, 4) : "") } }),
          y: axis("GWh per month", { stacked: true }),
        },
        plugins: { tooltip: { callbacks: { label: (c) => `${c.dataset.label}: ${fmt(c.parsed.y, 0)} GWh` } } },
      },
    });
  }

  // price vs residual load
  function systemScatter() {
    const h = DATA.history;
    const p = palette();
    const y = String(state.scatterYear);
    if (!h.scatter_rl_gw_price[y] || !h.median_by_rl[y]) throw new Error(`no scatter data for ${y}`);
    const others = Object.entries(h.scatter_rl_gw_price).filter(([k]) => k !== y).flatMap(([, v]) => v);
    const hl = y === "2022" ? p.s[1] : p.s[0];
    makeChart("c-scatter", {
      type: "scatter",
      data: {
        datasets: [
          { label: "Other years", data: others.map(([a, b]) => ({ x: a, y: b })), backgroundColor: p.context, pointRadius: 1.5, order: 3 },
          { label: y, data: h.scatter_rl_gw_price[y].map(([a, b]) => ({ x: a, y: b })), backgroundColor: withAlpha(hl, 0.7), pointRadius: 2.4, order: 2 },
          { type: "line", label: `${y} median`, data: h.median_by_rl[y].map(([a, b]) => ({ x: a, y: b })), borderColor: p.text, borderWidth: 2.5, pointRadius: 0, order: 1 },
        ],
      },
      options: {
        scales: { x: axis("Residual load (GW)", { type: "linear" }), y: axis("€/MWh", { min: -50, max: 500 }) },
        plugins: { tooltip: { callbacks: { label: (c) => `${c.dataset.label}: ${fmt(c.parsed.x, 1)} GW, ${fmt(c.parsed.y, 0)} €/MWh` } } },
      },
    });
  }

  function systemRegime() {
    const h = DATA.history;
    const p = palette();
    const ry = Object.keys(h.regime_share_pct);
    makeChart("c-regime", {
      type: "bar",
      data: {
        labels: ry,
        datasets: h.regime_order.map((k, j) => ({
          label: REGIME_LABELS[k], data: ry.map((yy) => h.regime_share_pct[yy][k] || 0), backgroundColor: p.s[j], borderWidth: 0,
        })),
      },
      options: {
        scales: { x: axis("", { stacked: true }), y: axis("% of hours", { stacked: true, max: 100 }) },
        plugins: { tooltip: { callbacks: { label: (c) => `${c.dataset.label}: ${fmt(c.parsed.y, 1)}%` } } },
      },
    });
    table("t-regime", ["Year", ...h.regime_order.map((k) => REGIME_LABELS[k])],
      ry.map((yy) => [yy, ...h.regime_order.map((k) => `${fmt(h.regime_share_pct[yy][k] || 0, 1)}%`)]));
  }

  // ------------------------------------------------------------ 3. renewables
  function logistic(v, q) { return q.c + q.L / (1 + Math.exp(-q.k * (v - q.v0))); }

  function initRenewablesControls() {
    const r = DATA.renewables;
    const years = Object.keys(r.monthly_cf);
    $("cf-year").innerHTML = years.map((y) => `<option value="${y}">${y}</option>`).join("");
    $("cf-year").value = String(state.cfYear);
    $("cf-year").addEventListener("input", () => { state.cfYear = Number($("cf-year").value); draw("c-cf", renewablesCf); });
    $("cap").value = r.calculator.default_capacity_mw;
    $("cap-note").textContent = `Default capacity: ${r.calculator.capacity_note}. Curve fitted on ${r.wind_curve.fit_years[0]}–${r.wind_curve.fit_years[r.wind_curve.fit_years.length - 1]} hours with positive prices (no curtailment).`;
    ["ws", "cap"].forEach((id) => $(id).addEventListener("input", updateCalc));
    updateCalc();
  }

  function updateCalc() {
    const q = DATA.renewables.wind_curve.params;
    const ws = Number($("ws").value);
    const cap = Math.max(0, Number($("cap").value) || 0);
    const cf = Math.max(0, logistic(ws, q));
    $("ws-label").textContent = `${ws.toFixed(1)} m/s`;
    $("calc-cf").textContent = `${(cf * 100).toFixed(0)}%`;
    $("calc-mw").textContent = `${Math.round(cf * cap).toLocaleString("en-US")} MW`;
  }

  function renewablesWind() {
    const r = DATA.renewables;
    const p = palette();
    makeChart("c-wind", {
      type: "scatter",
      data: {
        datasets: [
          { label: "Hours (sample)", data: r.wind_scatter_ws_cf.map(([a, b]) => ({ x: a, y: b })), backgroundColor: withAlpha(p.s[0], 0.3), pointRadius: 2 },
          { type: "line", label: "Fitted curve", data: r.wind_curve.ws.map((v, k) => ({ x: v, y: r.wind_curve.cf[k] })), borderColor: p.s[1], borderWidth: 2.5, pointRadius: 0 },
        ],
      },
      options: {
        scales: { x: axis("Mean 100 m wind speed (m/s)", { type: "linear", min: 0, max: 16 }), y: axis("Capacity factor", { min: 0, max: 1 }) },
        plugins: { tooltip: { callbacks: { label: (c) => `${fmt(c.parsed.x, 1)} m/s → ${(c.parsed.y * 100).toFixed(0)}%` } } },
      },
    });
  }

  function renewablesCf() {
    const r = DATA.renewables;
    const p = palette();
    const years = Object.keys(r.monthly_cf);
    const y = String(state.cfYear);
    const rangeOf = (key) => [
      MONTHS.map((_, m) => Math.min(...years.map((yy) => r.monthly_cf[yy][key][m]).filter((v) => v !== null))),
      MONTHS.map((_, m) => Math.max(...years.map((yy) => r.monthly_cf[yy][key][m]).filter((v) => v !== null))),
    ];
    const [wLo, wHi] = rangeOf("wind_cf");
    const [sLo, sHi] = rangeOf("solar_cf");
    makeChart("c-cf", {
      type: "line",
      data: {
        labels: MONTHS,
        datasets: [
          ...band(wLo, wHi, withAlpha(p.s[0], 0.15), "Wind, 2019–2025 range"),
          ...band(sLo, sHi, withAlpha(p.s[1], 0.15), "Solar, 2019–2025 range"),
          { label: `Wind ${y}`, data: r.monthly_cf[y].wind_cf, borderColor: p.s[0], borderWidth: 2, pointRadius: 3, pointBackgroundColor: p.s[0] },
          { label: `Solar ${y}`, data: r.monthly_cf[y].solar_cf, borderColor: p.s[1], borderWidth: 2, pointRadius: 3, pointBackgroundColor: p.s[1] },
        ],
      },
      options: {
        interaction: { mode: "index", intersect: false },
        scales: { x: axis(""), y: axis("Capacity factor", { min: 0, ticks: { callback: (v) => `${Math.round(v * 100)}%` } }) },
        plugins: { legend: hideBandLegend, tooltip: { filter: (it) => !it.dataset.isBand, callbacks: { label: (c) => `${c.dataset.label}: ${(c.parsed.y * 100).toFixed(0)}%` } } },
      },
    });
  }

  // ------------------------------------------------------------ boot
  async function boot() {
    initTheme();
    applyChartDefaults();
    const [impact, history, renewables] = await Promise.all(
      ["impact", "history", "renewables"].map((n) => fetch(`data/${n}.json`).then((r) => r.json())));
    Object.assign(DATA, { impact, history, renewables });
    initImpactControls();
    initSystemControls();
    initRenewablesControls();
    window.addEventListener("hashchange", showTab);
    showTab();
  }

  boot().catch((e) => {
    document.querySelector("main").insertAdjacentHTML("afterbegin",
      `<p class="note warn">Could not load the data (${e.message}). Open the page through a web server, not as a local file.</p>`);
  });
})();
