"""Browser tests for the static dashboard (dashboard/), driven with real input through Playwright.

Every control on every tab is exercised: each dropdown value, the sliders across their whole range
(keyboard steps), the PPA checkbox, the capacity box, tab switching and the theme toggle. After each
change every chart on the visible tab must have a Chart.js instance with data, have painted pixels,
show no error message, and the page must log no console errors.

    pip install -r requirements-dev.txt
    pytest tests/test_dashboard.py

Uses the installed Google Chrome if there is one, else Playwright's own Chromium
(`python -m playwright install chromium`). Set DASHBOARD_BROWSERS=chromium,firefox,webkit to run
other engines too.
"""

from __future__ import annotations

import functools
import json
import os
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

sync_api = pytest.importorskip("playwright.sync_api")

SITE = Path(__file__).resolve().parents[1] / "dashboard"
SETTLE_MS = 400  # chart animations are 250 ms
BROWSERS = os.environ.get("DASHBOARD_BROWSERS", "chromium").split(",")

# Per canvas on a tab: is there a chart, does every dataset have finite values, were pixels painted,
# is an error message shown? Pixel check samples the alpha channel of every 16th pixel.
CHART_STATE_JS = """(tab) => [...document.querySelectorAll(`#tab-${tab} canvas`)].map((c) => {
  const ch = Chart.getChart(c);
  const finite = (v) => v !== null && v !== undefined && (typeof v === 'object' ? Number.isFinite(v.y) : Number.isFinite(Number(v)));
  const datasets = ch ? ch.data.datasets.map((d) => d.data.filter(finite).length) : [];
  let painted = 0;
  if (c.width && c.height) {
    const px = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
    for (let k = 3; k < px.length; k += 64) if (px[k]) painted++;
  }
  return { id: c.id, chart: ch ? ch.id : null, datasets, painted, error: !!c.parentElement.querySelector('.chart-error') };
})"""

TAB_CHARTS = {
    "impact": ["c-pdc", "c-month", "c-hour", "c-size"],
    "system": ["c-mix", "c-scatter", "c-regime"],
    "renewables": ["c-wind", "c-cf"],
}


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


@pytest.fixture(scope="session")
def base_url():
    handler = functools.partial(_QuietHandler, directory=str(SITE))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/"
    server.shutdown()


@pytest.fixture(scope="session")
def playwright():
    with sync_api.sync_playwright() as p:
        yield p


@pytest.fixture(params=BROWSERS, scope="session")
def browser(request, playwright):
    name = request.param
    if name == "chromium":
        try:
            b = playwright.chromium.launch(channel="chrome")
        except Exception:
            b = playwright.chromium.launch()
    else:
        b = getattr(playwright, name).launch()
    yield b
    b.close()


class Page:
    """A dashboard page that records console errors."""

    def __init__(self, browser, url, **ctx):
        self.ctx = browser.new_context(**ctx)
        self.pg = self.ctx.new_page()
        self.errors: list[str] = []
        self.pg.on("console", lambda m: m.type == "error" and self.errors.append(m.text))
        self.pg.on("pageerror", lambda e: self.errors.append(f"uncaught: {e}"))
        self.pg.goto(url)
        self.pg.wait_for_function("() => typeof Chart !== 'undefined' && Chart.getChart('c-pdc')")

    def settle(self):
        self.pg.wait_for_timeout(SETTLE_MS)

    def charts(self, tab):
        return {c["id"]: c for c in self.pg.evaluate(CHART_STATE_JS, tab)}

    def check(self, tab, what):
        """Every chart on `tab` has data, painted pixels and no error; no console errors so far."""
        self.settle()
        state = self.charts(tab)
        assert sorted(state) == sorted(TAB_CHARTS[tab]), f"{what}: canvases {sorted(state)}"
        for cid, c in state.items():
            assert c["chart"] is not None, f"{what}: {cid} has no chart"
            assert c["datasets"] and all(n > 0 for n in c["datasets"]), f"{what}: {cid} empty dataset {c['datasets']}"
            assert c["painted"] > 0, f"{what}: {cid} drew nothing"
            assert not c["error"], f"{what}: {cid} shows an error"
        assert not self.errors, f"{what}: console errors {self.errors}"
        return state

    def options(self, sel):
        return self.pg.eval_on_selector(sel, "s => [...s.options].map(o => o.value)")

    def close(self):
        self.ctx.close()


@pytest.fixture
def page(browser, base_url):
    p = Page(browser, base_url)
    yield p
    p.close()


def text(page, sel):
    return page.pg.inner_text(sel).strip()


# ---------------------------------------------------------------- impact
def test_impact_defaults(page):
    """Without URL parameters the page opens at 300 MW, 2024, central, PPA off."""
    page.check("impact", "initial load")
    assert page.pg.input_value("#size") == "300"
    assert text(page, "#size-label") == "300 MW"
    assert page.pg.input_value("#year") == "2024"
    assert page.pg.input_value("#variant") == "central"
    assert not page.pg.is_checked("#ppa")
    assert text(page, "#stat-dprice").startswith("+")


def test_impact_url_state(browser, base_url):
    p = Page(browser, base_url + "?size=800&year=2025&ppa=1&variant=binned")
    try:
        p.check("impact", "URL state")
        assert p.pg.input_value("#size") == "800" and p.pg.input_value("#year") == "2025"
        assert p.pg.input_value("#variant") == "binned" and p.pg.is_checked("#ppa")
        assert p.pg.is_visible("#upper-note")
    finally:
        p.close()


def test_impact_size_slider(page):
    page.check("impact", "initial load")
    seen = []
    page.pg.focus("#size")
    page.pg.keyboard.press("Home")
    while True:
        v = page.pg.input_value("#size")
        page.check("impact", f"size {v} MW")
        assert text(page, "#size-label") == f"{v} MW"
        assert page.pg.is_hidden("#upper-note") == (int(v) <= 500), f"upper-bound note at {v} MW"
        seen.append(int(v))
        page.pg.keyboard.press("ArrowRight")
        if page.pg.input_value("#size") == v:
            break
    assert seen == list(range(0, 1001, 100))


@pytest.mark.parametrize("control", ["#year", "#variant"])
def test_impact_dropdowns(page, control):
    values = page.options(control)
    assert len(values) >= 3
    readings = set()
    for v in values:
        page.pg.select_option(control, v)
        page.check("impact", f"{control}={v}")
        readings.add(text(page, "#stat-dprice"))
    assert len(readings) == len(values), f"{control}: headline value did not change {readings}"


def test_impact_ppa_toggle(page):
    off = text(page, "#stat-dprice")
    assert text(page, "#stat-ppa") == "–"
    page.pg.click("#ppa")
    page.check("impact", "PPA on")
    assert text(page, "#stat-dprice") != off
    assert text(page, "#stat-ppa").endswith("MW")
    page.pg.click("#ppa")
    page.check("impact", "PPA off again")
    assert text(page, "#stat-dprice") == off


# ---------------------------------------------------------------- Finnish power system
def test_system_scatter_year_only_redraws_scatter(page):
    page.pg.click("[data-tab='system']")
    first = page.check("system", "open tab")
    for v in page.options("#scatter-year"):
        page.pg.select_option("#scatter-year", v)
        state = page.check("system", f"scatter year {v}")
        # the other two charts must be the same instances, i.e. not re-rendered
        assert state["c-mix"]["chart"] == first["c-mix"]["chart"]
        assert state["c-regime"]["chart"] == first["c-regime"]["chart"]
        label = page.pg.evaluate("() => Chart.getChart('c-scatter').data.datasets[1].label")
        assert label == v


# ---------------------------------------------------------------- renewables
def test_renewables_controls(page):
    page.pg.click("[data-tab='renewables']")
    first = page.check("renewables", "open tab")
    for v in page.options("#cf-year"):
        page.pg.select_option("#cf-year", v)
        state = page.check("renewables", f"cf year {v}")
        assert state["c-wind"]["chart"] == first["c-wind"]["chart"]

    # wind speed slider across its range: output rises with wind speed
    page.pg.focus("#ws")
    page.pg.keyboard.press("Home")
    outputs = []
    while True:
        v = page.pg.input_value("#ws")
        outputs.append(int(text(page, "#calc-mw").replace(",", "").split()[0]))
        page.pg.keyboard.press("ArrowRight")
        if page.pg.input_value("#ws") == v:
            break
    assert len(outputs) > 10 and outputs == sorted(outputs) and outputs[-1] > outputs[0]

    page.pg.fill("#cap", "5000")
    assert int(text(page, "#calc-mw").replace(",", "").split()[0]) <= 5000
    page.check("renewables", "calculator")


# ---------------------------------------------------------------- tabs and theme
def test_tabs_and_theme(page):
    for theme in ["light", "dark", "auto"]:
        page.pg.click("#theme-toggle")
        assert text(page, "#theme-toggle") == f"Theme: {theme}"
        for tab in TAB_CHARTS:
            page.pg.click(f"[data-tab='{tab}']")
            page.check(tab, f"theme {theme}, tab {tab}")
            bg = page.pg.evaluate("() => getComputedStyle(document.body).backgroundColor")
            if theme == "dark":
                assert bg == "rgb(18, 18, 17)"
            if theme == "light":
                assert bg == "rgb(244, 243, 240)"
        page.pg.click("[data-tab='method']")
        page.pg.wait_for_selector("#tab-method", state="visible")
        assert page.pg.is_hidden("#tab-impact")


def test_phone_width(browser, base_url):
    p = Page(browser, base_url, viewport={"width": 390, "height": 844}, color_scheme="dark")
    try:
        for tab in TAB_CHARTS:
            p.pg.click(f"[data-tab='{tab}']")
            p.check(tab, f"phone, {tab}")
            assert p.pg.evaluate("() => document.documentElement.scrollWidth") <= 390
        p.pg.click("[data-tab='system']")
        p.pg.select_option("#scatter-year", "2025")
        p.check("system", "phone, scatter 2025")
    finally:
        p.close()


# ---------------------------------------------------------------- isolation
def test_one_failing_chart_does_not_blank_the_others(browser, base_url):
    """Break the 2025 scatter data on purpose: only that chart may fail."""
    ctx = browser.new_context()
    history = json.loads((SITE / "data" / "history.json").read_text(encoding="utf-8"))
    del history["median_by_rl"]["2025"]
    ctx.route("**/data/history.json", lambda route: route.fulfill(
        status=200, content_type="application/json", body=json.dumps(history)))
    pg = ctx.new_page()
    errors = []
    pg.on("console", lambda m: m.type == "error" and errors.append(m.text))
    pg.on("pageerror", lambda e: errors.append(f"uncaught: {e}"))
    try:
        pg.goto(base_url + "#system")
        pg.wait_for_function("() => typeof Chart !== 'undefined' && Chart.getChart('c-mix')")
        pg.select_option("#scatter-year", "2025")
        pg.wait_for_timeout(SETTLE_MS)
        state = {c["id"]: c for c in pg.evaluate(CHART_STATE_JS, "system")}
        assert state["c-scatter"]["error"] and state["c-scatter"]["chart"] is None
        for cid in ["c-mix", "c-regime"]:
            assert state[cid]["chart"] is not None and state[cid]["painted"] > 0 and not state[cid]["error"]
        assert any("c-scatter" in e for e in errors), errors
        assert not any(e.startswith("uncaught") for e in errors), errors

        # recovers when a good year is chosen again
        pg.select_option("#scatter-year", "2024")
        pg.wait_for_timeout(SETTLE_MS)
        state = {c["id"]: c for c in pg.evaluate(CHART_STATE_JS, "system")}
        assert state["c-scatter"]["chart"] is not None and not state["c-scatter"]["error"]
    finally:
        ctx.close()
