#!/usr/bin/env node
// Circuit help qualification with real pointer input and a rendering WebGPU browser.
// Uses the same Playwright/browser environment variables as validate_visualization.cjs.
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');

const base = process.env.VISUALIZATION_URL || 'http://localhost:8080/';
const out = path.resolve(process.env.VISUALIZATION_ARTIFACTS || 'artifacts/visualization/circuit-hover');
fs.mkdirSync(out, { recursive: true });
const report = { url: base, timestamp: new Date().toISOString(), cases: [], interactions: [], errors: [] };

function sameTrajectory(actual, expected, location = 'trace') {
  assert.equal(typeof actual, typeof expected, `${location}: value type`);
  if (typeof expected === 'number') {
    assert.ok(Number.isFinite(actual) && Number.isFinite(expected), `${location}: finite numbers`);
    // Rust serde and JavaScript can round a decimal F64 literal to adjacent
    // representable values during import/export. Allow only that narrow noise,
    // retaining exact record keys, array order/shape and all nonnumeric data.
    const tolerance = 4 * Number.EPSILON * Math.max(1, Math.abs(actual), Math.abs(expected));
    assert.ok(Math.abs(actual - expected) <= tolerance, `${location}: ${actual} differs from ${expected} beyond JSON round-trip tolerance`);
  } else if (Array.isArray(expected)) {
    assert.ok(Array.isArray(actual), `${location}: array type`);
    assert.equal(actual.length, expected.length, `${location}: array length`);
    expected.forEach((value, index) => sameTrajectory(actual[index], value, `${location}[${index}]`));
  } else if (expected !== null && typeof expected === 'object') {
    assert.ok(actual !== null && !Array.isArray(actual), `${location}: record type`);
    assert.deepEqual(Object.keys(actual).sort(), Object.keys(expected).sort(), `${location}: record keys`);
    for (const key of Object.keys(expected)) sameTrajectory(actual[key], expected[key], `${location}.${key}`);
  } else {
    assert.equal(actual, expected, `${location}: exact value`);
  }
}

(async () => {
  const browser = await chromium.launch({
    ...(process.env.CHROME_BIN ? { executablePath: process.env.CHROME_BIN } : {}),
    headless: process.env.BROWSER_HEADLESS !== 'false',
    args: JSON.parse(process.env.BROWSER_ARGS || '[]'),
  });
  let page;
  try {
    report.browser = browser.version();
    page = await browser.newPage({ viewport: { width: 1600, height: 1200 }, deviceScaleFactor: 1 });
    const cdp = await page.context().newCDPSession(page);
    await cdp.send('Emulation.setFocusEmulationEnabled', { enabled: true });
    page.on('pageerror', error => report.errors.push(String(error)));
    page.on('console', message => { if (message.type() === 'error') report.errors.push(message.text()); });
    const diag = () => page.evaluate(() => JSON.parse(window.__qaApi.diagnostics_json()));
    const durable = snapshot => ({ frame: snapshot.frame, gate: snapshot.gate, revision: snapshot.revision, query: snapshot.query });

    async function ready() {
      await page.bringToFront();
      await page.waitForFunction(() => ['ready', 'error'].includes(window.__burnQuantumStatus?.state), null, { timeout: 60000 });
      assert.equal(await page.evaluate(() => window.__burnQuantumStatus.state), 'ready');
      await page.evaluate(async () => { window.__qaApi = await import('./pkg/burn_quantum_visualization.js'); });
      await page.waitForFunction(() => {
        const text = window.__qaApi.diagnostics_json();
        if (!text) return false;
        const d = JSON.parse(text);
        const actual = new URLSearchParams(location.search);
        return d.ready && Array.isArray(d.hover_regions)
          && [...new URLSearchParams(d.query)].every(([key, value]) => actual.get(key) === value);
      });
      await page.locator('canvas').scrollIntoViewIfNeeded();
      await page.waitForTimeout(400);
    }

    async function load(demo, extra = {}) {
      await page.mouse.move(10, 10);
      const url = new URL(base);
      url.search = new URLSearchParams({ demo, view: 'circuit', step: '0', playing: '0', zoom: '0.75', ...extra });
      await page.goto(url.href);
      await ready();
      assert.equal((await diag()).view, 'circuit');
    }

    async function leave() {
      await page.mouse.move(10, 10);
      await page.waitForFunction(() => JSON.parse(window.__qaApi.diagnostics_json()).tooltip === null);
    }

    function checkTooltip(snapshot, label) {
      const tip = snapshot.tooltip;
      assert.ok(tip, `${label}: tooltip is visible`);
      assert.ok(typeof tip.title === 'string' && tip.title.length > 2 && tip.title.length <= 120, `${label}: concise title`);
      assert.ok(typeof tip.summary === 'string' && tip.summary.length > 10 && tip.summary.length <= 800, `${label}: brief explanation`);
      assert.ok(Array.isArray(tip.rect) && tip.rect.length === 4 && tip.rect.every(Number.isFinite), `${label}: measured popup bounds`);
      const [x0, y0, x1, y1] = tip.rect;
      assert.ok(x1 > x0 && y1 > y0, `${label}: nonempty popup`);
      assert.ok(x0 >= -1 && y0 >= -1 && x1 <= snapshot.window_logical[0] + 1 && y1 <= snapshot.window_logical[1] + 1,
        `${label}: popup stays inside the canvas window: ${JSON.stringify({ rect: tip.rect, window: snapshot.window_logical })}`);
    }

    async function hover(predicate, label, { save = false, exactCenter = false } = {}) {
      const before = await diag();
      // Connectors and multiwire boxes can contain more specific control/target
      // regions. Follow the renderer's reverse hit precedence when choosing an
      // actual pointer position, rather than assuming a region's center is free.
      let target;
      for (const region of before.hover_regions.filter(predicate)) {
        const [x0, y0, x1, y1] = region.rect;
        if (exactCenter) {
          // Endpoint regressions must exercise the symbol itself, even when an
          // incorrectly ordered connector would obscure its hit region.
          target = { region, x: (x0 + x1) / 2, y: (y0 + y1) / 2 };
          break;
        }
        for (const fy of [0.5, 0.25, 0.75, 0.1, 0.9, 0.35, 0.65]) {
          for (const fx of [0.5, 0.1, 0.9]) {
            const x = x0 + (x1 - x0) * fx, y = y0 + (y1 - y0) * fy;
            const topmost = before.hover_regions.findLast(r => x >= r.rect[0] && x <= r.rect[2] && y >= r.rect[1] && y <= r.rect[3]);
            if (topmost?.title === region.title && topmost?.summary === region.summary) {
              target = { region, x, y };
              break;
            }
          }
          if (target) break;
        }
        if (target) break;
      }
      assert.ok(target, `${label}: exposed hover region exists; found ${JSON.stringify(before.hover_regions.map(r => r.title))}`);
      const { region, x, y } = target;
      const box = await page.locator('canvas').boundingBox();
      assert.ok(x >= 0 && y >= 0 && x <= box.width && y <= box.height, `${label}: pointer target is visible`);
      await page.mouse.move(box.x + x, box.y + y);
      await page.waitForFunction(expected => {
        const tip = JSON.parse(window.__qaApi.diagnostics_json()).tooltip;
        return tip && tip.title === expected.title && tip.summary === expected.summary;
      }, { title: region.title, summary: region.summary }, { timeout: 10000 });
      const after = await diag();
      checkTooltip(after, label);
      assert.deepEqual(durable(after), durable(before), `${label}: hover does not seek, select a gate, rebuild the plot or change the URL`);
      report.cases.push({ label, demo: after.demo, tooltip: after.tooltip, revision: after.revision, entity_count: after.entity_count });
      if (save) {
        await page.waitForFunction(frame => JSON.parse(window.__qaApi.diagnostics_json()).app_frames > frame + 2, after.app_frames);
        // Chromium's full-page capture temporarily resizes a narrow viewport,
        // changing canvas layout and correctly dismissing transient hover help.
        // Capture the actual visible viewport so the image proves rendering.
        await page.screenshot({ path: path.join(out, `${label}.png`), fullPage: false });
        const captured = await diag();
        checkTooltip(captured, `${label}: after capture`);
        assert.equal(captured.tooltip.title, region.title, `${label}: screenshot preserves the hovered symbol`);
        assert.equal(captured.tooltip.summary, region.summary, `${label}: screenshot preserves its explanation`);
      }
      return { snapshot: after, region, position: { x: box.x + x, y: box.y + y } };
    }

    async function clickControl(action) {
      const before = await diag();
      const control = before.controls.find(c => c.action === action);
      assert.ok(control, `visible control ${action}`);
      const box = await page.locator('canvas').boundingBox(), r = control.rect;
      await page.mouse.move(box.x + (r[0] + r[2]) / 2, box.y + (r[1] + r[3]) / 2);
      await page.waitForTimeout(100);
      await page.mouse.down();
      await page.waitForFunction(action => JSON.parse(window.__qaApi.diagnostics_json()).controls.find(c => c.action === action)?.pressed, action);
      await page.mouse.up();
      await page.waitForFunction(revision => JSON.parse(window.__qaApi.diagnostics_json()).revision !== revision, before.revision);
    }

    await load('bell');
    await hover(r => /hadamard/i.test(r.title), 'hadamard', { save: true });
    await hover(r => /control q/i.test(r.title), 'cnot-control');
    await hover(r => /target/i.test(r.title), 'cnot-target');
    await leave();
    const baseline = (await diag()).entity_count;
    for (let i = 0; i < 6; i++) {
      await hover(r => /hadamard/i.test(r.title), `repeat-${i}`);
      await leave();
    }
    assert.equal((await diag()).entity_count, baseline, 'repeated hover creates no retained entities');
    report.interactions.push('hover is transient; repeated entry/exit leaves plot revision, URL and entity count unchanged');

    const held = await hover(r => /hadamard/i.test(r.title), 'before-drag');
    await page.mouse.down();
    await page.waitForFunction(() => JSON.parse(window.__qaApi.diagnostics_json()).tooltip === null);
    await page.mouse.move(held.position.x + 40, held.position.y + 15, { steps: 5 });
    await page.waitForTimeout(450);
    assert.equal((await diag()).tooltip, null, 'tooltip stays hidden throughout a drag');
    await page.mouse.up();
    await leave();
    report.interactions.push('press and drag hide help without blocking normal circuit navigation');

    await load('bell');
    await hover(r => /hadamard/i.test(r.title), 'before-view-change');
    await clickControl('View(State)');
    await page.waitForFunction(() => {
      const d = JSON.parse(window.__qaApi.diagnostics_json());
      return d.view === 'state' && d.tooltip === null && d.hover_regions.length === 0;
    });
    report.interactions.push('switching views clears circuit help');

    await load('toffoli');
    await hover(r => /toffoli/i.test(`${r.title} ${r.summary}`), 'toffoli');
    await load('qft', { size: '3' });
    await hover(r => /phase/i.test(r.title), 'controlled-phase');
    await hover(r => /swap/i.test(r.title), 'swap');
    await load('teleportation');
    await hover(r => /measure/i.test(r.title), 'measurement');
    await hover(r => /classical.*control|feed.forward/i.test(r.title), 'classical-control', { save: true });

    // Add a condition that is already true for the displayed branch: this
    // changes its diagram while preserving every recorded state/probability.
    // Two connectors now cross classical rows at the final correction column.
    const multi = JSON.parse(await page.evaluate(() => window.__qaApi.export_document_json()));
    const trajectory = structuredClone(multi.trace);
    const correction = multi.protocol.program.instructions.findLast(i => i.Gate?.conditions.length > 0).Gate;
    const otherBit = multi.protocol.outcomes.findIndex((_, bit) => !correction.conditions.some(([used]) => used === bit));
    assert.ok(otherBit >= 0, 'teleportation has another measured bit for the same correction');
    correction.conditions.push([otherBit, multi.protocol.outcomes[otherBit]]);
    multi.title = 'Two classical controls on one correction';
    await leave();
    await page.evaluate(text => window.__qaApi.load_document_json(text), JSON.stringify(multi));
    await page.waitForFunction(title => JSON.parse(window.__qaApi.diagnostics_json()).title === title, multi.title);
    await ready();
    const imported = JSON.parse(await page.evaluate(() => window.__qaApi.export_document_json()));
    sameTrajectory(imported.trace, trajectory);
    assert.equal((await diag()).view, 'circuit');
    const endpoints = (await diag()).hover_regions.filter(r => /^Classical control/i.test(r.title)
      && r.rect[2] - r.rect[0] > r.rect[3] - r.rect[1]);
    const rightmostColumn = Math.max(...endpoints.map(r => (r.rect[0] + r.rect[2]) / 2));
    for (const [bit, value] of correction.conditions) {
      const title = `Classical control · c${bit} = ${value}`;
      const endpoint = endpoints.find(r => r.title === title && (r.rect[0] + r.rect[2]) / 2 === rightmostColumn);
      assert.ok(endpoint, `final correction has its c${bit} endpoint`);
      const result = await hover(r => r.title === title && r.rect.every((coordinate, i) => coordinate === endpoint.rect[i]),
        `shared-correction-c${bit}`, { exactCenter: true });
      assert.ok(result.snapshot.tooltip.summary.includes(`c${bit} = ${value}`), 'each dot describes its own condition');
    }
    report.interactions.push('two classical controls on one correction retain their own endpoint help despite crossing connectors');

    await load('qutrit');
    await hover(r => /fourier/i.test(r.title) && !/inverse/i.test(r.title), 'qutrit-fourier');
    await hover(r => /inverse.*fourier|fourier.*inverse/i.test(r.title), 'qutrit-inverse-fourier');
    await load('mixed');
    await hover(r => /unitary/i.test(r.title), 'generic-unitary');

    await page.setViewportSize({ width: 390, height: 1000 });
    await load('bell', { zoom: '1' });
    const mobile = await diag();
    const rightmost = mobile.hover_regions.filter(r => /control|target/i.test(r.title))
      .sort((a, b) => b.rect[2] - a.rect[2])[0];
    assert.ok(rightmost, 'narrow circuit retains a CNOT hover target');
    await hover(r => r.title === rightmost.title && r.summary === rightmost.summary, 'narrow-edge', { save: true });
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false, 'help adds no page overflow');
    await leave();
    report.interactions.push('390px viewport keeps the tooltip readable and inside canvas bounds');

    report.adapter = await page.evaluate(async () => {
      const a = await navigator.gpu.requestAdapter();
      return { vendor: a.info.vendor, architecture: a.info.architecture, description: a.info.description };
    });
    assert.deepEqual(report.errors, []);
    report.passed = true;
    console.log(JSON.stringify({ passed: true, hover_cases: report.cases.length, artifacts: out }));
  } catch (error) {
    report.failure = String(error.stack || error);
    if (page) {
      report.last_snapshot = await page.evaluate(() => window.__qaApi?.diagnostics_json()).catch(() => null);
      await page.screenshot({ path: path.join(out, 'failure.png'), fullPage: true }).catch(() => {});
    }
    throw error;
  } finally {
    await browser.close();
    fs.writeFileSync(path.join(out, 'results.json'), `${JSON.stringify(report, null, 2)}\n`);
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
