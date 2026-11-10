#!/usr/bin/env node
// Focused scientific application UI qualification. Uses the same Playwright/WebGPU
// environment as validate_visualization.cjs; no server or browser is implied.
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const base = process.env.VISUALIZATION_URL || 'http://localhost:8097/';
const out = path.resolve(process.env.VISUALIZATION_ARTIFACTS || 'artifacts/visualization/applications-focused');
fs.mkdirSync(out, {recursive:true});
const report = {url:base, timestamp:new Date().toISOString(), cases:[], errors:[]};

(async()=>{
 const browser = await chromium.launch({
  ...(process.env.CHROME_BIN ? {executablePath:process.env.CHROME_BIN} : {}),
  headless:process.env.BROWSER_HEADLESS !== 'false',
  args:JSON.parse(process.env.BROWSER_ARGS || '[]'),
 });
 try {
  report.browser=browser.version();
  const page=await browser.newPage({viewport:{width:1600,height:1200},deviceScaleFactor:1});
  const focus=await page.context().newCDPSession(page);
  await focus.send('Emulation.setFocusEmulationEnabled',{enabled:true});
  page.on('pageerror',error=>report.errors.push(String(error)));
  page.on('console',message=>{if(message.type()==='error')report.errors.push(message.text());});
  await page.addInitScript(()=>{
   window.__qaHeartbeat=[];
   let last=performance.now();
   setInterval(()=>{
    const now=performance.now();
    let diagnostics;
    try {if(window.__qaApi) diagnostics=JSON.parse(window.__qaApi.diagnostics_json());} catch {}
    if(window.__qaHeartbeat.length<20_000)window.__qaHeartbeat.push({
     time:now,gap:now-last,status:window.__burnQuantumStatus?.state,
     app_frames:diagnostics?.app_frames,ready:diagnostics?.ready,
    });
    last=now;
   },25);
  });
  const diag=()=>page.evaluate(()=>JSON.parse(window.__qaApi.diagnostics_json()));
  async function ready() {
   await page.bringToFront();
   await page.evaluate(async()=>{window.__qaApi=await import('./pkg/burn_quantum_visualization.js');});
   await page.waitForFunction(()=>['ready','error'].includes(window.__burnQuantumStatus?.state),null,{timeout:180000});
   const status=await page.evaluate(()=>window.__burnQuantumStatus);
   assert.equal(status.state,'ready',JSON.stringify(status));
   await page.waitForFunction(()=>{
    const text=window.__qaApi.diagnostics_json();
    return text && JSON.parse(text).ready && JSON.parse(text).controls;
   });
   await page.waitForTimeout(350);
  }
  async function click(action) {
   await page.locator('canvas').scrollIntoViewIfNeeded();
   const d=await diag(),button=d.controls.find(c=>c.action===action);
   assert.ok(button,`visible ${action}`);
   const b=await page.locator('canvas').boundingBox(),r=button.rect;
   await page.mouse.move(b.x+(r[0]+r[2])/2,b.y+(r[1]+r[3])/2);
   await page.waitForFunction(frame=>JSON.parse(window.__qaApi.diagnostics_json()).app_frames>frame+2,d.app_frames);
   await page.mouse.down();
   await page.waitForFunction(action=>JSON.parse(window.__qaApi.diagnostics_json()).controls.find(c=>c.action===action)?.pressed,action,{timeout:10000});
   await page.mouse.up();
   await page.waitForFunction(revision=>JSON.parse(window.__qaApi.diagnostics_json()).revision!==revision,d.revision);
   await page.waitForTimeout(200);
  }
  const cases=[
   ['wine-results-desktop',1600,1200,'demo=wine&view=results'],
   ['sat-results-desktop',1600,1200,'demo=three-sat&view=results'],
   ['photonic-results-desktop',1600,1200,'demo=photonic&view=results'],
   ['wine-results-mobile',390,1000,'demo=wine&view=results'],
   ['sat-results-mobile',390,1000,'demo=three-sat&view=results'],
   ['photonic-results-mobile',390,1000,'demo=photonic&view=results'],
   ['wine-learning-mobile',390,1000,'demo=wine&view=learning&frame=64'],
   ['photonic-vacuum-mobile',390,1000,'demo=photonic&size=3&variant=1&view=state&frame=12'],
   ['photonic-twelve-qubits',1600,1200,'demo=photonic&size=6&variant=0&view=state&frame=24'],
   ['wine-eight-qubits-three-layers',1600,1200,'demo=wine&size=8&layers=3&view=results'],
  ];
  for(const [name,width,height,query] of cases) {
   await page.setViewportSize({width,height});
   const start=Date.now();
   await page.goto(`${base}?${query}`);await ready();
   const startup_ms=Date.now()-start,d=await diag();
   const doc=await page.evaluate(()=>JSON.parse(window.__qaApi.export_document_json()));
   const params=new URLSearchParams(query);
   assert.equal(d.demo,params.get('demo'));assert.equal(d.view,params.get('view'));
   assert.equal(doc.study!==undefined,true,'task report retained in export');
   assert.ok(doc.trace.frames.every(frame=>Math.abs(frame.norm-1)<5e-4));
   assert.ok(d.controls.some(c=>c.action==='View(Results)'));
   assert.equal(d.controls.some(c=>c.action==='View(Learning)'),doc.optimization.length>0);
   if(d.view==='learning')assert.equal(d.drawing_dropped_lines,0,'validation curves/checkpoint must not be dropped by the renderer budget');
   const box=await page.locator('canvas').boundingBox();
   assert.ok(d.canvas[0]>=0&&d.canvas[1]>=0&&d.canvas[2]<=box.width+1&&d.canvas[3]<=box.height+1,`${name}: canvas bounded`);
   if(d.view==='results') {
    assert.ok(!d.controls.some(c=>c.action==='Scrubber'||c.action==='Play'),'aggregate results hide irrelevant playback');
    assert.equal(new URLSearchParams(d.query).get('view'),'results');
   }
   if(doc.study.photonic) {
    const p=doc.study.photonic;
    assert.ok(p.delivered+p.erased<=p.attempts);
    assert.equal(doc.circuit.register.length,p.logical_qubits);
    if(name==='photonic-vacuum-mobile') {
     assert.equal(p.includes_vacuum,true);
     assert.ok(doc.trace.frames.at(-1).subsystems.every(q=>q.dimension===3&&q.probabilities[2]<1e-6));
    }
    if(name==='photonic-twelve-qubits') {
     assert.equal(p.logical_qubits,12);assert.equal(p.includes_vacuum,false);
     assert.ok(doc.trace.frames.at(-1).subsystems.every(q=>Array.isArray(q.bloch)));
    }
   }
   const samples=await page.evaluate(()=>window.__qaHeartbeat);
   const loading=samples.filter(s=>s.ready===false&&s.app_frames>0);
   const timing={startup_ms,loading_ticks:loading.length,max_loading_gap_ms:Math.max(0,...loading.map(s=>s.gap)),max_page_gap_ms:Math.max(0,...samples.map(s=>s.gap))};
   if(name==='wine-eight-qubits-three-layers') {
    assert.ok(loading.length>=4,'browser timers execute while the largest QML experiment is still training');
    assert.ok(timing.max_loading_gap_ms<5000,`training starved the event loop: ${timing.max_loading_gap_ms} ms`);
    assert.equal(doc.circuit.register.length,8);
   }
   await page.screenshot({path:path.join(out,`${name}.png`),fullPage:true});
   fs.writeFileSync(path.join(out,`${name}.json`),JSON.stringify({diagnostics:d,study:doc.study,timing},null,2)+'\n');
   report.cases.push({name,viewport:[width,height],...timing,frames:doc.trace.frames.length,diagnostics:d});
   if(d.view==='results') {
    await click('Inspect');
    const details=await diag();
    assert.ok(details.inspection_text.includes('ASSUMPTIONS & PROVENANCE'));
    if(doc.study.sat)assert.ok(details.inspection_text.includes(' OR ')&&!details.inspection_text.includes('∨'));
    if(width<500)await page.screenshot({path:path.join(out,`${name}-details.png`),fullPage:true});
   }
  }
  assert.deepEqual(report.errors,[]);report.passed=true;
 } finally {
  await browser.close();
  fs.writeFileSync(path.join(out,'results.json'),JSON.stringify(report,null,2)+'\n');
 }
 console.log(JSON.stringify({passed:report.passed,cases:report.cases.length,artifacts:out}));
})().catch(error=>{console.error(error);process.exitCode=1;});
