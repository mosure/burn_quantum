#!/usr/bin/env node
// Actual browser rendering/interaction qualification. Requires Playwright and WebGPU.
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const base = process.env.VISUALIZATION_URL || 'http://localhost:8097/';
const out = path.resolve(process.env.VISUALIZATION_ARTIFACTS || 'artifacts/visualization/browser');
fs.mkdirSync(out, {recursive:true});
async function exerciseTouch(page) {
 await page.evaluate(async()=>{window.__qaApi=await import('./pkg/burn_quantum_visualization.js');});
 await page.waitForTimeout(500);await page.locator('canvas').scrollIntoViewIfNeeded();
 const diag=()=>page.evaluate(()=>JSON.parse(window.__qaApi.diagnostics_json()));
 const before=await diag(),box=await page.locator('canvas').boundingBox(),r=before.canvas;
 const center={x:box.x+(r[0]+r[2])/2,y:box.y+(r[1]+r[3])/2},cdp=await page.context().newCDPSession(page);
 await cdp.send('Input.dispatchTouchEvent',{type:'touchStart',touchPoints:[{x:center.x-25,y:center.y,id:1},{x:center.x+25,y:center.y,id:2}]});
 await page.waitForTimeout(100);
 await cdp.send('Input.dispatchTouchEvent',{type:'touchMove',touchPoints:[{x:center.x-45,y:center.y,id:1},{x:center.x+45,y:center.y,id:2}]});
 await page.waitForTimeout(150);await cdp.send('Input.dispatchTouchEvent',{type:'touchEnd',touchPoints:[]});await page.waitForTimeout(350);
 const pinched=await diag();assert.ok(pinched.zoom>1.5);
 const hit=pinched.controls.find(c=>c.action==='Inspect').rect,x=box.x+(hit[0]+hit[2])/2,y=box.y+(hit[1]+hit[3])/2;
 await cdp.send('Input.dispatchTouchEvent',{type:'touchStart',touchPoints:[{x,y,id:1}]});await page.waitForTimeout(100);
 await cdp.send('Input.dispatchTouchEvent',{type:'touchEnd',touchPoints:[]});await page.waitForTimeout(350);
 assert.equal((await diag()).show_details,true);
 return {viewport:[390,1000],before_zoom:before.zoom,after_pinch_zoom:pinched.zoom,details_after_touch:true};
}
const report = {url:base, timestamp:new Date().toISOString(), demos:[], interactions:[], errors:[]};
(async()=>{
 const browser = await chromium.launch({
   ...(process.env.CHROME_BIN ? {executablePath:process.env.CHROME_BIN} : {}),
   headless:process.env.BROWSER_HEADLESS !== 'false',
   args:JSON.parse(process.env.BROWSER_ARGS || '[]'),
 });
 try {
  report.browser = browser.version();
  const page = await browser.newPage({viewport:{width:1600,height:1200},deviceScaleFactor:1});
  page.on('pageerror', error => report.errors.push(String(error)));
  report.navigation=[];page.on('framenavigated',frame=>{if(frame===page.mainFrame())report.navigation.push({url:frame.url(),time:Date.now()});});
  page.on('console', message => {if(message.type()==='error') report.errors.push(message.text());});
  const diag = () => page.evaluate(async()=>JSON.parse((await import('./pkg/burn_quantum_visualization.js')).diagnostics_json()));
  async function ready() {
   await page.bringToFront();
   await page.waitForFunction(()=>['ready','error'].includes(window.__burnQuantumStatus?.state),null,{timeout:60000});
   assert.equal((await page.evaluate(()=>window.__burnQuantumStatus)).state,'ready');
   await page.evaluate(async()=>{window.__qaApi=await import('./pkg/burn_quantum_visualization.js');window.__qaPointer=[];for(const type of ['pointerdown','pointerup','mousedown','mouseup'])window.addEventListener(type,e=>{window.__qaPointer.push({type,x:e.clientX,y:e.clientY,target:e.target.tagName,focused:document.hasFocus()});window.__qaPointer=window.__qaPointer.slice(-12);},true);});
   await page.waitForFunction(()=>{
    const s=window.__qaApi.diagnostics_json();
    return s && JSON.parse(s).ready && JSON.parse(s).controls;
   });
   await page.waitForTimeout(400);
  }
  async function click(action) {
   await page.locator('canvas').scrollIntoViewIfNeeded();
   const d=await diag(), control=d.controls.find(c=>c.action===action) || d.plot_hits.find(c=>c.action===action);
   assert.ok(control,`visible control ${action}`);
   const b=await page.locator('canvas').boundingBox(),r=control.rect;
   assert.ok(r[0]>=-1 && r[1]>=-1 && r[2]<=b.width+1 && r[3]<=b.height+1,`${action} clipped: ${JSON.stringify({r,b})}`);
   await page.mouse.move(b.x+(r[0]+r[2])/2,b.y+(r[1]+r[3])/2);
   // Allow Bevy's frame-driven picking to observe hover before sending the press.
   await page.waitForFunction(frame=>JSON.parse(window.__qaApi.diagnostics_json()).app_frames>frame+2,d.app_frames);
   const hovered=await diag();
   await page.mouse.down();
   if(d.controls.some(c=>c.action===action) && !control.disabled) {
    await page.waitForFunction(action=>JSON.parse(window.__qaApi.diagnostics_json()).controls.find(c=>c.action===action)?.pressed,action,{timeout:10000});
   } else {
    await page.waitForFunction(frame=>JSON.parse(window.__qaApi.diagnostics_json()).app_frames>frame+2,hovered.app_frames);
   }
   const pressed=await diag();
   await page.mouse.up();
   if(action!=='Run'||d.config_valid) {
    try{await page.waitForFunction(revision=>JSON.parse(window.__qaApi.diagnostics_json()).revision!==revision,d.revision,{timeout:10000});}catch(error){const last=await diag();report.failed_click={action,before:d,pressed,last,events:await page.evaluate(()=>window.__qaPointer)};console.error(JSON.stringify(report.failed_click));await page.screenshot({path:path.join(out,'failed-click.png'),fullPage:true});throw error;}
   }
   await page.waitForTimeout(300);
  }
  const cases=[['bell','state',2],['ghz','state',6],['mixed','state',6],['teleportation','circuit',10],['toffoli','circuit',3],['qft','circuit',0],['deutsch-jozsa','circuit',0],['qpe','circuit',0],['grover','state',5],['qaoa','circuit',0],['bit-flip','circuit',0],['interference','state',2],['qutrit','state',2],['photonic','fabric',0],['optimization','learning',16]];
  for (const [demo,view,step] of cases) {
   const start=Date.now();await page.goto(`${base}?demo=${demo}&view=${view}&step=${step}`);await ready();
   const d=await diag();assert.equal(d.demo,demo);assert.equal(d.view,view);
   if(demo==='bell'){assert.ok(d.controls.length<=10,'default toolbar stays compact');assert.ok(!d.controls.some(c=>['Grid','Trails','Reset','BlochMode(Planar)'].includes(c.action)),'secondary controls are disclosed in Settings');}
   const document=await page.evaluate(async()=>JSON.parse((await import('./pkg/burn_quantum_visualization.js')).export_document_json()));
   assert.ok(document.trace.frames.length>1);assert.ok(document.trace.frames.every(f=>Math.abs(f.norm-1)<5e-4));
   const b=await page.locator('canvas').boundingBox();
   assert.ok(d.canvas[3]<=b.height,`plot exceeds canvas in ${demo}`);
   assert.ok(d.controls.some(c=>c.action==='Scrubber' && c.rect[3]<=b.height),`timeline clipped in ${demo}`);
   report.demos.push({demo,view,startup_ms:Date.now()-start,frames:document.trace.frames.length,diagnostics:d});
   await page.screenshot({path:path.join(out,`${demo}.png`),fullPage:true});
   await click('Inspect');let detail=await diag();assert.equal(detail.show_details,true);assert.ok(detail.inspection_text.length>30);
   if(demo==='optimization') {assert.ok(detail.inspection_text.includes(document.optimization[step].parameters[0].toFixed(5)));await page.screenshot({path:path.join(out,'learning-details.png'),fullPage:true});}
   await click('Inspect');assert.equal((await diag()).show_details,false);
   if(demo==='photonic') {
    await click('Step(1)');assert.equal((await diag()).tick,1);assert.equal((await diag()).frame,0);
    await click('Play');await page.waitForTimeout(1000);await click('Play');assert.ok((await diag()).tick>1);
    report.interactions.push('physical schedule playback preserves quantum frame');
   }
  }
  report.adapter=await page.evaluate(async()=>{const a=await navigator.gpu.requestAdapter();return {vendor:a.info.vendor,architecture:a.info.architecture,description:a.info.description};});
  await page.goto(`${base}?demo=bell&step=0`);await ready();
  await click('Step(1)');assert.equal((await diag()).frame,1);
  {const d=await diag(),b=await page.locator('canvas').boundingBox();await page.mouse.click(b.x+d.canvas[0]+10,b.y+d.canvas[1]+10,{delay:70});}
  await page.keyboard.press('ArrowRight');await page.waitForTimeout(300);assert.equal((await diag()).frame,2);
  await click('View(Circuit)');assert.equal((await diag()).view,'circuit');
  await click('View(State)');await click('Wire(1)');assert.equal((await diag()).qubit,1);
  const baseline=(await diag()).entity_count;
  for(let i=0;i<8;i++){await click('View(Circuit)');await click('View(State)');}
  assert.equal((await diag()).entity_count,baseline);
  report.interactions.push('step buttons and keyboard, view changes, qubit selection, stable entity count after 16 tab switches');
  const box=await page.locator('canvas').boundingBox(),area=(await diag()).canvas;
  await page.mouse.move(box.x+(area[0]+area[2])/2,box.y+(area[1]+area[3])/2);await page.mouse.wheel(0,-180);await page.waitForTimeout(300);
  assert.notEqual((await diag()).zoom,1);await click('Settings');await click('Reset');assert.equal((await diag()).zoom,1);
  await click('Grid');assert.equal((await diag()).guides,false);await click('Grid');await click('Settings');
  report.interactions.push('zoom/reset and guides');
  const before=await diag();
  const intervals=await page.evaluate(()=>new Promise(resolve=>{const values=[];let last;function next(t){if(last!==undefined)values.push(t-last);last=t;if(values.length===120)resolve(values);else requestAnimationFrame(next);}requestAnimationFrame(next);}));
  const after=await diag(),sorted=[...intervals].sort((a,b)=>a-b);
  report.steady_state={raf_samples:120,median_ms:sorted[60],p95_ms:sorted[114],app_frames_advanced:after.app_frames-before.app_frames,entity_count:after.entity_count,revision_unchanged:after.revision===before.revision};
  assert.ok(after.app_frames>before.app_frames);assert.ok(report.steady_state.revision_unchanged);
  const downloadEvent=page.waitForEvent('download');await page.locator('#export-button').click();const download=await downloadEvent;await download.saveAs(path.join(out,'export.json'));
  const saved=JSON.parse(fs.readFileSync(path.join(out,'export.json'),'utf8'));saved.title='Imported Bell validation';
  await page.locator('#import-file').setInputFiles({name:'valid.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify(saved))});
  await page.waitForFunction(() => JSON.parse(window.__qaApi.diagnostics_json()).title==='Imported Bell validation');
  report.interchange={accepted:await diag()};
  saved.trace.frames[0].probabilities[0]=-1;
  await page.locator('#import-file').setInputFiles({name:'invalid.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify(saved))});
  await page.waitForFunction(()=>document.querySelector('#io-notice').dataset.error==='true');
  report.interchange.rejected=await diag();report.interchange.exported_title=await page.evaluate(async()=>JSON.parse((await import('./pkg/burn_quantum_visualization.js')).export_document_json()).title);assert.equal(report.interchange.rejected.title,'Imported Bell validation');assert.equal(report.interchange.exported_title,'Imported Bell validation');
  report.interactions.push('actual JSON download, valid file import, invalid import preserves selected experiment');
  await page.setViewportSize({width:390,height:1000});await page.waitForTimeout(700);
  await click('Step(1)');await click('Wire(1)');await click('View(Circuit)');await click('View(State)');
  const small=await diag(),smallBox=await page.locator('canvas').boundingBox();assert.ok(small.canvas[3]<=smallBox.height);
  await page.screenshot({path:path.join(out,'narrow.png'),fullPage:true});report.narrow=small;
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false,'page overflows horizontally');
  await click('Inspect');const details=await diag();assert.equal(details.show_details,true);
  assert.ok(details.controls.find(c=>c.action==='Scrubber').rect[3]<=smallBox.height,'Details must preserve the entire timeline');
  assert.ok(details.scroll_regions.length>0,'narrow Details has a bounded inspection region');
  const region=details.scroll_regions.find(r=>r.content_height>r.viewport_height);
  await page.screenshot({path:path.join(out,'narrow-details.png'),fullPage:true});
  if(region){const b=await page.locator('canvas').boundingBox(),r=region.rect;await page.mouse.move(b.x+(r[0]+r[2])/2,b.y+(r[1]+r[3])/2);await page.mouse.wheel(0,250);await page.waitForTimeout(350);assert.ok((await diag()).scroll_regions.some(r=>r.offset_y>0),'overflowing Details actually scrolls');}
  await click('Inspect');
  report.interactions.push('390px viewport retains timeline, qubit, view and scientific Details controls');
  if(process.env.LARGE_TRACE_JSON) {
   await page.setViewportSize({width:1600,height:1200});await page.waitForTimeout(500);
   await page.locator('#import-file').setInputFiles(process.env.LARGE_TRACE_JSON);
   await page.waitForFunction(()=>JSON.parse(window.__qaApi.diagnostics_json()).title==='12-qubit bounded trace');
   for(let i=0;i<12;i++)await click('Step(1)');
   const large=await diag();assert.equal(large.frame,12);assert.equal(large.state_overview,true);assert.equal(large.plot_hits.filter(h=>h.action.startsWith('Wire(')).length,12);assert.ok(large.drawing.includes('4096 states'));assert.ok(large.entity_count<750);
   await page.screenshot({path:path.join(out,'twelve-qubits.png'),fullPage:true});report.large_trace=large;
   report.interactions.push('actual 12-qubit / 4096-state import and timeline retain bounded draw entities');
  }
  // Configurations are previewed first and only recomputed on Run.
  await page.setViewportSize({width:1600,height:1200});
  await page.goto(`${base}?demo=ghz&size=12&step=12`);await ready();
  assert.equal((await diag()).overview_capacity,12);
  await click('Wire(9)');assert.equal((await diag()).qubit,9);
  await click('ToggleFocus');assert.equal((await diag()).state_overview,false);
  await page.screenshot({path:path.join(out,'focused-q9.png'),fullPage:true});
  await click('ToggleFocus');
  await click('Settings');await click('Size(1)');assert.equal((await diag()).config.size,13);
  {const d=await page.evaluate(()=>JSON.parse(window.__qaApi.export_document_json()));assert.equal(d.circuit.register.length,12);}
  await click('Run');await page.waitForFunction(()=>{const d=JSON.parse(window.__qaApi.diagnostics_json());return d.ready&&JSON.parse(window.__qaApi.export_document_json()).circuit.register.length===13;});
  await click('Settings');await click('Size(1)');await click('Size(1)');await click('Size(1)');assert.equal((await diag()).config_valid,false);
  await click('Run');assert.equal((await diag()).ready,true);await click('Settings');
  await page.setViewportSize({width:390,height:1000});await page.waitForTimeout(700);
  const groupBefore=await diag();assert.ok(groupBefore.overview_capacity<13&&groupBefore.overview_capacity>=1);assert.ok(groupBefore.canvas[3]-groupBefore.canvas[1]>=170,'mobile register has useful height');
  await click('StateGroup(1)');const groupAfter=await diag();assert.ok(groupAfter.qubit>0);assert.ok(groupAfter.plot_hits.some(h=>h.action===`Wire(${groupAfter.qubit})`));
  await page.screenshot({path:path.join(out,'register-groups-mobile.png'),fullPage:true});
  await click('Settings');await click('BlochMode(Planar)');assert.equal((await diag()).bloch_mode,'Planar');
  await click('Plane');assert.equal((await diag()).bloch_plane,'Yz');await click('Settings');
  await page.screenshot({path:path.join(out,'planar-mobile.png'),fullPage:true});
  await click('Settings');await click('BlochMode(Spatial)');assert.equal((await diag()).bloch_mode,'Spatial');await click('Settings');
  await page.setViewportSize({width:1600,height:1200});await page.waitForTimeout(500);
  await page.goto(`${base}?demo=mixed&step=6`);await ready();
  await click('Settings');await click('BlochMode(Planar)');await click('Settings');await page.screenshot({path:path.join(out,'mixed-2d.png'),fullPage:true});
  await click('Settings');await click('Plane');await click('Plane');assert.equal((await diag()).bloch_plane,'Xy');
  await click('BlochMode(Spatial)');await click('Trails');assert.equal((await diag()).trails,true);await page.screenshot({path:path.join(out,'view-settings.png'),fullPage:true});await click('Settings');
  await page.screenshot({path:path.join(out,'mixed-3d-trails.png'),fullPage:true});
  await page.setViewportSize({width:390,height:1000});
  await page.goto(`${base}?demo=photonic&view=fabric`);await ready();
  {const d=await diag();assert.equal(d.plot_hits.filter(h=>h.action.startsWith('Node(')).length,4);for(const h of d.plot_hits){assert.ok(h.rect[0]>=d.canvas[0]&&h.rect[2]<=d.canvas[2],'mobile fabric components fit horizontally');}}
  await page.screenshot({path:path.join(out,'photonic-mobile.png'),fullPage:true});
  report.register_groups={before:groupBefore,after:groupAfter};
  report.interactions.push('12 simultaneous qubits, direct q9 selection, focused inspection, explicit 13-qubit run, oversize rejection, mobile groups');
  const unavailable=await browser.newPage();await unavailable.addInitScript(()=>Object.defineProperty(navigator,'gpu',{value:undefined}));await unavailable.goto(base);
  await unavailable.waitForFunction(()=>window.__burnQuantumStatus?.state==='error');
  assert.ok(await unavailable.locator('#native-link').isVisible());await unavailable.screenshot({path:path.join(out,'unsupported.png')});await unavailable.close();
  report.interactions.push('unsupported WebGPU displays native fallback');
  const lost=await browser.newPage({viewport:{width:390,height:1000},hasTouch:true,isMobile:true});
  await lost.addInitScript(()=>{const request=GPUAdapter.prototype.requestDevice;GPUAdapter.prototype.requestDevice=async function(...args){const device=await request.apply(this,args);(window.__qaDevices??=[]).push(device);return device;};});
  await lost.goto(base);await lost.waitForFunction(()=>window.__burnQuantumStatus?.state==='ready');
  report.touch=await exerciseTouch(lost);report.interactions.push('two-finger touch pinch and Feathers touch activation');
  await lost.evaluate(()=>window.__qaDevices.forEach(device=>device.destroy()));
  await lost.waitForFunction(()=>window.__burnQuantumStatus?.state==='error',null,{timeout:10000});
  assert.ok(await lost.locator('#overlay').isVisible());await lost.screenshot({path:path.join(out,'device-lost.png'),fullPage:true});await lost.close();
  report.interactions.push('actual WebGPU device loss reports an error instead of leaving a silent frozen canvas');
  assert.deepEqual(report.errors,[]);report.passed=true;
 } finally { await browser.close();fs.writeFileSync(path.join(out,'results.json'),JSON.stringify(report,null,2)+'\n'); }
 console.log(JSON.stringify({passed:report.passed,demos:report.demos.length,steady_state:report.steady_state,artifacts:out}));
})().catch(error=>{console.error(error);process.exitCode=1;});
