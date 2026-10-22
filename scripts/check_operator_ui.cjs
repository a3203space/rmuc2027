/* Run on a desktop-capable host after `bun run build`. No live match is modified. */
const {chromium}=require('playwright');
const {readFileSync,mkdirSync,writeFileSync}=require('node:fs');
const {resolve,extname,sep}=require('node:path');
const {spawnSync}=require('node:child_process');
const assert=require('node:assert/strict');
const root=resolve(__dirname,'..'),output=resolve(root,'build/operator-ui');
mkdirSync(output,{recursive:true});
const generated=spawnSync(process.env.RMUC_PYTHON||resolve(root,'.venv/bin/python'),[resolve(root,'scripts/operator_fixtures.py'),'--output',output],{encoding:'utf8'});
if(generated.status!==0)throw Error(generated.stderr);
const fixtures=JSON.parse(readFileSync(resolve(output,'fixtures.json'),'utf8'));
const types={'.html':'text/html','.js':'text/javascript','.css':'text/css','.png':'image/png','.glb':'model/gltf-binary','.json':'application/json'};
(async()=>{
 let browser;
 const report={screens:[],errors:[],rendered:false,visibleFpv:false};
 try{
  browser=await chromium.launch({headless:true,...(process.env.RMUC_CHROME?{executablePath:process.env.RMUC_CHROME}:{channel:'chrome'})});
  for(const [width,height,scale] of [[1280,720,1],[1920,1080,1],[1536,864,1.25],[1280,720,1.5]]){
   const context=await browser.newContext({viewport:{width,height},deviceScaleFactor:scale});
   await context.route('http://operator.local/**',route=>{const pathname=new URL(route.request().url()).pathname;
    const file=resolve(root,'build/web',pathname==='/play'?'apps/client/web/index.html':'.'+pathname);
    if(!file.startsWith(resolve(root,'build/web')+sep))return route.abort();
    try{return route.fulfill({body:readFileSync(file),contentType:types[extname(file)]||'application/octet-stream'});}catch{return route.fulfill({status:404,body:'missing fixture asset'});}
   });
   await context.addInitScript(catalog=>{
    window.__catalog=catalog;
    window.__commands=[];window.__frames=0;window.__visibleFpv=false;window.__last=null;
    window.qt={webChannelTransport:{}};
    window.QWebChannel=class{constructor(transport,callback){const handlers={};const signal=name=>({connect:fn=>{handlers[name]=fn}});
     window.__deliver=s=>{window.__last=s;handlers.snapshot?.(JSON.stringify(s));};
     setInterval(()=>{if(window.__last&&!window.__disconnect)handlers.snapshot?.(JSON.stringify(window.__last));},200);
     callback({objects:{arena:{response:signal('response'),snapshot:signal('snapshot'),notice:signal('notice'),room:signal('room'),input:payload=>{window.__input=JSON.parse(payload)},command:payload=>window.__commands.push(JSON.parse(payload)),viewportReady(){handlers.response?.('catalog',JSON.stringify(window.__catalog));},pause(){},renderStats(payload){window.__frames++;const stats=JSON.parse(payload);if(stats.phase==='running'&&stats.visible_samples>0)window.__visibleFpv=true;}}}});
    }};
   },fixtures._catalog);
   const page=await context.newPage();
   page.on('pageerror',e=>report.errors.push(e.message));
   await page.goto('http://operator.local/play');
   await page.waitForFunction(()=>typeof window.__deliver==='function');
   await page.getByRole('button',{name:'创建房间 ↗'}).waitFor();
   await page.screenshot({path:resolve(output,`home-${width}-${scale}.png`)});
   await page.evaluate(s=>window.__deliver(s),fixtures.lobby);
   await page.getByRole('button',{name:'机器人机库',exact:true}).click();
   await page.getByRole('button',{name:/I-T 变形隧道步兵/}).click();
   assert(await page.evaluate(()=>window.__commands.some(c=>c.name==='seat-config'&&c.body.model===301)));
   await page.screenshot({path:resolve(output,`hangar-${width}-${scale}.png`)});
   for(const phase of ['loading','intro','countdown']){await page.evaluate(s=>window.__deliver(s),fixtures[phase]);await page.locator('.ceremony-root').waitFor();await page.screenshot({path:resolve(output,`${phase}-${width}-${scale}.png`)});}
   for(const name of ['infantry','heavy','infantry4','sentry','hangar','contested','damaged','overheat','dead','resume','finished','penalty']){
    await page.evaluate(s=>window.__deliver(s),fixtures[name]);
    if(name==='finished'){await page.locator('.lan-results').waitFor();assert.equal(await page.locator('.results-actions button').count(),7);}else if(name==='penalty'){await page.locator('.penalty-cover').waitFor();assert.equal(await page.locator('.scoreboard').count(),0);}else{
     await page.locator('[data-testid="red-base"]').waitFor();
     assert.equal(await page.locator('.control-card').count(),4);
     assert.equal(await page.locator('.robot-card').count(),10);
     const header=await page.locator('.scoreboard').boundingBox(),points=await page.locator('.control-strip').boundingBox();
     assert(header&&points&&header.y+header.height<=points.y+1,'scoreboard overlaps control strip');
     for(const selector of ['.scoreboard','.control-strip','.own-state','.tactical-map']){const b=await page.locator(selector).boundingBox();assert(b&&b.x>=0&&b.y>=0&&b.x+b.width<=width+1&&b.y+b.height<=height+1,`${selector} clipped`);}
    }
    const filename=`${name}-${width}x${height}-${scale}.png`;await page.screenshot({path:resolve(output,filename)});report.screens.push(filename);
   }
   await page.evaluate(s=>window.__deliver(s),fixtures.infantry);
   await page.keyboard.down('Tab');await page.locator('.tactical-expanded').waitFor();await page.screenshot({path:resolve(output,`tactical-${width}-${scale}.png`)});await page.keyboard.up('Tab');
   await page.keyboard.press('Escape');await page.locator('.operator-menu').waitFor();
   await page.getByRole('button',{name:/远程补弹/}).click();
   assert(await page.evaluate(()=>window.__commands.some(c=>c.name==='ammo'&&c.value==='remote')));
   assert.equal(await page.evaluate(()=>window.__input.active),false);
   await page.getByRole('button',{name:'关闭',exact:true}).click();
   await page.evaluate(()=>{window.__disconnect=true;window.__deliver({transport:false});});
   await page.locator('.stale-banner').waitFor();
   report.rendered ||= await page.evaluate(()=>window.__frames>0);
   report.visibleFpv ||= await page.evaluate(()=>window.__visibleFpv);
   await context.close();
  }
  assert.equal(report.errors.length,0,JSON.stringify(report.errors));
  assert(report.rendered,'No rendered frames: investigate WebGL/GPU before accepting viewport');
  assert(report.visibleFpv,'FPV has no nonblack pixel samples; inspect render targets and shaders');
  report.passed=true;
 }catch(error){report.passed=false;report.failure=String(error);process.exitCode=1;}
 finally{if(browser)await browser.close();writeFileSync(resolve(output,'report.json'),JSON.stringify(report,null,2));console.log(JSON.stringify({passed:report.passed,screens:report.screens.length,rendered:report.rendered,failure:report.failure?.slice(-1500)}));}
})();
