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
const types={'.html':'text/html','.js':'text/javascript','.css':'text/css','.png':'image/png','.glb':'model/gltf-binary','.json':'application/json','.svg':'image/svg+xml','.m4a':'audio/mp4','.mp3':'audio/mpeg'};
(async()=>{
 let browser;
 const report={screens:[],errors:[],rendered:false,visibleFpv:false};
 try{
  browser=await chromium.launch({headless:true,...(process.env.RMUC_CHROME?{executablePath:process.env.RMUC_CHROME}:{channel:'chrome'})});
  for(const [width,height,scale] of [[1280,720,1],[1920,1080,1],[1536,864,1.25],[1280,720,1.5],[2560,1080,1],[1280,1024,1]]){
   const context=await browser.newContext({viewport:{width,height},deviceScaleFactor:scale});
   await context.route('http://operator.local/**',route=>{const pathname=new URL(route.request().url()).pathname;
    const file=resolve(root,'build/web',pathname==='/play'?'apps/client/web/index.html':'.'+pathname);
    if(!file.startsWith(resolve(root,'build/web')+sep))return route.abort();
    try{return route.fulfill({body:readFileSync(file),contentType:types[extname(file)]||'application/octet-stream'});}catch{return route.fulfill({status:404,body:'missing fixture asset'});}
   });
   await context.addInitScript(catalog=>{
    window.__catalog=catalog;
    window.__commands=[];window.__motionStats=[];window.__frames=0;window.__visibleFpv=false;window.__last=null;
    window.qt={webChannelTransport:{}};
    window.QWebChannel=class{constructor(transport,callback){const handlers={};const signal=name=>({connect:fn=>{handlers[name]=fn}});
     window.__deliver=s=>{window.__last=s;handlers.snapshot?.(JSON.stringify(s));};
     setInterval(()=>{if(window.__last&&!window.__disconnect)handlers.snapshot?.(JSON.stringify(window.__last));},200);
     callback({objects:{arena:{response:signal('response'),snapshot:signal('snapshot'),notice:signal('notice'),room:signal('room'),input:payload=>{window.__input=JSON.parse(payload)},command:payload=>{
      const command=JSON.parse(payload);window.__commands.push(command);
      if(command.name==='profile'){
       const result={request_id:command.body.request_id,ok:!window.__rejectProfile,message:'模拟保存失败'};
       window.__profileReply=()=>{
        if(result.ok){const s=structuredClone(window.__last),me=s.room.players.find(p=>p.id===s.player_id);me.profile={chassis:command.body.chassis,weapon:command.body.weapon};me.ready=false;s.room.revision=(s.room.revision||0)+1;window.__last=s;handlers.room?.(JSON.stringify(s.room));result.room=s.room;}
        handlers.response?.('profile-result',JSON.stringify(result));
       };
       if(!window.__holdProfileReply)window.__profileReply();
      }
      if(command.name==='seat-config'){
       const s=structuredClone(window.__last);Object.assign(s.room.seat_configs[command.body.team][command.body.slot],{model:command.body.model,strategy:command.body.strategy});s.room.revision=(s.room.revision||0)+1;window.__last=s;handlers.room?.(JSON.stringify(s.room));
      }
     },viewportReady(){handlers.response?.('catalog',JSON.stringify(window.__catalog));},pause(){},renderStats(payload){window.__frames++;const stats=JSON.parse(payload);if(stats.player_timeline)window.__motionStats.push(stats.player_timeline);if(stats.phase==='running'&&stats.visible_samples>0)window.__visibleFpv=true;}}}});
    }};
   },fixtures._catalog);
   const page=await context.newPage();
   page.on('pageerror',e=>report.errors.push(e.message));
   await page.goto('http://operator.local/play');
   await page.waitForFunction(()=>typeof window.__deliver==='function');
   await page.locator('.home-title').waitFor();
   await page.screenshot({path:resolve(output,`home-${width}-${scale}.png`)});
   await page.getByRole('button',{name:'对战大厅',exact:true}).click();
   await page.getByRole('button',{name:'创建房间 ↗'}).waitFor();
   await page.screenshot({path:resolve(output,`connect-${width}-${scale}.png`)});
   await page.evaluate(s=>window.__deliver(s),fixtures.lobby);
   await page.getByRole('button',{name:'机器人机库',exact:true}).click();
   await page.getByRole('button',{name:/I-T 变形隧道步兵/}).click();
   assert(await page.evaluate(()=>window.__commands.some(c=>c.name==='seat-config'&&c.body.model===301)));
   await page.screenshot({path:resolve(output,`hangar-${width}-${scale}.png`)});
   await page.getByRole('radio',{name:/血量优先/}).waitFor();
   await page.evaluate(()=>{window.__holdProfileReply=true;});
   await page.getByRole('radio',{name:/功率优先/}).click();
   await page.locator('.performance-picker[aria-busy="true"]').waitFor();
   await page.getByRole('button',{name:'对战大厅',exact:true}).click();
   assert(await page.getByRole('button',{name:'准备出战',exact:true}).isDisabled(),'ready must wait for profile acknowledgement');
   await page.evaluate(()=>{window.__profileReply();window.__holdProfileReply=false;});
   await page.getByRole('button',{name:'机器人机库',exact:true}).click();
   await page.locator('.performance-feedback.saved').waitFor();
   assert.equal(await page.getByRole('radio',{name:/功率优先/}).getAttribute('aria-checked'),'true');
   for(const name of ['爆发优先','血量优先','冷却优先']){
    await page.getByRole('radio',{name:new RegExp(name)}).click();await page.locator('.performance-feedback.saved').waitFor();
   }
   await page.evaluate(()=>{window.__rejectProfile=true;});
   await page.getByRole('radio',{name:/功率优先/}).click();await page.locator('.performance-feedback.error').waitFor();
   assert.equal(await page.getByRole('radio',{name:/血量优先/}).getAttribute('aria-checked'),'true');
   assert.equal(await page.getByRole('radio',{name:/冷却优先/}).getAttribute('aria-checked'),'true');
   await page.evaluate(()=>{window.__rejectProfile=false;});
   await page.getByRole('radio',{name:/功率优先/}).click();await page.locator('.performance-feedback.saved').waitFor();
   await page.getByRole('button',{name:/I-T 变形隧道步兵/}).click();
   assert.equal(await page.getByRole('radio',{name:/功率优先/}).getAttribute('aria-checked'),'true','hardware changes must preserve performance');
   for(const button of await page.getByRole('radio').all()){
    await button.scrollIntoViewIfNeeded();
    assert(await button.evaluate(el=>{const b=el.getBoundingClientRect();return el.contains(document.elementFromPoint(b.x+b.width/2,b.y+b.height/2));}),'performance button is covered');
   }
   await page.screenshot({path:resolve(output,`performance-${width}-${scale}.png`)});

   for(const phase of ['loading','intro','countdown']){await page.evaluate(s=>window.__deliver(s),fixtures[phase]);await page.locator('.bc-stage.cer-loading,.bc-stage.cer-lineup,.bc-stage.cer-countdown').waitFor();await page.screenshot({path:resolve(output,`${phase}-${width}-${scale}.png`)});}
   for(const name of ['infantry','heavy','infantry4','blue-infantry','blue-infantry4','sentry','hangar','contested','damaged','overheat','dead','resume','finished','penalty']){
    await page.evaluate(s=>window.__deliver(s),fixtures[name]);
    if(name==='finished'){await page.locator('.lan-results').waitFor();assert.equal(await page.locator('.lan-results-actions button').count(),7);}else if(name==='penalty'){await page.locator('.penalty-cover').waitFor();assert.equal(await page.locator('.operator-hud .bc-top').count(),0);assert.equal(await page.locator('.data-robot-row').count(),0);}else{
     await page.locator('[data-testid="red-base"]').waitFor();
     if(name==='dead'){
      await page.locator('.camera-layer.camera-grayscale canvas').waitFor();
      assert.equal(await page.locator('.camera-layer').evaluate(el=>getComputedStyle(el).filter),'grayscale(1)');
      assert.equal(await page.locator('.operator-hud').evaluate(el=>getComputedStyle(el).filter),'none','HUD must retain its colors');
      assert.equal(await page.evaluate(()=>window.__input?.fire||window.__input?.aim_assist||window.__input?.active||false),false,'death must release combat');
     }
     assert.equal(await page.locator('.control-card').count(),4);
     assert.equal(await page.locator('.bc-card').count(),10);
     const header=await page.locator('.operator-hud .bc-top').boundingBox(),points=await page.locator('.control-strip').boundingBox();
     assert(header&&points&&header.y+header.height<=points.y+1,'scoreboard overlaps control strip');
     const surface=await page.locator('.operator-surface').boundingBox(),own=await page.locator('.own-state').boundingBox(),map=await page.locator('.operator-hud .bc-mini').boundingBox();
     assert(surface&&own&&map);
     assert(Math.abs(surface.width/surface.height-16/9)<.002,'camera surface must stay 16:9');
     assert(own.x+own.width<surface.x+surface.width/2&&map.x>surface.x+surface.width/2,'health belongs on the left and map on the right');
     if(await page.locator('.sight-center').count()){
      const aim=await page.locator('.sight-center').boundingBox();
      assert(Math.abs(aim.x+aim.width/2-surface.x-surface.width/2)<1&&Math.abs(aim.y+aim.height/2-surface.y-surface.height/2)<1,'reticle must match camera center');
     }
     for(const roster of await page.locator('.bc-cards').all()){
      const cards=await roster.locator('.bc-card').all();
      const boxes=await Promise.all(cards.map(card=>card.boundingBox()));
      assert(boxes.every(box=>Math.abs(box.y-boxes[0].y)<1),'robot cards must form horizontal rows');
      for(const card of cards){
       assert(await card.evaluate(el=>el.scrollWidth<=el.clientWidth+1),'robot card text overflows');
      }
     }
     for(const selector of ['.operator-hud .bc-top','.control-strip','.own-state','.operator-hud .bc-mini']){const b=await page.locator(selector).boundingBox();assert(b&&b.x>=0&&b.y>=0&&b.x+b.width<=width+1&&b.y+b.height<=height+1,`${selector} clipped`);}
    }
    const filename=`${name}-${width}x${height}-${scale}.png`;await page.screenshot({path:resolve(output,filename)});report.screens.push(filename);
   }
   await page.evaluate(s=>window.__deliver(s),fixtures.infantry);
   await page.locator('.operator-hud').waitFor();
   await page.waitForTimeout(100);
   const kill=structuredClone(fixtures.infantry);
   kill.player.tick++;kill.player.scene.time+=.05;kill.player.scene.tick=kill.player.tick;
   kill.player.self.kills=1;
   const victim=kill.player.tactical.robots.find(r=>r.id==='blue-3');victim.alive=false;victim.hp=0;
   kill.player.public_events=[{id:'screenshot-kill',type:'destroy',tick:kill.player.tick,time:kill.player.scene.time,robot:'blue-3',killer:kill.player.self.id,team:'red'}];
   await page.evaluate(s=>window.__deliver(s),kill);
   await page.getByLabel('红方3号摧毁蓝方3号',{exact:true}).waitFor();
   await page.locator('[data-kill-announcement="first-blood"]').waitFor();
   await page.screenshot({path:resolve(output,`kill-announcement-${width}-${scale}.png`)});
   // Exercise live interpolation, including duplicate snapshots from the bridge timer.
   await page.evaluate(async seed=>{
    window.__motionStats=[];
    for(let tick=1;tick<=30;tick++){
     const s=structuredClone(seed),p=s.player,own=p.scene.robots.find(r=>r.id===p.self.id);
     p.tick+=tick;p.scene.tick=p.tick;p.scene.time+=tick*.05;
     const yaw=tick*.012,fold=Math.max(0,Math.min(1,(tick-8)/15));
     for(const r of [p.self,own]){r.x+=tick*.025;r.aim=yaw;r.pitch=.04;r.transform_progress=fold;r.muzzle_height_m=.4+(.145-.4)*fold;}
     window.__deliver(s);await new Promise(resolve=>setTimeout(resolve,50));
    }
   },fixtures.infantry4);
   const motion=await page.evaluate(()=>window.__motionStats);
   assert(motion.length>0,'no player timeline diagnostics');
   assert(motion.every(s=>s.buffer_ms===50&&s.samples<=12),'player buffer exceeds its limits');
   assert(motion.every((s,i)=>!i||s.render_time>=motion[i-1].render_time),'render clock moved backwards');
   await page.screenshot({path:resolve(output,`folding-motion-${width}-${scale}.png`)});
   await page.evaluate(s=>window.__deliver(s),fixtures.infantry);
   await page.getByRole('button',{name:'全屏操控',exact:true}).click();
   await page.locator('.viewport.page-fullscreen').waitFor();
   const fullscreenSurface=await page.locator('.operator-surface').boundingBox();
   assert(Math.abs(fullscreenSurface.x+fullscreenSurface.width/2-width/2)<1,'fullscreen camera is off center');
   await page.screenshot({path:resolve(output,`fullscreen-${width}-${scale}.png`)});
   await page.getByRole('button',{name:'退出全屏',exact:true}).click();
   await page.keyboard.down('Tab');await page.locator('.tactical-expanded').waitFor();
   const tacticalCards=await page.locator('.control-strip').boundingBox(),tacticalMap=await page.locator('.tactical-map').boundingBox();
   assert(tacticalCards.y+tacticalCards.height<tacticalMap.y,'expanded controls overlap tactical map');
   await page.locator('.own-state .own-details').waitFor();
   await page.screenshot({path:resolve(output,`tactical-${width}-${scale}.png`)});await page.keyboard.up('Tab');
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
