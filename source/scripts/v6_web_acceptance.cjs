#!/usr/bin/env node
/* Real Arena UI acceptance. --start is required to create the one short match.
 * The backend must already be running inside the authorized experiment window.
 * No mocks, training or simulator construction occurs in this script.
 */
const fs=require('node:fs');
const path=require('node:path');
const crypto=require('node:crypto');

function argumentsOf(argv){
 const args={url:'http://127.0.0.1:8767',seconds:20,start:false};
 for(let i=0;i<argv.length;i++){
  const key=argv[i];if(key==='--start'){args.start=true;continue;}if(key==='--help'){args.help=true;continue;}
  const names={'--url':'url','--run':'run','--deadline-unix':'deadline','--seconds':'seconds','--playwright-module':'playwrightModule','--chrome':'chrome','--checkpoint-id':'checkpoint'};
  if(!names[key]||argv[i+1]===undefined)throw Error('Unknown or incomplete argument: '+key);
  args[names[key]]=argv[++i];
 }
 return args;
}
function atomicJson(filename,value){const pending=filename+'.pending';fs.writeFileSync(pending,JSON.stringify(value,null,2)+'\n');fs.renameSync(pending,filename);}
function assert(condition,message){if(!condition)throw Error(message);}
async function fileHash(filename,remaining){const digest=crypto.createHash('sha256');for await(const chunk of fs.createReadStream(filename)){remaining();digest.update(chunk);}return digest.digest('hex');}
async function main(argv=process.argv.slice(2)){
 const args=argumentsOf(argv);
 if(args.help){console.log('node scripts/v6_web_acceptance.cjs --run RUN_DIRECTORY --deadline-unix UTC_UNIX [--start] [--url http://127.0.0.1:8767] [--seconds 20] [--checkpoint-id IMMUTABLE_ID] [--playwright-module PATH] [--chrome PATH]\nWithout --start: read-only catalog/UI inspection; writes web-inspection.json, never web-acceptance.json.\nWith --start: requires campaign.status=web_acceptance, performs one short frozen Rapier match plus a saved replay.');return;}
 const url=new URL(args.url);assert(['127.0.0.1','localhost'].includes(url.hostname),'Only the local Arena service is accepted');
 assert(args.run&&Number.isFinite(Number(args.deadline)),'--run and --deadline-unix are required');
 const root=path.resolve(args.run),manifest=JSON.parse(fs.readFileSync(path.join(root,'run-manifest.json'),'utf8'));
 assert(manifest.format==='rmuc-research-run-v1','Unsupported run manifest');
 let cutoff=Number(args.deadline);
 if(args.start){
  const campaign=JSON.parse(fs.readFileSync(path.join(root,'campaign.json'),'utf8'));
  assert(campaign.status==='web_acceptance','Live UI acceptance is allowed only in the campaign web_acceptance phase');
  const request=JSON.parse(fs.readFileSync(path.join(root,'web-acceptance-request.json'),'utf8'));
  assert(request.run_id===manifest.run_id,'Acceptance request belongs to another run');
  cutoff=Math.min(cutoff,Number(request.deadline_unix));
 }
 // Reserve ten seconds for stopping our own match, closing Chrome and reporting.
 const remaining=()=>{const left=(cutoff-Date.now()/1000-10)*1000;if(left<=0)throw Error('UI acceptance action deadline reached');return left;};
 remaining();
 const seconds=Number(args.seconds);assert(Number.isInteger(seconds)&&seconds>=1&&seconds<=60,'Short UI acceptance duration must be 1..60 seconds');
 const {chromium}=require(args.playwrightModule||process.env.PLAYWRIGHT_MODULE||'playwright-core');
 const output=path.join(root,args.start?'web-acceptance.json':'web-inspection.json'),shots=path.join(root,'web-evidence');fs.mkdirSync(shots,{recursive:true});
 const result={status:'running',started_utc:new Date().toISOString(),deadline_unix:cutoff,run_id:manifest.run_id,url:url.origin,api_stubbed:false,live_requested:args.start,gradient_updates:0,checks:{},errors:[],page_errors:[],console_errors:[]};
 let server,browser,page,ownedMatch=null;
 const watchdog=setTimeout(()=>{result.action_deadline_reached=true;if(page)void page.close().catch(()=>{});},remaining());
 async function get(endpoint){remaining();const response=await page.request.get(url.origin+endpoint,{timeout:Math.min(15000,remaining())});assert(response.ok(),endpoint+' returned '+response.status());return response.json();}
 const diagnose=()=>page.evaluate(()=>window.arenaDiagnostics?.());
 try{
  server=await chromium.launchServer({executablePath:args.chrome||'/usr/bin/google-chrome',headless:true,timeout:Math.min(30000,remaining()),args:['--no-sandbox','--disable-dev-shm-usage','--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader']});
  atomicJson(path.join(root,'web-browser-process.json'),{pid:server.process().pid,owner:'v6_web_acceptance',deadline_unix:cutoff,status:'running'});
  browser=await chromium.connect(server.wsEndpoint(),{timeout:Math.min(10000,remaining())});
  const context=await browser.newContext({viewport:{width:1440,height:1000}});page=await context.newPage();
  page.setDefaultTimeout(Math.min(20000,remaining()));
  page.on('pageerror',error=>result.page_errors.push(String(error)));
  page.on('console',message=>{if(message.type()==='error')result.console_errors.push(message.text());});
  const catalog=await get('/api/research/runs');
  const run=catalog.runs.find(row=>row.run_id===manifest.run_id);assert(run,'Registered run is absent from the web catalog');
  result.catalog={run_id:run.run_id,manifest_sha256:run.manifest_sha256,checkpoint_ids:run.checkpoints.map(row=>row.id),scenario_ids:run.scenarios.map(row=>row.id)};
  const checkpoint=run.checkpoints.find(row=>row.id===args.checkpoint)||(args.checkpoint?null:run.checkpoints.find(row=>row.label==='selected')||run.checkpoints.find(row=>row.label==='baseline'));
  assert(checkpoint,'Requested immutable checkpoint is unavailable');
  const scenario=run.scenarios.find(row=>row.learner_side==='red')||run.scenarios[0];assert(scenario,'No registered scenario');
  const profile=run.capability_profiles.find(row=>row.learner_side===scenario.learner_side);assert(profile,'No matching capability profile');
  await page.goto(url.origin,{waitUntil:'domcontentloaded',timeout:Math.min(30000,remaining())});
  await page.getByRole('button',{name:'研究',exact:true}).click();
  await page.getByLabel('研究运行').selectOption(run.run_id);
  await page.getByLabel('冻结权重').selectOption(checkpoint.id);
  await page.getByLabel('阵容与主哨兵所在方').selectOption(scenario.id);
  await page.getByLabel('整局能力配置').selectOption(profile.id);
  await page.getByLabel('物理后端').selectOption('rapier');
  await page.getByLabel('随机种子').fill('81001');
  await page.getByLabel('观测时长（秒）').fill(String(seconds));
  await page.screenshot({path:path.join(shots,'catalog.png'),fullPage:true,timeout:Math.min(15000,remaining())});
  result.checks.catalog_ui=true;
  if(args.start){
   assert(remaining()>30000,'Insufficient remaining web budget to start a new frozen match');
   const leaderboardBefore=await get('/api/leaderboard');
   const prior=await get('/api/state');assert(!prior.busy,'A web match is already running; this verifier will not stop it');
   const responsePromise=page.waitForResponse(response=>new URL(response.url()).pathname==='/api/start'&&response.request().method()==='POST',{timeout:Math.min(60000,remaining())});
   await page.getByRole('button',{name:'启动冻结对局',exact:true}).click();
   const response=await responsePromise;assert(response.ok(),'Frozen start failed: '+await response.text());
   const started=await response.json();ownedMatch=started.match_id;assert(ownedMatch,'Missing started match ID');
   result.start_request=JSON.parse(response.request().postData());
   let state;let maximumTick=0;
   while(true){
    state=await get('/api/state');assert(state.match_id===ownedMatch,'Another match replaced the owned frozen match');
    assert(!state.error,'Frozen match failed: '+state.error);maximumTick=Math.max(maximumTick,state.frame.tick);
    if(!state.busy)break;
    await page.waitForTimeout(Math.min(500,remaining()));
   }
   const record=await get('/api/matches/'+encodeURIComponent(ownedMatch));
   assert(record.has_replay,'The frozen UI match did not save its full replay');
   assert(record.metadata.test_match===true,'Research match was not excluded from Elo');
   assert(record.metadata.research?.checkpoint_sha256===checkpoint.sha256,'Displayed checkpoint hash differs from selected immutable model');
   assert(record.metadata.research?.scenario_id===scenario.id,'Frozen scenario differs from the chosen registry entry');
   assert(record.metadata.research?.capability_profile_id===profile.id,'Frozen capabilities differ from the chosen profile');
   assert(record.metadata.research?.capability_sha256===profile.sha256,'Frozen capability SHA differs from the registered profile');
   for(const [role,helper] of Object.entries(run.helpers))assert(record.metadata.research?.helpers?.[role]===helper.sha256,'Frozen helper hash differs for '+role);
   if(manifest.radar_classifier)assert(record.metadata.rl?.radar_classifier_sha256===manifest.radar_classifier.sha256,'Frozen radar classifier hash differs');
   assert(JSON.stringify(await get('/api/leaderboard'))===JSON.stringify(leaderboardBefore),'Research match changed Elo standings');
   assert(record.metadata.physics_backend==='rapier','The default frozen UI match did not use Rapier');
   assert(record.metadata.recording?.horizon_completed===true,'Frozen UI match did not complete its requested short horizon');
   await page.waitForFunction(tick=>window.arenaDiagnostics?.().mode==='live'&&window.arenaDiagnostics().tick>=tick,maximumTick,{timeout:Math.min(15000,remaining())});
   const live=await diagnose();assert(live.robots===10&&live.connected,'The actual live renderer did not receive the canonical ten-robot field');
   assert(await page.locator('#viewport canvas').count()>0,'The original Arena canvas is absent');
   await page.screenshot({path:path.join(shots,'frozen-live.png'),fullPage:true,timeout:Math.min(15000,remaining())});
   result.live={match_id:ownedMatch,maximum_tick:maximumTick,simulation_seconds:state.frame.time,metadata:record.metadata,result:record.result,diagnostics:live};
   // Preserve the actual canonical FrameLog inside the run before publishing the
   // acceptance marker; the service journal itself lives outside the run bundle.
   const liveReplay=await page.request.get(url.origin+'/api/replays/'+encodeURIComponent(ownedMatch),{timeout:Math.min(30000,remaining())});
   assert(liveReplay.ok(),'Could not archive the owned frozen match replay');
   const liveReplayBytes=await liveReplay.body();remaining();
   const liveReplayPath=path.join(shots,'frozen-live-replay.json');
   fs.writeFileSync(liveReplayPath+'.pending',liveReplayBytes);fs.renameSync(liveReplayPath+'.pending',liveReplayPath);
   result.live.replay={path:path.relative(root,liveReplayPath).split(path.sep).join('/'),bytes:liveReplayBytes.length,sha256:crypto.createHash('sha256').update(liveReplayBytes).digest('hex'),format:'canonical-full-frames',source:'owned-frozen-ui-match'};
   await liveReplay.dispose();
   result.checks.frozen_live_match=true;result.checks.test_match_no_elo=true;
   // A saved training/evaluation replay from the registered run is required.
   // Do not substitute the short web match we just created.
   const rows=await get('/api/research/runs/'+encodeURIComponent(run.run_id)+'/replays');
   const candidates=rows.filter(row=>row.seconds>.1&&['training','evaluation'].includes(row.kind));
   candidates.sort((a,b)=>(a.kind==='training'?0:1)-(b.kind==='training'?0:1)||a.bytes-b.bytes);
   const replay=candidates[0];assert(replay,'No registered saved training/evaluation replay has advancing frames');
   const descriptor=manifest.replays[replay.id];assert(descriptor,'Selected saved replay is missing from the frozen run manifest');
   const filename=path.resolve(root,descriptor.path);assert(filename.startsWith(root+path.sep),'Replay path escapes the registered run');
   assert(await fileHash(filename,remaining)===replay.sha256,'Saved replay SHA256 changed before playback');
   await page.getByRole('button',{name:'研究',exact:true}).click();
   await page.getByLabel('研究运行').selectOption(run.run_id);
   const row=page.getByRole('row').filter({has:page.getByText(replay.label,{exact:true})});
   await row.getByRole('button',{name:'观看',exact:true}).click();
   await page.waitForFunction(()=>window.arenaDiagnostics?.().mode==='replay',{timeout:Math.min(60000,remaining())});
   const timeline=page.locator('#timeline');await timeline.focus();await timeline.press('Home');await timeline.press('ArrowRight');
   await page.waitForFunction(()=>window.arenaDiagnostics?.().mode==='replay'&&window.arenaDiagnostics().tick>0,{timeout:Math.min(20000,remaining())});
   const playback=await diagnose();assert(playback.robots===10,'Saved replay did not render the canonical ten-robot field');
   await page.screenshot({path:path.join(shots,'saved-replay.png'),fullPage:true,timeout:Math.min(15000,remaining())});
   result.replay={...replay,hash_verified:true,diagnostics:playback};result.checks.saved_training_or_evaluation_replay=true;
   assert(result.page_errors.length===0,'Browser page exceptions occurred');
   result.status='passed';
  }else result.status='inspected_only';
 }catch(error){result.status='partial';result.errors.push(error.stack||String(error));}
 finally{
  clearTimeout(watchdog);
  if(ownedMatch&&page){try{const response=await page.request.get(url.origin+'/api/state',{timeout:1500});const state=await response.json();if(state.match_id===ownedMatch&&state.busy)await page.request.post(url.origin+'/api/control',{data:{command:'stop'},timeout:1500});}catch(error){result.cleanup_error=String(error);}}
  if(browser)await browser.close().catch(error=>{result.browser_close_error=String(error);});
  if(server){await server.close().catch(error=>{result.browser_server_close_error=String(error);});atomicJson(path.join(root,'web-browser-process.json'),{pid:server.process().pid,owner:'v6_web_acceptance',deadline_unix:cutoff,status:'closed'});}
  result.finished_utc=new Date().toISOString();result.within_web_deadline=Date.now()/1000<=cutoff;
  if(!result.within_web_deadline&&result.status==='passed')result.status='partial';
  atomicJson(output,result);console.log(JSON.stringify({status:result.status,path:output,checks:result.checks,errors:result.errors}));
 }
 if(args.start&&result.status!=='passed')process.exitCode=1;
}
if(require.main===module)main().catch(error=>{console.error(error.stack||String(error));process.exitCode=1;});
module.exports={argumentsOf};
