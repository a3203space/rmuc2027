import {test,expect} from 'bun:test';
import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {AnnouncerAudio} from '../../apps/client/web/arena/announcerAudio';
import {KILL_CUES,type KillAnnouncement,type KillCue} from '../../apps/client/web/arena/killAnnouncements';

const announcement=(cue:KillCue,killer='red-3'):KillAnnouncement=>({key:cue,cue,killer,victim:'blue-3',team:'red',at:1,count:2});
function fixture(){
 const nodes:any[]=[],audio=new AnnouncerAudio();
 const gain=()=>({gain:{value:0,cancelScheduledValues(){},setValueAtTime(){},setTargetAtTime(){}},connect(){return this;},disconnect(){}});
 const context:any={state:'running',currentTime:1,createGain:gain,createBufferSource(){const node:any={connect(){return this;},disconnect(){},start(){node.started=true;},stop(){node.stopped=true;node.onended?.();}};nodes.push(node);return node;},close(){return Promise.resolve();}};
 Object.assign(audio,{context,bus:gain()});audio.setVolume(.5);
 for(const cue of Object.keys(KILL_CUES)) (audio as any).buffers.set(cue,{duration:3});
 return {audio,context,nodes};
}
test('all six voice files retain source hashes',()=>{
 const records=JSON.parse(readFileSync('docs/legal/arena-audio.json','utf8')).filter((r:any)=>r.path.includes('/announcer/'));
 expect(records).toHaveLength(6);
 for(const record of records)expect(createHash('sha256').update(readFileSync(record.path)).digest('hex')).toBe(record.sha256);
});
test('original gains, serial voice, pending multi-kill upgrade and separate ACE',()=>{
 const {audio,nodes}=fixture(),shown:string[]=[];
 audio.announce(announcement('first-blood'),()=>shown.push('first'));
 audio.announce(announcement('double-kill'),()=>shown.push('double'));
 audio.announce(announcement('triple-kill'),()=>shown.push('triple'));
 audio.announce(announcement('aced'),()=>shown.push('ace'));
 expect(audio.diagnostics.starts).toBe(1);
 expect(audio.diagnostics.queued).toEqual(['triple-kill','aced']);
 expect(shown).toEqual(['first']);
 nodes[0].onended();expect(audio.diagnostics.starts).toBe(2);expect(shown).toEqual(['first','triple']);
 expect((audio as any).voice.gain.gain.value).toBe(KILL_CUES['triple-kill'].gain);
 nodes[1].onended();expect(shown).toEqual(['first','triple','ace']);audio.dispose();
});
test('mute keeps visual announcements and cancels queued or playing audio',()=>{
 const {audio,nodes}=fixture();let shown=0;
 audio.announce(announcement('first-blood'),()=>shown++);
 audio.announce(announcement('double-kill'),()=>shown++);
 audio.setVolume(0);
 expect(nodes[0].stopped).toBe(true);expect(audio.diagnostics.queued).toEqual([]);
 audio.announce(announcement('triple-kill'),seconds=>{expect(seconds).toBeGreaterThanOrEqual(3.4);shown++;});
 expect(shown).toBe(2);expect(audio.diagnostics.starts).toBe(1);audio.dispose();
});
test('hidden or disconnected surface cancels audio; old queued announcements expire',()=>{
 const {audio,context,nodes}=fixture();let shown=0;
 audio.announce(announcement('first-blood'),()=>shown++);
 audio.announce(announcement('aced'),()=>shown++);
 context.currentTime=10;nodes[0].onended();expect(shown).toBe(1);expect(audio.diagnostics.queued).toEqual([]);
 audio.announce(announcement('double-kill'),()=>shown++);audio.setActive(false);
 expect(nodes[1].stopped).toBe(true);
 audio.announce(announcement('aced'),()=>shown++);expect(shown).toBe(2);audio.dispose();
});
test('failed audio fetch still starts the original visual highlight',async()=>{
 const {audio}=fixture();(audio as any).buffers.clear();
 const previousFetch=globalThis.fetch,previousWarn=console.warn;let shown=0;
 globalThis.fetch=async()=>new Response('',{status:404});console.warn=()=>{};
 try{audio.announce(announcement('first-blood'),()=>shown++);await new Promise(resolve=>setTimeout(resolve,0));expect(shown).toBe(1);expect(audio.diagnostics.error).toContain('Announcer unavailable');expect(audio.diagnostics.queued).toEqual([]);}
 finally{globalThis.fetch=previousFetch;console.warn=previousWarn;audio.dispose();}
});
