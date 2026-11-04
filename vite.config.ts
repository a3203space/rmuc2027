import {defineConfig} from 'vite';
import react from '@vitejs/plugin-react';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {readFileSync,readdirSync} from 'node:fs';
import {webCatalog} from './scripts/webCatalog';
const hash=createHash('sha256');
function fingerprint(root:string){for(const name of readdirSync(root,{withFileTypes:true}).sort((a,b)=>a.name<b.name?-1:a.name>b.name?1:0)){const file=path.join(root,name.name);if(name.isDirectory())fingerprint(file);else if(/\.(tsx?|css|html)$/.test(name.name))hash.update(file.split(path.sep).join('/')+'\n').update(readFileSync(file,'utf8').replace(/\r\n/g,'\n'));}}
fingerprint('apps/client/web');fingerprint('packages/renderer/src');
for(const file of ['package.json','bun.lock','configs/rules.json','vite.config.ts','scripts/webCatalog.ts'])hash.update(readFileSync(file,'utf8').replace(/\r\n/g,'\n'));
const resourceVersion=hash.digest('hex').slice(0,20);
const localCatalog=()=>JSON.stringify(webCatalog(JSON.parse(readFileSync('configs/rules.json','utf8'))));
// Ubuntu 22.04 ships Qt WebEngine 6.2 with Chromium 87.
export default defineConfig({plugins:[react(),{name:'arena-version',generateBundle(){
 this.emitFile({type:'asset',fileName:'version.json',source:JSON.stringify({protocol:2,resource_version:resourceVersion})});
 this.emitFile({type:'asset',fileName:'catalog.json',source:localCatalog()});
},configureServer(server){server.middlewares.use('/catalog.json',(_request,response)=>{response.setHeader('Content-Type','application/json');response.setHeader('Cache-Control','no-cache');response.end(localCatalog());});}}],publicDir:'assets/web',build:{target:'chrome87',cssTarget:'chrome87',outDir:'build/web',emptyOutDir:true,rollupOptions:{input:{player:path.resolve('apps/client/web/index.html'),replay:path.resolve('apps/replay-viewer/index.html')}}}});
