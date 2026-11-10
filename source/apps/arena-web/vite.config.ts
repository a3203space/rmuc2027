import {defineConfig} from 'vite';
import react from '@vitejs/plugin-react';
import path from 'node:path';
export default defineConfig({
 root:path.resolve('apps/arena-web'),plugins:[react()],publicDir:path.resolve('assets/web'),
 build:{target:'chrome87',cssTarget:'chrome87',outDir:path.resolve('build/arena-web'),emptyOutDir:true},
 server:{host:'127.0.0.1',port:5174,proxy:{'/api':{target:'http://127.0.0.1:8767',ws:true}}},
});
