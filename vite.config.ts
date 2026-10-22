import {defineConfig} from 'vite';
import react from '@vitejs/plugin-react';
import path from 'node:path';
export default defineConfig({plugins:[react()],publicDir:'assets/web',build:{outDir:'build/web',emptyOutDir:true,rollupOptions:{input:{player:path.resolve('apps/client/web/index.html'),replay:path.resolve('apps/replay-viewer/index.html')}}}});
