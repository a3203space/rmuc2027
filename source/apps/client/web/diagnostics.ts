// Loaded before the renderer so failed module imports also produce a visible error.
function report(message: string) {
  console.error('ARENA_VIEWPORT:', message);
  let panel = document.getElementById('viewport-error');
  if (!panel) {
    panel = document.createElement('pre');
    panel.id = 'viewport-error';
    panel.style.cssText = 'position:fixed;inset:12px;z-index:99999;padding:20px;overflow:auto;white-space:pre-wrap;background:#101f2a;color:#ffceba;font:15px monospace';
    document.body.appendChild(panel);
  }
  panel.textContent = '仿真画面加载失败\n' + message + '\n请将此信息与终端 ARENA_WEB 日志一同反馈。';
}
window.addEventListener('error', event => {
  if (event.message) report(event.message);
});
window.addEventListener('unhandledrejection', event => report(String(event.reason?.message || event.reason)));
void import('./lan').catch(error => report(String(error?.message || error)));
