const t = getLastFrameTime();
if (performance.now() - t > 200) {
  console.warn('Watchdog detected stall, restarting worker');
  restartWorker();
}
