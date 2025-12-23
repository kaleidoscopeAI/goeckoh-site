let attempt = 0;
while (attempt <= task.retries) {
  try {
    const r = await fetch(task.url, task.opts);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    task.resolve(r);
    break;
  } catch (err) {
    attempt++;
    if (attempt > task.retries) {
      task.reject(err);
      break;
    }
    await new Promise(res => setTimeout(res, task.backoff * attempt));
  }
}
