const queue: any[] = [];
let active = 0;
const MAX_CONCURRENT = 1;
const DEFAULT_WAIT = 250;

export async function rateLimitedFetch(url: string, opts: any = {}, { retries = 1, backoff = 200 } = {}) {
  return new Promise((resolve, reject) => {
    queue.push({ url, opts, retries, backoff, resolve, reject });
    processQueue();
  });
}

async function processQueue() {
  if (active >= MAX_CONCURRENT || queue.length === 0) return;
  const task = queue.shift()!;
  active++;
  try {
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
  } finally {
    active--;
    setTimeout(processQueue, DEFAULT_WAIT);
  }
}
