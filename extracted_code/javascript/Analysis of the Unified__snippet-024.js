const msg = e.data;
if (!msg?.cmd) return;
if (msg.cmd === 'hypothesis') {
  this.handleHypothesis(msg.hypothesis, msg.systemContext, msg.requestId);
} else if (msg.cmd === 'log') console.log('[worker]', msg.msg);
