275        {/* Chart Section */}
276 -      <Card title="Coherence, drift, and latency over time">
277 -        <p className="text-sm text-slate-500 mb-6">Live GCL (green), dr
     ift (orange), and latency (indigo) from mirror telemetry.</p>
276 +      <Card>
277 +        <div className="flex items-start justify-between gap-4 mb-4">
278 +          <div>
279 +            <h3 className="text-xl font-semibold text-brand-black">Cohe
     rence, drift, and latency</h3>
280 +            <p className="text-sm text-slate-500">Live GCL (green), dri
     ft (orange), and latency (indigo) from mirror telemetry.</p>
281 +          </div>
282 +          <button
283 +            onClick={downloadTelemetryCsv}
284 +            disabled={!telemetryAll.length}
285 +            className="px-3 py-2 text-sm rounded-lg border border-slate
     -200 text-slate-700 hover:bg-slate-50 disabled:opacity-50 disabled:curs
     or-not-allowed transition-colors"
286 +          >
287 +            Download CSV
288 +          </button>
289 +        </div>
290          <div className="h-[300px] w-full">

