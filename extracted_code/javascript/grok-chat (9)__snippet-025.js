119    const telemetryRows = (mm.metrics?.telemetry || []).slice(-8).reverse
     ();
120 +  const telemetryAll = mm.metrics?.telemetry || [];
121 +
122 +  const downloadTelemetryCsv = () => {
123 +    if (!telemetryAll.length) return;
124 +    const header = [
125 +      "ts",
126 +      "gcl",
127 +      "drift",
128 +      "latency_ms",
129 +      "dur_s",
130 +      "rms",
131 +      "frag_score",
132 +      "harvested",
133 +      "best_ref",
134 +      "path",
135 +    ];
136 +    const rows = telemetryAll.map((t: any) =>
137 +      header
138 +        .map((k) => {
139 +          const v = t[k];
140 +          if (v === undefined || v === null) return "";
141 +          const s = String(v);
142 +          return s.includes(",") ? `"${s.replace(/"/g, '""')}"` : s;
143 +        })
144 +        .join(",")
145 +    );
146 +    const csv = [header.join(","), ...rows].join("\n");
147 +    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
148 +    const url = URL.createObjectURL(blob);
149 +    const a = document.createElement("a");
150 +    a.href = url;
151 +    a.download = "mirror_telemetry.csv";
152 +    a.click();
153 +    URL.revokeObjectURL(url);
154 +  };
155

