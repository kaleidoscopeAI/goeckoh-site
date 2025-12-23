      document.addEventListener("DOMContentLoaded", () => {
        const phraseLabels = {{ phrase_labels | tojson }};
        const phraseRates = {{ phrase_rates | tojson }};
        const timelineLabels = {{ timeline_labels | tojson }};
        const timelineRates = {{ timeline_rates | tojson }};

        const phraseCtx = document.getElementById("phraseChart");
        if (phraseCtx && phraseLabels.length) {
          new Chart(phraseCtx, {
            type: "bar",
            data: {
              labels: phraseLabels,
              datasets: [{
                label: "Correction rate (%)",
                data: phraseRates,
                borderWidth: 1
              }]
            },
            options: {
              responsive: true,
              maintainAspectRatio: false,
              plugins: {
                legend: {
                  labels: {
                    font: { size: 10 }
                  }
                }
              },
              scales: {
                x: {
                  ticks: {
                    maxRotation: 45,
                    minRotation: 0,
                    autoSkip: true,
                    font: { size: 9 }
                  },
                  grid: { display: false }
                },
                y: {
                  min: 0,
                  max: 100,
                  ticks: {
                    stepSize: 20,
                    font: { size: 9 },
                    callback: (value) => value + "%"
                  },
                  grid: {
                    borderDash: [4, 4]
                  }
                }
              }
            }
          });
        }

        const timelineCtx = document.getElementById("timelineChart");
        if (timelineCtx && timelineLabels.length) {
          new Chart(timelineCtx, {
            type: "line",
            data: {
              labels: timelineLabels,
              datasets: [{
                label: "Daily correction rate (%)",
                data: timelineRates,
                tension: 0.35,
                pointRadius: 2.5,
                borderWidth: 2
              }]
            },
            options: {
              responsive: true,
              maintainAspectRatio: false,
              plugins: {
                legend: {
                  labels: { font: { size: 10 } }
                }
              },
              scales: {
                x: {
                  ticks: { font: { size: 9 } },
                  grid: { display: false }
                },
                y: {
                  min: 0,
                  max: 100,
                  ticks: {
                    stepSize: 20,
                    font: { size: 9 },
                    callback: (value) => value + "%"
                  },
                  grid: { borderDash: [4, 4] }
                }
              }
            }
          });
        }

        // Phrase filter
        const filterInput = document.getElementById("phrase-filter");
        if (filterInput) {
          filterInput.addEventListener("input", (e) => {
            const q = e.target.value.toLowerCase();
            document.querySelectorAll(".phrase-row").forEach((row) => {
              const cell = row.querySelector("[data-phrase]");
              const text = (cell?.dataset.phrase || "").toLowerCase();
              row.style.display = text.includes(q) ? "" : "none";
            });
          });
        }

        // Live session updater: poll metrics API for most recent row
        async function refreshLive() {
          try {
            const res = await fetch("/api/metrics");
            if (!res.ok) return;
            const data = await res.json();
            if (!Array.isArray(data) || data.length === 0) return;

            const last = data[data.length - 1];
            const phrase = last.phrase_text || last.phrase_id || "—";
            const raw = last.raw_text || "—";
            const corrected = last.corrected_text || "—";

            const needsCorrection =
              last.needs_correction === "1" ||
              last.needs_correction === 1 ||
              last.needs_correction === true;

            const phraseEl = document.getElementById("live-phrase");
            const rawEl = document.getElementById("live-raw");
            const correctedEl = document.getElementById("live-corrected");
            const pill = document.getElementById("live-status-pill");

            if (phraseEl) phraseEl.textContent = phrase;
            if (rawEl) rawEl.textContent = raw;
            if (correctedEl) correctedEl.textContent = corrected;
            if (pill) {
              pill.querySelector("span:nth-child(2)").textContent =
                needsCorrection ? "Correction suggested" : "Sounded great";
            }
          } catch (err) {
            // fail silently in UI
          }
        }

        refreshLive();
        setInterval(refreshLive, 5000);
      });
