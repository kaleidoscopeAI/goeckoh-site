# quick demo: start job and let local controller propose
job = start_hybrid_job_from_target([0.4, 0.6], policy={"mix":0.0, "gamma":0.4})
print("Started job:", job)
# step a bit
print("Step:", step_job(job, dt=0.01, steps=20))
# create an LLMBridge pointing to local ollama (optional)
try:
    llm = LLMBridge(base_url="http://localhost:11434", model="mistral")
except Exception:
    llm = None
# if LLM available, call propose_and_apply_from_llm
if llm:
    res = propose_and_apply_from_llm(job, llm)
    print("LLM propose result:", res)
else:
    print("LLM not available; apply a manual small action")
    r = apply_action_safe(job, {"mix_delta": 0.05, "gamma_delta": 0.01}, sandbox_steps=8)
    print("Manual apply result:", r)
