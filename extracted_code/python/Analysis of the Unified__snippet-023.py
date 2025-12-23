from hybrid_orchestrator import start_hybrid_job_from_target, propose_and_apply_from_llm
from llm_bridge import LLMBridge
jid = start_hybrid_job_from_target([0.3,0.7])
llm = LLMBridge(base_url="http://localhost:11434", model="mistral")
res = propose_and_apply_from_llm(jid, llm)
print(res)
