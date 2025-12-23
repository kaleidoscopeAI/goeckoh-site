class MathSolver:
    @staticmethod
    def solve_expr(q:str)->Tuple[bool,str]:
        try:
            # quick detect equation vs expression
            if "=" in q:
                left,right=q.split("=",1)
                expr_l=sympify(left); expr_r=sympify(right)
                sol=solve(Eq(expr_l,expr_r))
                return True, f"solutions: {sol}"
            expr=sympify(q)
            return True, f"{simplify(expr)}"
        except Exception as e:
            return False, f"math_error: {e}"

class LogicPlanner:
    @staticmethod
    def plan(prompt:str)->List[str]:
        # simple heuristic planner (no LLM): split goals, produce steps
        steps=[]
        s=prompt.strip()
        if any(k in s.lower() for k in ["prove","show","why","because"]):
            steps += ["Define terms precisely","List known axioms/facts","Transform goal into subgoals","Check counterexamples","Synthesize final 
