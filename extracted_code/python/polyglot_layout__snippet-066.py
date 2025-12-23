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

