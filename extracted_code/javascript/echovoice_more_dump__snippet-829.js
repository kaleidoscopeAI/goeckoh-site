function F_apply(i, Ni): NodeState — forward model output.
function JF_apply_dot_v(i, v_i): NodeState — returns (∂F/∂N_i)·v_i. Provide analytic backprop or autodiff.
If F is an MLP, store its weights and implement JF_dot_v using linearized pass (backprop of linear output w.r.t. inputs times v), or use autodiff libs.
