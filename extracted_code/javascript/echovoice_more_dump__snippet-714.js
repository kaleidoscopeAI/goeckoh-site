function estimate_rho(Jv, D, opts) -> {rho, eigenvector, iters}
function estimate_sensitivities(engines, Jv, opts) -> map engine -> s_E
function autotune_gains(gains, sensitivities, rho, rho_target, opts) -> gains_new
