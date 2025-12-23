pub const fn stable_contraction<const N: usize>(s: StateVector<N>, w: [[f64; N]; N])
    requires
        symmetric(w),
        trace_lt_1(w),
    ensures
        contractive(s, w),
{ }
