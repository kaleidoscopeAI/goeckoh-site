void compute_geometric_derivative(const double *pos_i, const double *pos_j, double scalar_force, double *out_derivative) {
    for (int k = 0; k < 3; k++) {
        // Simple geometric force component proportional to the scalar force and displacement
        out_derivative[k] = scalar_force * (pos_j[k] - pos_i[k]);
    }
