// kaleidoscope_core.c
// C Backend for high-performance quantum/geometric calculations (Density Matrices, Purity, Complex Operations).
// Must be compiled into a shared library (e.g., .so or .dll) for use by uni_core.py (via ctypes).

#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>

// --- Data Structures ---

// Represents a 1D array of complex numbers (e.g., a quantum state vector)
typedef struct {
    int n; // Size of the array
    double complex *data;
} ComplexArray;

// --- Utility Functions ---

// Allocate complex array
ComplexArray* create_complex_array(int n) {
    ComplexArray *arr = (ComplexArray*) malloc(sizeof(ComplexArray));
    if (!arr) return NULL;
    arr->n = n;
    // calloc initializes the memory to zero
    arr->data = (double complex*) calloc(n, sizeof(double complex)); 
    if (!arr->data) {
        free(arr);
        return NULL;
    }
    return arr;
}

// Free complex array
void free_complex_array(ComplexArray* arr) {
    if (arr) {
        free(arr->data);
        free(arr);
    }
}

// --- Core Geometric/Quantum Operations ---

// Element-wise addition: out = a + b
void complex_array_add(ComplexArray* a, ComplexArray* b, ComplexArray* out) {
    if (a->n != b->n || a->n != out->n) return; // Size check
    for(int i = 0; i < a->n; i++) {
        out->data[i] = a->data[i] + b->data[i];
    }
}

// Element-wise multiplication: out = a * b
void complex_array_mul(ComplexArray* a, ComplexArray* b, ComplexArray* out) {
    if (a->n != b->n || a->n != out->n) return;
    for(int i = 0; i < a->n; i++) {
        out->data[i] = a->data[i] * b->data[i];
    }
}

// Compute Purity: Tr(rho^2)
// Assumes 'rho' is a flattened, square density matrix (size N*N, stored in ComplexArray of size N*N)
double compute_purity(ComplexArray* rho, int N) {
    if (rho->n != N * N) {
        fprintf(stderr, "Error: Density matrix size mismatch for purity calculation.\n");
        return 0.0;
    }

    double purity = 0.0;
    // Purity = Sum_{i,j} rho_{ij} * rho_{ji}
    // We compute Tr(rho * rho), where rho is N x N
    // This is mathematically equivalent to Sum_k rho_kk where rho = rho * rho.
    // However, a common simplification for Tr(rho^2) in a flattened array:
    // Purity = Sum_{i=0}^{N*N-1} |rho_i|^2 if rho is the vectorization of rho.
    // For the actual matrix product Tr(rho*rho), we calculate:
    
    // We will use the definition: Tr(A*B) = Sum_{i,j} A_{ij} * B_{ji}
    // Since rho is Hermitian, we calculate Sum_{i,j} rho_{ij} * rho_{ji} (Tr(rho^2))
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            // rho_ij is at index (i*N + j)
            double complex rho_ij = rho->data[i * N + j];
            // rho_ji is at index (j*N + i)
            double complex rho_ji = rho->data[j * N + i];
            purity += creal(rho_ij * rho_ji);
        }
    }
    return purity;
}

// Compute the Hamilton-Jacobi (H-J) derivative for node i
// This function calculates a term related to the rate of change of the geometric state.
// Input: pos_i (3D position vector), pos_j (3D position vector), scalar_force
// Output: 3D vector (double[3])
void compute_geometric_derivative(const double *pos_i, const double *pos_j, double scalar_force, double *out_derivative) {
    for (int k = 0; k < 3; k++) {
        // Simple geometric force component proportional to the scalar force and displacement
        out_derivative[k] = scalar_force * (pos_j[k] - pos_i[k]);
    }
}

// A simple function to test the C library integration
double test_c_call(double a, double b) {
    return a * a + b;
}

