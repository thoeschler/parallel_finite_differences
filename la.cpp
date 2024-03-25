#include "la.hpp"

#include <iostream>
#include <assert.h>
#include <cmath>

void matmul(CRSMatrix const&A, std::vector<double> const&b, std::vector<double> &out) {
    std::fill(out.begin(), out.end(), 0.0);
    std::size_t row = 0;
    std::size_t row_index = A.row_index(row + 1);
    for (std::size_t value_count = 0; value_count < A.size(); ++value_count) {
        if (value_count == row_index) {
            ++row;
            row_index = A.row_index(row + 1);
        }
        out[row] += A.value(value_count) * b[A.col_index(value_count)];
    }
}

double dot(std::vector<double> const& a, std::vector<double> const& b) {
    assert(a.size() == b.size());

    double result = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        result += a[i] * b[i];
    }

    return result;
}

void add(std::vector<double> &inout, std::vector<double> const&in) {
    assert(inout.size() == in.size());

    for (std::size_t i = 0; i < inout.size(); ++i) {
        inout[i] += in[i];
    }
}

void add_mult(std::vector<double> const&in1, std::vector<double> const&in2, std::vector<double> &out, double multiplier) {
    assert(in1.size() == in2.size());
    assert(in2.size() == out.size());

    for (std::size_t i = 0; i < out.size(); ++i) {
        out[i] = in1[i] + multiplier * in2[i];
    }
}

void add_mult_finout(std::vector<double> &inout, std::vector<double> const&in, double multiplier) {
    assert(inout.size() == in.size());

    for (std::size_t i = 0; i < inout.size(); ++i) {
        inout[i] += multiplier * in[i];
    }
}

void add_mult_sinout(std::vector<double> const&in, std::vector<double> &inout, double multiplier) {
    assert(inout.size() == in.size());

    for (std::size_t i = 0; i < inout.size(); ++i) {
        inout[i] = in[i] + multiplier * inout[i];
    }
}

void cg(CRSMatrix const&A, std::vector<double> const&b, std::vector<double> &u, const double tol) {
    bool converged = false;
    double alpha, gamma;
    std::size_t size = b.size();
    std::vector<double> p(size), r(size), Ap(size);

    double norm_b_squared = dot(b, b);
    double norm_r_old_squared, norm_r_squared;

    // initialize p (direction) and r (residual)
    matmul(A, u, Ap);
    add_mult(b, Ap, r, -1.0);
    p = r;
    norm_r_old_squared = dot(r, r);

    while (!converged) {
        matmul(A, p, Ap);
        alpha = dot(r, r) / dot(Ap, p);
        add_mult_finout(u, p, alpha);
        add_mult_finout(r, Ap, -alpha);
        norm_r_squared = dot(r, r);
        gamma = norm_r_squared / norm_r_old_squared;
        norm_r_old_squared = norm_r_squared;
        add_mult_sinout(r, p, gamma);

        converged = (norm_r_squared <= tol * tol * norm_b_squared);
    }
}

void parallel_cg() {
    // 0. compute p = r = b - Ax0
    // MPI_Allgather
    // before first step compute (r, r) (assumed to be know inside the loop)

    // ITERATION START

    // 1. compute Apk (locally) --> parallel matvec multiplication
    // global pk needed, local A (assemble locally), 

    // 2. compute ak = (rk, rk) / (Apk, pk)
    // (rk, rk) already known, compute (Apk, pk) locally, then MPI_Reduce

    // 3. xk+1 = xk + ak * pk
    // compute locally

    // 4. rk+1 = rk - ak Apk
    // compute locally

    // 5. gk = (rk+1, rk+1) / (rk, rk)
    // (rk+1, rk+1): compute locally, then MPI_Reduce
    // (rk, rk) already known

    // 6. pk+1 = rk+1 + gk * pk
    // compute locally then MPI_Allgather

    // ITERATION END

}
