#include "la.hpp"

#include <iostream>
#include <assert.h>
#include <cmath>

void matvec(CRSMatrix const&A, std::vector<double> const&b, std::vector<double> &out) {
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
    matvec(A, u, Ap);
    add_mult(b, Ap, r, -1.0);
    p = r;
    norm_r_old_squared = dot(r, r);

    while (!converged) {
        matvec(A, p, Ap);
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

void parallel_cg(CRSMatrix const&A_loc, std::vector<double> const&b_loc, std::vector<double> &u_loc, MPI_Comm comm_cart,
                 const double tol) {
    int rank;
    MPI_Comm_rank(comm_cart, &rank);
    std::cout << rank << "\n";

    std::size_t size_loc = b_loc.size();
    u_loc.resize(size_loc);
    std::vector<double> r_loc(size_loc), p_loc(size_loc), Ap_loc(size_loc);

    // 0. compute p = r = b - Ax0
    r_loc = p_loc = b_loc;
    double rr_loc = dot(r_loc, r_loc);
    double bb_loc = dot(b_loc, b_loc);
    double rr, bb;
    MPI_Allreduce(&rr_loc, &rr, 1, MPI_DOUBLE, MPI_SUM, comm_cart);
    MPI_Allreduce(&bb_loc, &bb, 1, MPI_DOUBLE, MPI_SUM, comm_cart);

    // initialize other variables
    double alpha, gamma, rr_old;

    bool converged = (rr <= tol * tol * bb);
    std::size_t counter = 0;
    while (!converged) {
        // 1. compute Apk (locally) --> parallel matvec multiplication
        matvec(A_loc, p_loc, Ap_loc);

        // 2. compute ak = (rk, rk) / (Apk, pk)
        double App_loc = dot(Ap_loc, p_loc);
        double App;
        MPI_Allreduce(&App_loc, &App, 1, MPI_DOUBLE, MPI_SUM, comm_cart);
        alpha = rr / App;

        // 3. xk+1 = xk + ak * pk
        add_mult_finout(u_loc, p_loc, alpha);

        // 4. rk+1 = rk - ak Apk
        add_mult_finout(r_loc, Ap_loc, -alpha);

        // 5. gk = (rk+1, rk+1) / (rk, rk)
        rr_old = rr;
        rr_loc = dot(r_loc, r_loc);
        MPI_Allreduce(&rr_loc, &rr, 1, MPI_DOUBLE, MPI_SUM, comm_cart);
        gamma = rr / rr_old;

        // 6. pk+1 = rk+1 + gk * pk
        add_mult_sinout(r_loc, p_loc, gamma);

        if (rank == 0) {
            std::cout << "it " << counter << ": rr = " << rr << std::endl;
        }
        ++counter;
        converged = (rr <= tol * tol * bb);
    }
}
