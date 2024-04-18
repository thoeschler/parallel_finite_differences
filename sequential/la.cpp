#include "la.hpp"

#include <iostream>
#include <assert.h>
#include <cmath>

void matmul(CRSMatrix const&A, std::vector<double> const&b, std::vector<double> &out) {
    std::fill(out.begin(), out.end(), 0.0);
    std::size_t row_start, row_end;
    for (std::size_t row = 0; row < A.nrows(); ++row) {
        row_start = A.row_index(row);
        row_end = A.row_index(row + 1);
        for (std::size_t value_count = row_start; value_count < row_end; ++value_count) {
            out[row] += A.value(value_count) * b[A.col_index(value_count)];
        }
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

void cg(CRSMatrix const&A, std::vector<double> const&b, std::vector<double> &u, const double tol, bool verbose) {
    bool converged = false;
    double alpha, gamma;
    std::size_t size = b.size();
    std::vector<double> p(size), r(size), Ap(size);

    double bb = dot(b, b);
    double rr, rr_old;

    // initialize p (direction) and r (residual)
    matmul(A, u, Ap);
    add_mult(b, Ap, r, -1.0);
    p = r;
    rr_old = dot(r, r);

    std::size_t counter = 0;

    while (!converged) {
        matmul(A, p, Ap);
        alpha = dot(r, r) / dot(Ap, p);
        add_mult_finout(u, p, alpha);
        add_mult_finout(r, Ap, -alpha);
        rr = dot(r, r);
        gamma = rr / rr_old;
        rr_old = rr;
        add_mult_sinout(r, p, gamma);

        converged = (rr <= tol * tol * bb);
        if (verbose && counter % 100 == 0) {
            std::cout << "it " << counter << ": rr / bb = " << std::sqrt(rr / bb) << "\n";
        }
        ++counter;
    }
}
