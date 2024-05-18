#include "la.hpp"

#include <iostream>
#include <assert.h>
#include <cmath>
#include <chrono>

void operator *=(std::vector<double> &v1, double multiplier) {
    for (std::size_t i = 0; i < v1.size(); ++i) {
        v1[i] *= multiplier;
    }
}

void operator +=(std::vector<double> &v1, const std::vector<double> &v2) {
    assert(v1.size() == v2.size());
    for (std::size_t i = 0; i < v1.size(); ++i) {
        v1[i] += v2[i];
    }
}

void operator -=(std::vector<double> &v1, const std::vector<double> &v2) {
    assert(v1.size() == v2.size());
    for (std::size_t i = 0; i < v1.size(); ++i) {
        v1[i] -= v2[i];
    }
}

std::vector<double> operator *(const std::vector<double> &v1, double multiplier) {
    std::vector<double> out(v1.size());
    for (std::size_t i = 0; i < v1.size(); ++i) {
        out[i] = multiplier * v1[i];
    }
    return out;
}

std::vector<double> operator *(double multiplier, const std::vector<double> &v1) {
    std::vector<double> out(v1.size());
    for (std::size_t i = 0; i < v1.size(); ++i) {
        out[i] = multiplier * v1[i];
    }
    return out;
}

std::vector<double> operator +(const std::vector<double> &v1, const std::vector<double> &v2) {
    assert(v1.size() == v2.size());
    std::vector<double> out(v1.size());
    for (std::size_t i = 0; i < v1.size(); ++i) {
        out[i] = v1[i] + v2[i];
    }
    return out;
}

std::vector<double> operator -(const std::vector<double> &v1, const std::vector<double> &v2) {
    assert(v1.size() == v2.size());
    std::vector<double> out(v1.size());
    for (std::size_t i = 0; i < v1.size(); ++i) {
        out[i] = v1[i] - v2[i];
    }
    return out;
}

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

void cg(CRSMatrix const&A, std::vector<double> const&b, std::vector<double> &u, const double tol, bool verbose) {
    bool converged = false;
    double alpha, gamma;
    std::size_t size = b.size();
    std::vector<double> p(size), Ap(size);

    double bb = dot(b, b);
    double rr, rr_old;

    // initialize p (direction) and r (residual)
    matmul(A, u, Ap);
    std::vector<double> r = b - Ap;
    p = r;
    rr_old = dot(r, r);

    std::size_t counter = 0;

    const auto start = std::chrono::high_resolution_clock::now();
    while (!converged) {
        matmul(A, p, Ap);
        alpha = dot(r, r) / dot(Ap, p);
        u += alpha * p;
        r -= alpha * Ap;
        rr = dot(r, r);
        gamma = rr / rr_old;
        rr_old = rr;
        p = r + gamma * p;

        converged = (rr <= tol * tol * bb);
        if (verbose && counter % 100 == 0) {
            std::cout << "it " << counter << ": rr / bb = " << std::sqrt(rr / bb) << "\n";
        }
        ++counter;
    }
    const auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> avg_time = (end - start) / counter;
    std::cout << "average time / it: " << avg_time.count() << "s\n";
}
