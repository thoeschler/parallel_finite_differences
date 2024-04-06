#include "la.hpp"

#include <assert.h>

void matvec(CRSMatrix const&A, std::vector<double> const&b, std::vector<double> &result) {
    std::size_t row_start;
    std::size_t row_end;

    #pragma omp parallel for private(row_index, value_count)
    for (std::size_t row = 0; row < A.nrows(); ++row) {
        row_start = A.row_index(row);
        row_end = A.row_index(row + 1);
        for (std::size_t value_count = row_start; value_count < row_end; ++value_count) {
            result[row] += A.value(value_count) * b[A.col_index(value_count)];
        }
    }
}

double dot(std::vector<double> const& a, std::vector<double> const& b) {
    assert(a.size() == b.size());

    double result = 0.0;
    #pragma omp parallel shared(a, b)
    {
        #pragma omp for reduction(+:result)
        for (std::size_t i = 0; i < a.size(); ++i) {
            result += a[i] * b[i];
        }
    }
    return result;
}

void add(std::vector<double> &inout, std::vector<double> const&in) {
    assert(inout.size() == in.size());

    #pragma omp parallel for
    for (std::size_t i = 0; i < inout.size(); ++i) {
        inout[i] += in[i];
    }
}

void add_mult(std::vector<double> const&in1, std::vector<double> const&in2, std::vector<double> &out, double multiplier) {
    assert(in1.size() == in2.size());
    assert(in2.size() == out.size());

    #pragma omp parallel for
    for (std::size_t i = 0; i < out.size(); ++i) {
        out[i] = in1[i] + multiplier * in2[i];
    }
}

void add_mult_finout(std::vector<double> &inout, std::vector<double> const&in, double multiplier) {
    assert(inout.size() == in.size());

    #pragma omp parallel for
    for (std::size_t i = 0; i < inout.size(); ++i) {
        inout[i] += multiplier * in[i];
    }
}

void add_mult_sinout(std::vector<double> const&in, std::vector<double> &inout, double multiplier) {
    assert(inout.size() == in.size());

    #pragma omp parallel for
    for (std::size_t i = 0; i < inout.size(); ++i) {
        inout[i] = in[i] + multiplier * inout[i];
    }
}
