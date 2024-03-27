#include "la.hpp"

#include <iostream>
#include <assert.h>
#include <cmath>

void matvec(CRSMatrix const&A, std::vector<double> const&b, std::vector<double> &result) {
    std::fill(result.begin(), result.end(), 0.0);
    std::size_t row = 0;
    std::size_t row_index = A.row_index(row + 1);
    for (std::size_t value_count = 0; value_count < A.size(); ++value_count) {
        if (value_count == row_index) {
            ++row;
            row_index = A.row_index(row + 1);
        }
        result[row] += A.value(value_count) * b[A.col_index(value_count)];
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

double dot(std::vector<double> const& a, std::vector<double>::const_iterator b) {
    double result = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i, ++b) {
        result += a[i] * (*b);
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

void add_mult_finout(std::vector<double> &inout, std::vector<double>::iterator in, double multiplier) {
    for (std::size_t i = 0; i < inout.size(); ++i, ++in) {
        inout[i] += multiplier * (*in);
    }
}

void add_mult_sinout(std::vector<double> const&in, std::vector<double> &inout, double multiplier) {
    assert(inout.size() == in.size());

    for (std::size_t i = 0; i < inout.size(); ++i) {
        inout[i] = in[i] + multiplier * inout[i];
    }
}

void add_mult_sinout(std::vector<double> const&in, std::vector<double>::iterator& inout, double multiplier) {
    for (std::size_t i = 0; i < in.size(); ++i, ++inout) {
        *inout = in[i] + multiplier * (*inout);
    }
}
