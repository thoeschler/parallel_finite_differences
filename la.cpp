#include "la.hpp"

#include <assert.h>

/**
 * @brief Matrix vector product for CRS Matrix.
 * 
 * @param A CRS Matrix.
 * @param b Input vector.
 * @param result Result vector.
 */
void matvec(CRSMatrix const&A, std::vector<double> const&b, std::vector<double> &result) {
    std::size_t row_start;
    std::size_t row_end;

    #pragma omp parallel for private(row_start, row_end)
    for (std::size_t row = 0; row < A.nrows(); ++row) {
        row_start = A.row_index(row);
        row_end = A.row_index(row + 1);
        for (std::size_t value_count = row_start; value_count < row_end; ++value_count) {
            result[row] += A.value(value_count) * b[A.col_index(value_count)];
        }
    }
}

/**
 * @brief Dot product.
 * 
 * @param a First vector.
 * @param b Second vector.
 * @return double: result.
 */
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

/**
 * @brief Vector addition.
 * 
 * @param inout First input vector (will be overwritten).
 * @param in Second input vector.
 */
void add(std::vector<double> &inout, std::vector<double> const&in) {
    assert(inout.size() == in.size());

    #pragma omp parallel for
    for (std::size_t i = 0; i < inout.size(); ++i) {
        inout[i] += in[i];
    }
}

/**
 * @brief Combined addition and multiplication, i.e. out = in1 + multiplier * in2.
 * 
 * @param in1 First input vector.
 * @param in2 Second input vector.
 * @param out Output vector.
 * @param multiplier Multiplier.
 */
void add_mult(std::vector<double> const&in1, std::vector<double> const&in2, std::vector<double> &out, double multiplier) {
    assert(in1.size() == in2.size());
    assert(in2.size() == out.size());

    #pragma omp parallel for
    for (std::size_t i = 0; i < out.size(); ++i) {
        out[i] = in1[i] + multiplier * in2[i];
    }
}

/**
 * @brief Combined addition and multiplication, i.e. inout = inout + multiplier * in.
 * 
 * @param inout First input vector (will be overwritten).
 * @param in Second input vector.
 * @param multiplier Multiplier.
 */
void add_mult_finout(std::vector<double> &inout, std::vector<double> const&in, double multiplier) {
    assert(inout.size() == in.size());

    #pragma omp parallel for
    for (std::size_t i = 0; i < inout.size(); ++i) {
        inout[i] += multiplier * in[i];
    }
}

/**
 * @brief Combined addition and multiplication, i.e. inout = inout + multiplier * in.
 * 
 * @param inout First input vector.
 * @param in Second input vector (will be overwritten).
 * @param multiplier Multiplier.
 */
void add_mult_sinout(std::vector<double> const&in, std::vector<double> &inout, double multiplier) {
    assert(inout.size() == in.size());

    #pragma omp parallel for
    for (std::size_t i = 0; i < inout.size(); ++i) {
        inout[i] = in[i] + multiplier * inout[i];
    }
}
