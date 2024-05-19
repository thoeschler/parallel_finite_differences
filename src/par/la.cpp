#include "la.hpp"

#include <assert.h>

void operator *=(std::vector<double> &v1, double multiplier) {
    #pragma omp parallel for
    for (std::size_t i = 0; i < v1.size(); ++i) {
        v1[i] *= multiplier;
    }
}

void operator +=(std::vector<double> &v1, const std::vector<double> &v2) {
    assert(v1.size() == v2.size());
    #pragma omp parallel for
    for (std::size_t i = 0; i < v1.size(); ++i) {
        v1[i] += v2[i];
    }
}

void operator -=(std::vector<double> &v1, const std::vector<double> &v2) {
    assert(v1.size() == v2.size());
    #pragma omp parallel for
    for (std::size_t i = 0; i < v1.size(); ++i) {
        v1[i] -= v2[i];
    }
}

std::vector<double> operator *(const std::vector<double> &v1, double multiplier) {
    std::vector<double> out(v1.size());
    #pragma omp parallel for
    for (std::size_t i = 0; i < v1.size(); ++i) {
        out[i] = multiplier * v1[i];
    }
    return out;
}

std::vector<double> operator *(double multiplier, const std::vector<double> &v1) {
    std::vector<double> out(v1.size());
    #pragma omp parallel for
    for (std::size_t i = 0; i < v1.size(); ++i) {
        out[i] = multiplier * v1[i];
    }
    return out;
}

std::vector<double> operator +(const std::vector<double> &v1, const std::vector<double> &v2) {
    assert(v1.size() == v2.size());
    std::vector<double> out(v1.size());
    #pragma omp parallel for
    for (std::size_t i = 0; i < v1.size(); ++i) {
        out[i] = v1[i] + v2[i];
    }
    return out;
}

std::vector<double> operator -(const std::vector<double> &v1, const std::vector<double> &v2) {
    assert(v1.size() == v2.size());
    std::vector<double> out(v1.size());
    #pragma omp parallel for
    for (std::size_t i = 0; i < v1.size(); ++i) {
        out[i] = v1[i] - v2[i];
    }
    return out;
}

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
