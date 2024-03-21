#include "crs.hpp"

#include <iostream>

void CRSMatrix::next_row() {
    this->_row_ptr.push_back(CRSMatrix::_values.size());
}

void CRSMatrix::append(double value, std::size_t col_index) {
    _values.push_back(value);
    _col_indices.push_back(col_index);
}

void CRSMatrix::print_values() const {
    std::size_t row = 0;
    std::size_t row_index = this->row_index(row + 1);
    for (std::size_t value_count = 0; value_count < this->size(); ++value_count) {
        if (value_count == row_index) {
            ++row;
            row_index = this-> row_index(row + 1);
            std::cout << "\n";
        }
        std::cout << _values[value_count] << " ";
    }
    std::cout << "\n";
}

FiniteDifferenceCRSMatrix::FiniteDifferenceCRSMatrix(std::size_t nrows, std::size_t ncols) {
    _values.reserve(5 * nrows); // TODO: what is the number of nonzero entries?
    _col_indices.reserve(5 * nrows);
    _row_ptr.reserve(nrows + 1);
}
