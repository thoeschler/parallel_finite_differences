#include "crs.hpp"

#include <iostream>

CRSMatrix::CRSMatrix(std::size_t num_values, std::size_t num_rows) {
    _values.reserve(num_values);
    _col_indices.reserve(num_values);
    _row_indices.reserve(num_rows);
}

void CRSMatrix::next_row() {
    this->_row_indices.push_back(CRSMatrix::_values.size());
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
