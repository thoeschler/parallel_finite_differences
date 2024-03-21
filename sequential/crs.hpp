#pragma once
#include <vector>
#include <iostream>

class CRSMatrix {
    protected:
        std::vector<double> _values; // nonzero values
        std::vector<std::size_t> _col_indices; // column indices
        std::vector<std::size_t> _row_ptr = {0}; // row information
    public:
        CRSMatrix() {};
        void append(double value, std::size_t col_index);
        void next_row();
        std::size_t nrows() const { return _row_ptr.size(); };
        std::size_t row_index(std::size_t row) const { return _row_ptr[row]; }
        std::size_t col_index(std::size_t index) const { return _col_indices[index]; }
        std::size_t size() const { return _values.size(); }
        double value(std::size_t index) const { return _values[index]; }
        void print_values() const;
};

class FiniteDifferenceCRSMatrix: private CRSMatrix {
    public:
        FiniteDifferenceCRSMatrix(std::size_t nrows, std::size_t ncols);

};
