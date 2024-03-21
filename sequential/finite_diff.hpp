#pragma once
#include "la.hpp"
#include "crs.hpp"

#include <vector>
#include <functional>

// class PoissonFiniteDifferenceProblem2D {
//     private:
//         std::function<double(double, double)> boundary_condition;
//         FiniteDifferenceCRSMatrix _A;
//         std::vector<double> u;
//     public:

// };

void assemble_rhs(std::vector<double> &b, const std::size_t Nx, const std::size_t Ny,
                  std::function<double(double, double)> bc);

void assemble_matrix(CRSMatrix &A, const std::size_t Nx, const std::size_t Ny);
