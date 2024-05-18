#pragma once
#include "crs.hpp"

#include <vector>

void operator *=(std::vector<double> &v1, double multiplier);
void operator +=(std::vector<double> &v1, const std::vector<double> &v2);
void operator -=(std::vector<double> &v1, const std::vector<double> &v2);
std::vector<double> operator *(const std::vector<double> &v1, double multiplier);
std::vector<double> operator *(double multiplier, const std::vector<double> &v1);
std::vector<double> operator +(const std::vector<double> &v1, const std::vector<double> &v2);
std::vector<double> operator -(const std::vector<double> &v1, const std::vector<double> &v2);

void matmul(CRSMatrix const&A, std::vector<double> const&b, std::vector<double> &out);
double dot(std::vector<double> const& a, std::vector<double> const& b);
void cg(CRSMatrix const&A, std::vector<double> const&b, std::vector<double> &u, const double tol=1e-7, bool verbose=false);
