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

void matvec(CRSMatrix const&A, std::vector<double> const&b, std::vector<double> &result);
double dot(std::vector<double> const& a, std::vector<double> const& b);
