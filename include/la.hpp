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

void matvec(const CRSMatrix &A, const std::vector<double> &b, std::vector<double> &result);
double dot(const std::vector<double> &a, const std::vector<double> &b);
