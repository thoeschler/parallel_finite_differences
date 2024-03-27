#pragma once
#include "crs.hpp"

#include <vector>
#include <mpi.h>

void matvec(CRSMatrix const&A, std::vector<double> const&b, std::vector<double> &result);
double dot(std::vector<double> const& a, std::vector<double> const& b);
double dot(std::vector<double> const& a, std::vector<double>::const_iterator b);
void add(std::vector<double> &inout, std::vector<double> const&in);
void add_mult(std::vector<double> const&in1, std::vector<double> const&in2, std::vector<double> &out, double multiplier);
void add_mult_finout(std::vector<double> &inout, std::vector<double> const&in, double multiplier);
void add_mult_finout(std::vector<double> &inout, std::vector<double>::iterator in, double multiplier);
void add_mult_sinout(std::vector<double> const&in, std::vector<double> &inout, double multiplier);
void add_mult_sinout(std::vector<double> const&in, std::vector<double>::iterator& inout, double multiplier);
