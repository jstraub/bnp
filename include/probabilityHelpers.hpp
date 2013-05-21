
#pragma once

#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/beta.hpp>
#include <armadillo>

using namespace std;
using namespace arma;

// digamma function
double digamma(double x);
// multivariate digamma function
double digamma_mult(double x,uint32_t d);
// cateorical distribution (Multionomial for one word)
double Cat(uint32_t x, Row<double> pi);
// log cateorical distribution (Multionomial for one word)
double logCat(uint32_t x, Row<double> pi);
// evaluate beta distribution at x
double Beta(double x, double alpha, double beta);
// log beta function
double betaln(double alpha, double beta);
// evaluate log beta distribution at x
double logBeta(double x, double alpha, double beta);
double logBeta(const Row<double>& x, double alpha, double beta);
double logDir(const Row<double>& x, const Row<double>& alpha);

void betaMode(Col<double>& v, const Col<double>& alpha, const Col<double>& beta);
void betaMode(Row<double>& v, const Col<double>& alpha, const Col<double>& beta);
// stick breaking proportions;  truncated stickbreaking -> stick breaks will be dim longer than proportions v 
void stickBreaking(Col<double>& prop, const Col<double>& v);
void stickBreaking(Row<double>& prop, const Row<double>& v);
uint32_t multinomialMode(const Row<double>& p);
void dirMode(Row<double>& mode, const Row<double>& alpha);
void dirMode(Col<double>& mode, const Col<double>& alpha);

template <class U>
Row<uint32_t> size(Mat<U> A)
{
  Row<uint32_t> s(2);
  s(0) = A.n_rows;
  s(1) = A.n_cols;
  return s;
};

template <class U>
Row<uint32_t> size(Col<U> A)
{
  Row<uint32_t> s(2);
  s(0) = A.n_rows;
  s(1) = 1;
  return s;
};

template <class U>
Row<uint32_t> size(Row<U> A)
{
  Row<uint32_t> s(2);
  s(0) = 1;
  s(1) = A.n_cols;
  return s;
};
