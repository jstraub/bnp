#include "probabilityHelpers.hpp"

// cateorical distribution (Multionomial for one word)
double Cat(uint32_t x, Row<double> pi)
{
  assert(x<pi.n_elem);
  double p = pi[x];
  for (uint32_t i=0; i< pi.n_elem; ++i)
    if (i!=x) p*=(1.0 - pi[i]);
  return p;
};

// log cateorical distribution (Multionomial for one word)
double logCat(uint32_t x, Row<double> pi)
{
  assert(x<pi.n_elem);
  double p = log(pi[x]);
  for (uint32_t i=0; i< pi.n_elem; ++i)
    if (i!=x) p += log(1.0 - pi[i]);
  return p;
};

double Beta(double x, double alpha, double beta)
{
  return (1.0/boost::math::beta(alpha,beta)) * pow(x,alpha-1.0) * pow(1.0-x,beta-1.0);
};

double betaln(double alpha, double beta)
{
  return -boost::math::lgamma(alpha+beta) + boost::math::lgamma(alpha) + boost::math::lgamma(beta);
};

double logBeta(double x, double alpha, double beta)
{
  if (alpha == 1.0) {
    return  -betaln(alpha, beta) + log(1.0-x)*(beta-1.0);
  }else if (beta == 1.0){
    return -betaln(alpha,beta) + log(x)*(alpha-1.0);
  }else if(alpha == 1.0 && beta == 1.0){
    return -betaln(alpha,beta); 
  }else{
    return -betaln(alpha,beta) + log(x)*(alpha-1.0) + log(1.0-x)*(beta-1.0);
  }
};

double logDir(const Row<double>& x, const Row<double>& alpha)
{ 
  assert(alpha.n_elem == x.n_elem);
  double logP=boost::math::lgamma(sum(alpha));
  for (uint32_t i=0; i<alpha.n_elem; ++i){
    logP += -boost::math::lgamma(alpha[i]) + (alpha[i]-1.0)*log(x[i]);
  }
  return logP;
};
