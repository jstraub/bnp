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
  if (pi[x] == 0.0) {
    return 0.0;
  }else{
    double p = log(pi[x]);
    for (uint32_t i=0; i< pi.n_elem; ++i){
      if (i!=x) p += log(1.0 - pi[i]);
      //cout<<"pi["<<i<<"]="<<pi[i]<<endl;
    }
    return p;
  }
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

double logBeta(const Row<double>& x, double alpha, double beta)
{
  double p=0.0;
  for (uint32_t i=0; i<x.n_cols; ++i){
    p += logBeta(x[i],alpha,beta);
  }
  return p;
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

void betaMode(Col<double>& v, const Col<double>& alpha, const Col<double>& beta)
{
  assert(alpha.n_elem == beta.n_elem);
  // breaking proportions
  v.set_size(alpha.n_elem);
  for (uint32_t i=0; i<v.n_elem; ++i){
    if (alpha[i]+beta[i] != 2.0) {
      v[i] = (alpha[i]-1.0)/(alpha[i]+beta[i]-2.0);
    }else{
      v[i] = 1.0;
    }
  }
};

// stick breaking proportions 
// truncated stickbreaking -> stick breaks will be dim longer than proportions v 
void stickBreaking(Col<double>& prop, const Col<double>& v)
{
  prop.set_size(v.n_elem+1);
  // stick breaking proportions
  for (uint32_t i=0; i<prop.n_elem; ++i){
    if (i == prop.n_elem-1){
      prop[i] = 1.0;
    }else{
      prop[i] = v[i];
    }
    for (uint32_t j=0; j<i; ++j){
      prop[i] *= (1.0 - v[j]);
    }
  }
};

uint32_t multinomialMode(const Row<double>& p )
{
  uint32_t ind =0;
  p.max(ind);
  return ind;
};

void dirMode(Row<double>& mode, const Row<double>& alpha)
{
  mode = (alpha-1.0)/sum(alpha-1.0);
};

void dirMode(Col<double>& mode, const Col<double>& alpha)
{
  mode = (alpha-1.0)/sum(alpha-1.0);
};
