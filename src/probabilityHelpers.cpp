#include "probabilityHelpers.hpp"


double digamma(double x)
{
  //http://en.wikipedia.org/wiki/Digamma_function#Computation_and_approximation
//  if(x<1e-50){
//    //cerr<<"\tdigamma param x near zero: "<<x<<" cutting of"<<endl;
//    x=1e-50;
//  }
  //double x_sq = x*x;
  //return log(x)-1.0/(2.0*x)-1.0/(12.0*x_sq)+1.0/(12*x_sq*x_sq)-1.0/(252.0*x_sq*x_sq*x_sq);
  if (is_finite(x)){
    return boost::math::digamma(x);
  }else{
    cout<<" warning digam of: "<<x<<endl;
    exit(1);
    return 709; // digamma(1e308) = 709. ... 
  }
}

double digamma_mult(double x,uint32_t d)
{
  double digam_d = 0.0;
  for (uint32_t i=1; i<d+1; ++i)
  {
//    cout<<"digamma_mult of "<<(x + (1.0-double(i))/2)<<" = "<<digamma(x + (1.0-double(i))/2)<<endl;
    digam_d += digamma(x + (1.0-double(i))/2);
  }
  return digam_d;
}


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
    if (alpha[i]+beta[i] == 2.0) {
      v[i] = 0.5;
    }else if (alpha[i]>1.0 && beta[i]>1.0) {
      v[i] = (alpha[i]-1.0)/(alpha[i]+beta[i]-2.0);
    }else if (alpha[i]<1.0 && beta[i]>1.0) {
      v[i] = 0.0;
    }else if (alpha[i]>1.0 && beta[i]<1.0) {
      v[i] = 1.0;
    }
  }
};
void betaMode(Row<double>& v, const Col<double>& alpha, const Col<double>& beta)
{
  assert(alpha.n_elem == beta.n_elem);
  // breaking proportions
  v.set_size(alpha.n_elem);
  for (uint32_t i=0; i<v.n_elem; ++i){
    if (alpha[i]+beta[i] == 2.0) {
      v[i] = 0.5;
    }else if (alpha[i]>1.0 && beta[i]>1.0) {
      v[i] = (alpha[i]-1.0)/(alpha[i]+beta[i]-2.0);
    }else if (alpha[i]<1.0 && beta[i]>1.0) {
      v[i] = 0.0;
    }else if (alpha[i]>1.0 && beta[i]<1.0) {
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
void stickBreaking(Row<double>& prop, const Row<double>& v)
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
  // see derivation in my notes
  Row<double> alpha_mod(alpha);
  double alpha_0 = sum(alpha);
  double K = alpha.n_elem;
  if (alpha_0 < K) {
    for (uint32_t i=0; i<K; ++i)
      if (alpha_mod[i] > 1.0) {alpha_mod[i] = 1.0;}
  }else if(alpha_0 > K) {
    for (uint32_t i=0; i<K; ++i)
      if (alpha_mod[i] < 1.0) {alpha_mod[i] = 1.0;}
  }
  mode = (alpha_mod-1.0)/sum(alpha_mod-1.0);
};

void dirMode(Col<double>& mode, const Col<double>& alpha)
{

  // see derivation in my notes
  Col<double> alpha_mod(alpha);
  double alpha_0 = sum(alpha);
  double K = alpha.n_elem;
  if (alpha_0 < K) {
    for (uint32_t i=0; i<K; ++i)
      if (alpha_mod[i] > 1.0) {alpha_mod[i] = 1.0;}
  }else if(alpha_0 > K) {
    for (uint32_t i=0; i<K; ++i)
      if (alpha_mod[i] < 1.0) {alpha_mod[i] = 1.0;}
  }
  mode = (alpha_mod-1.0)/sum(alpha_mod-1.0);
};
