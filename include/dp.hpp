/* Copyright (c) 2012, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See LICENSE.txt or 
 * http://www.opensource.org/licenses/mit-license.php */

#ifndef DP_HPP_
#define DP_HPP_

#include "baseMeasure.hpp"

#include <armadillo>

using namespace std;
using namespace arma;


double digamma(double x)
{
  // http://en.wikipedia.org/wiki/Digamma_function
  double x_sq = x*x;
  return ln(x) - 1.0/(2.0*x) - 1.0/(12.0*x_sq) + 1.0/(120.0*x_sq*x_sq) - 1.0/(252.0*x_sq*x_sq*x_sq);
}


template<class U>
class DP
{
public:
  DP(const BaseMeasure<U>& base, double alpha)
  : mH(base), mAlpha(alpha)
  {};

  ~DP()
  { };

  const BaseMeasure<U>& mH; // base measure
  double mAlpha;
private:
};

template<class U>
class DP_var
{
public:
  DP_var(const BaseMeasure<U>& base, double alpha)
   : mH(base), mAlpha(alpha)
  {};

  ~DP_var()
  { };

  Col<uint32_t> densityEst(const Mat<U>& x, uint32_t K0=10, uint32_t T0=10, uint32_t It=10)
  {
    
    Col<uint32_t> z(x.n_rows);

    uint32_t K=K0; // number of clusters

    // variables
    vector<double> gamma_1(K,0.0);
    vector<double> gamma_2(K,0.0);
    Row<U> tau_k1(x.n_cols);
    double tau_k2;
    Row<U> lambda_1(x.n_cols);
    double lambda_2;
    vector<double> S(K,0.0);
    
    // main loop
    for(uint32_t tt=0; tt<It; ++tt)
    {
      double S_sum=0.0;
      for(uint32_t k=0; k<K; ++k)
      {
        // TODO:  actually need to compute the exp of belows stuff => use different approximation!
        S[k] = digamma(gamma_1[k]) - digamma(gamma_1[k] + gamma_2[k]);
        for(uint32_t kt=0; kt<k-1; ++kt)
          S[k] += digamma(gamma_2[kt]) - digamma(gamma_1[kt] + gamma_2[kt]);
        S[k] += ; // TODO: two remaingin expected values
        S_sum += S[k];
      }


      for(uint32_t k=0; k<K; ++k)
      {
        double 
        gamma_k1=sum(z==k)+1.0;
        gamma_k2=alpha;
        for(uint32_t kj=k+1; kj<K; ++kj) gamma_k2 += sum(z==k);
        //Row<U> tau_k1=lambda_1; //TODO: what is lambda_1 and lambda_2
        tau_k2=lambda_2 + sum(z==k); // could reuse gamma_k1
      }
    }
  }

  const BaseMeasure<U>& mH; // base measure
  double mAlpha;
private:
   

};



#endif /* DP_HPP_ */
