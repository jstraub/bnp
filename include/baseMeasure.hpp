/*
 * baseMeasure.hpp
 *
 *  Created on: Feb 1, 2013
 *      Author: jstraub
 */

#ifndef BASEMEASURE_HPP_
#define BASEMEASURE_HPP_

#include "probabilityHelpers.hpp"

#include <armadillo>

#include <stddef.h>
#include <stdint.h>
#include <typeinfo>

using namespace std;
using namespace arma;


/*
 * templated on the unit
 */
template<class U>
class BaseMeasure
{
public:
  BaseMeasure()
    : mRowDim(0)
  {
    //cout<<"Creating "<<typeid(this).name()<<endl;
  };

  BaseMeasure(const BaseMeasure<U>& other)
    : mRowDim(0)
  { };

  virtual ~BaseMeasure()
  {};

  virtual double predictiveProb(const Col<U>& x_q, const Mat<U>& x_given) const
  {
    cerr<<"BaseMeasure:: Something gone wrong with virtual functions"<<endl;
    exit(0);
    return 0.0;
  };
  virtual double predictiveProb(const Col<U>& x_q) const
  {
    cerr<<"BaseMeasure:: Something gone wrong with virtual functions"<<endl;
    exit(0);
    return 0.0;
  };

  virtual BaseMeasure<U>* getCopy() const
  { 
    cerr<<"BaseMeasure:: getCopy()"<<endl;
    exit(0);
    return new BaseMeasure<U>(*this);
  };

  virtual Row<double> asRow() const
  {
    cerr<<"BaseMeasure:: asRow()"<<endl;
    exit(0); return Row<double>(); 
  };

  uint32_t rowDim() const
  {
    return mRowDim;
  };
  
  virtual void fromRow(const Row<double>& r)
  {
    cerr<<"BaseMeasure:: fromRow()"<<endl;
    exit(0);};

  virtual void mode(Row<double>& mode) const
  {
    cerr<<"BaseMeasure:: mode()"<<endl;
    exit(0);};

  virtual double Elog(const Col<U>& x) const
  {
    cerr<<"BaseMeasure:: Elog()"<<endl;
    exit(0);};

  virtual void posteriorHDP_var(const Col<double>& zeta, const Mat<double>& phi, uint32_t D, const Mat<U>& x)
  {
    cerr<<"BaseMeasure:: posteriorHDP_var()"<<endl;
    exit(0);};

protected:
  uint32_t mRowDim;

};

/*
 * Container for base measure pointers
 */
template <class U>
class DistriContainer
{
public:

  DistriContainer()
    : mDistris(0,NULL)
  {
  };

  DistriContainer(const BaseMeasure<U>& a, uint32_t d)
    : mDistris(d,NULL)
  {
    for (uint32_t i=0; i<d; ++i)
      mDistris[i] = a.getCopy();
  };

  DistriContainer(const vector<BaseMeasure<U>* >& a)
    : mDistris(a.size(),NULL)
  {
    for (uint32_t i=0; i<a.size(); ++i)
      mDistris[i] = a[i]->getCopy();
  };

  DistriContainer(const DistriContainer<U>& a)
    : mDistris(a.size(),NULL)
  {
    for (uint32_t i=0; i<a.size(); ++i)
      mDistris[i] = a[i]->getCopy();
  };


  ~DistriContainer()
  {
    for (uint32_t i=0; i<mDistris.size(); ++i)
    {
      cout<<"deleting "<<mDistris.size()<<" "<<i<<" :"<<mDistris[i]<<endl;
      delete mDistris[i];
    }
  };
  
  void init(const BaseMeasure<U>& a, uint32_t d)
  { 
    for (uint32_t i=0; i<mDistris.size(); ++i)
      delete mDistris[i];

    mDistris.resize(d,NULL);
    for (uint32_t i=0; i<d; ++i)
      mDistris[i] = a.getCopy();
  };

  BaseMeasure<U>* operator[](const uint32_t i) const
  {
    assert(i<mDistris.size());
    return mDistris[i];
  };

  uint32_t size() const
  {
    return mDistris.size();
  };

private:
  vector<BaseMeasure<U>* > mDistris;
};


class Dir : public BaseMeasure<uint32_t>
{
public:
  Dir(const Row<double>& alphas)
  : mAlphas(alphas), mAlpha0(sum(alphas))
  {
    mRowDim = mAlphas.n_elem;
  };

  Dir(const Dir& dir)
    : mAlphas(dir.mAlphas), mAlpha0(sum(dir.mAlphas))
  {
    mRowDim = mAlphas.n_elem;
  };

  /*
   * return a copy of this object - this uses the concept of covariance
   */
  virtual Dir* getCopy() const
  {
    return new Dir(*this);
  };

  virtual Row<double> asRow() const
  {
    return mAlphas;
  };
  
  virtual void fromRow(const Row<double>& r)
  {
    mAlphas = r;
    mAlpha0 = sum(r);
  };

  virtual void mode(Row<double>& mode) const
  {
    mode.set_size(mAlphas.size()); 
    dirMode(mode,mAlphas);
  };

  virtual double Elog(const Col<uint32_t>& x) const
  {
    // TODO: assumes that x is 1D word
    return digamma(mAlphas(x(0))) - digamma(mAlpha0);
  };

  /*
   * update parameters using observations x to form posterior for stochastic variational HDP
   */
  virtual void posteriorHDP_var(const Col<double>& zeta, const Mat<double>& phi, uint32_t D, const Mat<uint32_t>& x)
  { 

    uint32_t N = x.n_cols;
    uint32_t T = zeta.n_rows;
    uint32_t Nw = mAlphas.n_elem;

    Row<double> lambda(Nw);
    lambda.zeros();
    for (uint32_t i=0; i<T; ++i) 
    {
      Row<double> _lambda(Nw); 
      _lambda.zeros();
      for (uint32_t n=0; n<N; ++n){
        _lambda(x(n)) += phi(n,i);
      }
      lambda += zeta(i) * _lambda;
    }
    mAlphas += D*lambda;
    //cout<<"lambda-nu="<<d_lambda[k].t()<<endl;
    //cout<<"lambda="<<d_lambda[k].t()<<endl;
  };

  /*
   * used for gibbs sampling - seems to assume a categorical distribution
   */
  double predictiveProb(const Col<uint32_t>& x_q, const Mat<uint32_t>& x_given) const
  {
    // TODO: assumes x_q is 1D! (so actually just a uint32_t

    // compute posterior under x_given
    uint32_t k=x_q(0);
    Row<uint32_t> ids=(x_given.row(0)==k);
    uint32_t C_k=sum(ids);
    uint32_t L=x_given.n_cols;
    //    cout<<"k="<<k<<" C_k="<<C_k<<" L="<<L<<" alpha0="<<mAlpha0
    //      <<" alpha_k="<<mAlphas(k)
    //      <<" log(p)="<< log((C_k + mAlphas(k))/(L + mAlpha0))<<endl;
    //    cout<<x_q<<" -" <<x_given.t()<<endl;
    return log((C_k + mAlphas(k))/(L + mAlpha0));
  };
  double predictiveProb(const Col<uint32_t>& x_q) const
  {
    const uint32_t k=x_q(0);
    return log(mAlphas(k)/mAlpha0);
  };


  Row<double> mAlphas;
  double mAlpha0;
private:
};

class NIW : public BaseMeasure<double>
{
public:
  // make a copy of vtheta and Delta (that should only be a copy of the header anyway
  NIW(colvec vtheta, double kappa, mat Delta, double nu)
  : mVtheta(vtheta), mKappa(kappa), mDelta(Delta), mNu(nu)
  {
//    cout<<"Creating "<<typeid(this).name()<<endl;
    
    mRowDim = mVtheta.n_elem;
    mRowDim = mRowDim*mRowDim + mRowDim +2;
  };

  NIW(const NIW& niw)
  : mVtheta(niw.mVtheta), mKappa(niw.mKappa), mDelta(niw.mDelta), mNu(niw.mNu)
  {
    mRowDim = mVtheta.n_elem;
    mRowDim = mRowDim*mRowDim + mRowDim +2;
  };

  virtual NIW* getCopy() const
  {
    return new NIW(*this);
  };

  /*
   * puts all parameters, Vtheta, kappa, delta, and nu into one vector
   * (0 to d^2-1) Delta
   * (d^2 to d^2+d-1) Vtheta
   * (d^2+d) nu
   * (d^2+d+1) kappa 
   */
  virtual Row<double> asRow() const
  {
    uint32_t d = mVtheta.n_elem;
    Row<double> row(d*d+d+2);

    for (uint32_t i=0; i<d; ++i)
      for (uint32_t j=0; j<d; ++j)
        row(j+i*d) = mDelta(j,i); // column major
    for (uint32_t i=0; i<d; ++i)
      row(d*d+i) = mVtheta(i);
    row(d*d+d) = mNu;
    row(d*d+d+1) = mKappa;
    return row;
  };
  
  virtual void fromRow(const Row<double>& row)
  {
    uint32_t d = mVtheta.n_elem; // we already have a Vtheta from the init;
    for (uint32_t i=0; i<d; ++i)
      for (uint32_t j=0; j<d; ++j)
        mDelta(j,i) = row(j+i*d); // column major
    for (uint32_t i=0; i<d; ++i)
      mVtheta(i) = row(d*d+i);
    mNu = row(d*d+d);
    mKappa = row(d*d+d+1);
  };


  virtual void mode(Row<double>& mode) const
  { 
    uint32_t d = mVtheta.n_elem; // we already have a Vtheta from the init;
    mode.set_size(d*d+d); 
    //TODO: not sure about covariance...
    for (uint32_t i=0; i<d; ++i)
      for (uint32_t j=0; j<d; ++j)
        mode(j+i*d) = mDelta(j,i)/(mNu+d+1); // column major
    for (uint32_t i=0; i<d; ++i)
      mode(d*d+i) = mVtheta(i); // sure about the mean
  };

  virtual double Elog(const Col<double>& x) const
  {
    uint32_t d = mVtheta.n_elem; // we already have a Vtheta from the init;
    double a = (gamma((mNu+1.0)/2.0)/( pow((mNu-d+1.0)*datum::pi,d/2.0)* gamma((mNu-d+1.0)/2.0)* sqrt((mKappa+1.0)/(mKappa*(mNu-d+1.0)))* sqrt(det(mDelta)) )) ;
    double b = as_scalar((x-mVtheta).t()*solve(mDelta,x-mVtheta));
    double c = pow(1.0+(mKappa/(mKappa+1.0))*b,(mNu+1.0)/2.0);
    return a/c;
  };

  virtual void posteriorHDP_var(const Col<double>& zeta, const Mat<double>& phi, uint32_t D, const Mat<double>& x)
  {
    uint32_t n = x.n_cols;
    Col<double> x_hat=sum(x,1);
    Mat<double> S(mDelta);
    S.zeros();
    for (uint32_t i=0; i<n; ++i)
      S += (x.col(i) - x_hat) * (x.col(i) - x_hat).t();
    
    mDelta = mDelta + S + (mKappa*n)/(mKappa+n)*(x_hat - mVtheta)*(x_hat - mVtheta).t();
    mVtheta = mKappa/(mKappa+n)* mVtheta + n/(mKappa+n)*x_hat;
    mKappa += x.n_cols;
    mNu += x.n_cols;
  };

  double predictiveProb(const Col<double>& x_q, const Mat<double>& x_given) const
  {
    // compute posterior under x_given
    // M0: number of data points in the cluster
    // M1: sum over all data points in the cluster
    // M2: x*x.T over all data points in the cluster
    double M0=x_given.n_cols;
    colvec M1=sum(x_given,1);
    mat M2=x_given.t()*x_given;
    double kappa = mKappa+M0;
    double nu = mNu+M0;
    colvec vtheta = (mKappa*mVtheta + M1)/kappa;
    //    cout<<"vtheta="<<vtheta<<endl;
    //    cout<<"mVtheta="<<mVtheta<<endl;
    //    cout<<"M2="<<M2<<endl;
    //    cout<<"mDelta="<<mDelta<<endl;
    mat Delta = (mNu*mDelta + M2 + mKappa*(mVtheta*mVtheta.t()) - kappa*(vtheta*vtheta.t()))/nu;

    // compute the likelihood of x_q under the posterior
    // This can be approximated by using a moment matched gaussian
    // to the multivariate student-t distribution which arises in
    // when integrating over the parameters of the normal inverse
    // wishart
    mat C_matched=(((kappa+1.0)*nu)/(kappa*(nu-vtheta.n_rows-1.0)))*Delta;

    return logGaus(x_q, vtheta, C_matched);
  };

  double predictiveProb(const Col<double>& x_q) const
  {
    // compute the likelihood of x_q under the posterior
    // This can be approximated by using a moment matched gaussian
    // to the multivariate student-t distribution which arises in
    // when integrating over the parameters of the normal inverse
    // wishart
    mat C_matched=(((mKappa+1.0)*mNu)/(mKappa*(mNu-mVtheta.n_rows-1.0)))*mDelta;

    return logGaus(x_q, mVtheta, C_matched);
  };

  static double logGaus(const colvec& x, const colvec& mu, const mat& C)
  {
    //    cout<<"C"<<C<<endl;
    //    cout<<"mu"<<mu<<endl;
    //    cout<<"x"<<x<<endl;
    double detC=det(C);
    if(!is_finite(detC))
      return 0.0;
    mat CinvXMu=solve(C,x-mu);
    mat logXCX = (x-mu).t() * CinvXMu;
    //    cout<<"rows="<<C.n_rows<<" log(det(C))="<<log(det(C*0.001))-double(C.n_rows)*1000.0<<" logXC="<<logXCX(0)<<" p="<<-0.5*(double(C.n_rows)*1.8378770664093453 + log(det(C)) + logXCX(0))<<endl;

    return -0.5*(double(C.n_rows)*1.8378770664093453 + log(detC) + logXCX(0));
  };

  colvec mVtheta;
  double mKappa;
  mat mDelta;
  double mNu;
};


#endif /* BASEMEASURE_HPP_ */
