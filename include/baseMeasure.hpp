/*
 * baseMeasure.hpp
 *
 *  Created on: Feb 1, 2013
 *      Author: jstraub
 */

#ifndef BASEMEASURE_HPP_
#define BASEMEASURE_HPP_

#include <armadillo>

#include <stddef.h>
#include <stdint.h>
#include <typeinfo>

using namespace std;
using namespace arma;

// templated on the unit
template<class U>
class BaseMeasure
{
public:
  BaseMeasure()
  {
    //cout<<"Creating "<<typeid(this).name()<<endl;
  };

  BaseMeasure(const BaseMeasure<U>& other)
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
    exit(0);
    return new BaseMeasure<U>(*this);
  };

  virtual Row<U> asRow() const
  {
    exit(0);
    return Row<U>();
  };
  
  virtual fromRow(const Row<U>& r)
  {exit(0);};

  virtual mode(Row<double>& mode) const
  {exit(0);};

  virtual double Elog(U x) const
  {exit(0);};

};

class Dir : public BaseMeasure<uint32_t>
{
public:
  Dir(const Row<double>& alphas)
  : mAlphas(alphas), mAlpha0(sum(alphas))
  {};

  Dir(const Dir& dir)
    : mAlphas(dir.mAlphas), mAlpha0(sum(dir.mAlphas))
  {};

  /*
   * return a copy of this object - this uses the concept of covariance
   */
  virtual Dir* getCopy() const
  {
    return new Dir(*this);
  };

  virtual Row<U> asRow() const
  {
    return mAlphas;
  };
  
  virtual fromRow(const Row<U>& r)
  {
    mAlphas = r;
    mAlpha0 = sum(r);
  };

  virtual mode(Row<double>& mode) const
  {
    dirMode(mode,mAlphas);
  };

  virtual double Elog(uint32_t x) const
  {
    return digamma(mAlpha(x_u(i))) - digamma(mAlpha0);
  };

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
  };

  NIW(const NIW& niw)
  : mVtheta(niw.mVtheta), mKappa(niw.mKappa), mDelta(niw.mDelta), mNu(niw.mNu)
  { };

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
  virtual Row<U> asRow() const
  {
    uint32_t d = mVtheta.n_elem;
    Row<U> row(d*d+d+2);

    for (uint32_t i=0; i<d; ++i)
      for (uint32_t j=0; j<d; ++j)
        row(j+i*d) = mDelta(j,i); // column major
    for (uint32_t i=0; i<d; ++i)
      row(d*d+i) = mVtheta(i);
    row(d*d+d) = mNu;
    row(d*d+d+1) = mKappa;

    return row;
  };
  
  virtual fromRow(const Row<U>& r)
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


  virtual mode(Row<double>& mode) const
  { 
     
  };

  virtual double Elog(double x) const
  {
    return 0.0;
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
