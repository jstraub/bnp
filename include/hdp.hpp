/* Copyright (c) 2012, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See LICENSE.txt or 
 * http://www.opensource.org/licenses/mit-license.php */

#pragma once

#include "random.hpp"
#include "baseMeasure.hpp"
#include "dp.hpp"

#include <stddef.h>
#include <stdint.h>
#include <typeinfo>

#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/beta.hpp>
#include <armadillo>

using namespace std;
using namespace arma;

template <class U>
class HDP : public DP<U>
{
public:
  HDP(const BaseMeasure<U>& base, double alpha, double omega)
  : DP<U>(base, alpha), mOmega(omega)
    {
//    cout<<"Creating "<<typeid(this).name()<<endl;
    };

  ~HDP()
  {	};

  // method for "one shot" computation without storing data in this class
  vector<Col<uint32_t> > densityEst(const vector<Mat<U> >& x, uint32_t K0=10, uint32_t T0=10, uint32_t It=10)
      {
    RandDisc rndDisc;
    // x is a list of numpy arrays: one array per document
    uint32_t J=x.size(); // number of documents
    vector<uint32_t> N(J,0);
    for (uint32_t j=0; j<J; ++j)
      N.at(j)=int(x[j].n_rows); // number of datapoints in each document
    uint32_t d=x[0].n_cols;      // dimension (assumed to be equal among documents

    uint32_t K=K0; // number of clusters/dishes in the franchise
    vector<uint32_t> T(J,0);   // number of tables in each restaurant
    vector<Col<uint32_t> > t_ji(J); // assignment of a table in restaurant j to customer i -> stores table number for each customer (per restaurant)
    vector<Col<uint32_t> > k_jt(J); // assignment of a dish in restaurant j at table t -> stores dish number for each table (per restaurant)
    RandInt rndT(0,T0);
    RandInt rndK(0,K0);
    for (uint32_t j=0; j<J; ++j)
    {
      T[j]=T0;
      t_ji[j] = rndT.draw(N[j]);
      k_jt[j] = rndK.draw(T[j]);
    }

    vector<Col<uint32_t> > z_ji(J);
    vector<uint32_t> Tprev=T;   // number of tables in each restaurant
    uint32_t Kprev=K;
    for (uint32_t tt=0; tt<It; ++tt)
    {
      cout<<"---------------- Iteration "<<tt<<" K="<<K<<" -------------------"<<endl;
      // gibbs update for the customer assignments to tables
      for (uint32_t j=0; j<J; ++j)
      {
        cout<<"@j="<<j<<"; N_j="<<N[j]<<"; T_j="<<T[j]<<endl;
        uint32_t n_j=t_ji[j].n_rows;
        for (uint32_t i=0; i<N[j]; ++i)
        {
          colvec l=zeros<colvec>(T[j]+K+1,1);
          colvec f=zeros<colvec>(T[j]+K+1,1);
          for (uint32_t t=0; t<T[j]; ++t)
          {
            uint32_t n_jt=sum(t_ji[j]==t);
            if (t_ji[j](i) == t) n_jt--;
            if (n_jt == 0){
              l[t]=math::nan();
              continue;
            }
            Mat<U> x_k_ji = getXinK(x,j,i,k_jt[j](t),k_jt,t_ji);
            //HDP hdp_x_ji = posterior(x_k_ji); // compute posterior hdp given the data 
            double f_k_jt = this->mH.predictiveProb(x[j].row(i).t(),x_k_ji);
            //logGaus(x[j].row(i).t(), hdp_x_ji.mVtheta, hdp_x_ji.mmCov()); // marginal probability of x_ji in cluster k/dish k given all other data in that cluster
            f(t) = f_k_jt;
            l(t) = log(n_jt/(n_j + this->mAlpha)) + f_k_jt;
            //          if(x_k_ji.n_rows <3)
            //            cout<<"x_k_ji="<<x_k_ji<<"size="<<x_k_ji.n_rows<<" x "<<x_k_ji.n_cols<<"\t mu="<<hdp_x_ji.mVtheta(0)<<" \t cov="<<hdp_x_ji.mmCov()(0)<<"\t f="<<f_k_jt<<endl;
          }
          uint32_t m_=K; // number of dishes
          for (uint32_t k=0; k<K; ++k)
          {// handle cases where x_ji is seated at a new table with a existing dish
            //Col<uint32_t> m_k = zeros<Col<uint32_t> >(1,1); //number of tables serving dish k 
            uint32_t m_k = 0; //number of tables serving dish k 
            for (uint32_t jj=0; jj<J; ++jj) 
              m_k += sum(k_jt[jj] ==k);
            if(m_k ==0){
              l[T[j]+k] = math::nan();
              continue;
            }
            Mat<U> x_k_ji = getXinK(x,j,i,k,k_jt,t_ji,tt==It-1);
            //HDP hdp_x_ji = posterior(x_k_ji); // compute posterior hdp given the data in 
            double f_k =  this->mH.predictiveProb(x[j].row(i).t(),x_k_ji);
            //logGaus(x[j].row(i).t(), hdp_x_ji.mVtheta, hdp_x_ji.mmCov());  // marginal probability of x_ji in cluster k/dish k given all other data in that cluster
            f(T[j]+k) = f_k;
            l(T[j]+k) = log(this->mAlpha*m_k/((n_j+this->mAlpha)*(m_+mOmega))) + f_k; // TODO: shouldnt this be mAlpha of the posterior hdp?
          }
          // handle the case where x_ji sits at a new table with a new dish
          //        cout<<"ji=:"<<j<<" " <<i<<endl;
          //        cout<<"x_ji=:"<<x[j].row(i)<<endl;
          //        cout<<"mVtheta=:"<<mVtheta<<endl;
          //        cout<<"Cov=:"<<mmCov()<<endl;
          //
          double f_knew = this->mH.predictiveProb(x[j].row(i).t());
          //logGaus(x[j].row(i).t(), mVtheta, mmCov());
          f[T[j]+K] = f_knew;
          l[T[j]+K] = log(this->mAlpha*mOmega/((n_j+this->mAlpha)*(m_+mOmega))) + f_knew;

          //        cout<<"l.n_elem= "<<l.n_elem<<" l="<<l.t()<<endl;
          uint32_t z_i = sampleDiscLogProb(rndDisc,l);

#ifndef NDEBUG
          cout<<endl<<"l="<<l.t()<<" |l|="<<l.n_elem<<endl;
          cout<<"T_j="<<T[j]<<"; K="<<K<<"; z_i="<<z_i<<endl;
#endif
          if (z_i < T[j])
          { // customer sits at existing table 
            t_ji[j](i)=z_i; // update table information of customer i in restaurant j
#ifndef NDEBUG
            cout<<"customer sits at existing table "<<z_i<<endl;
#endif
          }else if (z_i-T[j] < K)
          { // customer sits at new table with a already existing dish
            t_ji[j](i)=T[j]; // update table information of customer i in restaurant j
            k_jt[j].resize(k_jt[j].n_elem+1);
            k_jt[j](k_jt[j].n_elem-1) = z_i-T[j]; // add a new table with the sampled dish
#ifndef NDEBUG
            cout<<"customer sits at new table with a already existing dish "<<z_i-T[j]<<" z_i="<<z_i<<" T_j="<<T[j]<<endl;
#endif
            T[j]++;
          }else if (z_i == T[j]+K)
          { // customer sits at a new table with a new dish
            t_ji[j](i)=T[j]; // update table information of customer i in restaurant j
            k_jt[j].resize(k_jt[j].n_elem+1);
            k_jt[j](k_jt[j].n_elem-1)=K; // add a new table with a new dish
            T[j]++;
            K++;
#ifndef NDEBUG
            cout<<"customer sits at a new table with a new dish"<<endl;
#endif
          }
        }
      }
      //remove unused tables
      for (uint32_t j=0; j<J; ++j) // find unused dishes 
        for (int32_t t=T[j]-1; t>-1; --t)
        {
          //Col<uint32_t> contained = (t_ji[j]==t);
          //contained=sum(contained);
          uint32_t contained = sum(t_ji[j]==t);
          if (contained == 0)
          {
            t_ji[j].elem(find(t_ji[j] >= t)) -=1;
            T[j]--;
            //cout<<"shed "<<t<<endl<<k_jt[j].t();
            k_jt[j].shed_row(t);
            //cout<<k_jt[j].t()<<endl;
          }
        }

      for (uint32_t j=0; j<J; ++j)
      {
        cout<<"-- T["<<j<<"]="<<T[j]<<"; Tprev["<<j<<"]="<<Tprev[j]<<" deltaT["<<j<<"]="<<int32_t(T[j])-int32_t(Tprev[j])<<endl;
      }
      Tprev=T;

      cout<<" Gibbs update for k_jt"<<endl;
      for (uint32_t j=0; j<J; ++j)
      {
        for (uint32_t t=0; t<T[j]; ++t)
        {
          colvec l=zeros<colvec>(K+1,1);
          colvec f=zeros<colvec>(K+1,1);
          uint32_t m_ = 0; // number of tables 
          for (uint32_t jj=0; jj<J; ++jj)
            m_ += T[jj];
          uvec i_jt=find(t == t_ji[j]);
          Mat<U> x_jt = zeros<Mat<U> >(i_jt.n_elem,d);
          for (uint32_t i=0; i<i_jt.n_elem; ++i)
            x_jt.row(i) = x[j].row(i_jt(i)); //all datapoints which are sitting at table t 
          for (uint32_t k=0; k<K; ++k)
          {
            uint32_t m_k = 0; // number of tables serving dish k 
            for (uint32_t jj=0; jj<J; ++jj)
              m_k += sum(k_jt[jj] == k);
            if (m_k == 0){
              l(k) = math::nan();
              continue;
            }
            double f_k=0.0;
            for (uint32_t i=0; i<x_jt.n_rows; ++i)
            { // product over independent x_ji_t
              Mat<U> x_k_ji = getXinK(x,j,i_jt(i),k,k_jt,t_ji); // for posterior computation
              //HDP hdp_x_ji = posterior(x_k_ji); // compute posterior hdp given the data in 
              f_k += this->mH.predictiveProb(x_jt.row(i).t(),x_k_ji);
              //logGaus(x_jt.row(i).t(), hdp_x_ji.mVtheta, hdp_x_ji.mmCov());
            }
            f(k)=f_k;
            l(k)=log(m_k/(m_+mOmega)) + f_k;
          }
          double f_knew=0.0;
          for (uint32_t i=0; i<x_jt.n_rows; ++i) // product over independent x_ji_t
            f_knew += this->mH.predictiveProb(x_jt.row(i).t());
          //logGaus(x_jt.row(i).t(), mVtheta, mmCov());
          f(K)=f_knew;
          l(K)=log(mOmega/(m_+mOmega)) + f_knew; // update dish at table t in restaurant j
          uint32_t z_jt = sampleDiscLogProb(rndDisc, l);
#ifndef NDEBUG
          cout<<endl<<"l="<<l.t()<<" |l|="<<l.n_elem<<endl;
          cout<<"T_j="<<T[j]<<"; K="<<K<<"; z_jt="<<z_jt<<endl;
#endif
          if (z_jt < K){
            k_jt[j](t)=z_jt;
#ifndef NDEBUG
            cout<<"Table "<<t<<" gets already existing meal "<<z_jt<<endl;
#endif
          }else if ( z_jt == K){
            k_jt[j](t)=K;
            K++;
#ifndef NDEBUG
            cout<<"Table "<<t<<" gets new meal "<<K-1<<endl;
#endif
          }
        }
      }

      //    for (uint32_t j=0; j<J; ++j)
      //      cout<<"k_jt="<<k_jt[j].t()<<endl;
      // remove unused dishes
      Col<uint32_t> k_used=k_jt[0];
      Col<uint32_t> k_unused=zeros<Col<uint32_t> >(K,1);
      for (uint32_t j=0; j<J; ++j)
      { 
        k_used.insert_rows(k_used.n_elem,k_jt[j]);
      }
      uint32_t k_sum;
      for (uint32_t k=0; k<K; ++k)
      {// find unused dishes
        k_sum = sum(k_used==k);
        k_unused(k) = k_sum>0?0:1;
      }
      for (int32_t k=K-1; k>-1; --k) // iterate from large to small so that the indices work out when deleting
        if (k_unused(k)==1){
          for (uint32_t j=0; j<J; ++j)
          {
            Col<uint32_t> ids=find(k_jt[j]>=k);
            for(uint32_t i=0; i<ids.n_elem; ++i)
              k_jt[j](ids(i)) -= 1;
            //cout<<k_jt[j].elem(find(k_jt[j]>=k))<<endl;
          }
          K--;
        }

      cout<<"-- K="<<K<<"; Kprev="<<Kprev<<" deltaK="<<int32_t(K)-int32_t(Kprev)<<endl;
      //    for (uint32_t j=0; j<J; ++j)
      //      cout<<"k_jt="<<k_jt[j].t()<<endl;
      for (uint32_t j=0; j<J; ++j)
      {
        z_ji[j].set_size(N[j]);
        z_ji[j].zeros();
        for (uint32_t i=0; i<N[j]; ++i)
          z_ji[j](i)=k_jt[j](t_ji[j](i));
        //cout<<"z_ji["<<j<<"]="<<z_ji[j].t()<<" |.|="<<z_ji[j].n_elem<<endl;
      }

      // TODO: compute log likelyhood of data under model
      //      for (uint32_t k=0; k<K; ++k)
      //      {
      //        Mat<U> x_k = zeros<Mat<U> >(d,1);
      //        for (uint32_t j=0; j<J; ++j)
      //        {
      //          Col<uint32_t> id=find(z_ji[j] == k);
      //          uint32_t offset=x_k.n_rows;
      //          x_k.resize(x_k.n_rows+id.n_elem, x_k.n_cols);
      //          for (uint32_t i=0; i<id.n_elem; ++i)
      //            x_k.row(offset+i) = x[j].row(id(i));
      //        }
      //        x_k.shed_row(0);
      //        mat mu=mean(x_k,0);
      //        mat c=var(x_k,0,0);
      //        cout<<"@"<<k<<": mu="<<mu(0,0)<<" var="<<c(0,0)<<endl;
      //      }
    }
    return z_ji;
      };
  // compute density estimate based on data previously fed into the class using addDoc
  bool densityEst(uint32_t K0=10, uint32_t T0=10, uint32_t It=10)
  {
    if(mX.size() > 0)
    {
      mZ = densityEst(mX,K0,T0,It);
      return true;
    }else{
      return false;
    }
  };

  // interface mainly for python
  uint32_t addDoc(const Mat<U>& x_i)
  {
    mX.push_back(x_i);
    return mX.size();
  };

  // interface mainly for python
  uint32_t addHeldOut(const Mat<U>& x_i)
  {
    //cout<<"added heldout doc with "<<x_i.size()<<" words"<<endl;
    mX_ho.push_back(x_i);
    return mX_ho.size();
  };

  // after computing the labels we can use this to get them.
  bool getClassLabels(Col<uint32_t>& z_i, uint32_t i)
  {
    if(mZ.size() > 0 && i < mZ.size())
    {
      z_i=mZ[i];
      return true;
    }else{
      return false;
    }
  };

  double mOmega;

protected:

  vector<Mat<U> > mX; // training data
  vector<Mat<U> > mX_ho; //  held out data
  vector<Col<uint32_t> > mZ;

private:

  Mat<U> getXinK(const vector<Mat<U> >& x, uint32_t j_x, uint32_t i_x, uint32_t k, 
      const vector<Col<uint32_t> >& k_jt, const vector<Col<uint32_t> >& t_ji, bool disp=false) const
          {
    uint32_t J = k_jt.size();
    uint32_t d = x[0].n_cols;
    Mat<U> x_k=zeros<Mat<U> >(1,d); // datapoints in cluster k
#ifndef NDEBUG
    if(disp) printf("----------- J=%d; d=%d; j_x=%d; i_x=%d; k=%d; ----------- \n",J,d,j_x,i_x,k);
#endif
    for (uint32_t j=0; j<J; ++j)
    {
      Col<uint32_t> T_k = find(k_jt[j] == k);
#ifndef NDEBUG
      if(disp) cout<<"k="<<k<<" k_jt[j]="<<k_jt[j].t();
      if(disp) cout<<"@j="<<j<<" T_k="<<T_k.t();
#endif
      for (uint32_t i=0; i < T_k.n_elem; ++i)
      { // for all tables with the dish k_jt
        uvec id=find(t_ji[j] == T_k(i));
        //uint32_t inThere=sum(id==i_x)
        if(j_x==j)
        {
          Col<uint32_t> i_x_id=find(id==i_x);
#ifndef NDEBUG
          if(disp) cout<<"T_k(i)="<<T_k(i)<<"; i_x_id="<<i_x_id.t();
#endif
          if(i_x_id.n_elem > 0)
            id.shed_row(i_x_id(0)); // make sure, that that j_x and i_x are not in the x_k
        }
#ifndef NDEBUG
        if(disp) cout<<"t_ji[j]="<<t_ji[j].t();
        if(disp) cout<<"id="<<id.t();
        if(disp) cout<<"x_k_before="<<x_k.t()<<endl;
#endif
        uint32_t offset=x_k.n_rows;
        x_k.resize(x_k.n_rows+id.n_elem, x_k.n_cols);
        for (uint32_t i=0; i<id.n_elem; ++i)
          x_k.row(offset+i) = x[j].row(id(i)); // append all datapoints which are sitting at a table with dish k
#ifndef NDEBUG
        if(disp) cout<<"x_k_after="<<x_k.t()<<endl;
#endif
      }
    }
    x_k.shed_row(0); // remove first row of zeros
#ifndef NDEBUG
    if(disp) cout<<x_k.t()<<endl;
#endif
    return x_k;
          };
};

class HDP_onl : public HDP<uint32_t>
{
public:

  HDP_onl(const BaseMeasure<uint32_t>& base, double alpha, double omega)
  : HDP<uint32_t>(base, alpha, omega), mT(0), mK(0), mNw(0)
  {};
  
  ~HDP_onl()
  {};

  double digamma(double x)
  {
    //http://en.wikipedia.org/wiki/Digamma_function#Computation_and_approximation
    if(x<1e-50){
      //cerr<<"\tdigamma param x near zero: "<<x<<" cutting of"<<endl;
      x=1e-50;
    }
    //double x_sq = x*x;
    //return log(x)-1.0/(2.0*x)-1.0/(12.0*x_sq)+1.0/(12*x_sq*x_sq)-1.0/(252.0*x_sq*x_sq*x_sq);
    return boost::math::digamma(x);
  }

  double ElogBeta(const Mat<double>& lambda, uint32_t k, uint32_t w_dn)
  {
    //if(lambda[k](w_dn)<1e-6){
    //  cout<<"\tlambda[k]("<<w_dn<<") near zero: "<<lambda[k](w_dn)<<endl;
    //}
    return digamma(lambda(k,w_dn)) - digamma(sum(lambda.row(k)));
  }

  double ElogSigma(const Mat<double>& a, uint32_t k)
  {
    double e=digamma(a(k,0)) - digamma(a(k,0) + a(k,1));
    for (uint32_t l=0; l<k; ++l)
      e+=digamma(a(k,1)) - digamma(a(k,0) + a(k,1));
    return e; 
  }


  //bool normalizeLogDistribution(Row<double>& r)
  bool normalizeLogDistribution(arma::subview_row<double> r)
  {
    //r.row(i)=exp(r.row(i));
    //cout<<" r="<<r<<endl;
    double minR = as_scalar(min(r));
    //cout<<" minR="<<minR<<endl;
    if(minR > -100.0) {
      //cout<<" logDenom="<<sum(exp(r),1)<<endl;
      double denom = as_scalar(sum(exp(r),1));
      //cout<<" logDenom="<<denom<<endl;
      r -= log(denom); // avoid division by 0
      //cout<<" r - logDenom="<<r<<endl;
      r = exp(r);
      r /= sum(r);
      //cout<<" exp(r - logDenom)="<<r<<endl;
      return true;
    }else{ // cannot compute this -> set the smallest r to 1.0 and the rest to 0
      double maxR = as_scalar(max(r));
      //cout<<"maxR="<<maxR<<" <-" <<arma::max(r) <<endl;
      uint32_t kMax=as_scalar(find(r==maxR,1));
      //cout<<"maxR="<<maxR<<" kMax="<<kMax<<" <-" <<arma::max(r) <<endl;
      r.zeros();
      r(kMax)=1.0;
      //cout<<" r ="<<r<<endl;
      return false;
    }
  }

//  double ElogSigma(const vector<double>& a, const vector<double>& b, uint32_t k)
//  {
//    double e=digamma(a[k]) - digamma(a[k] + b[k]);
//    for (uint32_t l=0; l<k; ++l)
//      e+=digamma(b[k]) - digamma(a[k] + b[k]);
//    return e; 
//  };
//
  void initZeta(Mat<double>& zeta, const Mat<double>& lambda, const Mat<uint32_t>& x_d)
  {
    uint32_t N = x_d.n_rows;
    uint32_t T = zeta.n_rows;
    uint32_t K = zeta.n_cols;
      //cerr<<"\tinit zeta"<<endl;
      for (uint32_t i=0; i<T; ++i) {
        for (uint32_t k=0; k<K; ++k) {
          zeta(i,k)=0.0;
          for (uint32_t n=0; n<N; ++n) {
            //if(i==0 && k==0) cout<<zeta(i,k)<<" -> ";
            zeta(i,k) += ElogBeta(lambda, k, x_d(n));
          }
        }
        normalizeLogDistribution(zeta.row(i));
//        if(normalizeLogDistribution(zeta.row(i)))
//        {
//          cerr<<"zeta normally computed"<<endl;
//        }else{
//          cerr<<"zeta thresholded"<<endl;
//        }

        //cout<<" normalized="<<zeta(0,0)<<endl;
      }
      //cerr<<"zeta>"<<endl<<zeta<<"<zeta"<<endl;
      //cerr<<"normalization check:"<<endl<<sum(zeta,1).t()<<endl; // sum over rows
  };

  void initPhi(Mat<double>& phi, const Mat<double>& zeta, const Mat<double>& lambda, const Mat<uint32_t>& x_d)
  {
    uint32_t N = x_d.n_rows;
    uint32_t T = zeta.n_rows;
    uint32_t K = zeta.n_cols;
    //cout<<"\tinit phi"<<endl;
    for (uint32_t n=0; n<N; ++n){
      for (uint32_t i=0; i<T; ++i) {
        phi(n,i)=0.0;
        for (uint32_t k=0; k<K; ++k) {
          phi(n,i)+=zeta(i,k)* ElogBeta(lambda, k, x_d(n));
        }
      }
      normalizeLogDistribution(phi.row(n));
      //        if(normalizeLogDistribution(phi.row(n)))
      //        {
      //          cerr<<"phi normally computed"<<endl;
      //        }else{
      //          cerr<<"phi thresholded"<<endl;
      //        }
      //
      //        phi.row(n)=exp(phi.row(n));
      //        double denom = sum(phi.row(n));
      //        if(denom > EPS)
      //        {
      //          phi.row(n)/=denom; // avoid division by 0
      //        }else{
      //          cout<<"Phi Init: denominator too small -> no division!
      //        }
    }
    //cerr<<"phi>"<<endl<<phi<<"<phi"<<endl;
  };

  void updateGamma(Mat<double>& gamma, const Mat<double>& phi)
  {
    uint32_t N = phi.n_rows;
    uint32_t T = phi.n_cols;

    gamma.ones();
    gamma.col(1) *= mAlpha;
    for (uint32_t i=0; i<T; ++i) 
    {
      for (uint32_t n=0; n<N; ++n){
        gamma(i,0) += phi(n,i);
        for (uint32_t j=i+1; j<T; ++j) {
          gamma(i,1) += phi(n,j);
        }
      }
    }
    //cout<<gamma.t()<<endl;
  };

  void updateZeta(Mat<double>& zeta, const Mat<double>& phi, const Mat<double>& a, const Mat<double>& lambda, const Mat<uint32_t>& x_d)
  {
    uint32_t N = x_d.n_rows;
    uint32_t T = zeta.n_rows;
    uint32_t K = zeta.n_cols;

    for (uint32_t i=0; i<T; ++i){
      //zeta(i,k)=0.0;
      for (uint32_t k=0; k<K; ++k) {
        zeta(i,k) = ElogSigma(a,k);
        //cout<<zeta(i,k)<<endl;
        for (uint32_t n=0; n<N; ++n){
          zeta(i,k) += phi(n,i)*ElogBeta(lambda,k,x_d(n));
        }
      }
      normalizeLogDistribution(zeta.row(i));
      //          if(normalizeLogDistribution(zeta.row(i)))
      //          {
      //            cerr<<"zeta normally computed"<<endl;
      //          }else{
      //            cerr<<"zeta thresholded"<<endl;
      //          }

      //          zeta.row(i)=exp(zeta.row(i));
      //          double denom = sum(zeta.row(i));
      //          if(denom > EPS) zeta.row(i)/=denom; // avoid division by 0
    }
  }


  void updatePhi(Mat<double>& phi, const Mat<double>& zeta, const Mat<double>& gamma, const Mat<double>& lambda, const Mat<uint32_t>& x_d)
  {
    uint32_t N = x_d.n_rows;
    uint32_t T = zeta.n_rows;
    uint32_t K = zeta.n_cols;

    for (uint32_t n=0; n<N; ++n){
      //phi(n,i)=0.0;
      for (uint32_t i=0; i<T; ++i) {
        phi(n,i) = ElogSigma(gamma,i);
        for (uint32_t k=0; k<K; ++k) {
          phi(n,i) += zeta(i,k)*ElogBeta(lambda,k,x_d(n)) ;
        }
      }
      normalizeLogDistribution(phi.row(n));
      //          if(normalizeLogDistribution(phi.row(n)))
      //          {
      //            cerr<<"phi normally computed"<<endl;
      //          }else{
      //            cerr<<"phi thresholded"<<endl;
      //          }
      //          phi.row(n)=exp(phi.row(n));
      //          double denom = sum(phi.row(n));
      //          if(denom > EPS) phi.row(n)/=denom; // avoid division by 0
    }
  }

  void computeNaturalGradients(Mat<double>& d_lambda, Mat<double>& d_a, const Mat<double>& zeta, const Mat<double>&  phi, double omega, uint32_t D, const Mat<uint32_t>& x_d)
  {
    uint32_t N = x_d.n_rows;
    uint32_t Nw = d_lambda.n_cols;
    uint32_t T = zeta.n_rows;
    uint32_t K = zeta.n_cols;

    d_lambda.zeros();
    d_a.zeros();
    for (uint32_t k=0; k<K; ++k) { // for all K corpus level topics
      for (uint32_t i=0; i<T; ++i) {
        Row<double> _lambda(Nw); _lambda.zeros();
        for (uint32_t n=0; n<N; ++n){
          _lambda(x_d(n)) += phi(n,i);
        }
        d_lambda.row(k) += zeta(i,k) * _lambda;
        d_a(k,0) += zeta(i,k);
        for (uint32_t l=k+1; l<K; ++l) {
          d_a(k,1) += zeta(i,l);
        }
      }
      d_lambda.row(k) = D*d_lambda.row(k);
      //cout<<"lambda-nu="<<d_lambda[k].t()<<endl;
      d_lambda.row(k) += ((Dir*)(&mH))->mAlphas;
      //cout<<"lambda="<<d_lambda[k].t()<<endl;
      d_a(k,0) = D*d_a(k,0)+1.0;
      d_a(k,1) = D*d_a(k,1)+omega;
    }
  }


  // method for "one shot" computation without storing data in this class
  //  Nw: number of different words
  //  kappa: forgetting rate
  //  uint32_t T=10; // truncation on document level
  //  uint32_t K=100; // truncation on corpus level
  vector<Col<uint32_t> > densityEst(const vector<Mat<uint32_t> >& x, uint32_t Nw, 
      double kappa=0.75, uint32_t K=100, uint32_t T=10)
  {

    // From: Online Variational Inference for the HDP
    // TODO: think whether it makes sense to use columns and Mat here
    //double mAlpha = 1;
    //double mGamma = 1;
    //double nu = ((Dir*)(&mH))->mAlpha0; // get nu from vbase measure (Dir)
    //double kappa = 0.75;

    // T-1 gamma draws determine a T dim multinomial
    // T = T-1;

    mT = T;
    mK = K;
    mNw = Nw;

    Mat<double> a(K,2);
    a.ones();
    a.col(1) *= mOmega; 
    uint32_t D=x.size();

    // initialize lambda
    GammaRnd gammaRnd(1.0,1.0);
    Mat<double> lambda(K,Nw);
    for (uint32_t k=0; k<K; ++k){
      for (uint32_t w=0; w<Nw; ++w) lambda(k,w) = gammaRnd.draw();
      lambda.row(k) *= double(D)*100.0/double(K*Nw);
      lambda.row(k) += ((Dir*)(&mH))->mAlphas;
    }

    mZeta.resize(D,Mat<double>());
    mPhi.resize(D,Mat<double>());
    mGamma.resize(D,Mat<double>());
    mPerp.set_size(D);

    vector<Col<uint32_t> > z_dn(D);
    Col<uint32_t> ind = shuffle(linspace<Col<uint32_t> >(0,D-1,D),0);
    #pragma omp parallel for ordered schedule(dynamic) 
    for (uint32_t dd=0; dd<D; ++dd)
    {
      uint32_t d=ind[dd];  
      uint32_t N=x[d].n_rows;
//      cout<<"---------------- Document "<<d<<" N="<<N<<" -------------------"<<endl;
      cout<<"----------------- dd="<<dd<<" -----------------"<<endl;
//      cout<<"a=\t"<<a.t();
//      for (uint32_t k=0; k<K; ++k)
//      {
//        cout<<"@"<<k<<" lambda=\t"<<lambda.row(k);
//      }
//
      Mat<double> zeta(T,K);
      initZeta(zeta,lambda,x[d]);
      Mat<double> phi(N,T);
      initPhi(phi,zeta,lambda,x[d]);

      // ------------------------ doc level updates --------------------
      bool converged = false;
      Mat<double> gamma(T,2);
      Mat<double> gamma_prev(T,2);
      gamma_prev.ones();
      gamma_prev.col(1) += mAlpha;

      uint32_t o=0;
      while(!converged){
//        cout<<"-------------- Iterating local params #"<<o<<" -------------------------"<<endl;
        updateGamma(gamma,phi);
        updateZeta(zeta,phi,a,lambda,x[d]);
        updatePhi(phi,zeta,gamma,lambda,x[d]);

        converged = (accu(gamma_prev != gamma))==0 || o>60 ;

        gamma_prev = gamma;
        //cout<<"zeta>"<<endl<<zeta<<"<zeta"<<endl;
        //cout<<"phi>"<<endl<<phi<<"<phi"<<endl;
        ++o;
      }

      mZeta[d] = Mat<double>(zeta);
      mPhi[d] = Mat<double>(phi);
      mGamma[d] = Mat<double>(gamma);

//      cout<<"z_dn: "<<endl;
//      z_dn[d].set_size(N);
//      for (uint32_t n=0; n<N; ++n){
//        uint32_t z=as_scalar(find(phi.row(n) == max(phi.row(n)),1));
//        uint32_t c_di=as_scalar(find(zeta.row(z) == max(zeta.row(z)),1));
//        z_dn[d](n) = c_di;
//        cout<<z_dn[d](n)<<" ("<<max(phi.row(n))<<"@"<< z<<" -> "<<max(zeta.row(z))<<"@"<<c_di<<") "<<endl ;
//        if (max(phi.row(n))>1.0)
//          cout<<phi.row(n)<<endl;
//      }; cout<<endl;
  
      //cerr<<"zeta>"<<endl<<zeta<<"<zeta"<<endl;
      //cerr<<"phi>"<<endl<<phi<<"<phi"<<endl;

//      cout<<" --------------------- natural gradients --------------------------- "<<endl;
//      cout<<"\tD="<<D<<" omega="<<mOmega<<endl;
//
      Mat<double> d_lambda(K,Nw);
      Mat<double> d_a(K,2); 
      computeNaturalGradients(d_lambda, d_a, zeta, phi, mOmega, D, x[d]);

      //cout<<"delta a= "<<d_a.t()<<endl;
      //for (uint32_t k=0; k<K; ++k)
      //  cout<<"delta lambda_"<<k<<" min="<<min(d_lambda.row(k))<<" max="<< max(d_lambda.row(k))<<" #greater 0.1="<<sum(d_lambda.row(k)>0.1)<<endl;

      // ----------------------- update global params -----------------------
      #pragma omp ordered
      {
        cout<<" ------------------- global parameter updates dd="<<dd<<" d="<<d<<" ---------------"<<endl;
        double ro = exp(-kappa*log(1+double(dd+1)));
        //cout<<"\tro="<<ro<<endl;
        lambda = (1.0-ro)*lambda + ro*d_lambda;
        a = (1.0-ro)*a + ro*d_a;

        mA=a;
        mLambda = lambda;

        mPerp[d] = 0.0;
        if (dd > 10){
          cout<<"computing "<<mX_ho.size()<<" perplexities"<<endl;
          for (uint32_t i=0; i<mX_ho.size(); ++i)
          {
            double perp_i =  perplexity(mX_ho[i],dd,ro); //perplexity(mX_ho[i], mZeta[d], mPhi[d], mGamma[d], lambda);
            cout<<"perp_"<<i<<"="<<perp_i<<endl;
            mPerp[d] += perp_i;
          }
          mPerp[d] /= double(mX_ho.size());
          //cout<<"Perplexity="<<mPerp[d]<<endl;
        }else{
          cout<<"skipping "<<dd<<endl;
        }
      }
    }


    return z_dn;
  };

    // compute density estimate based on data previously fed into the class using addDoc
  bool densityEst(uint32_t Nw, double kappa=0.75, uint32_t K=100, uint32_t T=10)
  {
    if(mX.size() > 0)
    {
      mZ = densityEst(mX,Nw,kappa,K,T);
      return true;
    }else{
      return false;
    }
  };

  // after an initial densitiy estimate has been made using densityEst()
  // can use this to update the estimate with information from additional x 
bool  updateEst(const Mat<uint32_t>& x, double kappa=0.75)
{
  if (mX.size() > 0 && mX.size() == mPhi.size()) { // this should indicate that there exists a estimate already
    uint32_t N = x.n_rows;
    uint32_t T = mT; //mZeta[0].n_rows;
    uint32_t K = mK; //mZeta[0].n_cols;
    mX.push_back(x);
    mZeta.push_back(Mat<double>(T,K));
    mPhi.push_back(Mat<double>(N,T));
    //    mZeta.set_size(T,K);
    //    mPhi.set_size(N,T);
    mGamma.push_back(Mat<double>(T,2));
    uint32_t d = mX.size()-1;
    mPerp.resize(d+1);

    if(updateEst(mX[d],mZeta[d],mPhi[d],mGamma[d],mA,mLambda,mOmega,d,kappa))
    {
      mPerp[d] = 0.0;
      for (uint32_t i=0; i<mX_ho.size(); ++i)
        mPerp[d] += perplexity(mX_ho[i], mZeta[d], mPhi[d], mGamma[d], mLambda);
      mPerp[d] /= double(mX_ho.size());
      cout<<"Perplexity="<<mPerp[d]<<endl;
      return true; 
    }else{
      return false;
    } 
  }else{
    return false;
  }
};

// compute the perplexity of a given document x
double perplexity(const Mat<uint32_t>& x, uint32_t d, double kappa=0.75)
{
  if (mX.size() > 0 && mX.size() == mPhi.size()) { // this should indicate that there exists a estimate already
    uint32_t N = x.n_rows;
    //uint32_t Nw = mLambda.n_cols;
    uint32_t T = mT; //mZeta[0].n_rows; // TODO: most likely seg fault because of this
    uint32_t K = mK; //mZeta[0].n_cols;

    Mat<double> zeta = Mat<double>(T,K);
    Mat<double> phi = Mat<double>(N,T);
    Mat<double> gamma = Mat<double>(T,2);
    //uint32_t d = mX.size()-1;

    Mat<double> a = mA; // TODO: make deep copy here!
    Mat<double> lambda = mLambda;
    double omega = mOmega;
    
    //cout<<"updating copied model with x"<<endl;
    updateEst(x,zeta,phi,gamma,a,lambda,omega,d,kappa);
    //cout<<"computing perplexity under updated model"<<endl;

    return perplexity(x, zeta, phi, gamma, lambda);
  }else{
    return 1.0/0.0;
  }
};

// compute the perplexity given a document x and the model paremeters of it (after incorporating x)
double perplexity(const Mat<uint32_t>& x, Mat<double>& zeta, Mat<double>& phi, Mat<double>& gamma, Mat<double>& lambda)
{
    //cout<<"Computing Perplexity"<<endl;
    uint32_t N = x.n_rows;
    uint32_t T = mT; //mZeta[0].n_rows;
    // find most likely pi_di and c_di
    Col<double> pi;
    Col<double> sigPi; 
    Col<uint32_t> c(T);
    getDocTopics(pi, sigPi, c, gamma, zeta);
    // find most likely z_dn
    Col<uint32_t> z(N);
    getWordTopics(z, phi);
    // find most likely topics 
    Mat<double> topics;

    //cout<<" lambda.shape="<<lambda.n_rows<<" "<<lambda.n_cols<<endl;
    getCorpTopic(topics, lambda);

    double perp = 0.0;
    for (uint32_t n=0; n<x.n_rows; ++n){
      //cout<<"c_z_n = "<<c[z[n]]<<" z_n="<<z[n]<<" n="<<n<<" N="<<x.n_rows<<" x_n="<<x[n]<<" topics.shape="<<topics.n_rows<<" "<<topics.n_cols<<endl;
      perp -= logCat(x[n],topics.row(c[z[n]]));
    } 
    perp /= double(x.n_elem);
    perp /= log(2.0); // since it is log base 2 in the perplexity formulation!
    perp = pow(2.0,perp);

//        return logCat(self.x[d][n], self.beta[ self.c[d][ self.z[d][n]]]) \
//    + logCat(self.c[d][ self.z[d][n]], self.sigV) \
//    + logBeta(self.v, 1.0, self.omega) \
//    + logCat(self.z[d][n], self.sigPi[d]) \
//    + logBeta(self.pi[d], 1.0, self.alpha) \
//    + logDir(self.beta[ self.c[d][ self.z[d][n]]], self.Lambda)
//
   
    return perp;
}

  
bool updateEst(const Mat<uint32_t>& x, Mat<double>& zeta, Mat<double>& phi, Mat<double>& gamma, Mat<double>& a, Mat<double>& lambda, double omega, uint32_t d, double kappa= 0.75)
{
    uint32_t D = d+1; // assume that doc d is appended to the end  
    //uint32_t N = x.n_rows;
    uint32_t Nw = lambda.n_cols;
    uint32_t T = zeta.n_rows;
    uint32_t K = zeta.n_cols;

//    cout<<"---------------- Document "<<d<<" N="<<N<<" -------------------"<<endl;
//    cout<<"a=\t"<<a.t();
//    for (uint32_t k=0; k<K; ++k)
//    {
//      cout<<"@"<<k<<" lambda=\t"<<lambda.row(k);
//    }

    initZeta(zeta,lambda,x);
    initPhi(phi,zeta,lambda,x);

    // ------------------------ doc level updates --------------------
    bool converged = false;
    Mat<double> gamma_prev(T,2);
    gamma_prev.ones();

    uint32_t o=0;
    while(!converged){
//      cout<<"-------------- Iterating local params #"<<o<<" -------------------------"<<endl;
      updateGamma(gamma,phi);
      updateZeta(zeta,phi,a,lambda,x);
      updatePhi(phi,zeta,gamma,lambda,x);

      converged = (accu(gamma_prev != gamma))==0 || o>60 ;

      gamma_prev = gamma;
      //cout<<"zeta>"<<endl<<zeta<<"<zeta"<<endl;
      //cout<<"phi>"<<endl<<phi<<"<phi"<<endl;
      ++o;
    }

//    cout<<" --------------------- natural gradients --------------------------- "<<endl;
//    cout<<"\tD="<<D<<" omega="<<omega<<endl;

    Mat<double> d_lambda(K,Nw);
    Mat<double> d_a(K,2); 
    computeNaturalGradients(d_lambda, d_a, zeta, phi, omega, D, x);

//    cout<<" ------------------- global parameter updates: ---------------"<<endl;
    //cout<<"delta a= "<<d_a.t()<<endl;
    //for (uint32_t k=0; k<K; ++k)
    //  cout<<"delta lambda_"<<k<<" min="<<min(d_lambda.row(k))<<" max="<< max(d_lambda.row(k))<<" #greater 0.1="<<sum(d_lambda.row(k)>0.1)<<endl;

    // ----------------------- update global params -----------------------
    double ro = exp(-kappa*log(1+double(d+1)));
//    cout<<"\tro="<<ro<<endl;
    lambda = (1.0-ro)*lambda + ro*d_lambda;
    a = (1.0-ro)*a+ ro*d_a;

  return true;
};

  void getA(Col<double>& a)
  {
    a=mA.col(0);
  };

  void getB(Col<double>& b)
  {
    b=mA.col(1);
  };

  bool getLambda(Col<double>& lambda, uint32_t k)
  {
    if(mLambda.n_rows > 0 && k < mLambda.n_rows)
    {
      lambda=mLambda.row(k).t();
      return true;
    }else{
      return false;
    }
  };


  static void betaMode(Col<double>& v, const Col<double>& alpha, const Col<double>& beta)
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
  static void stickBreaking(Col<double>& prop, const Col<double>& v)
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

  static uint32_t multinomialMode(const Row<double>& p )
  {
    uint32_t ind =0;
    p.max(ind);
    return ind;
  };

static void dirMode(Row<double>& mode, const Row<double>& alpha)
{
  mode = (alpha-1.0)/sum(alpha-1.0);
};

static void dirMode(Col<double>& mode, const Col<double>& alpha)
{
  mode = (alpha-1.0)/sum(alpha-1.0);
};

  bool getDocTopics(Col<double>& pi, Col<double>& sigPi, Col<uint32_t>& c, uint32_t d)
  {
    return getDocTopics(pi,sigPi,c,mGamma[d],mZeta[d]);
  };

  bool getDocTopics(Col<double>& pi, Col<double>& sigPi, Col<uint32_t>& c, const Mat<double>& gamma, const Mat<double>& zeta)
  {
    uint32_t T = gamma.n_rows; // doc level topics

    sigPi.set_size(T+1);
    pi.set_size(T);
    c.set_size(T);

    //cout<<"K="<<K<<" T="<<T<<endl;
    betaMode(pi,gamma.col(0),gamma.col(1));
    stickBreaking(sigPi,pi);
    //cout<<"pi="<<pi<<endl;
    //cout<<"sigPi="<<sigPi<<endl;
    //cout<<"mGamma="<<mGamma[d]<<endl;
    for (uint32_t i=0; i<T; ++i){
      c[i] = multinomialMode(zeta.row(i));
    }
    return true;
  };


  bool getWordTopics(Col<uint32_t>& z, uint32_t d){
    return getWordTopics(z,mPhi[d]);
  };

  bool getWordTopics(Col<uint32_t>& z, const Mat<double>& phi){
    z.set_size(phi.n_rows);
    for (uint32_t i=0; i<z.n_elem; ++i){
      z[i] = multinomialMode(phi.row(i));
    }
    return true;
  };

  bool getCorpTopicProportions(Col<double>& v, Col<double>& sigV)
  {
    return getCorpTopicProportions(v,sigV,mA);
  };

  // a are the parameters of the beta distribution from which v is drawn
  bool getCorpTopicProportions(Col<double>& v, Col<double>& sigV, const Mat<double>& a)
  {
    uint32_t K = a.n_rows; // corp level topics

    sigV.set_size(K+1);
    v.set_size(K);

    betaMode(v, a.col(0), a.col(1));
    stickBreaking(sigV,v);
    return true;
  };


  bool getCorpTopic(Col<double>& topic, uint32_t k)
  {
    if(mLambda.n_rows > 0 && k < mLambda.n_rows)
    {
      return getCorpTopic(topic,mLambda.row(k));
    }else{
      return false;
    }
  };

  bool getCorpTopic(Col<double>& topic, const Row<double>& lambda)
  {
      // mode of dirichlet (MAP estimate)
      dirMode(topic, lambda.t());
      return true;
  };
  
bool getCorpTopic(Mat<double>& topics, const Mat<double>& lambda)
{
  uint32_t K = lambda.n_rows;
  uint32_t Nw = lambda.n_cols;
  topics.set_size(K,Nw);
  for (uint32_t k=0; k<K; k++){
    // mode of dirichlet (MAP estimate)

    topics.row(k) = (lambda.row(k)-1.0)/sum(lambda.row(k)-1.0);
    //dirMode(topics.row(k), lambda.row(k));
  }
  return true;
};


  // cateorical distribution (Multionomial for one word)
  static double Cat(uint32_t x, Row<double> pi)
  {
    assert(x<pi.n_elem);
    double p = pi[x];
    for (uint32_t i=0; i< pi.n_elem; ++i)
      if (i!=x) p*=(1.0 - pi[i]);
    return p;
  };

  // log cateorical distribution (Multionomial for one word)
  static double logCat(uint32_t x, Row<double> pi)
  {
    assert(x<pi.n_elem);
    double p = log(pi[x]);
    for (uint32_t i=0; i< pi.n_elem; ++i)
      if (i!=x) p += log(1.0 - pi[i]);
    return p;
  };

  static double Beta(double x, double alpha, double beta)
  {
    return (1.0/boost::math::beta(alpha,beta)) * pow(x,alpha-1.0) * pow(1.0-x,beta-1.0);
  };

static double betaln(double alpha, double beta)
{
  return -boost::math::lgamma(alpha+beta) + boost::math::lgamma(alpha) + boost::math::lgamma(beta);
};

static double logBeta(double x, double alpha, double beta)
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

  static double logDir(const Row<double>& x, const Row<double>& alpha)
  { 
    assert(alpha.n_elem == x.n_elem);
    double logP=boost::math::lgamma(sum(alpha));
    for (uint32_t i=0; i<alpha.n_elem; ++i){
      logP += -boost::math::lgamma(alpha[i]) + (alpha[i]-1.0)*log(x[i]);
    }
    return logP;
  };

  // eval probability of a word under the estimated model
  double logProb(uint32_t w)
  {
    // TODO : For now I dont need it in here -> do it in python
    return 0.0;
  };

protected:
  Mat<double> mLambda; // corpus level topics (Dirichlet)
  Mat<double> mA; // corpus level Beta process alpha parameter for stickbreaking
  //Col<double> mB; // corpus level Beta process beta parameter for stickbreaking
  vector<Mat<double> > mZeta; // document level topic indices/pointers to corpus level topics (Multinomial) 
  vector<Mat<double> > mPhi; // document level word to doc level topic assignment (Multinomial)
  vector<Mat<double> > mGamma; // document level Beta distribution alpha parameter for stickbreaking
  //vector<Col<double> > mGammaB; // document level Beta distribution beta parameter for stickbreaking

  Col<double> mPerp; // perplexity for each document

  uint32_t mT; // Doc level truncation
  uint32_t mK; // Corp level truncation
  uint32_t mNw; // size of dictionary

};

