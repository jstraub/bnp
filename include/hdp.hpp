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
  HDP(const BaseMeasure<U>& base, double alpha, double gamma)
  : DP<U>(base, alpha), mGamma(gamma)
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
            l(T[j]+k) = log(this->mAlpha*m_k/((n_j+this->mAlpha)*(m_+mGamma))) + f_k; // TODO: shouldnt this be mAlpha of the posterior hdp?
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
          l[T[j]+K] = log(this->mAlpha*mGamma/((n_j+this->mAlpha)*(m_+mGamma))) + f_knew;

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
            l(k)=log(m_k/(m_+mGamma)) + f_k;
          }
          double f_knew=0.0;
          for (uint32_t i=0; i<x_jt.n_rows; ++i) // product over independent x_ji_t
            f_knew += this->mH.predictiveProb(x_jt.row(i).t());
          //logGaus(x_jt.row(i).t(), mVtheta, mmCov());
          f(K)=f_knew;
          l(K)=log(mGamma/(m_+mGamma)) + f_knew; // update dish at table t in restaurant j
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

  double mGamma;

protected:

  vector<Mat<U> > mX;
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

  HDP_onl(const BaseMeasure<uint32_t>& base, double alpha, double gamma)
  : HDP<uint32_t>(base, alpha, gamma)
  {};
  
  ~HDP_onl()
  {};

  double digamma(double x)
  {
    //http://en.wikipedia.org/wiki/Digamma_function#Computation_and_approximation
    if(x<1e-50){
      cerr<<"\tdigamma param x near zero: "<<x<<" cutting of"<<endl;
      x=1e-50;
    }
    //double x_sq = x*x;
    //return log(x)-1.0/(2.0*x)-1.0/(12.0*x_sq)+1.0/(12*x_sq*x_sq)-1.0/(252.0*x_sq*x_sq*x_sq);
    return boost::math::digamma(x);
  }

  double ElogBeta(const vector<Col<double> >& lambda, uint32_t k, uint32_t w_dn)
  {
    //if(lambda[k](w_dn)<1e-6){
    //  cout<<"\tlambda[k]("<<w_dn<<") near zero: "<<lambda[k](w_dn)<<endl;
    //}
    return digamma(lambda[k](w_dn)) - digamma(sum(lambda[k]));
  }

  double ElogSigma(const Col<double>& a, const Col<double>& b, uint32_t k)
  {
    double e=digamma(a(k)) - digamma(a(k) + b(k));
    for (uint32_t l=0; l<k; ++l)
      e+=digamma(b(k)) - digamma(a(k) + b(k));
    return e; 
  }


  //bool normalizeLogDistribution(Row<double>& r)
  bool normalizeLogDistribution(arma::subview_row<double> r)
  {
    //r.row(i)=exp(r.row(i));
    double minR = as_scalar(min(r));
    if(minR > -100.0) {
      double denom = as_scalar(sum(exp(r)));
      //cout<<" logDenom="<<denom<<endl;
      r -= log(denom); // avoid division by 0
      //cout<<" r - logDenom="<<r<<endl;
      r = exp(r);
      r /= sum(r);
      //cout<<" exp(r - logDenom)="<<r<<endl;
      return true;
    }else{ // cannot compute this -> set the smallest r to 1.0 and the rest to 0
      double maxR = as_scalar(max(r));
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

  // method for "one shot" computation without storing data in this class
  //  Nw: number of different words
  //  kappa: forgetting rate
  //  uint32_t T=10; // truncation on document level
  //  uint32_t K=100; // truncation on corpus level
  vector<Col<uint32_t> > densityEst(const vector<Mat<uint32_t> >& x, uint32_t Nw, double kappa=0.75, uint32_t K=100, uint32_t T=10)
  {

    // From: Online Variational Inference for the HDP
    // TODO: think whether it makes sense to use columns and Mat here
    //double mAlpha = 1;
    //double mGamma = 1;
    //double nu = ((Dir*)(&mH))->mAlpha0; // get nu from vbase measure (Dir)
    //double kappa = 0.75;

    // T-1 gamma draws determine a T dim multinomial
    // T = T-1;

    Col<double> a(K); a.ones();
    Col<double> b(K); b.ones(); b*=mGamma;
    uint32_t D=x.size();

    // initialize lambda
    GammaRnd gammaRnd(1.0,1.0);
    vector<Col<double> > lambda(K,Col<double>(Nw));
    for (uint32_t k=0; k<K; ++k){
        gammaRnd.draw(lambda[k]);
        lambda[k]*=double(D)*100.0/double(K*Nw);
        lambda[k]+=((Dir*)(&mH))->mAlphas.t();
    }

    mZeta.resize(D,Mat<double>());
    mPhi.resize(D,Mat<double>());
    mGammaA.resize(D,Col<double>());
    mGammaB.resize(D,Col<double>());

    vector<Col<uint32_t> > z_dn(D);
    Col<uint32_t> ind = shuffle(linspace<Col<uint32_t> >(0,D-1,D),0);
    for (uint32_t dd=0; dd<D; ++dd)
    {
      uint32_t d=ind[dd];  
      uint32_t N=x[d].n_rows;
      cout<<"---------------- Document "<<d<<" N="<<N<<" -------------------"<<endl;
      cout<<"----------------- dd="<<dd<<" -----------------"<<endl;
      cout<<"a=\t"<<a.t();
      cout<<"b=\t"<<b.t();
      for (uint32_t k=0; k<K; ++k)
      {
        cout<<"@"<<k<<" lambda=\t"<<lambda[k].t();
      }

      cout<<"\tinit zeta"<<endl;
      Mat<double> zeta(T,K);
      for (uint32_t i=0; i<T; ++i) {
        for (uint32_t k=0; k<K; ++k) {
          zeta(i,k)=0.0;
          for (uint32_t n=0; n<N; ++n) {
            if(i==0 && k==0) cout<<zeta(i,k)<<" -> ";
            zeta(i,k) += ElogBeta(lambda, k, x[d](n));
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
      cerr<<"zeta>"<<endl<<zeta<<"<zeta"<<endl;
      cerr<<"normalization check:"<<endl<<sum(zeta,1).t()<<endl; // sum over rows
      cout<<"\tinit phi"<<endl;
      Mat<double> phi(N,T);
      for (uint32_t n=0; n<N; ++n){
        for (uint32_t i=0; i<T; ++i) {
          phi(n,i)=0.0;
          for (uint32_t k=0; k<K; ++k) {
            phi(n,i)+=zeta(i,k)* ElogBeta(lambda, k, x[d](n));
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
        cerr<<"phi>"<<endl<<phi<<"<phi"<<endl;

      // ------------------------ doc level updates --------------------
      bool converged = false;
      Col<double> gamma_1(T); 
      Col<double> gamma_2(T);
  
      Col<double> gamma_1_prev(T);
      Col<double> gamma_2_prev(T);
      gamma_1_prev.ones();
      gamma_2_prev.ones(); gamma_2_prev *= mAlpha;

      uint32_t o=0;
      while(!converged){
        cout<<"-------------- Iterating local params #"<<o<<" -------------------------"<<endl;
        // gammas -------------------------
        gamma_1.ones();
        gamma_2.ones(); gamma_2 *= mAlpha;
        for (uint32_t i=0; i<T; ++i) 
        {
          for (uint32_t n=0; n<N; ++n){
            gamma_1(i)+=phi(n,i);
            for (uint32_t j=i+1; j<T; ++j) {
              gamma_2(i) += phi(n,j);
            }
          }  
          cout<<gamma_1(i)<<" "<<gamma_2(i)<<endl;
        }
        // zeta --------------------------
        for (uint32_t i=0; i<T; ++i){
          //zeta(i,k)=0.0;
          for (uint32_t k=0; k<K; ++k) {
            zeta(i,k) = ElogSigma(a,b,k);
            //cout<<zeta(i,k)<<endl;
            for (uint32_t n=0; n<N; ++n){
              zeta(i,k) += phi(n,i)*ElogBeta(lambda,k, x[d](n));
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

        // phi ------------------------
        for (uint32_t n=0; n<N; ++n){
          //phi(n,i)=0.0;
          for (uint32_t i=0; i<T; ++i) {
            phi(n,i) = ElogSigma(gamma_1,gamma_2,i);
            for (uint32_t k=0; k<K; ++k) {
              phi(n,i) += zeta(i,k)*ElogBeta(lambda,k,x[d](n)) ;
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

        converged = ((sum(gamma_1_prev != gamma_1)==0) && (sum(gamma_2_prev != gamma_2)==0)) || o>60 ;
        
        //cout<<gamma_1_prev<<endl<<gamma_1<<endl<<" sum="<<sum(gamma_1_prev != gamma_1)<<endl;
        //cout<<gamma_2_prev<<endl<<gamma_2<<endl<<" sum="<<sum(gamma_2_prev != gamma_2)<<endl;

        gamma_1_prev = gamma_1;
        gamma_2_prev = gamma_2;
        //cout<<"zeta>"<<endl<<zeta<<"<zeta"<<endl;
        //cout<<"phi>"<<endl<<phi<<"<phi"<<endl;
        ++o;
      }

      mZeta[d] = Mat<double>(zeta);
      mPhi[d] = Mat<double>(phi);
      mGammaA[d] = Col<double>(gamma_1);
      mGammaB[d] = Col<double>(gamma_2);

      cout<<"z_dn: "<<endl;
      z_dn[d].set_size(N);
      for (uint32_t n=0; n<N; ++n){
        //cout<<phi.row(n);
        uint32_t z=as_scalar(find(phi.row(n) == max(phi.row(n)),1));
        uint32_t c_di=as_scalar(find(zeta.row(z) == max(zeta.row(z)),1));
        z_dn[d](n) = c_di;
        cout<<z_dn[d](n)<<" ("<<max(phi.row(n))<<"@"<< z<<" -> "<<max(zeta.row(z))<<"@"<<c_di<<") "<<endl ;
        if (max(phi.row(n))>1.0)
          cout<<phi.row(n)<<endl;
      }; cout<<endl;
  
      cerr<<"zeta>"<<endl<<zeta<<"<zeta"<<endl;
      cerr<<"phi>"<<endl<<phi<<"<phi"<<endl;

      cout<<" --------------------- natural gradients --------------------------- "<<endl;
      cout<<"\tD="<<D<<" gamma="<<mGamma<<endl;
      vector<Col<double> > lambda_kv(K,Col<double>(Nw));
      for (uint32_t k=0; k<K; ++k){
        lambda_kv[k].zeros();
      }
      Col<double> a_k(K); a_k.zeros();
      Col<double> b_k(K); b_k.zeros();
      for (uint32_t k=0; k<K; ++k) { // for all K corpus level topics
        for (uint32_t i=0; i<T; ++i) {
          Col<double> _lambda(Nw); _lambda.zeros();
          for (uint32_t n=0; n<N; ++n){
            _lambda(x[d](n))+=phi(n,i);
          }
          lambda_kv[k]+=zeta(i,k)*_lambda;
          a_k(k)+=zeta(i,k);
          for (uint32_t l=k+1; l<K; ++l) {
            b_k(k)+=zeta(i,l);
          }
        }
        lambda_kv[k] = D*lambda_kv[k];
        //cout<<"lambda-nu="<<lambda_kv[k].t()<<endl;
        lambda_kv[k] += ((Dir*)(&mH))->mAlphas.t();
        //cout<<"lambda="<<lambda_kv[k].t()<<endl;
        a_k(k) = D*a_k(k)+1.0;
        b_k(k) = D*b_k(k)+mGamma;
      }

      cout<<" ------------------- global parameter updates: ---------------"<<endl;
      cout<<"delta a= "<<a_k.t()<<endl;
      cout<<"delta b= "<<b_k.t()<<endl;
      for (uint32_t k=0; k<K; ++k)
        cout<<"delta lambda_"<<k<<" min="<<min(lambda_kv[k])<<" max="<< max(lambda_kv[k])<<" #greater 0.1="<<sum(lambda_kv[k]>0.1)<<endl;

      // ----------------------- update global params -----------------------
      double ro = exp(-kappa*log(1+double(dd+1)));
      cout<<"\tro="<<ro<<endl;

      for (uint32_t k=0; k<K; ++k)
        lambda[k] = (1.0-ro)*lambda[k] + ro*lambda_kv[k];
      a = (1.0-ro)*a + ro*a_k;
      b = (1.0-ro)*b + ro*b_k;

    }


    mA=a;
    mB=b;
    for (uint32_t k=0; k<K; ++k)
      mLambda.push_back(lambda[k]);

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

  void getA(Col<double>& a)
  {
    a=mA;
  };

  void getB(Col<double>& b)
  {
    b=mB;
  };

  bool getLambda(Col<double>& lambda, uint32_t k)
  {
    if(mLambda.size() > 0 && k < mLambda.size())
    {
      lambda=mLambda[k];
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

  bool getDocTopics(Col<double>& pi, Col<double>& sigPi, Col<uint32_t>& c, uint32_t d)
  {
    uint32_t K = mLambda.size(); // corp level topics
    uint32_t T = mGammaA[0].n_elem; // doc level topics

    sigPi.set_size(T+1);
    pi.set_size(T);
    c.set_size(T);

//    if (T != sigPi.n_elem || T != c.n_elem){
//      cerr<<"getDocTopics:: proportions was not properly preallocated "<<T<<"!="<<sigPi.n_elem<<" and "<<c.n_elem<<endl;
//      return false;
//    }

    betaMode(pi,mGammaA[d],mGammaB[d]);
    stickBreaking(sigPi,pi);
    for (uint32_t i=0; i<T; ++i){
      c[i] = multinomialMode(mZeta[d].row(i));
    }
    return true;
  }

  bool getWordTopics(Col<uint32_t>& z, uint32_t d){
    z.set_size(mX[d].n_elem);

    for (uint32_t i=0; i<z.n_elem; ++i){
      z[i] = multinomialMode(mPhi[d].row(i));
    }
    return true;
  };

  bool getCorpTopicProportions(Col<double>& v, Col<double>& sigV)
  {
    uint32_t K = mLambda.size(); // corp level topics

    sigV.set_size(K+1);
    v.set_size(K);

//    if (K != sigV.n_elem){
//      cerr<<"getCorpTopicProportions:: proportions was not properly preallocated "<<K<<" != "<<sigV.n_elem<<endl;
//      return false;
//    }

    betaMode(v, mA,mB);
    stickBreaking(sigV,v);
    return true;
  }

  bool getCorpTopic(Col<double>& topic, uint32_t k)
  {
    if(mLambda.size() > 0 && k < mLambda.size())
    {
      double norm = 1.0/(sum(mLambda[k]) - mLambda[k].n_elem);
      topic = (mLambda[k]-1)*norm;
      return true;
    }else{
      return false;
    }
  };

  // cateorical distribution (Multionomial for one word)
  static double Cat(Row<double> pi, uint32_t x)
  {
    assert(x<pi.n_elem);
    double p = pi[x];
    for (uint32_t i=0; i< pi.n_elem; ++i)
      if (i!=x) p*=(1.0 - pi[i]);
    return p;
  };

  // log cateorical distribution (Multionomial for one word)
  static double logCat(Row<double> pi, uint32_t x)
  {
    assert(x<pi.n_elem);
    double p = log(pi[x]);
    for (uint32_t i=0; i< pi.n_elem; ++i)
      if (i!=x) p += log(1.0 - pi[i]);
    return p;
  };

  static double Beta(double alpha, double beta, double x)
  {
    return (1.0/boost::math::beta(alpha,beta)) * pow(x,alpha-1.0) * pow(1.0-x,beta-1.0);
  };

  static double logBeta(double alpha, double beta, double x)
  {
    return boost::math::lgamma(alpha+beta) - boost::math::lgamma(alpha) 
      - boost::math::lgamma(beta) + log(x)*(alpha-1.0) + log(1.0-x)*(beta-1.0);
  };

  static double logDir(Row<double> alpha, Row<double> x)
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
  vector<Col<double> > mLambda; // corpus level topics (Dirichlet)
  Col<double> mA; // corpus level Beta process alpha parameter for stickbreaking
  Col<double> mB; // corpus level Beta process beta parameter for stickbreaking
  vector<Mat<double> > mZeta; // document level topic indices/pointers to corpus level topics (Multinomial) 
  vector<Mat<double> > mPhi; // document level word to doc level topic assignment (Multinomial)
  vector<Col<double> > mGammaA; // document level Beta distribution alpha parameter for stickbreaking
  vector<Col<double> > mGammaB; // document level Beta distribution beta parameter for stickbreaking

};

