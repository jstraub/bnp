/* Copyright (c) 2012, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See LICENSE.txt or 
 * http://www.opensource.org/licenses/mit-license.php */

#pragma once

#include "random.hpp"
#include "baseMeasure.hpp"
#include "hdp.hpp"
#include "probabilityHelpers.hpp"

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
class HDP_gibbs : public HDP<U>
{
  public:
    HDP_gibbs(const BaseMeasure<U>& base, double alpha, double omega)
      : HDP<U>(base, alpha, omega)
    {
      //    cout<<"Creating "<<typeid(this).name()<<endl;
    };

    ~HDP_gibbs()
    {	};


    // method for "one shot" computation without storing data in this class
    vector<Row<uint32_t> > densityEst(const vector<Mat<U> >& x, uint32_t Nw, uint32_t K0, uint32_t T0, uint32_t It)
    {
      mNw = Nw;

      RandDisc rndDisc;
      // x is a list of numpy arrays: one array per document
      uint32_t J=x.size(); // number of documents
      vector<uint32_t> N(J,0);
      for (uint32_t j=0; j<J; ++j)
        N.at(j)=int(x[j].n_cols); // number of datapoints in each document

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

      vector<Row<uint32_t> > z_ji(J);
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
              //cout<<"x_j="<<x[j].col(i)<<endl;
              //cout<<"x_k_ji="<<x_k_ji<<endl;
              double f_k_jt = this->mH.predictiveProb(x[j].col(i),x_k_ji);
              //cout<<"f_k_jt="<<f_k_jt<<endl;
              //logGaus(x[j].col(i), hdp_x_ji.mVtheta, hdp_x_ji.mmCov()); // marginal probability of x_ji in cluster k/dish k given all other data in that cluster
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
              double f_k =  this->mH.predictiveProb(x[j].col(i),x_k_ji);
              //logGaus(x[j].col(i), hdp_x_ji.mVtheta, hdp_x_ji.mmCov());  // marginal probability of x_ji in cluster k/dish k given all other data in that cluster
              f(T[j]+k) = f_k;
              l(T[j]+k) = log(this->mAlpha*m_k/((n_j+this->mAlpha)*(m_ + HDP<U>::mOmega))) + f_k; // TODO: shouldnt this be mAlpha of the posterior hdp?
            }
            // handle the case where x_ji sits at a new table with a new dish
            //        cout<<"ji=:"<<j<<" " <<i<<endl;
            //        cout<<"x_ji=:"<<x[j].col(i)<<endl;
            //        cout<<"mVtheta=:"<<mVtheta<<endl;
            //        cout<<"Cov=:"<<mmCov()<<endl;
            //
            double f_knew = this->mH.predictiveProb(x[j].col(i));
            //logGaus(x[j].col(i), mVtheta, mmCov());
            f[T[j]+K] = f_knew;
            l[T[j]+K] = log(this->mAlpha*HDP<U>::mOmega/((n_j+this->mAlpha)*(m_+HDP<U>::mOmega))) + f_knew;

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
            Mat<U> x_jt = zeros<Mat<U> >(x[j].n_rows,i_jt.n_elem);
            for (uint32_t i=0; i<i_jt.n_elem; ++i)
              x_jt.col(i) = x[j].col(i_jt(i)); //all datapoints which are sitting at table t 
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
              for (uint32_t i=0; i<x_jt.n_cols; ++i)
              { // product over independent x_ji_t
                Mat<U> x_k_ji = getXinK(x,j,i_jt(i),k,k_jt,t_ji); // for posterior computation
                //HDP hdp_x_ji = posterior(x_k_ji); // compute posterior hdp given the data in 
                f_k += this->mH.predictiveProb(x_jt.col(i),x_k_ji);
                //logGaus(x_jt.col(i), hdp_x_ji.mVtheta, hdp_x_ji.mmCov());
              }
              f(k)=f_k;
              l(k)=log(m_k/(m_ + HDP<U>::mOmega)) + f_k;
            }
            double f_knew=0.0;
            for (uint32_t i=0; i<x_jt.n_cols; ++i) // product over independent x_ji_t
              f_knew += this->mH.predictiveProb(x_jt.col(i));
            //logGaus(x_jt.col(i), mVtheta, mmCov());
            f(K)=f_knew;
            l(K)=log(HDP<U>::mOmega/(m_ + HDP<U>::mOmega)) + f_knew; // update dish at table t in restaurant j
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

      }
      mZ_ji = z_ji;
      mK = K;
      mT = T;

      computeTopics(); // compute the corpus level topic distributions from the labels z_ji

      return z_ji;
    };
    // compute density estimate based on data previously fed into the class using addDoc
    bool densityEst(uint32_t Nw, uint32_t K0, uint32_t T0, uint32_t It)
    {
      if(HDP<U>::mX.size() > 0)
      {
        if(HDP<U>::mX_te.size() > 0){
          // put the test docs mX_te into th normal docs and record their index
          mX_id_test.resize(HDP<U>::mX_te.size());
          for(uint32_t i=0; i<HDP<U>::mX_te.size(); ++i){
            mX_id_test[i]= HDP<U>::mX.size(); // helps locate the documents wich are trained in order to get a topic model
            HDP<U>::mX.push_back(HDP<U>::mX_te[i]);
          }
        }
        mZ_ji = densityEst(HDP<U>::mX,Nw,K0,T0,It);
        return true;
      }else{
        return false;
      }
    };

    // after computing the labels we can use this to get them.
    bool getClassLabels(Col<uint32_t>& z_i, uint32_t i)
    {
      if(mZ_ji.size() > 0 && i < mZ_ji.size())
      {
        z_i=mZ_ji[i];
        return true;
      }else{
        return false;
      }
    };

    Row<double> logP_w(uint32_t d) const
    {
      Row<double> logP(mNw);
      logP.zeros();
      uint32_t N = mZ_ji[d].n_elem;
      for (uint32_t i=0; i<N; ++i){
        logP[ HDP<U>::mX[d](i) ] += log(mBeta( mZ_ji[d](i), HDP<U>::mX[d](i)));
      }
      return logP;
    };


    /* compute the perplexity of all test docs after gibbs sampling is done
     */
    Row<double> perplexity(){
      mPerp.set_size(HDP<U>::mX_ho.size());
      for (uint32_t i=0; i<mX_id_test.size(); ++i){
        // iterate over all held out data and compute the perplexity
        uint32_t d=mX_id_test[i];
        Row<double> logP = logP_w(d);
        mPerp(i) = HDP<U>::perplexity(HDP<U>::mX_ho[i],logP);
      }
      return mPerp;
    };

  protected:

    Row<uint32_t> mX_id_test; //  the id of the half of the test data in mX

    vector<Row<uint32_t> > mZ_ji;
    uint32_t mNw; // number of different words 
    uint32_t mK;
    vector<uint32_t> mT;
    Mat<double> mBeta; // corpus level topics
    Row<double> mPerp; // perplexities of all test docs after sampling is finished

  private:

    //TODO: compute topics from the labeling
    void computeTopics(void)
    {
      mBeta.set_size(mK,mNw); // corpus level topics
      mBeta.zeros();
      uint32_t D=HDP<U>::mX.size(); // D=J -> number of documents in corpus; document d/j
      for (uint32_t d=0; d<D; ++d){
        uint32_t N = mZ_ji[d].n_elem;
        for (uint32_t i=0; i<N; ++i){
          assert(mZ_ji[d](i)<mK);
          assert(HDP<U>::mX[d](i)<mNw);
          mBeta(mZ_ji[d](i), HDP<U>::mX[d](i)) ++; // increment count for word x_di in topic z_di 
        }
      }
      //normalize
      for (uint32_t k=0; k<mK; ++k){
        mBeta.row(k) /= sum(mBeta.row(k));
      }
    };

    Mat<U> getXinK(const vector<Mat<U> >& x, uint32_t j_x, uint32_t i_x, uint32_t k, 
        const vector<Col<uint32_t> >& k_jt, const vector<Col<uint32_t> >& t_ji, bool disp=false) const
    {
      uint32_t J = k_jt.size();
      Mat<U> x_k; // datapoints in cluster k
#ifndef NDEBUG
      if(disp) printf("----------- J=%d; j_x=%d; i_x=%d; k=%d; ----------- \n",J,j_x,i_x,k);
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
          if(disp) cout<<"x_k_before="<<x_k<<endl;
#endif
          uint32_t offset=x_k.n_cols;
          x_k.resize(x[j].n_rows,x_k.n_cols+id.n_elem);
          for (uint32_t i=0; i<id.n_elem; ++i)
            x_k.col(offset+i) = x[j].col(id(i)); // append all datapoints which are sitting at a table with dish k
#ifndef NDEBUG
          if(disp) cout<<"x_k_after="<<x_k<<endl;
#endif
        }
      }
      //x_k.shed_col(0); // remove first row of zeros
#ifndef NDEBUG
      if(disp) cout<<x_k<<endl;
#endif
      return x_k;
    };
};

