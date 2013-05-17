/* Copyright (c) 2012, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See LICENSE.txt or 
 * http://www.opensource.org/licenses/mit-license.php */

#pragma once

#include "hdp.hpp"

#include "random.hpp"
#include "baseMeasure.hpp"
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
class DistriContainer:
{
public:
  DistriContainer(const BaseMeasure<U>& a, uint32_t d)
    : mDistris(d,NULL)
  {
    for (uint32_t i=0; i<d; ++i)
      mDistris[i] = a.getClopy();
  };


  DistriContainer(const vector<BaseMeasure<U>* >& a)
    : mDistris(d,NULL)
  {
    for (uint32_t i=0; i<d; ++i)
      mDistris[i] = a[i]->getCopy();
  };

  ~DistriContainer()
  {
    for (uint32_t i=0; i<d; ++i)
      delete mDistris[i];
  };

  BaseMeasure<U>* operator[](uint32_t i)
  {
    assert(i<mDistris.size());
    return mDistris[i];
  };

private:
  vector<BaseMeasure<U>* > mDistris;
}


/*
 * this one assumes that the number of words per document is 
 * smaller than the dictionary size
 *
 * http://en.wikipedia.org/wiki/Virtual_inheritance
 */
class HDP_var: public HDP<uint32_t>, public virtual HDP_var_base
{
  public:

    HDP_var(const BaseMeasure<uint32_t>& base, double alpha, double omega)
      : HDP_var_base(0,0,0), HDP<uint32_t>(base, alpha, omega)
    {};

    ~HDP_var()
    {};

    // interface mainly for python
    uint32_t addDoc(const Mat<uint32_t>& x_i)
    {
      uint32_t x_ind = HDP<uint32_t>::addDoc(x_i);
      // TODO: potentially slow
      // add the index of the added x_i
      mInd2Proc.resize(mInd2Proc.n_elem+1);
      mInd2Proc[mInd2Proc.n_elem-1] = x_ind;
      return x_ind;
    };


    /* 
     * Initializes the corpus level parameters mA and mLambda according 
     * to Blei's Stochastic Variational paper
     *
     * @param D is the assumed number of documents for init
     */
    void initCorpusParams(uint32_t Nw, uint32_t K, uint32_t T, uint32_t D)
    {

      mT = T;
      mK = K;
      mNw = Nw;

      mA.ones(K,2);
      mA.col(1) *= mOmega; 

      // initialize lambda
      mLambda.resize(K,NULL);
      for (uint32_t k=0; k<K; ++k){
        mLambda[k] = mH0.getCopy(); // initialize the priors of the topics from the base measure.
      }
//      mLambda.zeros(K,Nw);
//      GammaRnd gammaRnd(1.0, 1.0);
//      for (uint32_t k=0; k<K; ++k){
//        for (uint32_t w=0; w<Nw; ++w) mLambda(k,w) = gammaRnd.draw();
//        mLambda.row(k) *= double(D)*100.0/double(K*Nw);
//        mLambda.row(k) += ((Dir*)(&mH0))->mAlphas;
//      }
    };

    /* From: Online Variational Inference for the HDP
     * method for "one shot" computation without storing data in this class
     *  Nw: number of different words
     *  kappa=0.9: forgetting rate
     *  uint32_t T=10; // truncation on document level
     *  uint32_t K=100; // truncation on corpus level
     *  S = batch size
     */
    void densityEst(const vector<Mat<uint32_t> >& x, uint32_t Nw, 
        double kappa, uint32_t K, uint32_t T, uint32_t S)
    {
      cout<<"densityEstimate with: K="<<K<<"; T="<<T<<"; kappa="<<kappa<<"; Nw="<<Nw<<"; S="<<S<<endl;

      mX = x;
      uint32_t D=mX.size();
      cout<<"D="<<D<<endl;
      cout<<"mX[0].shape= "<<mX[0].n_rows<<"x"<<mX[0].n_cols<<endl;

      mInd2Proc = linspace<Row<uint32_t> >(0,D-1,D);

      mT = T;
      mK = K;
      mNw = Nw;

      initCorpusParams(mNw,mK,mT,D);
      cout<<"Init of corpus params done"<<endl;
      Row<uint32_t> ind = updateEst_batch(mInd2Proc,mZeta,mPhi,mGamma,mA,mLambda,mPerp,mOmega,kappa,S,true);
//      cout<<"mPhi -> D="<<mPhi.size()<<endl;
//      cout<<"mPhi -> D="<<HDP_var_base::mPhi.size()<<endl;
//      cout<<"mPerp="<<mPerp.t()<<endl;

      Mat<double> pi(D,T);
      Mat<double> sigPi(D,T+1);
      Mat<uint32_t> c(D,T);
      getDocTopics(pi,sigPi,c);
//      cout<<"c:"<<c<<endl;

      mInd2Proc.set_size(0); // all processed

    };

    /*
     * compute density estimate based on data previously fed into the class using addDoc
     */
    bool densityEst(uint32_t Nw, double kappa, uint32_t K, uint32_t T, uint32_t S)
    {
      if(mX.size() > 0)
      {
        densityEst(mX,Nw,kappa,K,T,S);
        //TODO: return p_d(x)
        return true;
      }else{
        return false;
      }
    };


    /* 
     * updated the estimate using newly added docs in mX
     * "newly added": the ones that are indicated in mInd2Proc
     */
    bool updateEst_batch(double kappa, uint32_t S)
    {
      uint32_t Db=mInd2Proc.n_elem;
      if (Db >0){  
        cout<<"updatedEstimate with: K="<<mK<<"; T="<<mT<<"; kappa="<<kappa<<"; Nw="<<mNw<<"; S="<<S<<endl;
        vector<Mat<double> > zeta; // will get resized accordingly inside updateEst_batch
        vector<Mat<double> > phi;
        vector<Mat<double> > gamma;
        Col<double> perp;

        Row<uint32_t> ind = updateEst_batch(mInd2Proc,zeta,phi,gamma,mA,mLambda,perp,mOmega,kappa,S);

        mZeta.resize(mZeta.size()+Db);
        mPhi.resize(mPhi.size()+Db);
        mGamma.resize(mGamma.size()+Db);
        mPerp.resize(mPerp.n_elem+Db);
        mPerp.rows(mPerp.n_elem-Db,mPerp.n_elem) = perp;
        for (uint32_t i=0; i<ind.n_elem; ++i){
          mZeta[ind[i]] = zeta[i];
          mPhi[ind[i]] = phi[i];
          mGamma[ind[i]] = gamma[i];
        }

        mInd2Proc.set_size(0); // all processed

        return true;
      }else{
        cout<<"add more documents before starting to process"<<endl;
      }
      return false;
    };

    /*
     * after an initial densitiy estimate has been made using densityEst()
     * can use this to update the estimate with information from additional x 
     */
    bool  updateEst(const Mat<uint32_t>& x, double kappa)
    {
      if (mX.size() > 0 && mX.size() == mPhi.size()) { // this should indicate that there exists an estimate already
        uint32_t N = x.n_cols;
        uint32_t T = mT; 
        uint32_t K = mK;
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
          {    
            Row<double> logP=logP_w(mX[d],mPhi[d], mZeta[d], mGamma[d], mLambda);
            mPerp[d] += HDP<uint32_t>::perplexity(mX_ho[i], logP);
            //mPerp[d] += perplexity(mX_ho[i], mZeta[d], mPhi[d], mGamma[d], mLambda);
          }
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

    /*
     *
     */
    bool updateEst(const Mat<uint32_t>& x, Mat<double>& zeta, Mat<double>& phi, Mat<double>& gamma, Mat<double>& a, Mat<double>& lambda, double omega, uint32_t d, double kappa)
    {
      uint32_t D = d+1; // assume that doc d is appended to the end  
      uint32_t Nw = lambda.n_cols;
      uint32_t T = zeta.n_rows;
      uint32_t K = zeta.n_cols;

      Mat<double> eLogBeta(mK,mNw);
      Col<double> digam_lamb_sum(mK);
      compElogBeta(eLogBeta,  lambda, x);

      Col<double> eLogSig_a(K);
      compElogSig(eLogSig_a, a);

      //    cout<<"---------------- Document "<<d<<" N="<<N<<" -------------------"<<endl;
      initZeta(zeta,eLogBeta, x);
      initPhi(phi,zeta,eLogBeta, x);

      // ------------------------ doc level updates --------------------
      bool converged = false;
      Col<double> eLogSig_gam(T);
      Mat<double> gamma_prev(T,2);
      gamma_prev.ones();

      uint32_t o=0;
      while(!converged){
        //      cout<<"-------------- Iterating local params #"<<o<<" -------------------------"<<endl;
        updateGamma(gamma,phi);
        
        compElogSig(eLogSig_gam,gamma); // precompute 

        updateZeta(zeta,phi,eLogSig_a,eLogBeta, x);
        updatePhi(phi,zeta,eLogSig_gam,eLogBeta, x);

        converged = (accu(gamma_prev != gamma))==0 || o>60 ;
        gamma_prev = gamma;
        ++o;
      }

      //    cout<<" --------------------- natural gradients --------------------------- "<<endl;
      //    cout<<"\tD="<<D<<" omega="<<omega<<endl;
      Mat<double> d_lambda(K,Nw);
      Mat<double> d_a(K,2); 
      computeNaturalGradients(d_lambda, d_a, zeta, phi, omega, D, x);

      //    cout<<" ------------------- global parameter updates: ---------------"<<endl;
      double ro = exp(-kappa*log(1+double(d+1)));
      //    cout<<"\tro="<<ro<<endl;
      lambda = (1.0-ro)*lambda + ro*d_lambda;
      a = (1.0-ro)*a+ ro*d_a;
      return true;
    };


    /*
     * Updates the estimate using mini batches
     * @param ind_x indices of docs to process within docs mX. !these are assumed to be in order!
     * @param sameIndAsX == true -> zeta,phi,gamma have same indices as x (ind_x). This typically happens for the initial batch update. Setting this to true eliminates the need of reordering the results afterwords to match the indices of the docs x.
     * @return the randomly shuffled indices to show how the data was processed -> this allows association of zetas, phis and gammas with docs in mX
     */
    Row<uint32_t> updateEst_batch(const Row<uint32_t>& ind_x, vector<Mat<double> >& zeta, vector<Mat<double> >& phi, vector<Mat<double> >& gamma, Mat<double>& a, vector<BaseMeasure<uint32_t> >& lambda, Col<double>& perp, double omega, double kappa, uint32_t S, bool sameIndAsX=false)
    {
      uint32_t d_0 = min(ind_x); // thats the doc number that we start with -> needed for ro computation; assumes that all indices in mX prior to d_0 have already been processed.
      uint32_t D= max(ind_x)+1; // D is the maximal index of docs that we are processing +1

      Row<uint32_t> ind = shuffle(ind_x,1);
//        cout<<"ind_x: "<<ind_x.cols(0,S)<<endl;
//        cout<<"ind  : "<<ind.cols(0,S)<<endl;

      zeta.resize(ind.n_elem,Mat<double>(mT,mK));
      phi.resize(ind.n_elem);
      gamma.resize(ind.n_elem,Mat<double>(mT,2));
      perp.zeros(ind.n_elem);

      for (uint32_t dd=0; dd<ind.n_elem; dd += S)
      {
        DistriContainer<uint32_t> db_lambda(mH0,mK);

        Mat<double> db_a(mK,2); 
        db_a.zeros();

        Col<double> eLogSig_a(mK);
        compElogSig(eLogSig_a, a);

#pragma omp parallel for schedule(dynamic) 
        for (uint32_t db=dd; db<min(dd+S,ind.n_elem); db++)
        {
          uint32_t d=ind[db];  
          uint32_t dout=sameIndAsX?d:db;
          uint32_t N=mX[d].n_cols;
          //      cout<<"---------------- Document "<<d<<" N="<<N<<" -------------------"<<endl;

          cout<<"-- db="<<db<<" d="<<d<<" N="<<N<<endl;

          Mat<double> eLogBeta(mK,mNw);
          Col<double> digam_lamb_sum(mK);
          compElogBeta(eLogBeta, lambda, mX[d]);

          //Mat<double> zeta(T,K);
          phi[dout].resize(N,mT);
          initZeta(zeta[dout],eLogBeta, mX[d]);
          initPhi(phi[dout],zeta[dout],eLogBeta, mX[d]);

          //cout<<" ------------------------ doc level updates --------------------"<<endl;
          //Mat<double> gamma(T,2);
          Col<double> eLogSig_gam(mT);
          Mat<double> gamma_prev(mT,2);
          gamma_prev.ones();
          gamma_prev.col(1) += mAlpha;
          bool converged = false;
          uint32_t o=0;
          while(!converged){
            //           cout<<"-------------- Iterating local params #"<<o<<" -------------------------"<<endl;
            updateGamma(gamma[dout],phi[dout]);

            compElogSig(eLogSig_gam,gamma[dout]); // precompute 

            updateZeta(zeta[dout],phi[dout],eLogSig_a,eLogBeta, mX[d]);
            updatePhi(phi[dout],zeta[dout],eLogSig_gam,eLogBeta, mX[d]);

            converged = (accu(gamma_prev != gamma[dout]))==0 || o>60 ;
            gamma_prev = gamma[dout];
            ++o;
//            cout<<"o="<<o<<endl;
          }

          DistriContainer<uint32_t> d_lambda(mH0,mK); // batch updates
          Mat<double> d_a(mK,2); 
          //      cout<<" --------------------- natural gradients --------------------------- "<<endl;
          computeNaturalGradients(d_lambda, d_a, zeta[dout], phi[dout], mOmega, D, mX[d]);
#pragma omp critical
          {
            for (uint32_t k=0; k<d_lambda.size(); ++k)
              db_lambda[k]->fromRow(db_lambda[k]->asRow() + d_lambda[k]->asRow());
            db_a += d_a;
          }
        }
        //for (uint32_t k=0; k<K; ++k)
        //  cout<<"delta lambda_"<<k<<" min="<<min(d_lambda.row(k))<<" max="<< max(d_lambda.row(k))<<" #greater 0.1="<<sum(d_lambda.row(k)>0.1)<<endl;
        // ----------------------- update global params -----------------------
        uint32_t t=dd+d_0; // d_0 is the timestep of the the first index to process; dd is the index in the current batch
        uint32_t bS = min(S,ind.n_elem-dd); // necessary for the last batch, which migth not form a complete batch
        //TODO: what is the time dd? d_0 needed?
        double ro = exp(-kappa*log(1+double(t)+double(bS)/2.0)); // as "time" use the middle of the batch 
        cout<<" -- global parameter updates t="<<t<<" bS="<<bS<<" ro="<<ro<<endl;
        //cout<<"d_a="<<d_a<<endl;
        //cout<<"a="<<a<<endl;
        
        for (uint32_t k=0; k<d_lambda.size(); ++k)
          lambda[k]->fromRow((1.0-ro)*lambda[k]->asRow() + (ro/S)*db_lambda[k]->asRow()); //TODO: doies this make sense for NIW prior???
        //lambda = (1.0-ro)*lambda + (ro/S)*db_lambda;
        a = (1.0-ro)*a + (ro/S)*db_a;

        perp[dd+bS/2] = 0.0;
        if (mX_te.size() > 0) {
          cout<<"computing "<<mX_te.size()<<" perplexities"<<endl;
#pragma omp parallel for schedule(dynamic) 
          for (uint32_t i=0; i<mX_te.size(); ++i)
          {
            //cout<<"mX_te: "<< mX_te[i].n_rows << "x"<< mX_te[i].n_cols<<endl;
            //cout<<"mX_ho: "<< mX_ho[i].n_rows << "x"<< mX_ho[i].n_cols<<endl;
            double perp_i =  perplexity(mX_te[i],mX_ho[i],dd+bS/2+1,ro); //perplexity(mX_ho[i], mZeta[d], mPhi[d], mGamma[d], lambda);
            //cout<<"perp_"<<i<<"="<<perp_i<<endl;
#pragma omp critical
            {
              perp[dd+bS/2] += perp_i;
            }
          }
          perp[dd+bS/2] /= double(mX_te.size());
          //cout<<"Perplexity="<<mPerp[d]<<endl;
        }
      }
      cout<<"perp="<<perp.t()<<endl;
      return ind;
    };


    // compute the perplexity of a given document split into x_test (to find a topic model for the doc) and x_ho (to evaluate the perplexity)
    double perplexity(const Mat<uint32_t>& x_te, const Mat<uint32_t>& x_ho, uint32_t d, double kappa=0.75)
    {
      if (mX.size() > 0 && mX.size() == mPhi.size()) { // this should indicate that there exists a estimate already
        uint32_t N = x_te.n_cols;
        uint32_t T = mT; 
        uint32_t K = mK; 

        Mat<double> zeta(T,K);
        Mat<double> phi(N,T);
        Mat<double> gamma(T,2);
        //uint32_t d = mX.size()-1;

        Mat<double> a(mA);// DONE: make deep copy here!

        vector<BaseMeasure<uint32_t>* > lambda(mLambda.size(),NULL);
        for (uint32_t i=0; i<lambda.size(); ++i)
          lambda[i] = mLambda[i]->getCopy();
        double omega = mOmega;

        cout<<"updating copied model with x"<<endl;
        updateEst(x_te,zeta,phi,gamma,a,lambda,omega,d,kappa);
        cout<<" lambda.shape="<<lambda.size()<<endl;
        cout<<"computing perplexity under updated model"<<endl;
        //TODO: compute probabilities then use that to compute perplexity

        cout<<"x_te: "<<size(x_te);
        Row<double> logP=logP_w(x_te,phi, zeta, gamma, lambda);
        return HDP<uint32_t>::perplexity(x_ho, logP);
        //return perplexity(x_ho, zeta, phi, gamma, lambda);
      }else{
        return 1.0/0.0;
      }
    };

    /* Probability distribution over the words in document d
     *
     * TODO: so is that here not some MAP or ML estimate?!
     */
    Row<double> logP_w(uint32_t d) const {
//      cout<<"mX.size="<<mX.size()<<endl;
      return logP_w(mX[d],mPhi[d],mZeta[d],mGamma[d],mLambda);
    };

    /* 
     * log probability using the samples x (not using sufficient statistics -> x is just a list of words)
     */
    Row<double> logP_w(const Mat<uint32_t>& x, const Mat<double>& phi, const Mat<double>& zeta, const Mat<double>& gamma, const vector<BaseMeasure<uint32_t>* >& lambda) const
    {
      Row<double> p(mNw);
      p.zeros();

//      cout<<"x:\t"<<size(x);
//      cout<<"phi:\t"<<size(phi);
//      cout<<"zeta:\t"<<size(zeta);
//      cout<<"gamma:\t"<<size(gamma);
//      cout<<"lambda:\t"<<size(lambda);

      Col<double> pi;
      Col<double> sigPi;
      Col<uint32_t> c;
      getDocTopics(pi,sigPi,c,gamma,zeta);
      //cout<<"getDocTopics done"<<endl;
      //cout<<"c="<<c.t()<<size(c);
      Col<uint32_t> z(mNw);
      getWordTopics(z, phi);
      //cout<<"getWordTopics done"<<endl;
      //cout<<"z="<<z.t()<<size(z);
      Mat<double> beta;
      getCorpTopics(beta,lambda);
      //cout<<"getCorpTopics done"<<endl;
      //cout<<"beta:\t"<<size(beta);

      for (uint32_t i=0; i<x.n_cols; ++i){
        //cout<<"z_"<<i<<"="<<z[i]<<endl;
        p[x[i]] += logCat(x[i], beta.row( c[ z[i] ]));
        //cout<<"p_"<<x[i]<<"="<<p[x[i]]<<endl;
      }
      for (uint32_t w=0; w<mNw; ++w)
        p[w] = p[w]==0.0?-1e10:p[w];

      //cout<<"p="<<p<<endl;
      return p;
    };

  protected:

    Row<uint32_t> mInd2Proc; // indices of docs that have not been processed

  private:

    /*
     * precompute necessary digamma function values, because these are slowing the whole algorithm down
     * all the update methods for zeta and phi need these values very often! I can precumpute these once after updating the global parameters (and hence lambda)
     */
    void compElogBeta(Mat<double>& eLogBeta, const vector<BaseMeasure<uint32_t>* >& lambda, const Mat<uint32_t>& x_d) const 
    { 
      eLogBeta.zeros(mK,mNw);

      Col<double> digam_lamb_sum(mK);
      Mat<uint32_t> x_u = unique(x_d);
//      cout<<"x_d="<<x_d<<endl;
//      cout<<"x_u="<<x_u<<endl;
 
      for (uint32_t i = 0; i < x_u.n_elem ; i++) {
        for (uint32_t k = 0; k < mK; k++) {
           eLogBeta(k,x_u(i)) = lambda[k]->Elog(x_u(i)); // E[log beta] computation in paper
        }
      }
//      for (uint32_t k = 0; k < mK; k++) {
//        digam_lamb_sum(k) = digamma(sum(lambda.row(k)));
//      }
//      for (uint32_t i = 0; i < x_u.n_elem ; i++) {
//        for (uint32_t k = 0; k < mK; k++) {
//           eLogBeta(k,x_u(i)) = digamma(lambda(k,x_u(i))) - digam_lamb_sum(k);
//        }
//      }
    }

    void compElogSig(Col<double>& eLogSig, const Mat<double>& a) const
    {
      for (uint32_t k=0; k<a.n_rows; ++k){
        eLogSig(k)=digamma(a(k,0)) - digamma(a(k,0) + a(k,1));
        for (uint32_t l=0; l<k; ++l)
          eLogSig(k) += digamma(a(l,1)) - digamma(a(l,0) + a(l,1));
      }
    }

    void initZeta(Mat<double>& zeta, const Mat<double>& eLogBeta, const Mat<uint32_t>& x_d)
    {
      uint32_t N = x_d.n_cols;
      uint32_t T = zeta.n_rows;
      uint32_t K = zeta.n_cols;
      //cerr<<"\tinit zeta"<<endl;
      for (uint32_t i=0; i<T; ++i) {
        for (uint32_t k=0; k<K; ++k) {
          zeta(i,k)=0.0;
          for (uint32_t n=0; n<N; ++n) {
            //if(i==0 && k==0) cout<<zeta(i,k)<<" -> ";
            zeta(i,k) += eLogBeta(k,x_d(n)); //ElogBeta(lambda, k, x_d(n));
          }
        }
        normalizeLogDistribution(zeta.row(i));
        //cout<<" normalized="<<zeta(0,0)<<endl;
      }
      //cerr<<"zeta>"<<endl<<zeta<<"<zeta"<<endl;
      //cerr<<"normalization check:"<<endl<<sum(zeta,1).t()<<endl; // sum over rows
    };

    void initPhi(Mat<double>& phi, const Mat<double>& zeta, const Mat<double>& eLogBeta, const Mat<uint32_t>& x_d)
    {
      uint32_t N = x_d.n_cols;
      uint32_t T = zeta.n_rows;
      uint32_t K = zeta.n_cols;
      //cout<<"\tinit phi"<<endl;
      for (uint32_t n=0; n<N; ++n){
        for (uint32_t i=0; i<T; ++i) {
          phi(n,i)=0.0;
          for (uint32_t k=0; k<K; ++k) {
            phi(n,i)+=zeta(i,k)* eLogBeta(k,x_d(n)); // ElogBeta(lambda, k, x_d(n));
          }
        }
        normalizeLogDistribution(phi.row(n));
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

    void updateZeta(Mat<double>& zeta, const Mat<double>& phi, const Col<double>& eLogSig_a, const Mat<double>& eLogBeta, const Mat<uint32_t>& x_d)
    {

      assert(x_d.n_rows == 1);

      uint32_t N = x_d.n_cols;
      uint32_t T = zeta.n_rows;
      uint32_t K = zeta.n_cols;

      for (uint32_t i=0; i<T; ++i){
        //zeta(i,k)=0.0;
        for (uint32_t k=0; k<K; ++k) {
          zeta(i,k) = eLogSig_a(k); //ElogSigma(a,k);
          //cout<<zeta(i,k)<<endl;
          for (uint32_t n=0; n<N; ++n){
            zeta(i,k) += phi(n,i)* eLogBeta(k,x_d(n)); //ElogBeta(lambda,k,x_d(n));
          }
        }
        normalizeLogDistribution(zeta.row(i));
      }
    }


    void updatePhi(Mat<double>& phi, const Mat<double>& zeta, const Col<double>& eLogSig_gam, const Mat<double>& eLogBeta, const Mat<uint32_t>& x_d)
    {

      assert(x_d.n_rows == 1);

      uint32_t N = x_d.n_cols;
      uint32_t T = zeta.n_rows;
      uint32_t K = zeta.n_cols;

      for (uint32_t n=0; n<N; ++n){
        //phi(n,i)=0.0;
        for (uint32_t i=0; i<T; ++i) {
          phi(n,i) = eLogSig_gam(i); //ElogSigma(gamma,i);
          for (uint32_t k=0; k<K; ++k) {
            phi(n,i) += zeta(i,k)* eLogBeta(k,x_d(n)); //ElogBeta(lambda,k,x_d(n)) ;
          }
        }
        normalizeLogDistribution(phi.row(n));
      }
    }

    void computeNaturalGradients(Mat<double>& d_lambda, Mat<double>& d_a, const Mat<double>& zeta, const Mat<double>&  phi, double omega, uint32_t D, const Mat<uint32_t>& x_d)
    {
      uint32_t N = x_d.n_cols;
      uint32_t Nw = d_lambda.n_cols;
      uint32_t T = zeta.n_rows;
      uint32_t K = zeta.n_cols;

      d_lambda.zeros();
      d_a.zeros();
      for (uint32_t k=0; k<K; ++k) 
      { // for all K corpus level topics
        for (uint32_t i=0; i<T; ++i) 
        {
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
        d_lambda.row(k) += ((Dir*)(&mH0))->mAlphas;
        //cout<<"lambda="<<d_lambda[k].t()<<endl;
        d_a(k,0) = D*d_a(k,0)+1.0;
        d_a(k,1) = D*d_a(k,1)+omega;
      }
      //cout<<"da="<<d_a<<endl;
    }


    double digamma(double x) const
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

//    double ElogBeta(const Mat<double>& lambda, uint32_t k, uint32_t w_dn)
//    {
//      //if(lambda[k](w_dn)<1e-6){
//      //  cout<<"\tlambda[k]("<<w_dn<<") near zero: "<<lambda[k](w_dn)<<endl;
//      //}
//      return digamma(lambda(k,w_dn)) - digamma(sum(lambda.row(k)));
//    }
//
//    double ElogSigma(const Mat<double>& a, uint32_t k)
//    {
//      double e=digamma(a(k,0)) - digamma(a(k,0) + a(k,1));
//      for (uint32_t l=0; l<k; ++l)
//        e+=digamma(a(l,1)) - digamma(a(l,0) + a(l,1));
//      return e; 
//    }

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

};

