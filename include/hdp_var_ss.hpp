/* Copyright (c) 2012, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See LICENSE.txt or 
 * http://www.opensource.org/licenses/mit-license.php */

#pragma once

#include "random.hpp"
#include "baseMeasure.hpp"
//#include "dp.hpp"
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

// base HDP for using sufficient statistics
template <class U>
class HDP_ss // : public DP<U>
{
  public:
    HDP_ss(const BaseMeasure<U>& base, double alpha, double omega)
      : mH(base), mAlpha(alpha), mOmega(omega)
    {
      //    cout<<"Creating "<<typeid(this).name()<<endl;
    };

    ~HDP_ss()
    { };

    //virtual Row<double> logP_w(uint32_t d) const=0;

    // compute the perplexity given a heldout data from document x_ho and the model paremeters of it (after incorporating x)
    double perplexity(const Row<U>& x_ho, const Row<double>& logP)
    {
//      //cout<<"Computing Perplexity"<<endl;
      uint32_t Nw = x_ho.n_cols;
      uint32_t N = sum(x_ho);
      double perp = 0.0;
      for (uint32_t w=0; w<Nw; ++w){
        //cout<<"c_z_n = "<<c[z[w]]<<" z_n="<<z[w]<<" w="<<w<<" N="<<N<<" x_w="<<x_ho[w]<<" topics.shape="<<topics.n_rows<<" "<<topics.n_cols;
        if (x_ho[w] > 0) {
          perp -= x_ho[w]*logP[w];
          cout<<"w="<<w<<"\tx_ho_w="<<x_ho[w]<<"\tlogP="<<logP[w]<<"\tperp+="<<-double(x_ho[w])*logP[w]<<endl;
        }
      } cout<<endl;
      perp /= double(N);
      perp /= log(2.0); // since it is log base 2 in the perplexity formulation!
      perp = pow(2.0,perp);

      return perp;
    }

protected:

    const BaseMeasure<U>& mH; // base measure
    double mAlpha; 
    double mOmega;
    Mat<U> mX; // training data
    Mat<U> mX_ho; // held out data
    Mat<U> mX_te; // test data

};


/* 
 * this one assumes that the number of words per document are bigger 
 * than the number of individual words and is optimized for that case
 * 
 * http://en.wikipedia.org/wiki/Virtual_inheritance
 */
class HDP_var_ss: public HDP_ss<uint32_t>, public virtual HDP_var_base
{
  public:

    HDP_var_ss(const BaseMeasure<uint32_t>& base, double alpha, double omega)
      : HDP_var_base(0,0,0), HDP_ss<uint32_t>(base, alpha, omega)
    {};

    ~HDP_var_ss()
    {};


    // method for "one shot" computation without storing data in this class
    //  Nw: number of different words
    //  kappa=0.9: forgetting rate
    //  uint32_t T=10; // truncation on document level
    //  uint32_t K=100; // truncation on corpus level
    // S = batch size
    void densityEst(const Mat<uint32_t>& x, const Mat<uint32_t>& x_test, double kappa, uint32_t K, uint32_t T, uint32_t S)
    {

      // From: Online Variational Inference for the HDP
      // TODO: think whether it makes sense to use columns and Mat here
      //double mAlpha = 1;
      //double mGamma = 1;
      //double nu = ((Dir*)(&mH))->mAlpha0; // get nu from vbase measure (Dir)
      //double kappa = 0.75;
      // TODO: x is a matrix of counts of words per row. so columns are counts of specific words

      // T-1 gamma draws determine a T dim multinomial
      // T = T-1;

      mX = x;
      mT = T;
      mK = K;
      mNw = x.n_cols;
      uint32_t D=x.n_rows;

      // split up x_test into test and held out data points
      mX_te.set_size(x_test.n_rows,x_test.n_cols);
      mX_ho.set_size(x_test.n_rows,x_test.n_cols);
      for (uint32_t d=0; d<x_test.n_rows; ++d){
        uint32_t N=sum(x_test.row(d));
        Row<uint32_t> x_s(N); // construct a vector with the words as given by the counts in x_test
        uint32_t offset =0;
        for (uint32_t w=0; w<mNw; ++w){
          for (uint32_t i=0; i<x_test(d,w); ++i){
            x_s[i+offset] = w;
          }
          offset += x_test(d,w);
        }
        vector<uint32_t> rndInds(N);
        for (uint32_t i=0; i<N; ++i)
          rndInds[i]=i;
        std::random_shuffle(rndInds.begin(),rndInds.end());

        Row<uint32_t> x_te(N/2);
        Row<uint32_t> x_ho(N/2+N%2);
        for (uint32_t i=0; i<N/2; ++i)
          x_te[i] = x_s[rndInds[i]];
        for (uint32_t i=N/2; i<N; ++i)
          x_ho[i-(N/2)] = x_s[rndInds[i]];

        cout<<"x_te.n_elem="<<x_te.n_elem<<" x_ho.n_elem="<<x_ho.n_elem<<" N="<<N<<endl;
        // now convert back into counts
        for (uint32_t w=0; w<mNw; ++w){
          mX_ho.row(d)[w] = sum(x_ho == w);
          mX_te.row(d)[w] = sum(x_te == w);
        }
        cout<<"x_test=\t"<<x_test.row(d)<<endl;
        cout<<"x_te+ho=\t"<<mX_te.row(d)+mX_ho.row(d)<<endl;
        cout<<"x_ho=\t"<<mX_ho.row(d)<<endl;
        cout<<"x_te=\t"<<mX_te.row(d)<<endl;
      }

      Mat<double> a(K,2);
      a.ones();
      a.col(1) *= mOmega; 

      // initialize lambda
      GammaRnd gammaRnd(1.0,1.0);
      Mat<double> lambda(K,mNw);
      for (uint32_t k=0; k<K; ++k){
        for (uint32_t w=0; w<mNw; ++w) lambda(k,w) = gammaRnd.draw();
        lambda.row(k) *= double(D)*100.0/double(K*mNw);
        lambda.row(k) += ((Dir*)(&mH))->mAlphas;
      }

      mZeta.resize(D,Mat<double>());
      mPhi.resize(D,Mat<double>());
      mGamma.resize(D,Mat<double>());
      mPerp.zeros(D);

      vector<Col<uint32_t> > z_dn(D);
      Col<uint32_t> ind = shuffle(linspace<Col<uint32_t> >(0,D-1,D),0);
//#pragma omp parallel private(dd,db)
//#pragma omp parallel private(d,dd,N,zeta,phi,converged,gamma,gamma_prev,o,d_lambda,d_a,ro,i,perp_i)
      //shared(x,mZeta,mPhi,mGamma,mOmega,D,T,K,Nw,mA,mLambda,mPerp,mX_ho)
      {
//#pragma omp for schedule(dynamic)
//#pragma omp for schedule(dynamic) ordered
        for (uint32_t dd=0; dd<D; dd += S)
        {
          Mat<double> db_lambda(K,mNw); // batch updates
          Mat<double> db_a(K,2); 
#pragma omp parallel for schedule(dynamic) 
          for (uint32_t db=dd; db<min(dd+S,D); db++)
          {
            uint32_t d=ind[db];  
            uint32_t N=sum(x.row(d));
            //      cout<<"---------------- Document "<<d<<" N="<<N<<" -------------------"<<endl;

            cout<<"-- db="<<db<<" d="<<d<<" N="<<N<<endl;
            Mat<double> zeta(T,K);
            Mat<double> phi(mNw,T);
            initZeta(zeta,lambda,x.row(d));
            initPhi(phi,zeta,lambda,x.row(d));

            //cout<<" ------------------------ doc level updates --------------------"<<endl;
            Mat<double> gamma(T,2);
            Mat<double> gamma_prev(T,2);
            gamma_prev.ones();
            gamma_prev.col(1) += mAlpha;
            bool converged = false;
            uint32_t o=0;
            while(!converged){
              //           cout<<"-------------- Iterating local params #"<<o<<" -------------------------"<<endl;
              updateGamma(gamma,phi);
              updateZeta(zeta,phi,a,lambda,x.row(d));
              updatePhi(phi,zeta,gamma,lambda,x.row(d));

              converged = (accu(gamma_prev != gamma))==0 || o>60 ;
              gamma_prev = gamma;
              ++o;
            }

            mZeta[d] = Mat<double>(zeta);
            mPhi[d] = Mat<double>(phi);
            mGamma[d] = Mat<double>(gamma);
            Mat<double> d_lambda(K,mNw);
            Mat<double> d_a(K,2); 
            //      cout<<" --------------------- natural gradients --------------------------- "<<endl;
            computeNaturalGradients(d_lambda, d_a, zeta, phi, mOmega, D, x.row(d));
 #pragma omp critical
            {
              db_lambda += d_lambda;
              db_a += d_a;
            }
          }
          //for (uint32_t k=0; k<K; ++k)
          //  cout<<"delta lambda_"<<k<<" min="<<min(d_lambda.row(k))<<" max="<< max(d_lambda.row(k))<<" #greater 0.1="<<sum(d_lambda.row(k)>0.1)<<endl;
          // ----------------------- update global params -----------------------
          double bS = min(S,D-dd); // necessary for the last batch, which migth not form a complete batch
          double ro = exp(-kappa*log(1+double(dd)+double(bS)/2.0)); // as "time" use the middle of the batch 
          cout<<" -- global parameter updates dd="<<dd<<" bS="<<bS<<" ro="<<ro<<endl;
          //cout<<"d_a="<<d_a<<endl;
          //cout<<"a="<<a<<endl;
          lambda = (1.0-ro)*lambda + (ro/S)*db_lambda;
          a = (1.0-ro)*a + (ro/S)*db_a;
          mA=a;
          mLambda = lambda;

          mPerp[dd+bS/2] = 0.0;
          if (mX_ho.n_rows > 0) {
            cout<<"computing "<<mX_ho.n_rows<<" perplexities"<<endl;
#pragma omp parallel for schedule(dynamic) 
            for (uint32_t i=0; i<mX_ho.n_rows; ++i)
            {
              double perp_i =  perplexity(mX_te.row(i),mX_ho.row(i),dd+bS/2+1,kappa); //perplexity(mX_ho[i], mZeta[d], mPhi[d], mGamma[d], lambda);
              cout<<"perp_"<<i<<"="<<perp_i<<endl;
#pragma omp critical
              {
                mPerp[dd] += perp_i;
              }
            }
            mPerp[dd] /= double(mX_ho.n_rows);
            //cout<<"Perplexity="<<mPerp[d]<<endl;
          }
        }
      }

      //return z_dn; //TODO: return p_d(x)
    };

//    // compute density estimate based on data previously fed into the class using addDoc
//    bool densityEst(double kappa, uint32_t K, uint32_t T, uint32_t S)
//    {
//      if(mX.n_rows > 0)
//      {
//        densityEst(mX,kappa,K,T,S);
//        //TODO: return p_d(x)
//        return true;
//      }else{
//        return false;
//      }
//    };

    // after an initial densitiy estimate has been made using densityEst()
    // can use this to update the estimate with information from additional x 
    bool  updateEst(const Mat<uint32_t>& x, double kappa)
    {
      if (mX.n_rows > 0 && mX.n_rows == mPhi.size()) { // this should indicate that there exists a estimate already
        uint32_t d = mX.n_rows;
        uint32_t Nw = x.n_cols;
        uint32_t T = mT; //mZeta[0].n_rows;
        uint32_t K = mK; //mZeta[0].n_cols;
        mX.resize(d+1,mX.n_cols);
        mX.row(d) = x;
        mZeta.push_back(Mat<double>(T,K));
        mPhi.push_back(Mat<double>(Nw,T));
        //    mZeta.set_size(T,K);
        //    mPhi.set_size(N,T);
        mGamma.push_back(Mat<double>(T,2));
        mPerp.resize(d+1);

        if(updateEst(mX.row(d),mZeta[d],mPhi[d],mGamma[d],mA,mLambda,mOmega,d,kappa))
        {
          mPerp[d] = 0.0;
          for (uint32_t i=0; i<mX_ho.n_rows; ++i)
            mPerp[d] += perplexity(mX_te.row(i),mX_ho.row(i), d, kappa); //mZeta[d], mPhi[d], mGamma[d], mLambda);
          mPerp[d] /= double(mX_ho.n_rows);
          cout<<"Perplexity="<<mPerp[d]<<endl;
          return true; 
        }else{
          return false;
        } 
      }else{
        return false;
      }
    };

    // compute the perplexity of a given document x_ho (held out for perplexity) and x_te (used to update the model) of a document
    double perplexity(const Row<uint32_t>& x_te, const Row<uint32_t>& x_ho, uint32_t d, double kappa)
    {
      if (mX.n_rows > 0 && mX.n_rows == mPhi.size()) { // this should indicate that there exists a estimate already
        uint32_t Nw = x_te.n_cols;
        //uint32_t Nw = mLambda.n_cols;
        uint32_t T = mT; //mZeta[0].n_rows; 
        uint32_t K = mK; //mZeta[0].n_cols;

        Mat<double> zeta(T,K);
        Mat<double> phi(Nw,T);
        Mat<double> gamma(T,2);
        //uint32_t d = mX.size()-1;

        Mat<double> a(mA);// DONE: make deep copy here!
        Mat<double> lambda(mLambda);
        double omega = mOmega;

        //cout<<"updating copied model with x"<<endl;
        updateEst(x_te,zeta,phi,gamma,a,lambda,omega,d,kappa);
        //cout<<"computing perplexity under updated model"<<endl;
        Row<double> logP=logP_w(phi, zeta, gamma, lambda);
        return HDP_ss<uint32_t>::perplexity(x_ho, logP);
        //return perplexity(x_ho, zeta, phi, gamma, lambda);
      }else{
        return 1.0/0.0;
      }
    };

    // compute the perplexity given a heldout data from document x_ho and the model paremeters of it (after incorporating x)
    double perplexity(const Row<uint32_t>& x_ho, const Mat<double>& zeta, const Mat<double>& phi, const Mat<double>& gamma, const Mat<double>& lambda)
    {
      Row<double> logP=logP_w(phi, zeta, gamma, lambda);

//      //cout<<"Computing Perplexity"<<endl;
      uint32_t Nw = x_ho.n_cols;
      uint32_t N = sum(x_ho);
      double perp = 0.0;
      for (uint32_t w=0; w<Nw; ++w){
        //cout<<"c_z_n = "<<c[z[w]]<<" z_n="<<z[w]<<" w="<<w<<" N="<<N<<" x_w="<<x_ho[w]<<" topics.shape="<<topics.n_rows<<" "<<topics.n_cols;
        if (x_ho[w] > 0) {
          perp -= x_ho[w]*logP[w];
          cout<<"w="<<w<<"\tx_ho_w="<<x_ho[w]<<"\tlogP="<<logP[w]<<"\tperp+="<<-double(x_ho[w])*logP[w]<<endl;
        }
      } cout<<endl;
      perp /= double(N);
      perp /= log(2.0); // since it is log base 2 in the perplexity formulation!
      perp = pow(2.0,perp);

      return perp;
    }


    bool updateEst(const Row<uint32_t>& x, Mat<double>& zeta, Mat<double>& phi, Mat<double>& gamma, Mat<double>& a, Mat<double>& lambda, double omega, uint32_t d, double kappa)
    {
      uint32_t D = d+1; // assume that doc d is appended to the end  
      //uint32_t N = x.n_rows;
      uint32_t Nw = x.n_cols;
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

    /* Probability distribution over the words in document d
     *
     * TODO: so is that here not some MAP or ML estimate?!
     */
    Row<double> logP_w(uint32_t d) const {
      return logP_w(mPhi[d],mZeta[d],mGamma[d],mLambda);
    };

    /* 
     * probability of assuming the use of suffient statistics
     */
    // TODO: this was the old version - presumably working for sufficient statistics
    Row<double> logP_w(const Mat<double>& phi, const Mat<double>& zeta, const Mat<double>& gamma, const Mat<double>& lambda) const
    {
      Row<double> p(mNw);
      p.zeros();

      cout<<"phi:\t"<<size(phi);
      cout<<"zeta:\t"<<size(zeta);
      cout<<"gamma:\t"<<size(gamma);
      cout<<"lambda:\t"<<size(lambda);

      Col<double> pi;
      Col<double> sigPi;
      Col<uint32_t> c;
      getDocTopics(pi,sigPi,c,gamma,zeta);
      cout<<"getDocTopics done"<<endl;
      cout<<"c="<<c<<endl;
      Col<uint32_t> z(mNw);
      getWordTopics(z, phi);
      cout<<"getWordTopics done"<<endl;
      cout<<"z="<<z<<endl;
      Mat<double> beta;
      getCorpTopics(beta,lambda);
      cout<<"getCorpTopics done"<<endl;
      cout<<"beta:\t"<<size(beta);


      for (uint32_t w=0; w<mNw; ++w){
        cout<<"w="<<w<<"; Nw="<<mNw<<endl;
        cout<<"z_w="<<z[w]<<endl;
        p[w] = logCat(w, beta.row( c[ z[w] ]));                                   
        cout<<"p_"<<w<<"="<<p[w]<<endl;
      }      
      cout<<"p="<<p<<endl;
      return p;
    };


    /* TODO: its not realy the joint... or is it?!
     * joint probability distribution
     */
    Row<double> logP_joint(uint32_t d) const {
      return logP_joint(mPhi[d],mZeta[d],mGamma[d],mLambda,mA);
    };
    Row<double> logP_joint(const Mat<double>& phi, const Mat<double>& zeta, const Mat<double>& gamma, const Mat<double>& lambda, const Mat<double>& a) const
    {
      Row<double> p(mNw);
      p.zeros();

      Col<double> v;
      Col<double> sigV;
      getCorpTopicProportions(v,sigV,a);
      Col<double> pi;
      Col<double> sigPi;
      Col<uint32_t> c;
      getDocTopics(pi,sigPi,c,gamma,zeta);
      Col<uint32_t> z(mNw);
      getWordTopics(z, phi);
      Mat<double> beta;
      getCorpTopics(beta,lambda);

      for (uint32_t w=0; w<mNw; ++w){
        p[w] = logCat(w, beta.row( c[ z[w] ]).t()) + 
          logCat(c[ z[w] ], sigV.t()) + 
          logBeta(v, 1.0, mOmega) + 
          logCat(z[w], sigPi.t()) +
          logBeta(pi, 1.0, mAlpha) +
          logDir(beta.row( c[ z[w] ]).t(), mLambda.t());
      }
      return p;
    };

  private:


    void initZeta(Mat<double>& zeta, const Mat<double>& lambda, const Row<uint32_t>& x_d)
    {
      uint32_t Nw = x_d.n_cols;
      uint32_t T = zeta.n_rows;
      uint32_t K = zeta.n_cols;
      //cerr<<"\tinit zeta"<<endl;
      for (uint32_t i=0; i<T; ++i) {
        for (uint32_t k=0; k<K; ++k) {
          zeta(i,k)=0.0;
          for (uint32_t w=0; w<Nw; ++w) {
            //if(i==0 && k==0) cout<<zeta(i,k)<<" -> ";
            zeta(i,k) += x_d(w) * ElogBeta(lambda, k, w);
          }
        }
        normalizeLogDistribution(zeta.row(i));
      }
      //cerr<<"zeta>"<<endl<<zeta<<"<zeta"<<endl;
      //cerr<<"normalization check:"<<endl<<sum(zeta,1).t()<<endl; // sum over rows
    };

    void initPhi(Mat<double>& phi, const Mat<double>& zeta, const Mat<double>& lambda, const Row<uint32_t>& x_d)
    {
      uint32_t Nw = x_d.n_cols;
      uint32_t T = zeta.n_rows;
      uint32_t K = zeta.n_cols;
      //cout<<"\tinit phi"<<endl;
      for (uint32_t w=0; w<Nw; ++w){
        for (uint32_t i=0; i<T; ++i) {
          phi(w,i)=0.0;
          for (uint32_t k=0; k<K; ++k) {
            phi(w,i)+=zeta(i,k)* x_d(w) * ElogBeta(lambda, k, w);
          }
        }
        normalizeLogDistribution(phi.row(w));
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

    void updateZeta(Mat<double>& zeta, const Mat<double>& phi, const Mat<double>& a, const Mat<double>& lambda, const Row<uint32_t>& x_d)
    {
      uint32_t Nw = x_d.n_cols;
      uint32_t T = zeta.n_rows;
      uint32_t K = zeta.n_cols;

      for (uint32_t i=0; i<T; ++i){
        //zeta(i,k)=0.0;
        for (uint32_t k=0; k<K; ++k) {
          zeta(i,k) = ElogSigma(a,k);
          //cout<<zeta(i,k)<<endl;
          for (uint32_t w=0; w<Nw; ++w){
            zeta(i,k) += phi(w,i)*x_d(w)*ElogBeta(lambda,k,w);
          }
        }
        normalizeLogDistribution(zeta.row(i));
      }
    }


    void updatePhi(Mat<double>& phi, const Mat<double>& zeta, const Mat<double>& gamma, const Mat<double>& lambda, const Row<uint32_t>& x_d)
    {
      uint32_t Nw = x_d.n_cols;
      uint32_t T = zeta.n_rows;
      uint32_t K = zeta.n_cols;

      for (uint32_t w=0; w<Nw; ++w){
        //phi(n,i)=0.0;
        for (uint32_t i=0; i<T; ++i) {
          phi(w,i) = ElogSigma(gamma,i);
          for (uint32_t k=0; k<K; ++k) {
            phi(w,i) += zeta(i,k)*x_d(w)*ElogBeta(lambda,k,w) ;
          }
        }
        normalizeLogDistribution(phi.row(w));
      }
    }

    void computeNaturalGradients(Mat<double>& d_lambda, Mat<double>& d_a, const Mat<double>& zeta, const Mat<double>&  phi, double omega, uint32_t D, const Row<uint32_t>& x_d)
    {
      uint32_t Nw = x_d.n_cols;
      uint32_t T = zeta.n_rows;
      uint32_t K = zeta.n_cols;

      d_lambda.zeros();
      d_a.zeros();
      for (uint32_t k=0; k<K; ++k) { // for all K corpus level topics
        for (uint32_t i=0; i<T; ++i) {
          Row<double> _lambda(Nw); _lambda.zeros();
          for (uint32_t w=0; w<Nw; ++w){
            _lambda(w) += phi(w,i); // i think if I multiply by x_d(w) again here I am doing it twice (see updatePhi)  x_d(w)*phi(w,i);
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
      //cout<<"da="<<d_a<<endl;
    }


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

    double ElogBeta(const Mat<double>& lambda, uint32_t k, uint32_t w)
    {
      //if(lambda[k](w_dn)<1e-6){
      //  cout<<"\tlambda[k]("<<w_dn<<") near zero: "<<lambda[k](w_dn)<<endl;
      //}
      return digamma(lambda(k,w)) - digamma(sum(lambda.row(k)));
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

};

