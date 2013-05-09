/* Copyright (c) 2012, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See LICENSE.txt or 
 * http://www.opensource.org/licenses/mit-license.php */

#pragma once

#include "baseMeasure.hpp"
#include "probabilityHelpers.hpp"

#include <stddef.h>
#include <stdint.h>
#include <typeinfo>

#include <armadillo>

using namespace std;
using namespace arma;

template <class U>
class HDP // : public DP<U>
{
  public:
    HDP(const BaseMeasure<U>& base, double alpha, double omega)
      : mH(base), mAlpha(alpha), mOmega(omega)
    { };

    ~HDP()
    {	};

    /* 
     * interface mainly for python
     * @return the index of the added x_i in the set of all documents
     */
    uint32_t addDoc(const Mat<U>& x_i)
    {
      mX.push_back(x_i);
      return mX.size()-1; 
    };

    // interface mainly for python
    uint32_t addHeldOut(const Mat<U>& x_i)
    {
      //cout<<"added heldout doc with "<<x_i.size()<<" words"<<endl;
      uint32_t N = x_i.n_cols;
      
      //DONE: this shuffle might not be working
      cout<<"x_i: "<<x_i.cols(0,10)<<endl;
      Mat<U> x_s = shuffle(x_i,1); // suffle the columns (words)
      cout<<"x_s: "<<x_s.cols(0,10)<<endl;
      Mat<U> x_te = x_s.cols(0,(N/2)-1) ; // select first half to train on
      Mat<U> x_ho = x_s.cols(N/2,N-1) ; // select second half as held out

      cout<<"x_te: "<<x_te.n_rows<<"x"<<x_te.n_cols<<endl;
      cout<<"x_ho: "<<x_ho.n_rows<<"x"<<x_ho.n_cols<<endl;
      
  
      mX_te.push_back(x_te); // on half of the data we train as usual to get a topic model for the document
      mX_ho.push_back(x_ho); // held out data to evaluate the perplexity
      return mX_ho.size();
    };

    //virtual Row<double> logP_w(uint32_t d) const=0;


    double perplexity(const Mat<U>& x_ho, const Row<double>& logP) const
    {
      assert(x_ho.n_rows==1);

      uint32_t N = x_ho.n_cols;
      double perp = 0.0;
      for (uint32_t i=0; i<N; ++i){
        //cout<<"c_z_n = "<<c[z[w]]<<" z_n="<<z[w]<<" w="<<w<<" N="<<N<<" x_w="<<x_ho[w]<<" topics.shape="<<topics.n_rows<<" "<<topics.n_cols;
        cout<<"x_ho_i="<<x_ho(i)<<"; logP_x_ho_i="<<logP(x_ho(i))<<endl;
          perp -= logP(x_ho(i));
        //cout<<"w="<<w<<"\tx_ho_w="<<x_ho[w]<<"\tlogP="<<logP[w]<<"\tperp+="<<-double(x_ho[w])*logP[w]<<endl;
      } cout<<endl;
      perp /= double(N);
      perp /= log(2.0); // since it is log base 2 in the perplexity formulation!
      perp = pow(2.0,perp);
      cout<<"perp="<<perp<<endl;

      return perp;
    };

protected:

    const BaseMeasure<U>& mH; // base measure
    double mAlpha; 
    double mOmega;
    vector<Mat<U> > mX; // training data
    vector<Mat<U> > mX_te; //  test data
    vector<Mat<U> > mX_ho; //  held out data

private:

};

class HDP_var_base 
{
  public:
    
    HDP_var_base(uint32_t K=0, uint32_t T=0, uint32_t Nw=0)
      : mK(K), mT(T), mNw(Nw)
    {};

    void getA(Mat<double>& a)
    {
      a=mA;
    };

    void getPerplexity(Mat<double>& perp)
    {
      perp=mPerp;
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

//    bool getDocTopics(Col<double>& pi, Col<double>& sigPi, Col<uint32_t>& c, uint32_t d) const
//    {
//      if (d < mGamma.size())
//        return getDocTopics(pi,sigPi,c,mGamma[d],mZeta[d]);
//      else{
//        cout<<"asking for out of range doc "<<d<<" have only "<<mGamma.size()<<endl;
//        return false;
//      }
//    };

    bool getDocTopics(Mat<double>& pi, Mat<double>& sigPi, Mat<uint32_t>& c) const
    {
      uint32_t D=mZeta.size();

      pi.set_size(D,mT);
      sigPi.set_size(D,mT+1);
      c.set_size(D,mT);
      for (uint32_t d=0; d<D; ++d){
        Row<double> cpi;
        Row<double> csigPi;
        Row<uint32_t> cc;

        betaMode(cpi,mGamma[d].col(0),mGamma[d].col(1));
        stickBreaking(csigPi,cpi);
        for (uint32_t i=0; i<T; ++i){
          cc[i] = multinomialMode(mZeta[d].row(i));
        }

        //getDocTopics(cpi,csigPi,cc,mGamma[d],mZeta[d]);
        pi.row(d) = cpi;
        sigPi.row(d) = csigPi;
        c.row(d) = cc;

        cout<<csigPi.t()<<" = "<<sum(csigPi)<<endl;
        cout<<sigPi.row(d)<<" = "<<sum(sigPi.row(d))<<endl;
      }
      return true;
    };

  /*
   * Gets the indicators for each word pointing to a doc level topic.
   * Need to give it the doc of interest, because all documents have
   * different numbers of words and hence z is different for all docs
   *
   * @param d points to the document of interest. 
   */
    bool getWordTopics(Col<uint32_t>& z, uint32_t d) const {
//      cout<<"mPhi -> D="<<mPhi.size()<<endl;
      if (z.n_rows != mPhi[d].n_rows)
      {
        cout<<"z.rows="<<z.n_rows<<" vs phi_d.rows="<<mPhi[d].n_rows<<endl;
        return false;
      }else{
      return getWordTopics(z,mPhi[d]);
      }
    };

    bool getCorpTopicProportions(Col<double>& v, Col<double>& sigV) const
    {
      return getCorpTopicProportions(v,sigV,mA);
    };


    /* The mode of the estimate dirichlet distribution parameterized by lambda is used
     * as an estimate for the Multinomial distribution of the respective topics
     *
     */
    bool getCorpTopic(Col<double>& topic, uint32_t k) const
    {
      if(mLambda.n_rows > 0 && k < mLambda.n_rows)
      {
        return getCorpTopic(topic,mLambda.row(k));
      }else{
        return false;
      }
    };
    
    bool getCorpTopics(Mat<double>& topics) const
    {
      return getCorpTopics(topics,mLambda);
    };

    virtual Row<double> logP_w(uint32_t d) const 
    {
      cout<<"has to be implemented in the child classes!!"<<endl;
      assert(false);
      return Row<double>(1);
    };

    bool getWordDistr(Mat<double>& p){
      uint32_t D=mZeta.size();
      assert(p.n_rows == D);
      assert(p.n_cols == mNw);
      cout<<"D="<<D<<endl;
      for (uint32_t d=0; d<D; ++d){
        p.row(d) = logP_w(d);
      }
      return true;
    };


  protected:
    Mat<double> mLambda; // corpus level topics (Dirichlet)
    Mat<double> mA; // corpus level Beta process alpha parameter for stickbreaking
    vector<Mat<double> > mZeta; // document level topic indices/pointers to corpus level topics (Multinomial) 
    vector<Mat<double> > mPhi; // document level word to doc level topic assignment (Multinomial)
    vector<Mat<double> > mGamma; // document level Beta distribution alpha parameter for stickbreaking

    Col<double> mPerp; // perplexity for each document

    uint32_t mK; // Corp level truncation
    uint32_t mT; // Doc level truncation
    uint32_t mNw; // size of dictionary



    bool getCorpTopic(Col<double>& topic, const Row<double>& lambda) const
    {
      // mode of dirichlet (MAP estimate)
      dirMode(topic, lambda.t());
      return true;
    };

    bool getCorpTopics(Mat<double>& topics, const Mat<double>& lambda) const
    {
      uint32_t K = lambda.n_rows;
      uint32_t Nw = lambda.n_cols;
      topics.set_size(K,Nw);
      for (uint32_t k=0; k<K; k++){
        // mode of dirichlet (MAP estimate)
        Row<double> lamb = lambda.row(k);
        Row<double> beta(Nw);
        dirMode(beta, lamb);
        topics.row(k) = beta;
//        cout<<"lambda_"<<k<<"="<<lambda.row(k)<<endl;
//        cout<<"topic_"<<k<<"="<<topics.row(k)<<endl;
//        cout<<"sum over topic_"<<k<<"="<<sum(topics.row(k))<<endl<<endl;
        //dirMode(topics.row(k), lambda.row(k));
      }
      return true;
    };

    /* 
     * @param a are the parameters of the beta distribution from which v is drawn
     */
    bool getCorpTopicProportions(Col<double>& v, Col<double>& sigV, const Mat<double>& a) const
    {
      uint32_t K = a.n_rows; // corp level topics

      sigV.set_size(K+1);
      v.set_size(K);

      betaMode(v, a.col(0), a.col(1));
      stickBreaking(sigV,v);
      return true;
    };

    bool getDocTopics(Col<double>& pi, Col<double>& sigPi, Col<uint32_t>& c, const Mat<double>& gamma, const Mat<double>& zeta) const
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


    bool getWordTopics(Col<uint32_t>& z, const Mat<double>& phi) const {
      cout<<phi.n_rows<<" x "<<phi.n_cols<<endl;
      z.set_size(phi.n_rows);
      for (uint32_t i=0; i<z.n_elem; ++i){
        z[i] = multinomialMode(phi.row(i));
      }
      return true;
    };
};


