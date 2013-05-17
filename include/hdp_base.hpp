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
      : mH0(base), mAlpha(alpha), mOmega(omega)
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
//      cout<<"x_i: "<<x_i.cols(0,10)<<endl;
      Mat<U> x_s = shuffle(x_i,1); // suffle the columns (words)
//      cout<<"x_s: "<<x_s.cols(0,10)<<endl;
      Mat<U> x_te = x_s.cols(0,(N/2)-1) ; // select first half to train on
      Mat<U> x_ho = x_s.cols(N/2,N-1) ; // select second half as held out

//      cout<<"x_te: "<<x_te.n_rows<<"x"<<x_te.n_cols<<endl;
//      cout<<"x_ho: "<<x_ho.n_rows<<"x"<<x_ho.n_cols<<endl;
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
//        cout<<"x_ho_i="<<x_ho(i)<<"; logP_x_ho_i="<<logP(x_ho(i))<<endl;
          perp -= logP(x_ho(i));
        //cout<<"w="<<w<<"\tx_ho_w="<<x_ho[w]<<"\tlogP="<<logP[w]<<"\tperp+="<<-double(x_ho[w])*logP[w]<<endl;
      } cout<<endl;
      perp /= double(N);
      perp /= log(2.0); // since it is log base 2 in the perplexity formulation!
      perp = pow(2.0,perp);
//      cout<<"perp="<<perp<<endl;

      return perp;
    };


    bool getLambda(Row<double>& lambda, uint32_t k)
    {
      if(mLambda.size()> 0 && k < mLambda.size())
      {
        lambda=mLambda[k]->asRow();
        return true;
      }else{
        return false;
      }
    };

    /* 
     * The mode of the estimate dirichlet distribution parameterized by lambda is used
     * as an estimate for the Multinomial distribution of the respective topics
     */
    bool getCorpTopic(Row<double>& topic, uint32_t k) const
    {
      if(mLambda.size() > 0 && k < mLambda.size())
      {
        mLambda[k]->mode(topic);
        return true; 
      }else{
        return false;
      }
    };
    
    bool getCorpTopics(Mat<double>& topics) const
    {
      return getCorpTopics(topics,mLambda);
    };

protected:

    const BaseMeasure<U>& mH0; // base measure
    double mAlpha; 
    double mOmega;
    vector<Mat<U> > mX; // training data
    vector<Mat<U> > mX_te; //  test data
    vector<Mat<U> > mX_ho; //  held out data

    DistriContainer<U> mLambda;
    //Mat<double> mLambda; // corpus level topics (Dirichlet)


    bool getCorpTopic(Row<double>& topic, const BaseMeasure<U>* lambda) const
    {
      // mode of dirichlet (MAP estimate)
      lambda->mode(topic);
      return true;
    };

    bool getCorpTopics(Mat<double>& topics, const DistriContainer<U>& lambda) const
    {
      uint32_t K = lambda.size();
      Row<double> beta;
      lambda[0]->mode(beta);
      uint32_t Nw = beta.n_elem;
      topics.set_size(K,Nw);
      for (uint32_t k=0; k<K; k++){
        // mode of dirichlet (MAP estimate)
        Row<double> beta;
        lambda[k]->mode(beta);
        topics.row(k) = beta;
//        cout<<"lambda_"<<k<<"="<<lambda.row(k)<<endl;
//        cout<<"topic_"<<k<<"="<<topics.row(k)<<endl;
//        cout<<"sum over topic_"<<k<<"="<<sum(topics.row(k))<<endl<<endl;
        //dirMode(topics.row(k), lambda.row(k));
      }
      return true;
    };
private:

};


