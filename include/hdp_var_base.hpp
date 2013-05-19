/* Copyright (c) 2012, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See LICENSE.txt or 
 * http://www.opensource.org/licenses/mit-license.php */

#pragma once

#include "probabilityHelpers.hpp"

#include <stddef.h>
#include <stdint.h>
#include <typeinfo>

#include <armadillo>

using namespace std;
using namespace arma;

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


    bool getDocTopics(Mat<double>& pi, Mat<double>& sigPi, Mat<uint32_t>& c) const
    {
      uint32_t D=mZeta.size();

      pi.set_size(D,mT);
      sigPi.set_size(D,mT+1);
      c.set_size(D,mT);
      for (uint32_t d=0; d<D; ++d){
        Row<double> cpi(mT);
        Row<double> csigPi(mT+1);
        Row<uint32_t> cc(mT);

        betaMode(cpi,mGamma[d].col(0),mGamma[d].col(1));
        stickBreaking(csigPi,cpi);
        if ( sum(csigPi>1.0)>0)
        {
          cout<<"d="<<d<<endl;
          cout<<"gamma: "<<mGamma[d]<<endl;
          cout<<"cpi="<<cpi<<endl;
          cout<<"csigPi="<<csigPi<<endl;
          cout<<"|csigPi|="<<sum(csigPi)<<endl;
          assert(0);
        }
        for (uint32_t i=0; i<mT; ++i){
          cc[i] = multinomialMode(mZeta[d].row(i));
        }

        //getDocTopics(cpi,csigPi,cc,mGamma[d],mZeta[d]);
        pi.row(d) = cpi;
        sigPi.row(d) = csigPi;
        c.row(d) = cc;
//        cout<<csigPi.t()<<" = "<<sum(csigPi)<<endl;
//        cout<<sigPi.row(d)<<" = "<<sum(sigPi.row(d))<<endl;
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
//        cout<<"z.rows="<<z.n_rows<<" vs phi_d.rows="<<mPhi[d].n_rows<<endl;
        return false;
      }else{
      return getWordTopics(z,mPhi[d]);
      }
    };

    bool getCorpTopicProportions(Col<double>& v, Col<double>& sigV) const
    {
      return getCorpTopicProportions(v,sigV,mA);
    };

    virtual Row<double> logP_w(uint32_t d) const 
    {
      cerr<<"has to be implemented in the child classes!!"<<endl;
      exit(0);
      return Row<double>(1);
    };

//    bool getWordDistr(Mat<double>& p){
//      uint32_t D=mZeta.size();
//      p.set_size(D,mNw);
////      assert(p.n_rows == D);
////      assert(p.n_cols == mNw);
////      cout<<"D="<<D<<endl;
//      for (uint32_t d=0; d<D; ++d){
//        p.row(d) = logP_w(d);
//      }
//      return true;
//    };

protected:
    Mat<double> mA; // corpus level Beta process alpha parameter for stickbreaking
    vector<Mat<double> > mZeta; // document level topic indices/pointers to corpus level topics (Multinomial) 
    vector<Mat<double> > mPhi; // document level word to doc level topic assignment (Multinomial)
    vector<Mat<double> > mGamma; // document level Beta distribution alpha parameter for stickbreaking

    Col<double> mPerp; // perplexity for each document

    uint32_t mK; // Corp level truncation
    uint32_t mT; // Doc level truncation
    uint32_t mNw; // size of dictionary



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
//      cout<<phi.n_rows<<" x "<<phi.n_cols<<endl;
      z.set_size(phi.n_rows);
      for (uint32_t i=0; i<z.n_elem; ++i){
        z[i] = multinomialMode(phi.row(i));
      }
      return true;
    };
};

