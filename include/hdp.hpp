/* Copyright (c) 2012, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See LICENSE.txt or 
 * http://www.opensource.org/licenses/mit-license.php */

#pragma once

#include "baseMeasure.hpp"

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
    {
      //    cout<<"Creating "<<typeid(this).name()<<endl;
    };

    ~HDP()
    {	};

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

    virtual Row<double> logP_w(uint32_t d) const=0;


    double perplexity(const Mat<U>& x_ho, const Row<double>& logP) const
    {
      assert(x_ho.n_rows==1);

      uint32_t N = x_ho.n_cols;
      double perp = 0.0;
      for (uint32_t i=0; i<N; ++i){
        //cout<<"c_z_n = "<<c[z[w]]<<" z_n="<<z[w]<<" w="<<w<<" N="<<N<<" x_w="<<x_ho[w]<<" topics.shape="<<topics.n_rows<<" "<<topics.n_cols;
          perp -= logP(x_ho(i));
        //cout<<"w="<<w<<"\tx_ho_w="<<x_ho[w]<<"\tlogP="<<logP[w]<<"\tperp+="<<-double(x_ho[w])*logP[w]<<endl;
      } cout<<endl;
      perp /= double(N);
      perp /= log(2.0); // since it is log base 2 in the perplexity formulation!
      perp = pow(2.0,perp);

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

