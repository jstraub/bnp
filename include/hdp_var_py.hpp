/* Copyright (c) 2012, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See LICENSE.txt or 
 * http://www.opensource.org/licenses/mit-license.php */

#pragma once

#include <hdp_var.hpp>
#include <hdp_var_base_py.hpp>

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <typeinfo>

#include <armadillo>

#include <boost/python.hpp>
#include <boost/python/wrapper.hpp>
#include <numpy/ndarrayobject.h> // for PyArrayObject

#ifdef PYTHON_2_6
  #include <python2.6/object.h> // for PyArray_FROM_O
#endif 
#ifdef PYTHON_2_7
  #include <python2.7/object.h> // for PyArray_FROM_O
#endif

using namespace boost::python;

template <class U>
class HDP_var_py : public HDP_var_base_py, public HDP_var<U>
{
public:
  HDP_var_py(const BaseMeasure<U>& base, double alpha, double gamma)
  :  HDP_var_base_py(0,0,0), HDP_var<U>(base,alpha,gamma)
  { };

  bool densityEst(uint32_t Nw, double kappa, uint32_t K, uint32_t T, uint32_t S)
  {
    return HDP_var<U>::densityEst(Nw,kappa,K,T,S);
  }

  /*
   * makes no copy of the external data x_i
   */
  uint32_t addDoc(const numeric::array& x_i)
  {
    return HDP_var<U>::addDoc(np2mat<U>(x_i));
  };

  uint32_t addHeldOut(const numeric::array& x_i)
  {
    return HDP_var<U>::addHeldOut(np2mat<U>(x_i));
  };


  /* 
   * after an initial densitiy estimate has been made using addDoc() and densityEst()
   * can use this to update the estimate with information from additional x 
   */
  bool updateEst(const numeric::array& x, double ro=0.75)
  {
    return HDP_var<U>::updateEst(np2mat<U>(x),ro);
  }
  bool updateEst_batch(double kappa, uint32_t S){
    return HDP_var<U>::updateEst_batch(kappa,S);
  }

  /* 
   * works on the data in lambda -> size has to be correct in order for this to work!
   * makes a copy of the internal labels vector
   */
  bool getLambda_py(numeric::array& lambda, uint32_t k)
  {
    Row<double> lambda_row;
    if(!HDP_var<U>::getLambda(lambda_row, k)){return false;} // works on the data in _mat
    Row<double> lambda_wrap=np2row<double>(lambda); 
    if(lambda_row.n_cols != lambda_wrap.n_cols)
      return false;
    else{
      lambda_wrap = lambda_row;
      return true;
    }
  };

  bool getCorpTopics_py(numeric::array& beta)
  {
    Mat<double> beta_mat;
    if(!HDP_var<U>::getCorpTopics(beta_mat)){return false;} // works on the data in _mat
    assignMat2np(beta_mat,beta);
    return true;
  };

};

typedef HDP_var_py<uint32_t> HDP_var_Dir_py;
typedef HDP_var_py<double> HDP_var_NIW_py;

//class HDP_var_ss_py : public HDP_var_base_py, public HDP_var_ss
//{
//public:
//  HDP_var_ss_py(const BaseMeasure<uint32_t>& base, double alpha, double gamma)
//  : HDP_var_base_py(0,0,0), HDP_var_ss(base,alpha,gamma)
//  {
//    //cout<<"Creating "<<typeid(this).name()<<endl;
//  };
//
//  bool densityEst(const numeric::array& x, const numeric::array& x_ho, double kappa, uint32_t K, uint32_t T, uint32_t S)
//  {
////    cout<<"mX.n_rows="<<x_mat.n_rows<<endl;
////    cout<<"mX_ho.n_rows="<<x_ho_mat.n_rows<<endl;
//    HDP_var_ss::densityEst(np2mat<uint32_t>(x),np2mat<uint32_t>(x_ho),kappa,K,T,S);
//    return true;
//  }
//
//  /* 
//   * after an initial densitiy estimate has been made using addDoc() and densityEst()
//   * can use this to update the estimate with information from additional x 
//   */
//  bool updateEst(const numeric::array& x, double kappa)
//  {
//    return HDP_var_ss::updateEst(np2row<uint32_t>(x),kappa);
//  }
//
//};

