/* Copyright (c) 2012, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See LICENSE.txt or 
 * http://www.opensource.org/licenses/mit-license.php */

#pragma once

//#include <baseMeasure_py.hpp>
#include <hdp_var_base.hpp>
// using the hdp which utilizes sufficient statistics 
//#include <hdp_var_ss.hpp>

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


/*
 * Class to handle access to all comon variables for var HDPs
 *
 * http://en.wikipedia.org/wiki/Virtual_inheritance
 */
class HDP_var_base_py : public virtual HDP_var_base 
{
  public:
  HDP_var_base_py(uint32_t K, uint32_t T, uint32_t Nw)
    : HDP_var_base(K,T,Nw)
  {};

  bool getWordDistr_py(const numeric::array& p)
  {
    Mat<double> p_mat; // can do this since x_mat gets copied inside    
    if (HDP_var_base::getWordDistr(p_mat))
    {
      assignMat2np(p_mat,p);
      return true;
    }else{
      return false;
    }
  }


  void getA_py(numeric::array& a)
  {
    assignMat2np(HDP_var_base::mA,a);
  };

  void getPerplexity_py(numeric::array& perp)
  {
    Col<double> perp_wrap=np2col<double>(perp); 
    HDP_var_base::getPerplexity(perp_wrap);

   //perp_wrap = HDP_var_base::mPerp;
    //perp_wrap = mPerp;
//    for (uint32_t i=0; i<perp_wrap.n_rows; ++i)
//      perp_wrap.at(i)=mPerp.at(i);
  };

  /*
   * @param sigPi get topic probabilities sigPi
   * @param c pointer to corpus level topics
   */
  bool getDocTopics_py(numeric::array& pi, numeric::array& sigPi, numeric::array& c)
  {
    Mat<double> sigPi_mat;
    Mat<double> pi_mat;
    Mat<uint32_t> c_mat;
    if(!HDP_var_base::getDocTopics(pi_mat, sigPi_mat, c_mat)){return false;} // works on the data in _mat
//    Mat<double> sigPi_wrap=np2mat<double>(sigPi); 
//    Mat<double> pi_wrap=np2mat<double>(pi); 
//    Mat<uint32_t> c_wrap=np2mat<uint32_t>(c); 
//    if((sigPi_mat.n_rows != sigPi_wrap.n_rows) || (c_mat.n_rows != c_wrap.n_rows) || (pi_mat.n_rows != pi_wrap.n_rows) || 
//        (sigPi_mat.n_cols != sigPi_wrap.n_cols) || (c_mat.n_cols != c_wrap.n_cols) || (pi_mat.n_cols != pi_wrap.n_cols)) {
//      cout<<"sigPi_mat: "<<size(sigPi_mat)<<" vs sigPi_wrap: "<<size(sigPi_wrap)<<endl;
//      cout<<"pi_mat: "<<size(pi_mat)<<" vs pi_wrap: "<<size(pi_wrap)<<endl;
//      cout<<"c_mat: "<<size(c_mat)<<" vs c_wrap: "<<size(c_wrap)<<endl;
//      return false;
//    }else{

      assignMat2np(pi_mat,pi);
      assignMat2np(sigPi_mat,sigPi);
      assignMat2np(c_mat,c);
      return true;
//    }
  };

  bool getCorpTopicProportions_py(numeric::array& v, numeric::array& sigV)
  {
    Col<double> sigV_col;
    Col<double> v_col;
    if(!HDP_var_base::getCorpTopicProportions(v_col,sigV_col)){return false;} // works on the data in _mat
    Col<double> sigV_wrap=np2col<double>(sigV); 
    Col<double> v_wrap=np2col<double>(v); 
    if((sigV_col.n_rows != sigV_wrap.n_rows) || (v_col.n_rows != v_wrap.n_rows))
      return false;
    else{
      v_wrap = v_col;
      sigV_wrap = sigV_col;
//      for (uint32_t i=0; i<v_wrap.n_rows; ++i)
//        v_wrap.at(i)=v_col.at(i);
//      for (uint32_t i=0; i<sigV_wrap.n_rows; ++i)
//        sigV_wrap.at(i)=sigV_col.at(i);
      return true;
    }
  }; 


  /*
   * Gets the indicators for each word pointing to a doc level topic.
   * Need to give it the doc of interest, because all documents have 
   * different numbers of words and hence z is different for all docs
   *
   * @param d points to the document of interest. 
   */
  bool getWordTopics_py(numeric::array& z, uint32_t d)
  {
    Col<uint32_t> z_col;
    if(!HDP_var_base::getWordTopics(z_col, d)){return false;} // works on the data in _mat
    //cout<<"x_col="<<z_col.t()<<endl;
    Col<uint32_t> z_wrap=np2col<uint32_t>(z); 
    if(z_col.n_rows != z_wrap.n_rows)
      return false;
    else{
      z_wrap = z_col;
//      for (uint32_t i=0; i<z_wrap.n_rows; ++i)
//        z_wrap.at(i)=z_col.at(i);
      return true;
    }
  };
};
