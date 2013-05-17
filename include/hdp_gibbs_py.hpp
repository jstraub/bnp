/* Copyright (c) 2012, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See LICENSE.txt or 
 * http://www.opensource.org/licenses/mit-license.php */

#include <baseMeasure.hpp>
#include <hdp_gibbs.hpp>

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
class HDP_gibbs_py : public HDP_gibbs<U>
{
public:
  HDP_gibbs_py(const BaseMeasure<U>& base, double alpha, double gamma)
  : HDP_gibbs<U>(base,alpha,gamma)
  {
    //cout<<"Creating "<<typeid(this).name()<<endl;
  };

  bool densityEst(uint32_t Nw, uint32_t K0, uint32_t T0, uint32_t It)
  {
//    cout<<"mX.size()="<<HDP<U>::mX.size()<<endl;
//    for (uint32_t i=0; i<HDP<U>::mX.size(); ++i)
//      cout<<"  x_"<<i<<": "<<HDP<U>::mX[i].n_cols<<": "<<HDP<U>::mX[i]<<endl;

    return HDP_gibbs<U>::densityEst(Nw, K0, T0, It);
  }

  // makes no copy of the external data x_i
  uint32_t addDoc(const numeric::array& x_i)
  {
    return HDP_gibbs<U>::addDoc(np2mat<U>(x_i));
  };

  uint32_t addHeldOut(const numeric::array& x_i)
  {
    return HDP_gibbs<U>::addHeldOut(np2mat<U>(x_i));
  };


  void getPerplexity(numeric::array& perp)
  {
    Row<double> perp_wrap=np2row<double>(perp); 
    perp_wrap = HDP_gibbs<U>::perplexity();
  };

  /* 
   * works on the data in z_i -> size has to be correct in order for this to work!
   * makes a copy of the internal labels vector
   */
  bool getClassLabels(numeric::array& z_i, uint32_t i)
  {
    Col<uint32_t> z_i_col;
    if(!HDP_gibbs<U>::getClassLabels(z_i_col, i)){return false;} // works on the data in z_i_mat
    Col<uint32_t> z_i_wrap=np2col<uint32_t>(z_i); // can do this since x_i_mat gets copied inside
    if(z_i_col.n_rows != z_i_wrap.n_rows)
      return false;
    else{
      for (uint32_t i=0; i<z_i_wrap.n_rows; ++i)
        z_i_wrap.at(i)=z_i_col.at(i);
      return true;
    }
  };
};

typedef HDP_gibbs_py<uint32_t> HDP_gibbs_Dir;
typedef HDP_gibbs_py<double> HDP_gibbs_NIW;

