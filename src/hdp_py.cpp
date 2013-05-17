/* Copyright (c) 2012, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See LICENSE.txt or 
 * http://www.opensource.org/licenses/mit-license.php */

#include <baseMeasure.hpp>
#include <hdp_gibbs.hpp>
#include <hdp_var.hpp>
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

enum NpTypes{
  INT32 = 5,
  UINT32 = 6,
  INT64 = 7,
  UINT64 = 8,
  FLT32 = 11,
  FLT64 = 12, // default of python on my 64bit
}; // for value in PyArrayObject->descr->type_num

template<class U>
struct NpTyp
{
  const static NpTypes Num;
};
template<>
struct NpTyp<float>
{
  const static NpTypes Num=FLT32;
};
template<>
struct NpTyp<double>
{
  const static NpTypes Num=FLT64;
};
template<>
struct NpTyp<uint32_t>
{
  const static NpTypes Num=UINT32;
};
template<>
struct NpTyp<uint64_t>
{
  const static NpTypes Num=UINT64;
};
template<>
struct NpTyp<int32_t>
{
  const static NpTypes Num=INT32;
};
template<>
struct NpTyp<int64_t>
{
  const static NpTypes Num=INT64;
};

bool checkPyArr(PyArrayObject* a, const int ndims, const NpTypes npType)
{
	if (a == NULL) {
		//throw std::exception("Could not get NP array.");
		cerr<<"Could not get NP array."<<endl;
		return false;
//	}else if (a->descr->elsize != sizeof(double))
//	{
//		//throw std::exception("Must be double ndarray");
//		cerr<<"Must be double ndarray"<<endl;
//		return false;
	}else if(a->descr->type_num != npType)
	{
	  cerr<<"Wrong datatype on array ("<<a->descr->type_num<<" != "<<npType<<")"<<endl;
	  return false;
	}else if(a->descr->type_num == npType)
	{
	  if ((npType == FLT64 && a->descr->elsize != sizeof(double)) ||
	      (npType == FLT32 && a->descr->elsize != sizeof(float)) ||
	      (npType == UINT32 && a->descr->elsize != sizeof(uint32_t)) ||
	      (npType == UINT64 && a->descr->elsize != sizeof(uint64_t)) ||
	      (npType == INT32 && a->descr->elsize != sizeof(int32_t)) ||
	      (npType == INT64 && a->descr->elsize != sizeof(int64_t)))
	  {
	    cerr<<"Ensure that numpy datatype definitions are matching your architectures"<<endl;
	    return false;
	  }
	}else if (a->nd != ndims)
	{
		//throw std::exception("Wrong dimension on array.");
		cerr<<"Wrong dimension on array ("<<a->nd<<" != "<<ndims<<")"<<endl;
		return false;
	}else if (ndims == 2 && (a->strides[0] != a->dimensions[0] || a->strides[1] != 1))
		//((ndims == 1 && (a->strides[0] != 1))
	{
		cerr<<"Strides are not right (ndims="<< ndims <<" strides: "
				<< a->strides[0] <<"; "<< a->strides[1] <<")";
		return false;
	}

	//cout<<"Type="<<a->descr->type<< " Type_num=" <<a->descr->type_num<<endl;
	return true;
}

/*
 * converter from numpy to arma for read only const copies of the np matrix
 * the compllicating factor is that arma is col major and np is row major
 */
template<class U>
const Mat<U> np2mat(const numeric::array& np)
{
  // Get pointer to np array
  PyArrayObject* a = (PyArrayObject*)PyArray_FROM_O(np.ptr());
  if(!checkPyArr(a,2,NpTyp<U>::Num)) exit(0);
  // do not copy the data!
  //cout<<a->nd<<": "<<a->dimensions[0]<<"x"<<a->dimensions[1]<<endl;
  arma::Mat<U> A=arma::Mat<U>((U*)a->data,a->dimensions[1],a->dimensions[0],false,true);
  return A.t();
}

/*
 * for assigning data from an arma matrix to a numpy matrix
 * the compllicating factor is that arma is col major and np is row major
 */
template<class U>
void assignMat2np(const Mat<U>& A_mat, const numeric::array& np)
{
  // Get pointer to np array
  PyArrayObject* a = (PyArrayObject*)PyArray_FROM_O(np.ptr());
  if(!checkPyArr(a,2,NpTyp<U>::Num)) exit(0);
  // do not copy the data!
  //cout<<a->nd<<": "<<a->dimensions[0]<<"x"<<a->dimensions[1]<<endl;

  arma::Mat<U> a_wrap=arma::Mat<U>((U*)a->data,a->dimensions[1],a->dimensions[0],false,true); // verse the dimensions to getthe expected behavior
  if ((a_wrap.n_rows != A_mat.n_cols) || (a_wrap.n_cols != A_mat.n_rows)){
    cout<<"assignMat2np:: Problem  with size of arma and np matrices!"<<endl;
    assert(0);
  }
  a_wrap = A_mat.t(); // IMPORTANT: Arma is col major and np is row major -> let Arma do the conversion by assigning .t() all over the place!!!
}

template<class U>
Row<U> np2row(const numeric::array& np)
{
	// Get pointer to np array
	PyArrayObject* a = (PyArrayObject*)PyArray_FROM_O(np.ptr());
	if(!checkPyArr(a,1,NpTyp<U>::Num)) exit(0);
  if (a->nd == 1) { // I get a numpy 1 dim vector that I want to convert into an arma row
	  // do not copy the data!
	  return Row<U>((U*)a->data,a->dimensions[0],false,true);
  }else{ // I get a numpy 2 dim matrix (with hopefully only 1 row) that I want to convert into an arma row
	  // do not copy the data!
	  return Row<U>((U*)a->data,a->dimensions[1],false,true);
  }
}

template<class U>
Col<U> np2col(const numeric::array& np)
{
	// Get pointer to np array
	PyArrayObject* a = (PyArrayObject*)PyArray_FROM_O(np.ptr());
	if(!checkPyArr(a,1,NpTyp<U>::Num)) exit(0);
	// do not copy the data!
	return Col<U>((U*)a->data,a->dimensions[0],false,true);
}

class Dir_py : public Dir
{
public:
  Dir_py(const numeric::array& alphas) :
	  Dir(np2row<double>(alphas))
  {
	  //cout<<"alphas: "<<mAlphas<<endl;
  };
  Dir_py(const Dir_py& dir) :
	  Dir(dir)
  {
	  //cout<<"alphas: "<<mAlphas<<endl;
  };
};

class NIW_py : public NIW
{
public:
	NIW_py(const numeric::array& vtheta, double kappa,
			const numeric::array& Delta, double nu) :
        NIW(np2col<double>(vtheta),kappa,np2mat<double>(Delta),nu)
    //        cpVtheta(np2col<double>(vtheta)), cpDelta(np2mat<double>(Delta)),
//  NIW(cpVtheta,kappa,cpDelta,nu)
	{
	  //cout<<"Creating "<<typeid(this).name()<<endl;
	};
	NIW_py(const NIW_py& inw) :
		NIW(inw)
	{};
private:
	colvec cpVtheta;
	mat cpDelta;
};

template <class U>
class HDP_py : public HDP_gibbs<U>
{
public:
  HDP_py(const BaseMeasure<U>& base, double alpha, double gamma)
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


  // works on the data in z_i -> size has to be correct in order for this to work!
  // makes a copy of the internal labels vector
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

//typedef DP<uint32_t> DP_Dir;
//typedef DP<double> DP_INW;
typedef HDP_py<uint32_t> HDP_Dir;
typedef HDP_py<double> HDP_INW;

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

  /* 
   * works on the data in lambda -> size has to be correct in order for this to work!
   * makes a copy of the internal labels vector
   */
  bool getLambda_py(numeric::array& lambda, uint32_t k)
  {
    Row<double> lambda_row;
    if(!HDP_var_base::getLambda(lambda_row, k)){return false;} // works on the data in _mat
    Row<double> lambda_wrap=np2row<double>(lambda); 
    if(lambda_row.n_cols != lambda_wrap.n_cols)
      return false;
    else{
      lambda_wrap = lambda_row;
//      for (uint32_t i=0; i<lambda_wrap.n_rows; ++i)
//        lambda_wrap.at(i)=lambda_col.at(i);
      return true;
    }
  };

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

  bool getCorpTopics_py(numeric::array& beta)
  {
    Mat<double> beta_mat;
    if(!HDP_var_base::getCorpTopics(beta_mat)){return false;} // works on the data in _mat
//    Mat<double> beta_wrap=np2mat<double>(beta); 
//    if((beta_mat.n_rows != beta_wrap.n_rows)||(beta_mat.n_cols != beta_wrap.n_cols))
//      return false;
//    else{

      assignMat2np(beta_mat,beta);
      return true;
//    }
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

class HDP_var_py : public HDP_var_base_py, public HDP_var
{
public:
  HDP_var_py(const BaseMeasure<uint32_t>& base, double alpha, double gamma)
  :  HDP_var_base_py(0,0,0), HDP_var(base,alpha,gamma)
  {
    //cout<<"Creating "<<typeid(this).name()<<endl;
  };

  bool densityEst(uint32_t Nw, double kappa, uint32_t K, uint32_t T, uint32_t S)
  {
//    cout<<"mX.size()="<<HDP_var::mX.size()<<endl;
//    cout<<"mX_ho.size()="<<HDP_var::mX_ho.size()<<endl;
//    for (uint32_t i=0; i<HDP_var::mX.size(); ++i)
//      cout<<"  x_"<<i<<": "<<HDP_var::mX[i].n_rows<<"x"<<HDP_var::mX[i].n_cols<<endl;
//
    return HDP_var::densityEst(Nw,kappa,K,T,S);
  }

  // makes no copy of the external data x_i
  uint32_t addDoc(const numeric::array& x_i)
  {
    return HDP_var::addDoc(np2mat<uint32_t>(x_i));
  };

  uint32_t addHeldOut(const numeric::array& x_i)
  {
    return HDP_var::addHeldOut(np2mat<uint32_t>(x_i));
  };


  // after an initial densitiy estimate has been made using addDoc() and densityEst()
  // can use this to update the estimate with information from additional x 
  bool updateEst(const numeric::array& x, double ro=0.75)
  {
    return HDP_var::updateEst(np2mat<uint32_t>(x),ro);
  }

  bool updateEst_batch(double kappa, uint32_t S){
    return HDP_var::updateEst_batch(kappa,S);
  }

};

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


class TestNp2Arma_py
{
  public:
    TestNp2Arma_py()
    {
      Amat = Mat<double>(3,3);
      Amat<<1<<2<<3<<endr
          <<4<<5<<6<<endr
          <<7<<8<<9<<endr;
      Arect = Mat<double>(4,3);
      Arect<<1<<2<<3<<endr
          <<4<<5<<6<<endr
          <<7<<8<<9<<endr
          <<10<<11<<12<<endr;
      Acol = Col<double>(3);
      Acol<<1<<2<<3;
      Arow = Row<double>(3);
      Arow<<1<<2<<3;
    };

  void getAmat(const numeric::array& a)
  {
    cout<<Amat<<endl;
    assignMat2np(Amat,a);
  }

  void getArect(const numeric::array& a)
  {
    cout<<Arect<<endl;
    assignMat2np(Arect,a);
  }

  void putArect(const numeric::array& a)
  {
    const Mat<double> a_wrap=np2mat<double>(a);
    cout<<"a_wrap:"<<a_wrap<<endl;
  }

  void getAcol(const numeric::array& a)
  {
    cout<<Acol<<endl;
    Col<double> a_wrap=np2col<double>(a);
    a_wrap = Acol;
  }

  void getArow(const numeric::array& a)
  {
    cout<<Arow<<endl;
    Row<double> a_wrap=np2row<double>(a);
    a_wrap = Arow;
  }

  private:
  Mat<double> Amat;
  Mat<double> Arect;
  Col<double> Acol;
  Row<double> Arow;

};

BOOST_PYTHON_MODULE(libbnp)
{
	import_array();
	boost::python::numeric::array::set_module_and_type("numpy", "ndarray");

	class_<Dir_py>("Dir", init<numeric::array>())
			.def(init<Dir_py>());
	class_<NIW_py>("NIW",init<const numeric::array, double,
			const numeric::array, double>())
			.def(init<NIW_py>());

	//	class_<DP_Dir>("DP_Dir",init<Dir_py,double>());
	//	class_<DP_INW>("DP_INW",init<NIW_py,double>());

	class_<HDP_Dir>("HDP_Dir",init<Dir_py&,double,double>())
        .def("densityEst",&HDP_Dir::densityEst)
        .def("getClassLabels",&HDP_Dir::getClassLabels)
        .def("addDoc",&HDP_Dir::addDoc)
        .def("addHeldOut",&HDP_Dir::addHeldOut)
        .def("getPerplexity",&HDP_Dir::getPerplexity);
  //      .def_readonly("mGamma", &HDP_Dir::mGamma);

	class_<HDP_INW>("HDP_INW",init<NIW_py&,double,double>())
        .def("densityEst",&HDP_INW::densityEst)
        .def("getClassLabels",&HDP_INW::getClassLabels)
        .def("addDoc",&HDP_INW::addDoc);
  //      .def_readonly("mGamma", &HDP_INW::mGamma);

	class_<HDP_var_py>("HDP_var",init<Dir_py&,double,double>())
        .def("densityEst",&HDP_var_py::densityEst)
        //TODO: not sure that one works: .def("updateEst",&HDP_var_py::updateEst)
        .def("updateEst_batch",&HDP_var_py::updateEst_batch)
        .def("addDoc",&HDP_var_py::addDoc)
        .def("addHeldOut",&HDP_var_py::addHeldOut)
        .def("getPerplexity",&HDP_var_py::getPerplexity_py)
        .def("getA",&HDP_var_py::getA_py)
        .def("getLambda",&HDP_var_py::getLambda_py)
        .def("getDocTopics",&HDP_var_py::getDocTopics_py)
        .def("getWordTopics",&HDP_var_py::getWordTopics_py)
        .def("getCorpTopicProportions",&HDP_var_py::getCorpTopicProportions_py)
        .def("getCorpTopics",&HDP_var_py::getCorpTopics_py)
        .def("getWordDistr",&HDP_var_py::getWordDistr_py);
   //     .def_readonly("mGamma", &HDP_var_py::mGamma);
//        .def("perplexity",&HDP_var_py::perplexity)


//	class_<HDP_var_ss_py>("HDP_var_ss",init<Dir_py&,double,double>())
//        .def("densityEst",&HDP_var_ss_py::densityEst)
//        //TODO: not sure that one works .def("updateEst",&HDP_var_ss_py::updateEst)
//        .def("getPerplexity",&HDP_var_ss_py::getPerplexity_py)
//        .def("getA",&HDP_var_ss_py::getA_py)
//        .def("getLambda",&HDP_var_ss_py::getLambda_py)
//        .def("getDocTopics",&HDP_var_ss_py::getDocTopics_py)
//        .def("getWordTopics",&HDP_var_ss_py::getWordTopics_py)
//        .def("getCorpTopicProportions",&HDP_var_ss_py::getCorpTopicProportions_py)
//        .def("getCorpTopics",&HDP_var_ss_py::getCorpTopics_py)
//        .def("getWordDistr",&HDP_var_ss_py::getWordDistr_py);
//        .def("perplexity",&HDP_var_ss_py::perplexity)

  class_<TestNp2Arma_py>("TestNp2Arma",init<>())
    .def("getAmat",&TestNp2Arma_py::getAmat)
    .def("getArect",&TestNp2Arma_py::getArect)
    .def("putArect",&TestNp2Arma_py::putArect)
    .def("getAcol",&TestNp2Arma_py::getAcol)
    .def("getArow",&TestNp2Arma_py::getArow);

}

