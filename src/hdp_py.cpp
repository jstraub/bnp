/* Copyright (c) 2012, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See LICENSE.txt or 
 * http://www.opensource.org/licenses/mit-license.php */

#include <hdp.hpp>

#include <assert.h>
#include <typeinfo>

#include <armadillo>

#include <boost/python.hpp>
#include <boost/python/wrapper.hpp>
#include <python2.7/object.h> // for PyArray_FROM_O
#include <numpy/ndarrayobject.h> // for PyArrayObject

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

	cout<<"Type="<<a->descr->type<< " Type_num=" <<a->descr->type_num<<endl;
	return true;
}

template<class U>
Mat<U> np2mat(const numeric::array& np)
{
  // Get pointer to np array
  PyArrayObject* a = (PyArrayObject*)PyArray_FROM_O(np.ptr());
  if(!checkPyArr(a,2,NpTyp<U>::Num)) exit(0);
  // do not copy the data!
  return arma::Mat<U>((U*)a->data,a->dimensions[0],a->dimensions[1],false,true);
}

template<class U>
Row<U> np2row(const numeric::array& np)
{
	// Get pointer to np array
	PyArrayObject* a = (PyArrayObject*)PyArray_FROM_O(np.ptr());
	if(!checkPyArr(a,1,NpTyp<U>::Num)) exit(0);
	// do not copy the data!
	return Row<U>((U*)a->data,a->dimensions[0],false,true);
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

//class Dir_wrap : public BaseMeasure<uint32_t>
//{
//public:
//  Dir_wrap(const numeric::array& alphas)
//  : mAlphas(np2row<double>(alphas)), mAlpha0(sum(mAlphas))
//  {};
//
//  double predictiveProb(const Col<uint32_t>& x_q, const Mat<uint32_t>& x_given) const
//  {
//    return BaseMeasure<uint32_t>::predictiveProb(x_q, x_given);
//  };
//  double predictiveProb(const Col<uint32_t>& x_q) const
//  {
//    return BaseMeasure<uint32_t>::predictiveProb(x_q);
//  };
//
//  Row<double> mAlphas;
//  double mAlpha0;
//};

class Dir_py : public Dir
{
public:
  Dir_py(const numeric::array& alphas) :
	  Dir(np2row<double>(alphas))
  {
	  cout<<"alphas: "<<mAlphas<<endl;
  };
  Dir_py(const Dir_py& dir) :
	  Dir(dir)
  {
	  cout<<"alphas: "<<mAlphas<<endl;
  };
};

class InvNormWishart_py : public InvNormWishart
{
public:
	InvNormWishart_py(const numeric::array& vtheta, double kappa,
			const numeric::array& Delta, double nu) :
        InvNormWishart(np2col<double>(vtheta),kappa,np2mat<double>(Delta),nu)
    //        cpVtheta(np2col<double>(vtheta)), cpDelta(np2mat<double>(Delta)),
//  InvNormWishart(cpVtheta,kappa,cpDelta,nu)
	{
	  cout<<"Creating "<<typeid(this).name()<<endl;
	};
	InvNormWishart_py(const InvNormWishart_py& inw) :
		InvNormWishart(inw)
	{};
private:
	colvec cpVtheta;
	mat cpDelta;
};

template <class U>
class HDP_py : public HDP<U>
{
public:
  HDP_py(const BaseMeasure<U>& base, double alpha, double gamma)
  : HDP<U>(base,alpha,gamma)
  {
    cout<<"Creating "<<typeid(this).name()<<endl;
  };

  bool densityEst(uint32_t K0=10, uint32_t T0=10, uint32_t It=10)
  {
    cout<<"mX.size()="<<HDP<U>::mX.size()<<endl;
    for (uint32_t i=0; i<HDP<U>::mX.size(); ++i)
      cout<<"  x_"<<i<<": "<<HDP<U>::mX[i].n_rows<<"x"<<HDP<U>::mX[i].n_cols<<endl;

    return HDP<U>::densityEst(K0, T0, It);
  }

  // makes no copy of the external data x_i
  uint32_t addDoc(const numeric::array& x_i)
  {
    Mat<U> x_i_mat=np2mat<U>(x_i); // can do this since x_i_mat gets copied inside
    return HDP<U>::addDoc(x_i_mat);
  };

  // works on the data in z_i -> size has to be correct in order for this to work!
  // makes a copy of the internal labels vector
  bool getClassLabels(numeric::array& z_i, uint32_t i)
  {
    Col<uint32_t> z_i_col;
    if(!HDP<U>::getClassLabels(z_i_col, i)){return false;} // works on the data in z_i_mat
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


BOOST_PYTHON_MODULE(libbnp)
{
	import_array();
	boost::python::numeric::array::set_module_and_type("numpy", "ndarray");

	//def("np2arma",&np2arma);
//	class_<BaseMeasure<uint32_t> >("BaseMeasure_uint", no_init);
//	class_<BaseMeasure<double> >("BaseMeasure_double", no_init);

//	class_<Dir_wrap, bases<BaseMeasure<uint32_t> > >("Dir", init<numeric::array>());

	class_<Dir_py>("Dir_py", init<numeric::array>())
			.def(init<Dir_py>());
	class_<InvNormWishart_py>("INW",init<const numeric::array, double,
			const numeric::array, double>())
			.def(init<InvNormWishart_py>());

//	class_<DP_Dir>("DP_Dir",init<Dir_py,double>());
	//	class_<DP_INW>("DP_INW",init<InvNormWishart_py,double>());
	class_<HDP_Dir>("HDP_Dir",init<Dir_py&,double,double>())
        .def("densityEst",&HDP_Dir::densityEst)
        .def("getClassLabels",&HDP_Dir::getClassLabels)
        .def("addDoc",&HDP_Dir::addDoc)
        .def_readonly("mGamma", &HDP_Dir::mGamma);

	class_<HDP_INW>("HDP_INW",init<InvNormWishart_py&,double,double>())
        .def("densityEst",&HDP_INW::densityEst)
        .def("getClassLabels",&HDP_INW::getClassLabels)
        .def("addDoc",&HDP_INW::addDoc)
        .def_readonly("mGamma", &HDP_INW::mGamma);
}

