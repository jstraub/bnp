/* Copyright (c) 2012, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See LICENSE.txt or 
 * http://www.opensource.org/licenses/mit-license.php */

#pragma once

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

