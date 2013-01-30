/* Copyright (c) 2012, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See LICENSE.txt or 
 * http://www.opensource.org/licenses/mit-license.php */

#include <hdp.hpp>

#include <assert.h>
#include <armadillo>

#include <boost/python.hpp>
#include <python2.7/object.h> // for PyArray_FROM_O
#include <numpy/ndarrayobject.h> // for PyArrayObject

using namespace boost::python;

bool checkPyArr(PyArrayObject* a, const int ndims)
{
	if (a == NULL) {
		//throw std::exception("Could not get NP array.");
		cerr<<"Could not get NP array."<<endl;
		return false;
	}else if (a->descr->elsize != sizeof(double))
	{
		//throw std::exception("Must be double ndarray");
		cerr<<"Must be double ndarray"<<endl;
		return false;
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
	return true;
}

arma::Mat<double> np2mat(const numeric::array& np)
{
	// Get pointer to np array
	PyArrayObject* a = (PyArrayObject*)PyArray_FROM_O(np.ptr());
	if(!checkPyArr(a,2)) exit(0);
	// do not copy the data!
	return arma::Mat<double>((double*)a->data,a->dimensions[0],a->dimensions[1],false,true);
}

Row<double> np2row(const numeric::array& np)
{
	// Get pointer to np array
	PyArrayObject* a = (PyArrayObject*)PyArray_FROM_O(np.ptr());
	if(!checkPyArr(a,1)) exit(0);
	// do not copy the data!
	return arma::Row<double>((double*)a->data,a->dimensions[0],false,true);
}

Col<double> np2col(const numeric::array& np)
{
	// Get pointer to np array
	PyArrayObject* a = (PyArrayObject*)PyArray_FROM_O(np.ptr());
	if(!checkPyArr(a,1)) exit(0);
	// do not copy the data!
	return arma::Row<double>((double*)a->data,a->dimensions[0],false,true);
}

class Dir_py : public Dir
{
public:
  Dir_py(const numeric::array& alphas) :
	  Dir(np2row(alphas))
  {
	  cout<<"alphas: "<<mAlphas<<endl;
  };
  void print()
  {
	  cout<<"alphas: "<<mAlphas<<endl;
  };
};



BOOST_PYTHON_MODULE(libhdp)
{
	import_array();
	boost::python::numeric::array::set_module_and_type("numpy", "ndarray");

	//def("np2arma",&np2arma);

	class_<Dir_py>("Dir", init<numeric::array>())
			.def("print", &Dir_py::print);
	//
}

