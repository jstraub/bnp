/* Copyright (c) 2012, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See LICENSE.txt or 
 * http://www.opensource.org/licenses/mit-license.php */

#include <baseMeasure.hpp>
#include <numpyArma.hpp>

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

  void asRow(const numeric::array& row)
  {
    Row<double> row_arma = Dir::asRow();
    Row<double> row_wrap = np2row<double>(row);
    row_wrap = row_arma;
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


  void asRow(const numeric::array& row)
  {
    Row<double> row_arma = NIW::asRow();
    Row<double> row_wrap = np2row<double>(row);
    row_wrap = row_arma;
  };

};

