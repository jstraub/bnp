/* Copyright (c) 2012, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See LICENSE.txt or 
 * http://www.opensource.org/licenses/mit-license.php */

#include <baseMeasure_py.hpp>
#include <hdp_gibbs_py.hpp>
#include <hdp_var_py.hpp>
#include <hdp_var_base_py.hpp>
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


BOOST_PYTHON_MODULE(libbnp)
{
	import_array();
	boost::python::numeric::array::set_module_and_type("numpy", "ndarray");

	class_<Dir_py>("Dir", init<numeric::array>())
			.def(init<Dir_py>())
      .def("asRow",&Dir_py::asRow)
      .def("rowDim",&Dir_py::rowDim);

	class_<NIW_py>("NIW",init<const numeric::array, double, const numeric::array, double>())
			.def(init<NIW_py>())
      .def("asRow",&NIW_py::asRow)
      .def("rowDim",&NIW_py::rowDim);

	//	class_<DP_Dir>("DP_Dir",init<Dir_py,double>());
	//	class_<DP_INW>("DP_INW",init<NIW_py,double>());

	class_<HDP_gibbs_Dir>("HDP_gibbs_Dir",init<Dir_py&,double,double>())
        .def("densityEst",&HDP_gibbs_Dir::densityEst)
        .def("getClassLabels",&HDP_gibbs_Dir::getClassLabels)
        .def("addDoc",&HDP_gibbs_Dir::addDoc)
        .def("addHeldOut",&HDP_gibbs_Dir::addHeldOut)
        .def("getPerplexity",&HDP_gibbs_Dir::getPerplexity);
  //      .def_readonly("mGamma", &HDP_Dir::mGamma);

	class_<HDP_gibbs_NIW>("HDP_gibbs_NIW",init<NIW_py&,double,double>())
        .def("densityEst",&HDP_gibbs_NIW::densityEst)
        .def("getClassLabels",&HDP_gibbs_NIW::getClassLabels)
        .def("addDoc",&HDP_gibbs_NIW::addDoc);
  //      .def_readonly("mGamma", &HDP_NIW::mGamma);

	class_<HDP_var_Dir_py>("HDP_var_Dir",init<Dir_py&,double,double>())
        .def("densityEst",&HDP_var_Dir_py::densityEst)
        //TODO: not sure that one works: .def("updateEst",&HDP_var_Dir_py::updateEst)
        .def("updateEst_batch",&HDP_var_Dir_py::updateEst_batch)
        .def("addDoc",&HDP_var_Dir_py::addDoc)
        .def("addHeldOut",&HDP_var_Dir_py::addHeldOut)
        .def("getPerplexity",&HDP_var_Dir_py::getPerplexity_py)
        .def("getA",&HDP_var_Dir_py::getA_py)
        .def("getTopicPriorDescriptionLength",&HDP_var_Dir_py::getTopicPriorDescriptionLength)
        .def("getLambda",&HDP_var_Dir_py::getLambda_py)
        .def("getDocTopics",&HDP_var_Dir_py::getDocTopics_py)
        .def("getWordTopics",&HDP_var_Dir_py::getWordTopics_py)
        .def("getCorpTopicProportions",&HDP_var_Dir_py::getCorpTopicProportions_py)
        .def("getTopicsDescriptionLength",&HDP_var_Dir_py::getTopicsDescriptionLength)
        .def("getCorpTopics",&HDP_var_Dir_py::getCorpTopics_py)
        .def("getWordDistr",&HDP_var_Dir_py::getWordDistr_py);
   //     .def_readonly("mGamma", &HDP_var_Dir_py::mGamma);
//        .def("perplexity",&HDP_var_Dir_py::perplexity)

	class_<HDP_var_NIW_py>("HDP_var_NIW",init<NIW_py&,double,double>())
        .def("densityEst",&HDP_var_NIW_py::densityEst)
        //TODO: not sure that one works: .def("updateEst",&HDP_var_NIW_py::updateEst)
        .def("updateEst_batch",&HDP_var_NIW_py::updateEst_batch)
        .def("addDoc",&HDP_var_NIW_py::addDoc)
        .def("addHeldOut",&HDP_var_NIW_py::addHeldOut)
        .def("getPerplexity",&HDP_var_NIW_py::getPerplexity_py)
        .def("getA",&HDP_var_NIW_py::getA_py)
        .def("getTopicPriorDescriptionLength",&HDP_var_NIW_py::getTopicPriorDescriptionLength)
        .def("getLambda",&HDP_var_NIW_py::getLambda_py)
        .def("getDocTopics",&HDP_var_NIW_py::getDocTopics_py)
        .def("getWordTopics",&HDP_var_NIW_py::getWordTopics_py)
        .def("getCorpTopicProportions",&HDP_var_NIW_py::getCorpTopicProportions_py)
        .def("getTopicsDescriptionLength",&HDP_var_NIW_py::getTopicsDescriptionLength)
        .def("getCorpTopics",&HDP_var_NIW_py::getCorpTopics_py)
        .def("getWordDistr",&HDP_var_NIW_py::getWordDistr_py);
   //     .def_readonly("mGamma", &HDP_var_NIW_py::mGamma);
//        .def("perplexity",&HDP_var_NIW_py::perplexity)

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
    .def("resizeAmat",&TestNp2Arma_py::resizeAmat)
    .def("getArow",&TestNp2Arma_py::getArow);

}

