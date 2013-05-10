

#include <armadillo>

#include "probabilityHelpers.hpp"

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE probabilityHelpers
//#include <boost/test/auto_unit_test.hpp>
#include <boost/test/unit_test.hpp>

using namespace std;
using namespace arma;

BOOST_AUTO_TEST_CASE( probabilityHelpersTest )
{
  uint32_t T = 5;
  Col<double> prop(T+1);
  Col<double> prop_true(T+1);
  Col<double> v(T);

  v.zeros();
  stickBreaking(prop, v);  
  cout<<prop.t();
  prop_true << 0.0 << 0.0 << 0.0 << 0.0 <<0.0 << 1.0;
  BOOST_CHECK_EQUAL( int32_t(sum(prop != prop_true)), int32_t(0) ); 

  v <<0.5<<0.5<<0.5<<0.5<<0.5;
  stickBreaking(prop, v);  
  cout<<prop.t();
  prop_true << 0.5 << 0.25 << 0.125 << 0.0625 <<0.03125 << 0.03125;
  BOOST_CHECK_EQUAL( int32_t(sum(prop != prop_true)), int32_t(0) ); 

  v <<1.0<<1.0<<1.0<<1.0<<1.0;
  stickBreaking(prop, v);  
  cout<<prop.t();
  prop_true << 1.0 << 0.0 << 0.0 << 0.0 <<0.0 << 0.0;
  BOOST_CHECK_EQUAL( int32_t(sum(prop != prop_true)), int32_t(0) ); 



}
