/*
 * dp.hpp
 *
 *  Created on: Feb 1, 2013
 *      Author: jstraub
 */

#ifndef DP_HPP_
#define DP_HPP_

#include "baseMeasure.hpp"

#include <armadillo>

using namespace std;
using namespace arma;

template<class U>
class DP
{
public:
  DP(const BaseMeasure<U>& base, double alpha)
  : mH(base), mAlpha(alpha)
  {};

  ~DP()
  { };

  const BaseMeasure<U>& mH; // base measure
  double mAlpha;
private:
};





#endif /* DP_HPP_ */
