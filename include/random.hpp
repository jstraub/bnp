/* Copyright (c) 2012, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See LICENSE.txt or 
 * http://www.opensource.org/licenses/mit-license.php */

#pragma once

#include <boost/random/mersenne_twister.hpp>
//#include <boost/random/uniform_int_distribution.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/uniform_01.hpp>

#include <armadillo>
#include <time.h>

using namespace arma;

class RandInt
{
public:
  RandInt(uint32_t limLower, uint32_t limUpper)
    : mGen(time(0)),  mDist(limLower,limUpper-1) // so we generate numbers in the range( upper - lower)
  {};

  uint32_t draw(void)
  {
    return mDist(mGen);
  };

  void draw(Col<uint32_t>& c)
  {
    for (uint32_t i=0; i<c.n_rows; ++i)
    {
      c(i)=mDist(mGen);
    }
  }
  Col<uint32_t> draw(uint32_t N)
  {
    Col<uint32_t> c(N);
    draw(c);
    return c;
  }


private:
  boost::mt19937 mGen;
  boost::uniform_int<> mDist;
};

class RandDisc
{
public:
  RandDisc() : mGen(time(0))
  { };

  double draw(void)
  {
    return mDist(mGen);
  };

  uint32_t draw(const Col<double>& pdf)
  {
    Col<double> cdf=cumsum(pdf);
    double r=mDist(mGen);
    for (uint32_t i=0; i<pdf.n_rows; ++i)
      if (r<cdf(i)){return i;}
    return pdf.n_rows-1; 
  };

private:
  Col<double> mPdf;
  boost::mt19937 mGen;
  boost::uniform_01<> mDist;
};


