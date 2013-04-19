/* Copyright (c) 2012, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See LICENSE.txt or 
 * http://www.opensource.org/licenses/mit-license.php */

#include "hdp_var.hpp"
#include "hdp_gibbs.hpp"

#include <iostream>
#include <fstream>
#include <vector>

#include <armadillo>

using namespace std;
using namespace arma;

int main(int argc, char** argv)
{

  cout<<" ------------------------ Dir base measure online ------------------- "<<endl;
  // inverse normal wishart base measure
//  mat x_rand(90,1);
//  x_rand.randu();
//  x_rand += 12.0;
  vector<Row<uint32_t> > x(12,zeros<Mat<uint32_t> >(90,1));

  for (uint32_t d=0; d<12; d+=3)
  {
    x[d].zeros();
    x[d].cols(0,29) += 1;
    x[d].cols(30,59) += 2;
    x[d+1].zeros();
    x[d+1].cols(0,29) += 1;
    x[d+1].cols(60,89) += 1;
    x[d+2].zeros();
    //x[d+2].rows(0,29) += 1;
    //x[d+2].rows(60,89) += 1;
  }
  uint32_t Nw = 3;
//  x[2].randn();
//  x[2].rows(0,29) -= 8;
//  x[3].randn();
//  x[3].rows(0,59) += 4;
//  x[4].randn();
//  x[4].rows(0,29) += 4;
  Row<double> alphas(3);
  alphas.ones();
  alphas(0)=30;
  alphas(1)=30;
  alphas(2)=30;
  //alphas *= 1.1; // smaller alpha means more uncertainty in the where the good distributions are
  double alpha =1.0, gamma=1000.0;
  Dir dir(alphas);
  HDP_var hdp_onl(dir, alpha, gamma);

  hdp_onl.densityEst(x,4,0.6,10,2,1);

  cout<<"x:"<<endl;
  for(uint32_t j=0; j<x.size(); ++j)
  {
    for(uint32_t i=0; i<x[j].n_cols; ++i)
      cout<<x[j](i)<<" ";
    cout<<endl;
  }

  return 0;

  cout<<" ---------------------------------- Dir base measure ---------------------------" <<endl;
  HDP_gibbs<uint32_t> hdp_dir(dir, alpha, gamma);

  vector<Row<uint32_t> > z_ji = hdp_dir.densityEst(x,Nw,10,10,100);
  uint32_t J=z_ji.size();

  cout<<"z_ji:"<<endl;
  for(uint32_t j=0; j<J; ++j)
  {
    for(uint32_t i=0; i<z_ji[j].n_elem; ++i)
      cout<<z_ji[j](i)<<" ";
    cout<<endl;
  }
  cout<<"x:"<<endl;
  for(uint32_t j=0; j<x.size(); ++j)
  {
    for(uint32_t i=0; i<x[j].n_cols; ++i)
      cout<<x[j](i)<<" ";
    cout<<endl;
  }

  
  return 0;

// not working for now - do not see a use for it currently!
//
//  cout<<" ------------------------ inverse normal wishart base measure ------------------- "<<endl;
//  // inverse normal wishart base measure
//  vector<mat> xx(2,zeros<mat>(90,2));
//  xx[0].randn();
//  xx[0].rows(0,29) += 8;
//  xx[0].rows(30,59) -= 8;
//  xx[1].randn();
//  xx[1].rows(0,29) += 8;
////  xx[2].randn();
////  xx[2].rows(0,29) -= 8;
////  xx[3].randn();
////  xx[3].rows(0,59) += 4;
////  xx[4].randn();
////  xx[4].rows(0,29) += 4;
////
//  colvec vtheta;
//  vtheta << 0.0 << 0.0;
//  mat Delta;
//  Delta << 2.0 << 0.0 <<endr
//        << 0.0 << 2.0 <<endr;
//  double kappa=1.0, nu=3.1;
//  alpha=1.0;
//  gamma=1.0;
//  InvNormWishart inw(vtheta, kappa,Delta, nu);
//  HDP_gibbs<double> hdp_inw(inw, alpha, gamma);
//
//  z_ji = hdp_inw.densityEst(xx,10,10,20);
//  J=z_ji.size();
//  cout<<"z_ji:"<<endl;
//  for(uint32_t j=0; j<J; ++j)
//  {
//    for(uint32_t i=0; i<z_ji[j].n_elem; ++i)
//      cout<<z_ji[j](i)<<" ";
//    cout<<endl;
//  }


  return 0;
}
