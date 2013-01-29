/* Copyright (c) 2012, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See LICENSE.txt or 
 * http://www.opensource.org/licenses/mit-license.php */

#include "random.hpp"

#include <iostream>

using namespace std;
using namespace arma;

int main(int argc, char** argv)
{
  
  RandInt rndInt(0,10);
  cout<<"RandInt (0,10)"<<endl;
  for (uint32_t i=0; i<1000; ++i)
  {
    cout<<rndInt.draw()<<"\t";
  } cout<<endl;

  Col<uint32_t> rIs(100);
  rndInt.draw(rIs);
  cout<<rIs.t()<<endl<<endl;
  cout<<rndInt.draw(100)<<endl;

  cout<<"RandDisc:"<<endl;
  Col<double> pdf(10);
  pdf<<0.001<<0.2<<0.3<<0.4<<0.5<<0.5<<0.7<<0.8<<0.9<<1.0;
  pdf=pdf/sum(pdf);
  cout<<"pdf:\t"<<pdf.t();

  RandDisc rndDisc;
  uint32_t Ns=10000;
  colvec sample(Ns);
  for (uint32_t i=0; i<Ns; ++i)
    sample(i)=rndDisc.draw(pdf);
  
  Col<double> s_pdf(10);
  for(uint32_t j=0; j<pdf.n_elem; ++j)
    s_pdf(j)=sum(sample==j);
  s_pdf=s_pdf/sum(s_pdf);
  cout<<"s_pdf:\t"<<s_pdf.t();
  cout<<"diff:\t"<<pdf.t()-s_pdf.t();



  cout<<"--------------- disc sample ------------"<<endl;
  Col<double> l(7);
  l << -1.0471e+01<<  -7.1581e+01<<  -1.2912e+01<<  -1.1772e+01<<  -8.8031e+00<<  -1.6699e+01<<  -1.1533e+01;
  l(4)=math::nan();
  cout<<"l:\t"<<l.t();

  double lmax=l.max();
  double lmin=l.min();
  for(uint32_t i=0; i<l.n_elem; ++i)
    if(!is_finite(l(i)))
      l(i)=0.0;
    else
      l(i)=exp(l(i)+ (lmax - lmin)*0.5);

  cout<<"l:\t"<<l.t();
  for (uint32_t i=0; i<Ns; ++i)
  {
    sample(i)=rndDisc.draw(l/sum(l));
  }
  Col<double> s_l(l.n_elem);
  for(uint32_t j=0; j<l.n_elem; ++j)
    s_l(j)=sum(sample==j);
  s_l=s_l/sum(s_l);

  cout<<"l:\t"<<l.t()/sum(l);
  cout<<"s_l:\t"<<s_l.t();
  cout<<"diff:\t"<<l.t()/sum(l)-s_l.t();

  return 0;
}
