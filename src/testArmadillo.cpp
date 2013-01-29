/* Copyright (c) 2012, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See LICENSE.txt or 
 * http://www.opensource.org/licenses/mit-license.php */

#include <armadillo>
#include <iostream>

#include <stdint.h>


using namespace arma;
using namespace std;

int main(int argc, char** argv)
{
  

  Col<uint32_t> a;
  a<<1<<2<<4<<3<<1<<6<<7<<2<<4;

  cout<<a.t()<<endl;
  cout<<(a==1).t()<<endl;
  cout<<sum(a==1)<<endl;

  mat A;
  A.randn(10,2);
  A.rows(0,4) += 4;
  cout<<A<<endl;

  Col<uint32_t> b=a;
  b.resize(b.n_elem+1);
  b(b.n_elem-1)=99;
  cout<<a.t()<<endl;
  cout<<b.t()<<endl;

  colvec c(10);
  c.zeros();
  c(4)=math::inf();
  c(5)=math::nan();
  cout<<c<<endl;
  cout<<is_finite(c(3))<<endl;
  cout<<is_finite(c(4))<<endl;
  cout<<is_finite(c(5))<<endl;
  cout<<is_finite(c)<<endl;

  colvec d(2);
  d.ones();
  c.insert_rows(c.n_elem,d);
  cout<<c<<endl;

  colvec l(2);
  l << -8.0 << -16.0;
  cout<<l.t()<<endl;
  cout<<exp(l).t()<<endl;

  return 0;
}
