/* Copyright (c) 2012, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See LICENSE.txt or 
 * http://www.opensource.org/licenses/mit-license.php */

#include "hdp.hpp"

#include <time.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>

#include <armadillo>

using namespace std;
using namespace arma;


#define DIRICHLET_BASE


long int writeZji(const vector<Col<uint32_t> >& z_ji, string dir, long int timeTag=0, string tag=string("noTag"))
{
  if(timeTag==0)
  {
    time_t rawtime; time(&rawtime);
    timeTag = rawtime;
  }
  char* buf = new char[dir.size()+tag.size()+100];
  if(tag.compare("noTag")==0)
    sprintf(buf,"%s/%ld_zji.txt",dir.c_str(),timeTag);
  else
    sprintf(buf,"%s/%ld_%s.txt",dir.c_str(),timeTag,tag.c_str());
  cout<<"Writing z_ji to "<<buf<<endl;
  ofstream out(buf);
  if(out)
  {
    for(uint32_t j=0; j < z_ji.size(); ++j){
      out<<z_ji[j].t();
    }
    out.close();
  }else{
    cout<< "Could not open "<<buf<<" for writing"<<endl;
  }
  return timeTag;
}


int main(int argc, char** argv)
{
  string featType;
  int c,count=1;
  uint32_t maxN_j=99999;
  uint32_t maxJ=99999;
  uint32_t Niter=10;
  bool classesAsDocs=false;
  while ((c = getopt (argc, argv, "hm:i:cJ:")) != -1)
  {
    switch (c)
    {
//      case 'f':
//        featType=string(optarg);
//        cout<<"Selected Featuretype = "<<featType<<endl;
//        break;
      case 'i':
        Niter=atoi(optarg);
        cout<<"Selected number of iterations for hdp = "<<Niter<<endl;
        count+=2;
        break;
      case 'm':
        maxN_j=atoi(optarg);
        cout<<"Selected maximal number of words per document = "<<maxN_j<<endl;
        count+=2;
        break;
      case 'J':
        maxJ=atoi(optarg);
        cout<<"Selected maximal number of documents = "<<maxJ<<endl;
        count+=2;
        break;
      case 'c':
        cout<<"Use class labels for document assignment (see -h for description)"<<endl;
        classesAsDocs=true;
        count++;
        break;
      case 'h':
      default:
        cout<<"Help:"<<endl
          <<"featExtract <options> <path to data file> "<<endl
          <<" The data file is assumed to have a first column which indicates the membership"<<endl
          <<" to a document. Document numbers start at 0. The second and third column are used to identify how the data in the following columns is interpreted (i.e. column two is number of rows and column three is number of columns). A row is one observation and a column is the value of a feature of an observation. Elements are in row major (since this is a c program)"<<endl
          <<"\t-m\t\tMaximal number of words per document"<<endl
          <<"\t-J\t\tMaximal number of documents"<<endl
          <<"\t-c\t\tClasses as documents - without that option each row is considered a set of observations for one document and class ids (first column) are ignored"<<endl
          <<"\t-i\t\tSelect number of iterations for hdp"<<endl
          <<"\t-h\t\tDisplay help"<<endl;
        abort();
    }
  }

  uint32_t J=0;
  uint32_t d=0;
  vector<mat> x_i; // documents per line
  vector<uint32_t> class_i;
  vector<mat> x_j; // one document are all documents in a class
  vector<mat>* x=&x_i; // pointer to the respective document vector (x_i or x_j depending on classesAsDocs)
  string dataFile;
  if (count<argc)
  {
    dataFile=string(argv[count]);
    ifstream in(dataFile.c_str());
    if (!in)
    {
      cout<<"Loading from "<<dataFile<<" did not work!"<<endl;
      return 1;
    }
    // load all documents into x_i
    uint32_t classId=0, nrows=0, ncols=0, i=0;
    while(in >> classId){
      in >> nrows >> ncols;
      mat xx(nrows,ncols);
      for (uint32_t c=0; c<ncols; ++c)
        for (uint32_t r=0; r<nrows; ++r)
        {
          in >> xx(r,c);
        }
      if (!classesAsDocs){
        uint32_t N_i=min(xx.n_rows,maxN_j);
        x_i.push_back(xx.rows(0,N_i-1));
      }else
        x_i.push_back(xx);
      class_i.push_back(classId);
      i++;
      cout<<" Document "<<i<<" of class "<< class_i[i-1] <<" |.|="<<x_i[i-1].n_rows<<" x "<<x_i[i-1].n_cols<<endl;
    }
    J=classId+1; // largest class id from last document +1 because we start at 0
    d=ncols;

    if (classesAsDocs)
    { // we want to concatenate all documents in a class into one document
      cout<<"Concatenate all documents in a class into one document."<<endl;
      x_j.resize(J);
      uint32_t i=0;
      for (uint32_t j=0; j<J; ++j)
      {
        while(class_i[i]==j)
        {
          x_j[j] = join_cols(x_j[j],x_i[i]);
          ++i;
        }
      uint32_t N_j=min(x_j[j].n_rows,maxN_j);
      x_j[j]=x_j[j].rows(0,N_j-1);
      cout<<" Document "<<j<<" of class "<< j <<" |.|="<<x_j[j].n_rows<<" x "<<x_j[j].n_cols<<endl;
      }
      x=&x_j;
    }
  }else{
    cout<<"need at least one input file with data!"<<endl;
    return 1;
  }

  // remove all documents which are to many
  while(x->size()>maxJ)
    x->pop_back();
  J=x->size();

#ifdef DIRICHLET_BASE
  
  uint32_t K=0;
  vector<Mat<uint32_t> > x_int(J);
  for (uint32_t j=0; j<J; ++j)
  {
    x_int[j] = conv_to<Mat<uint32_t> >::from(x->at(j));
    K=max(K, x_int[j].max());
  }

  cout<<"K="<<K<<endl;

  Row<double> alphas(K);
  alphas.ones();
  alphas *= 1.1;
  double alpha =1.0, gamma=1.0;
  Dir dir(alphas);
  HDP<Dir> hdp(dir, alpha, gamma);
  vector<Col<uint32_t> > z_ji = hdp.densityEst(x_int,10,10,Niter);
#else
  mat means(J,d);
  double x_max=-999999., x_min=999999.;
  for(uint32_t j=0; j<J; ++j)
  {
    double xi_max=max(max(x->at(j)));
    double xi_min=min(min(x->at(j)));
    x_max=max(xi_max,x_max);
    x_min=min(xi_min,x_min);
    means.row(j)=mean(x->at(j),0);
  }

  cout<<"mean of means:"<<mean(means,0)<<endl;
  cout<<"x_min="<<x_min<<" x_max="<<x_max<<endl;
  colvec vtheta(d);
  vtheta.zeros();
  vtheta=mean(means,0).t();
  mat Delta = eye<mat>(d,d)*((x_max-x_min)/6.0)*((x_max-x_min)/6.0); // Variance matrix! not std
  double kappa=1.0, alpha =1., gamma=10000.0, nu=d+1.1;

  HDP<InvNormWish> hdp(vtheta, kappa, Delta, nu, alpha, gamma);
  vector<Col<uint32_t> > z_ji = hdp.densityEst(*x,10,10,Niter);
#endif

  // print z_ji
  string pathToResults(".");
  long int timeTag=writeZji(z_ji, pathToResults,0,string("z_ji"));

  // print x
  char* buf = new char[100];
  sprintf(buf,"%s/%ld_x_j.txt",pathToResults.c_str(),timeTag);
  ofstream out(buf);
  for (uint32_t j=0; j<J; ++j)
  {
    out<<j<<"\t"<<x->at(j).n_rows<<"\t"<<x->at(j).n_cols<<"\t";
    for(uint32_t c=0; c<x->at(j).n_cols; ++c)
      for(uint32_t r=0; r<x->at(j).n_rows; ++r)
      {
        out<<x->at(j)(r,c)<<"\t";
      }
    out<<endl;
  }
  out.close();

  // print config
  sprintf(buf,"%s/%ld_config.txt",pathToResults.c_str(),timeTag);
  out.open(buf);
  out<<"dataFile="<<dataFile<<endl
    <<"maxJ="<<maxJ<<endl
    <<"maxN_j="<<maxN_j<<endl
    <<"Niter="<<Niter<<endl
    <<"classesAsDocs="<<(classesAsDocs?1:0)<<endl;
  out.close();

  cout<<"z_ji:"<<endl;
  for(uint32_t j=0; j<z_ji.size(); ++j)
  {
    for(uint32_t i=0; i<z_ji[j].n_elem; ++i)
      cout<<z_ji[j](i)<<" ";
    cout<<endl;
  }


  return 0;
}
