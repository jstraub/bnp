#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/beta.hpp>
#include <armadillo>

using namespace std;
using namespace arma;

// cateorical distribution (Multionomial for one word)
double Cat(uint32_t x, Row<double> pi);
// log cateorical distribution (Multionomial for one word)
double logCat(uint32_t x, Row<double> pi);
// evaluate beta distribution at x
double Beta(double x, double alpha, double beta);
// log beta function
double betaln(double alpha, double beta);
// evaluate log beta distribution at x
double logBeta(double x, double alpha, double beta);
double logDir(const Row<double>& x, const Row<double>& alpha);

