/*
  The contents of this file are dedicated by all of its authors, including

    Michael S. Gashler,
    anonymous contributors,

  to the public domain (http://creativecommons.org/publicdomain/zero/1.0/).

  Note that some moral obligations still exist in the absence of legal ones.
  For example, it would still be dishonest to deliberately misrepresent the
  origin of a work. Although we impose no legal requirements to obtain a
  license, it is beseeming for those who build on the works of others to
  give back useful improvements, or pay it forward in their own field. If
  you would like to cite us, a published paper about Waffles can be found
  at http://jmlr.org/papers/volume12/gashler11a/gashler11a.pdf. If you find
  our code to be useful, the Waffles team would love to hear how you use it.
*/

#include <exception>
#include <iostream>
#include <GClasses/GApp.h>
#include <GClasses/GError.h>
#include <GClasses/GMatrix.h>
#include <eigen3/Eigen/Dense>

using namespace GClasses;
using std::cout;
using Eigen::MatrixXd;

int main(int argc, char *argv[])
{
	GApp::enableFloatingPointExceptions();
	if(argc > 1)
		throw Ex("No args were expected");

	cout << "Making a Waffles matrix...\n";
	GMatrix waf_m(2, 2);
	waf_m[0][0] = 1.0; waf_m[0][1] = 2.0;
	waf_m[1][0] = 3.0; waf_m[1][1] = 4.0;
	waf_m.print(cout);

	cout << "\n\nConverting to an Eigen matrix...\n";
	MatrixXd eig_m(waf_m.rows(), waf_m.cols());
	for(size_t j = 0; j < waf_m.rows(); j++)
	{
		for(size_t i = 0; i < waf_m.cols(); i++)
			eig_m(j, i) = waf_m[j][i];
	}
	cout << eig_m << "\n";

	cout << "\n\nConverting back to a Waffles matrix...\n";
	GMatrix waf_2(2, 2);
	for(size_t j = 0; j < 2; j++)
	{
		for(size_t i = 0; i < 2; i++)
			waf_2[j][i] = eig_m(j, i);
	}
	waf_2.print(cout);

	return 0;
}

