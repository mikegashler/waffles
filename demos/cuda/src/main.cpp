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
#include <GClasses/GVec.h>
#include <GClasses/GMatrix.h>
#include <depends/GCuda/GCudaMatrix.h>

using namespace GClasses;
using std::cerr;
using std::cout;

void doit(GArgReader& args)
{
	GVec v1(2);
	v1.v[0] = 1.0;
	v1.v[1] = 2.0;
	GMatrix m(2, 3);
	m[0][0] = 1.0; m[0][1] = 2.0; m[0][2] = 3.0;
	m[1][0] = 4.0; m[1][1] = 5.0; m[1][2] = 6.0;

	GCudaEngine e;
	GCudaVector cv1;
	cv1.upload(v1.v, 2);
	GCudaMatrix cm;
	cm.upload(m);
	GCudaVector cv2;
	cm.rowVectorTimesThis(e, cv1, cv2);

	GVec v2(3);
	cv2.download(v2.v);
	std::cout << "This should print 9, 12, 15\n";
	std::cout << to_str(v2.v[0]) << ", " << to_str(v2.v[1]) << ", " << to_str(v2.v[2]) << "\n";

	v2.v[0] = 1.0; v2.v[1] = 2.0; v2.v[2] = 3.0;
	GCudaVector cv3;
	cv2.upload(v2.v, 3);
	cv3.upload(v2.v, 3);
	cv2.addAndApplyTanh(e, cv3);
	cv2.download(v2.v);
	std::cout << "This should print 0.96402758007582, 0.99932929973907, 0.9999877116508\n";
	std::cout << to_str(v2.v[0]) << ", " << to_str(v2.v[1]) << ", " << to_str(v2.v[2]) << "\n";
}

int main(int argc, char *argv[])
{
#ifdef _DEBUG
	GApp::enableFloatingPointExceptions();
#endif
	int nRet = 0;
	try
	{
		GArgReader args(argc, argv);
		doit(args);
	}
	catch(const std::exception& e)
	{
		cerr << e.what() << "\n";
		nRet = 1;
	}

	return nRet;
}

