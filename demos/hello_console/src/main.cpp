// -------------------------------------------------------------
// The contents of this file may be distributed under the CC0
// license (http://creativecommons.org/publicdomain/zero/1.0/).
// Of course you may also distribute it under a more restrictive
// license, such as any of the OSI-approved licenses
// (http://www.opensource.org/licenses).
// -------------------------------------------------------------

#include <exception>
#include <iostream>
#include <GClasses/GApp.h>
#include <GClasses/GError.h>

using namespace GClasses;
using std::cerr;
using std::cout;

void doit(GArgReader& args)
{
	cout << "Hello console!\n";
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

