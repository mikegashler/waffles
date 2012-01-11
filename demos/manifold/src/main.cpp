// -------------------------------------------------------------
// The contents of this file may be distributed under the CC0
// license (http://creativecommons.org/publicdomain/zero/1.0/).
// Of course you may also distribute it under a more restrictive
// license, such as any of the OSI-approved licenses
// (http://www.opensource.org/licenses).
// -------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <GClasses/GApp.h>
#include <GClasses/GError.h>
#include "Manifold.h"
#ifdef WINDOWS
#	include <direct.h>
#endif
#include <exception>
#include <iostream>

using namespace GClasses;
using std::cerr;

int main(int argc, char *argv[])
{
	char szAppPath[300];
	GApp::appPath(szAppPath, 300, true);
#ifdef WINDOWS
	if(_chdir(szAppPath) != 0)
#else
	if(chdir(szAppPath) != 0)
#endif
	{}

	int nRet = 0;
	try
	{
		ManifoldMenuController c;
		c.RunModal();
	}
	catch(const std::exception& e)
	{
		cerr << e.what() << "\n";
		nRet = 1;
	}

	return nRet;
}

