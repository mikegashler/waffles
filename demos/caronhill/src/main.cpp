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

#include <stdio.h>
#include <stdlib.h>
#include <GClasses/GApp.h>
#include <GClasses/GError.h>
#include "CarOnHill.h"
#ifdef WINDOWS
#	include <direct.h>
#endif
#include <exception>
#include <iostream>

using namespace GClasses;
using std::cerr;
using std::cout;

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
		CarOnHillController c;
		c.RunModal();
	}
	catch(const std::exception& e)
	{
		cerr << e.what() << "\n";
		nRet = 1;
	}

	return nRet;
}

