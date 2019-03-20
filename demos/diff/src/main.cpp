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
#ifdef WINDOWS
#	include <windows.h>
#	include <process.h>
#	include <direct.h>
#else
#	include <unistd.h>
#endif
#include <GClasses/GRand.h>
#include <GClasses/GTime.h>
#include <GClasses/GApp.h>
#include <GClasses/GHolders.h>
#include <GClasses/GThread.h>
#include "server.h"


using namespace GClasses;
using std::cout;
using std::cerr;
using std::vector;
using std::string;
using std::ostream;
using std::map;
using std::set;
using std::pair;
using std::make_pair;
using std::multimap;

class View;
class Account;
class ViewStats;


void LaunchBrowser(const char* szAddress, GRand* pRand)
{
	cout << "Opening URL: " << szAddress << "\n";
	if(!GApp::openUrlInBrowser(szAddress))
	{
		cout << "Failed to open the URL: " << szAddress << "\nPlease open this URL manually.\n";
		cout.flush();
	}
}

void doit(const char* szFile1, const char* szFile2)
{
	int port = 8985;
	size_t seed = getpid() * (size_t)time(NULL);
	GRand prng(seed);
	Server* pServer = new Server(port, &prng);
	Holder<Server> hServer(pServer);
	string s = pServer->myAddress();
	s += "?left=";
	s += szFile1;
	s += "\\&right=";
	s += szFile2;
	LaunchBrowser(s.c_str(), &prng);

	// Handle requests from the browser
	double starttime = GTime::seconds();
	while(true)
	{
		if(!pServer->process())
		{
			if(GTime::seconds() - starttime > 8.0)
			{
				cout << "Whelp, the browser should probably have it all by now, so I'm exiting...\n";
				break;
			}
			GThread::sleep(100);
		}
	}
	cout << "Goodbye.\n";
}

int main(int nArgs, char* pArgs[])
{
	if(nArgs != 3)
		throw Ex("Expected two args--two filenames to diff.");

	int nRet = 1;
	try
	{
		doit(pArgs[1], pArgs[2]);
	}
	catch(std::exception& e)
	{
		cerr << e.what() << "\n";
	}
	return nRet;
}
