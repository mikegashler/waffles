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
	string s = szAddress;
	s += "/tools/survey";
	if(!GApp::openUrlInBrowser(s.c_str()))
	{
		cout << "Failed to open the URL: " << s.c_str() << "\nPlease open this URL manually.\n";
		cout.flush();
	}
}

void redirectStandardStreams(const char* pPath)
{
	string s1(pPath);
	s1 += "stdout.log";
	if(!freopen(s1.c_str(), "a", stdout))
	{
		cout << "Error redirecting stdout\n";
		cerr << "Error redirecting stdout\n";
		throw Ex("Error redirecting stdout");
	}
	string s2(pPath);
	s2 += "stderr.log";
	if(!freopen(s2.c_str(), "a", stderr))
	{
		cout << "Error redirecting stderr\n";
		cerr << "Error redirecting stderr\n";
		throw Ex("Error redirecting stderr");
	}
}

void doit(void* pArg)
{
	{
#ifdef _DEBUG
		int port = 8987;
#else
		int port = 8988;
#endif
		size_t seed = getpid() * (size_t)time(NULL);
		GRand prng(seed);
		char statePath[300];
		Server::getLocalStorageFolder(statePath);
		strcat(statePath, "state.json");
		Server* pServer = new Server(port, &prng);
		Holder<Server> hServer(pServer);
		if(GFile::doesFileExist(statePath))
		{
			GDom doc;
			cout << "Loading state...\n";
			cout.flush();
			doc.loadJson(statePath);
			pServer->deserializeState(doc.root());
			cout << "Server state loaded from " << statePath << "\n";
			cout.flush();

/*
			// Do some training to make sure the model is in good shape
			cout << "doing some training...\n";
			for(size_t i = 0; i < recommender().topics().size(); i++)
				recommender().refineModel(i, ON_STARTUP_TRAINING_ITERS);
*/
		}
		else
			cout << "No saved state. Made a new server\n";

		char buf[300];
		GTime::asciiTime(buf, 256, false);
		cout << "Server started at: " << buf << "\n";
		cout.flush();

		LaunchBrowser(pServer->myAddress(), &prng);

		// Process web requests
		double dLastMaintenance = GTime::seconds();
		GSignalHandler sh;
		while(pServer->m_keepGoing)
		{
			int sig = sh.check();
			if(sig != 0)
			{
				cout << "Received signal " << to_str(sig) << "(" << sh.to_str(sig) << ").\n";
				cout.flush();
				if(sig != 13) // Don't break for SIGPIPE. That just means the other end of the connection hung up, which happens all the time.
					break;
			}
			if(!pServer->process())
			{
				if(GTime::seconds() - dLastMaintenance > 14400)	// 4 hours
				{
					pServer->doSomeRecommenderTraining();
					dLastMaintenance = GTime::seconds();
				}
				else
					GThread::sleep(100);
			}
		}
		pServer->onShutDown();
	}
	cout << "Goodbye.\n";
}

void doItAsDaemon()
{
	char path[300];
	Server::getLocalStorageFolder(path);
	string s1 = path;
	s1 += "stdout.log";
	string s2 = path;
	s2 += "stderr.log";
	if(chdir(path) != 0)
		throw Ex("Failed to change dir to ", path);
	cout << "Launching daemon...\n";
	GApp::launchDaemon(doit, path, s1.c_str(), s2.c_str());
	if(!getcwd(path, 300))
	{
	}
	cout << "Daemon running in " << path << ".\n	stdout >> " << s1.c_str() << "\n	stderr >> " << s2.c_str() << "\n";
}

int main(int nArgs, char* pArgs[])
{
	//cout << "an1=" << to_str(AUTO_NAME_1_COUNT) << "," << to_str(AUTO_NAME_2_COUNT) << "," << to_str(AUTO_NAME_3_COUNT) << "\n";
	int nRet = 1;
	try
	{
		if(nArgs > 1 && strcmp(pArgs[1], "daemon") == 0)
			doItAsDaemon();
		else
			doit(NULL);
	}
	catch(std::exception& e)
	{
		cerr << e.what() << "\n";
	}
	return nRet;
}
