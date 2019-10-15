/*
  The contents of this file are dedicated by all of its authors, including

    Michael S. Gashler,
    Eric Moyer,
    anonymous contributors,

  to the public domain (http://creativecommons.org/publicdomain/zero/1.0/).

  Note that some moral obligations still exist in the absence of legal ones.
  For example, it would still be dishonest to deliberately misrepresent the
  origin of a work. Although we impose no legal requirements to obtain a
  license, it is beseeming for those who build on the works of others to
  give back useful improvements, or find a way to pay it forward. If
  you would like to cite us, a published paper about Waffles can be found
  at http://jmlr.org/papers/volume12/gashler11a/gashler11a.pdf. If you find
  our code to be useful, the Waffles team would love to hear how you use it.
*/

#include "GError.h"
#include <stdarg.h>
#include <wchar.h>
#include <exception>
#include <signal.h>
#include <sys/stat.h>
#include <string.h>
#include <string>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#ifndef WINDOWS
#	include <execinfo.h>
#endif
#include "GString.h"

using std::exception;
using std::string;
using std::cerr;

namespace GClasses {

bool g_exceptionExpected = false;

void Ex::setMessage(std::string message)
{
	if(g_exceptionExpected)
		m_message = message;
	else
	{
		m_message = message;
#ifdef _DEBUG
		/*
		// Attempt to add a backtrace to the error message. (This will only produce human-readable results if the "-rdynamic" flag is used with the linker.)
		m_message += "\n";
		void* stackPointers[50];
		size_t stackSize = backtrace(stackPointers, 50);
		char** stackNames = backtrace_symbols(stackPointers, stackSize);
		for(size_t i = 0; i < stackSize; i++)
		{
			m_message += stackNames[i];
			m_message += "\n";
		}
		free(stackNames);
		*/
		// Stop in the debugger
		cerr << "Unexpected exception: " << m_message << "\nRaising SIGINT...";
		cerr.flush();
		raise(SIGINT);
#endif
	}
}

const char* Ex::what() const throw()
{
	return m_message.c_str();
}



GExpectException::GExpectException()
{
	m_prev = g_exceptionExpected;
	g_exceptionExpected = true;
}

GExpectException::~GExpectException()
{
	g_exceptionExpected = m_prev;
}



void TestEqual(char const*expected, char const*got, std::string desc){
  TestEqual(std::string(expected), std::string(got), desc);
}

void TestEqual(char const* expected, char* got, std::string desc){
  TestEqual(std::string(expected), std::string(got), desc);
}

void TestEqual(char* expected, char* got, std::string desc){
  TestEqual(std::string(expected), std::string(got), desc);
}

void TestContains(std::string expectedSubstring, std::string got,
                  std::string descr){
	using std::endl;
	if(got.find(expectedSubstring) == std::string::npos){
		std::cerr
			<< endl
			<< "Substring match failed: " << descr << endl
			<< endl
			<< "Expected substring: " << expectedSubstring << endl
			<< "Got               : " << got << endl
			;
		throw Ex("Substring match test failed: ", descr);
	}
}



#ifdef WINDOWS
void GAssertFailed(const char* filename, int line)
{
	cerr << "Debug Assert Failed in " << filename << ":" << line << std::endl;
	cerr.flush();
	__debugbreak();
}
void GAssertFailed(const char* filename, int line, const char* message)
{
	cerr << "Debug Assert Failed in " << filename << ":" << line << std::endl;
	cerr << "Message: " << message << std::endl;
	cerr.flush();
	__debugbreak();
}
#else
void GAssertFailed(const char* filename, int line)
{
	cerr << "Debug Assert Failed in " << filename << ":" << line << std::endl;
	cerr.flush();
	raise(SIGINT);
}

void GAssertFailed(const char* filename, int line, const char* message)
{
	cerr << "Debug Assert Failed in " << filename << ":" << line << std::endl;
	cerr << "Message: " << message << std::endl;
	cerr.flush();
	raise(SIGINT);
}

int _stricmp(const char* szA, const char* szB)
{
	while(*szA)
	{
		if((*szA | 32) < (*szB | 32))
			return -1;
		if((*szA | 32) > (*szB | 32))
			return 1;
		szA++;
		szB++;
	}
	if(*szB)
		return -1;
	return 0;
}

int _strnicmp(const char* szA, const char* szB, int len)
{
	int n;
	for(n = 0; n < len; n++)
	{
		if((*szA | 32) < (*szB | 32))
			return -1;
		if((*szA | 32) > (*szB | 32))
			return 1;
		szA++;
		szB++;
	}
	return 0;
}

long filelength(int filedes)
{
	struct stat s;
	if(fstat(filedes, &s) == -1)
		return 0;
	return s.st_size;
}
#endif


} // namespace GClasses
