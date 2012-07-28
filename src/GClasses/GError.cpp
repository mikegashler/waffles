/*
	Copyright (C) 2006, Mike Gashler

	This library is free software; you can redistribute it and/or
	modify it under the terms of the GNU Lesser General Public
	License as published by the Free Software Foundation; either
	version 2.1 of the License, or (at your option) any later version.

	see http://www.gnu.org/copyleft/lesser.html
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
#ifndef MIN_PREDICT
#include "GString.h"
#endif // MIN_PREDICT

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
		m_message = message; // (This is a good place to put a breakpoint. All unexpected exceptions pass through here.)
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




std::string to_str(const std::vector<bool>& vv){
	std::deque<bool> v(vv.begin(), vv.end());
	return to_str(v.begin(), v.end(),"vector");
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
void GAssertFailed()
{
	cerr << "Debug Assert Failed!\n";
	cerr.flush();
	__debugbreak();
}
#else
void GAssertFailed()
{
	cerr << "Debug Assert Failed!\n";
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

#ifndef MIN_PREDICT
namespace{
	template<class container, class T>
		void CFill(container& c, const T a){
		c.insert(c.end(), a);
	}


	template<class container, class T>
		void CFill(container& c, const T a, const T b){
		CFill(c,a);
		CFill(c,b);
	}

	template<class container, class T>
		void CFill(container& c, const T a, const T b, const T d){
		CFill(c,a);
		CFill(c,b,d);
	}

	template<class container, class T>
		void CFill(container& c, const T a, const T b, const T d, const T e){
		CFill(c,a,b);
		CFill(c,d,e);
	}
}

void test_to_str(){
	using namespace std;
	//Test some basic types (not exhaustive)
	TestEqual("12",to_str(12),"Failed to_str(12)");
	TestEqual("12.123456789012",to_str(12.12345678901234),
						"Failed to_str(12.12345678901234)");
	TestEqual("A string",to_str("A string"),
						"Failed to_str(\"A string\")");


	//Test vector: empty, with one item and with 4 items
	{
		vector<int> v;	CFill(v, 10, 5, 2, 1);
		TestEqual("[vector:10,5,2,1]",to_str(v),
							"Failed to_str([vector:10,5,2,1])"); }

	{
		vector<int> v;	CFill(v, 2);
		TestEqual("[vector:2]",to_str(v),
							"Failed to_str([vector:2])"); }

	{
		vector<int> v;
		TestEqual("[vector:]",to_str(v),
							"Failed to_str([vector:])"); }

	//Test the other individual STL containers (vector<bool> is a
	//different container than vector)
	{
		vector<bool> v;	CFill(v, false, true, true, false);
		TestEqual("[vector:0,1,1,0]",to_str(v),
							"Failed to_str([vector:false, true, true, false])"); }

	{
		list<int> v;	CFill(v, 10, 5, 2);
		TestEqual("[list:10,5,2]",to_str(v),
							"Failed to_str([list:10,5,2])"); }

	{
		deque<int> v;	CFill(v, 10, 5, 2, 1);
		TestEqual("[deque:10,5,2,1]",to_str(v),
							"Failed to_str([deque:10,5,2,1])"); }

	{
		set<int> v;	CFill(v, 10, 5, 2, 1);
		TestEqual("[set:1,2,5,10]",to_str(v),
							"Failed to_str([set:10,5,2,1])"); }

	{
		multiset<int> v;	CFill(v, 10, 5, 2, 1);
		TestEqual("[multiset:1,2,5,10]",to_str(v),
							"Failed to_str([multiset:10,5,2,1])"); }


	{
		map<int,int> v;	CFill(v, make_pair(10,2), make_pair(5,1), make_pair(2,3),
													make_pair(1,7));
		TestEqual("[map:<1,7>,<2,3>,<5,1>,<10,2>]",to_str(v),
							"Failed to_str([map:<1,7>,<2,3>,<5,1>,<10,2>])"); }

	{
		multimap<int,int> v;	CFill(v, make_pair(10,2), make_pair(5,1),
																make_pair(2,3), make_pair(1,7));
		TestEqual("[multimap:<1,7>,<2,3>,<5,1>,<10,2>]",to_str(v),
							"Failed to_str([multimap:<1,7>,<2,3>,<5,1>,<10,2>])"); }

	//Test some nested containers
	{
		list<int> l1;	CFill(l1, 18, 19, 16);
		list<int> l2;	CFill(l2, 28, 29, 26, 24);

		vector<int> v3; CFill(v3,33,34,35);
		vector<int> v4; CFill(v4,45,46,47);
		map<list<int>,vector<int> > m;	CFill(m, make_pair(l1,v3),
																					make_pair(l2,v4));

		TestEqual("[map:<[list:18,19,16],[vector:33,34,35]>,"
							"<[list:28,29,26,24],[vector:45,46,47]>]",
							to_str(m),
							"Failed to_str on map<list<int>,vector<int> >");
	}
}
#endif // MIN_PREDICT

} // namespace GClasses

