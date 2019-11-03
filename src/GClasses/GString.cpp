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

#include "GString.h"
#include "GError.h"
#include <string.h>
#include <stdlib.h>
#include <stdarg.h>

using std::string;

namespace GClasses {



std::string to_str(const std::vector<bool>& vv){
	std::deque<bool> v(vv.begin(), vv.end());
	return to_str(v.begin(), v.end(),"vector");
}


size_t safe_strcpy(char* szDest, const char* szSrc, size_t nMaxSize)
{
	if(nMaxSize == 0)
		return 0;
	nMaxSize--;
	size_t n;
	for(n = 0; szSrc[n] != '\0' && n < nMaxSize; n++)
		szDest[n] = szSrc[n];
	szDest[n] = '\0';
	return n;
}


std::string to_fixed_str(size_t val, size_t total_chars, char pad)
{
	std::string s = to_str(val);
	if(s.length() < total_chars)
		return std::string(total_chars - s.length(), pad) + s;
	else if(s.length() > total_chars)
		throw Ex("Not enough characters to represent the value ", s);
	return s;
}



std::string to_fixed_str(double val, size_t total_chars, char pad)
{
	std::ostringstream os;
	os.precision(total_chars);
	os << val;
	std::string s = os.str();
	if(s.length() < total_chars)
		return std::string(total_chars - s.length(), pad) + s;
	else if(s.length() > total_chars)
	{
		size_t dot = s.find_last_of(".");
		if(dot == string::npos)
			throw Ex("Not enough characters to represent the value ", s);
		size_t e = s.find_last_of("eE");
		if(e == string::npos)
		{
			if(dot + 1 > total_chars)
				throw Ex("Not enough characters to represent the value ", s);
			while(s.length() > total_chars)
				s.pop_back();
			return s;
		}
		else
		{
			size_t b = e - (s.length() - total_chars);
			if(b <= dot || b >= e)
				throw Ex("Not enough characters to represent the value ", s);
			s.erase(b, e - b);
			return s;
		}
	}
	else
		return s;
}


std::string& ltrim(std::string& str, const std::string& chars)
{
    str.erase(0, str.find_first_not_of(chars));
    return str;
}

std::string& rtrim(std::string& str, const std::string& chars)
{
    str.erase(str.find_last_not_of(chars) + 1);
    return str;
}

std::string& trim(std::string& str, const std::string& chars)
{
    return ltrim(rtrim(str, chars), chars);
}

bool ends_with(std::string const &fullString, std::string const &ending)
{
    if (fullString.length() >= ending.length())
        return fullString.compare(fullString.length() - ending.length(), ending.length(), ending) == 0;
	else
		return false;
}




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
	TestEqual("12",to_str((size_t)12),"Failed to_str(12)");
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

	TestEqual("00012", to_fixed_str((size_t)12, 5, '0'),"Failed to_fixed_str(1)");
	TestEqual("0.333", to_fixed_str(1.0 / 3.0, 5, '0'),"Failed to_fixed_str(2)");
	TestEqual("1.2e+29", to_fixed_str(123456789012345678901234567890.1, 7, '0'),"Failed to_fixed_str(4)");
}



GStringChopper::GStringChopper(const char* szString, size_t nMinLength, size_t nMaxLength, bool bDropLeadingWhitespace)
{
	GAssert(nMinLength > 0 && nMaxLength >= nMinLength); // lengths out of range
	m_bDropLeadingWhitespace = bDropLeadingWhitespace;
	if(nMinLength < 1)
		nMinLength = 1;
	if(nMaxLength < nMinLength)
		nMaxLength = nMinLength;
	m_nMinLen = nMinLength;
	m_nMaxLen = nMaxLength;
	m_szString = szString;
	m_nLen = strlen(szString);
	if(m_nLen > nMaxLength)
		m_pBuf = new char[nMaxLength + 1];
	else
		m_pBuf = NULL;
}

GStringChopper::~GStringChopper()
{
	delete[] m_pBuf;
}

void GStringChopper::reset(const char* szString)
{
	m_szString = szString;
	m_nLen = strlen(szString);
	if(!m_pBuf && m_nLen > m_nMaxLen)
		m_pBuf = new char[m_nMaxLen + 1];
}

const char* GStringChopper::next()
{
	if(m_nLen <= 0)
		return NULL;
	if(m_nLen <= m_nMaxLen)
	{
		m_nLen = 0;
		return m_szString;
	}
	size_t i;
	for(i = m_nMaxLen; i >= m_nMinLen && m_szString[i] > ' '; i--)
	{
	}
	if(i < m_nMinLen)
		i = m_nMaxLen;
	memcpy(m_pBuf, m_szString, i);
	m_pBuf[i] = '\0';
	m_szString += i;
	m_nLen -= i;
	if(m_bDropLeadingWhitespace)
	{
		while(m_nLen > 0 && m_szString[0] <= ' ')
		{
			m_szString++;
			m_nLen--;
		}
	}
	return m_pBuf;
}


} // namespace GClasses
