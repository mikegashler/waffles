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

#ifndef __GERROR_H__
#define __GERROR_H__


#ifdef WINDOWS
// Throw out the min and max macros supplied by Microsoft that collide with std::min and std::max
#	define NOMINMAX
#	undef min
#	undef max
#	include <BaseTsd.h>
#	include <algorithm>
#endif
#include <string>
#include <sstream>
#include <iostream>
#include <vector>
#include <deque>
#include <list>
#include <set>
#include <map>
#include <utility> //for pair



namespace GClasses {




///\brief Convert another type that has a stream-insertion operator <<
///to a string
template<typename T>
std::string to_str(const T& n)
{
	std::ostringstream os;
	os.precision(14);
	os << n;
	return os.str();
}

class GMatrix; //Forward declare GMatrix so that I can declare a
               //to_str override for it

///\brief Allow GMatrix objects to easily be converted into a string
///for debugging
///
///\note this is defined with the rest of the GMatrix code in
///      GMatrix.cpp.  It is only declared in GError.h
///
///\param m the matrix that will be converted to a string
///
///\return a string representing the matrix \a m
///
///\see template<class T> to_str(const T& n)
std::string to_str(const GMatrix& m);


//Forward declaration of the special container class templates
template<typename T>
std::string to_str(const std::vector<T>& v);
template<typename T>
std::string to_str(const std::list<T>& v);
template<typename T>
std::string to_str(const std::set<T>& v);
template<typename T>
std::string to_str(const std::deque<T>& v);
template<typename T>
std::string to_str(const std::multiset<T>& v);
template<typename Key, typename T>
std::string to_str(const std::multimap<Key, T>& v);
template<typename Key, typename T>
std::string to_str(const std::map<Key, T>& v);

///\brief Convert a vector<bool> to a string (should allow converting nested
///stl containers like vector<list ...> to strings)
std::string to_str(const std::vector<bool>& vv);


///\brief Convert a pair to a string (should allow converting nested
///stl containers like pair<list<vector...>, multiset<foo> > to strings)
template<typename T, typename U>
std::string to_str(std::pair<T,U> v){
  return std::string("<")+to_str(v.first)+","+to_str(v.second)+">";
}

///\brief Convert an stl container-like object to a string
template<typename T>
std::string to_str(T begin, T end, std::string spec = "")
{
	std::ostringstream os;
	os.precision(14);
	os << "[" << spec;
	if(spec != ""){
	  os << ":";
	}
	while(begin != end){
	  os << to_str(*begin); ++begin;
	  if(begin != end){ os << ","; }
	}
	os << "]";
	return os.str();
}

///\brief Convert a vector to a string (should allow converting nested
///stl containers like vector<list ...> to strings)
template<typename T>
std::string to_str(const std::vector<T>& v){
  return to_str(v.begin(), v.end(),"vector");
}

///\brief Convert a list to a string (should allow converting nested
///stl containers like vector<list ...> to strings)
template<typename T>
std::string to_str(const std::list<T>& v){
  return to_str(v.begin(), v.end(),"list");
}

///\brief Convert a set to a string (should allow converting nested
///stl containers like set<vector..> to string)
template<typename T>
std::string to_str(const std::set<T>& v){
  return to_str(v.begin(), v.end(),"set");
}

///\brief Convert a deque to a string (should allow converting nested
///stl containers like deque<list ...> to strings)
template<typename T>
std::string to_str(const std::deque<T>& v){
  return to_str(v.begin(), v.end(),"deque");
}

///\brief Convert a multiset to a string (should allow converting nested
///stl containers like multiset<list ...> to strings)
template<typename T>
std::string to_str(const std::multiset<T>& v){
  return to_str(v.begin(), v.end(),"multiset");
}

///\brief Convert a multimap to a string (should allow converting nested
///stl containers like multimap<list< ...> , vector< ... > > to strings)
template<typename Key, typename T>
std::string to_str(const std::multimap<Key, T>& v){
  return to_str(v.begin(), v.end(),"multimap");
}

///\brief Convert a map to a string (should allow converting nested
///stl containers like map<list< ...> , vector< ... > > to strings)
template<typename Key, typename T>
std::string to_str(const std::map<Key, T>& v){
  return to_str(v.begin(), v.end(),"map");
}


///\brief The class of all exceptions thrown by this library
/// A simple exception object that wraps a string message
class Ex : public std::exception
{
protected:
	std::string m_message;

public:
	typedef std::string s;
	Ex(s a) { setMessage(a); }
	Ex(s a, s b) { setMessage(a + b); }
	Ex(s a, s b, s c) { setMessage(a + b + c); }
	Ex(s a, s b, s c, s d) { setMessage(a + b + c + d); }
	Ex(s a, s b, s c, s d, s e) { setMessage(a + b + c + d + e); }
	Ex(s a, s b, s c, s d, s e, s f) { setMessage(a + b + c + d + e + f); }
	Ex(s a, s b, s c, s d, s e, s f, s g) { setMessage(a + b + c + d + e + f + g); }
	Ex(s a, s b, s c, s d, s e, s f, s g, s h) { setMessage(a + b + c + d + e + f + g + h); }
	Ex(s a, s b, s c, s d, s e, s f, s g, s h, s i) { setMessage(a + b + c + d + e + f + g + h + i); }
	Ex(s a, s b, s c, s d, s e, s f, s g, s h, s i, s j) { setMessage(a + b + c + d + e + f + g + h + i + j); }

	virtual ~Ex() throw()
	{
	}

	/// Sets the message on the exception. (This method is called by all constructors of this object.)
	void setMessage(std::string message);

	/// Returns the error message corresponding to this exception
	virtual const char* what() const throw();
};


#define INVALID_INDEX ((size_t)-1)


void GAssertFailed(const char* filename, int line);
void GAssertFailed(const char* filename, int line, const char* message);

#ifdef _DEBUG
#define GASSERT_HELPER1(x)\
	{\
		if(!(x))\
			GAssertFailed(__FILE__, __LINE__);\
	}
#define GASSERT_HELPER2(x, msg)\
	{\
		if(!(x))\
			GAssertFailed(__FILE__, __LINE__, msg);\
	}
#else // _DEBUG
#define GASSERT_HELPER1(x) ((void)0)
#define GASSERT_HELPER2(x, msg) ((void)0)
#endif // else _DEBUG

#define COUNT_GASSERT_ARGS_IMPL2(_1, _2, count, ...) \
   count
#define COUNT_GASSERT_ARGS_IMPL(args) \
   COUNT_GASSERT_ARGS_IMPL2 args
#define COUNT_GASSERT_ARGS(...) \
   COUNT_GASSERT_ARGS_IMPL((__VA_ARGS__, 2, 1, 0))
 /* Pick the right helper macro to invoke. */
#define GASSERT_CHOOSE_HELPER2(count) GASSERT_HELPER##count
#define GASSERT_CHOOSE_HELPER1(count) GASSERT_CHOOSE_HELPER2(count)
#define GASSERT_CHOOSE_HELPER(count) GASSERT_CHOOSE_HELPER1(count)
 /* The actual macro. */
#define GASSERT_GLUE(x, y) x y
#define GAssert(...) \
   GASSERT_GLUE(GASSERT_CHOOSE_HELPER(COUNT_GASSERT_ARGS(__VA_ARGS__)), \
               (__VA_ARGS__))



///\brief Instantiating an object of this class specifies that any exceptions thrown
/// during the life of this object should be treated as "expected".
class GExpectException
{
protected:
	bool m_prev;

public:
	GExpectException();
	~GExpectException();
};





// ----------------------------
// Platform Compatability Stuff
// ----------------------------

#ifdef WINDOWS
#else
int _stricmp(const char* szA, const char* szB);
int _strnicmp(const char* szA, const char* szB, int len);
long filelength(int filedes);
#endif

///\brief Verify that \a expected and \a got are equal for test code. Unlike Assert, this check does not disappear in optimized builds.
///
///If expected==got then does nothing.  Otherwise prints to stderr:
///
///<pre>
///Test for equality failed: ---------test_descr goes here ---------------
///
///Expected: ------------expected goes here---------------
///Got     : ------------got goes here     ---------------
///</pre>
///
///Then it throws an exception using ThrowError
///
///Calls operator==(const T1&,const T2&) to determine equality.
///
///Calls GClasses::to_str to form the string representation of \a expected
///and \a got
///
///\param expected The value expected from specifications
///
///\param got      The value actually produced by the code
///
///\param test_descr A short test description to allow a human to
///                  easily find the failing test in the code and
///                  understand why it was written and have some help
///                  in diagnosing the bug.
template<class T1, class T2>
void TestEqual(const T1& expected, const T2& got, std::string test_descr)
{
	using std::endl;
	if(!(expected == got)){
		std::cerr
			<< endl
			<< "Test for equality failed: " << test_descr << endl
			<< endl
			<< "Expected: " << GClasses::to_str(expected) << endl
			<< "Got     : " << GClasses::to_str(got) << endl
			;
		throw Ex("Test for equality failed: ", test_descr);
	}
}

///"Specialization" of TestEqual for c-strings done using overloading
void TestEqual(char const* expected, char const* got, std::string desc);

///"Specialization" of TestEqual for c-strings done using overloading
void TestEqual(char const* expected, char* got, std::string desc);

///"Specialization" of TestEqual for c-strings done using overloading
void TestEqual(char* expected, char* got, std::string desc);

///\brief Verify that \a expectedSubstring is a substring of \a got for test code. Unlike Assert, this check does not disappear in optimized builds.
///
///If \a got contains \a expectedSubstring then does nothing.
///Otherwise prints to stderr:
///
///<pre>
///Substring match failed: ---------test_descr goes here ---------------
///
///Expected substring: ------------expectedSubstring goes here---------------
///Got               : ------------got goes here     ---------------
///</pre>
///
///Then it throws an exception using ThrowError
///
///\param expectedSubstring The value expected as a substring from
///                         specifications
///
///\param got      The value actually produced by the code
///
///\param test_descr A short test description to allow a human to
///                  easily find the failing test in the code and
///                  understand why it was written and have some help
///                  in diagnosing the bug.
void TestContains(std::string expectedSubstring, std::string got,
                  std::string desc);

} // namespace GClasses

#endif // __GERROR_H__
