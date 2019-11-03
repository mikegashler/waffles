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

#ifndef __GSTRING_H__
#define __GSTRING_H__

#include <stdlib.h>
#include <string>

namespace GClasses {


///\brief Run unit tests for the to_str functions.  Throws an
///       exception if it detects an error
void test_to_str();



/// This is similar to strncpy, but it always makes sure that
/// there is a null-terminating '\0' at the end of the new string.
/// Returns the length of the new string.
size_t safe_strcpy(char* szDest, const char* szSrc, size_t nDestBufferSize);

/// Converts a size_t to a string of fixed size padded in front with 'pad' as necessary
std::string to_fixed_str(size_t val, size_t chars, char pad);

/// Converts a double to a string of fixed size padded in front with 'pad' as necessary
std::string to_fixed_str(double val, size_t chars, char pad);

/// Removes leading whitespace (or any other set of characters) from a string
std::string& ltrim(std::string& str, const std::string& chars = "\t\n\v\f\r ");

/// Removes trailing whitespace (or any other set of characters) from a string
std::string& rtrim(std::string& str, const std::string& chars = "\t\n\v\f\r ");

/// Removes leading and trailing whitespace (or any other set of characters) from a string
std::string& trim(std::string& str, const std::string& chars = "\t\n\v\f\r ");

bool ends_with(std::string const &fullString, std::string const &ending);

/// This class chops a big string at word breaks so you can display it intelligently
/// on multiple lines
class GStringChopper
{
protected:
	const char* m_szString;
	size_t m_nLen;
	size_t m_nMaxLen;
	size_t m_nMinLen;
	char* m_pBuf;
	bool m_bDropLeadingWhitespace;

public:
	GStringChopper(const char* szString, size_t nMinLength, size_t nMaxLength, bool bDropLeadingWhitespace);
	~GStringChopper();

	/// Starts over with szString
	void reset(const char* szString);

	/// Returns NULL when there are no more lines left
	const char* next();
};




} // namespace GClasses

#endif // __GSTRING_H__
