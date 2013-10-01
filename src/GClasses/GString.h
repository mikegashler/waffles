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

namespace GClasses {

// This is similar to strncpy, but it always makes sure that
// there is a null-terminating '\0' at the end of the new string.
// Returns the length of the new string.
size_t safe_strcpy(char* szDest, const char* szSrc, size_t nDestBufferSize);


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
