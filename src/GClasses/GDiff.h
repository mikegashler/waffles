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

#ifndef __GDIFF_H__
#define __GDIFF_H__

namespace GClasses {

/// This is a helper struct used by GDiff.
struct GDiffLine
{
	const char* pLine;
	size_t nLength;
	size_t nLineNumber1;
	size_t nLineNumber2;
};


/// This class finds the differences between two text files
/// It is case and whitespace sensitive, but is tolerant of Unix/Windows/Mac
/// line endings. It uses lines as the atomic unit. It accepts
/// matching lines in a greedy manner.
class GDiff
{
protected:
	const char* m_pFile1;
	const char* m_pFile2;
	size_t m_nPos1, m_nPos2;
	size_t m_nNextMatch1, m_nNextMatch2, m_nNextMatchLen;
	size_t m_nLine1, m_nLine2;

public:
	GDiff(const char* szFile1, const char* szFile2);
	virtual ~GDiff();

#ifndef NO_TEST_CODE
	/// Performs unit tests for this class. Throws an exception if there is a failure.
	static void test();
#endif // !NO_TEST_CODE

	bool nextLine(struct GDiffLine* pLine);

protected:
	static size_t measureLineLength(const char* pLine);
	size_t findNextMatchingLine(size_t* pPos1, size_t* pPos2);
};


} // namespace GClasses

#endif // __GDIFF_H__
