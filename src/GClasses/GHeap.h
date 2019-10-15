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

#ifndef __GHEAP_H__
#define __GHEAP_H__

#include <stddef.h>
#include <string.h>
#include "GError.h"

namespace GClasses {

#define ALIGN_BYTES 1 // For byte-addressable machines
//#define ALIGN_BYTES (sizeof(void*)) // For pointer-addressable machines
#define ALIGN_DOWN(p) (((p) / ALIGN_BYTES) * ALIGN_BYTES)
#define ALIGN_UP(p) ALIGN_DOWN((p) + ALIGN_BYTES - 1)

/// Provides a heap in which to put strings or whatever
/// you need to store. If you need to allocate space for
/// a lot of small objects, it's much more efficient to
/// use this class than the C++ heap. Plus, you can
/// delete them all by simply deleting the heap. You can't,
/// however, reuse the space for individual objects in
/// this heap.
class GHeap
{
protected:
	char* m_pCurrentBlock;
	size_t m_nMinBlockSize;
	size_t m_nCurrentPos;

public:
	GHeap(size_t nMinBlockSize)
	{
		m_pCurrentBlock = NULL;
		m_nMinBlockSize = nMinBlockSize;
		m_nCurrentPos = nMinBlockSize;
	}

	GHeap(const GHeap& that)
	{
		throw Ex("This object is not intended to be copied by value");
	}

	virtual ~GHeap();

	/// Deletes all the blocks and frees up memory
	void clear();

	/// Allocate space in the heap and copy a string to it.  Returns
	/// a pointer to the string
	char* add(const char* szString)
	{
		return add(szString, (int)strlen(szString));
	}

	/// Allocate space in the heap and copy a string to it.  Returns
	/// a pointer to the string
	char* add(const char* pString, size_t nLength)
	{
		char* pNewString = allocate(nLength + 1);
		memcpy(pNewString, pString, nLength);
		pNewString[nLength] = '\0';
		return pNewString;
	}

	/// Allocate space in the heap and return a pointer to it
	char* allocate(size_t nLength)
	{
		if(m_nCurrentPos + nLength > m_nMinBlockSize)
		{
			char* pNewBlock = new char[sizeof(char*) + std::max(nLength, m_nMinBlockSize)];
			*(char**)pNewBlock = m_pCurrentBlock;
			m_pCurrentBlock = pNewBlock;
			m_nCurrentPos = 0;
		}
		char* pNewBytes = m_pCurrentBlock + sizeof(char*) + m_nCurrentPos;
		m_nCurrentPos += nLength;
		return pNewBytes;
	}

	/// Allocate space in the heap and return a pointer to it. The returned pointer
	/// will be aligned to start at a location divisible by the size of a pointer,
	/// so it will be suitable for use with placement new even on architectures that
	/// require aligned pointers.
	char* allocAligned(size_t nLength)
	{
		size_t nAlignedCurPos = ALIGN_UP(m_nCurrentPos);
		if(nAlignedCurPos + nLength > m_nMinBlockSize)
		{
			char* pNewBlock = new char[sizeof(char*) + std::max(nLength, m_nMinBlockSize)];
			*(char**)pNewBlock = m_pCurrentBlock;
			m_pCurrentBlock = pNewBlock;
			m_nCurrentPos = 0;
			nAlignedCurPos = 0;
		}
		char* pNewBytes = m_pCurrentBlock + sizeof(char*) + nAlignedCurPos;
		m_nCurrentPos = nAlignedCurPos + nLength;
		return pNewBytes;
	}
};

} // namespace GClasses

#endif // __GHEAP_H__
