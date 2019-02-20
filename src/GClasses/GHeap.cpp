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

#include "GHeap.h"
#include "GError.h"

using namespace GClasses;

// virtual
GHeap::~GHeap()
{
	clear();
}

void GHeap::clear()
{
	while(m_pCurrentBlock)
	{
		char* pNext = *(char**)m_pCurrentBlock;
		delete[] m_pCurrentBlock;
		m_pCurrentBlock = pNext;
	}
	m_nCurrentPos = m_nMinBlockSize;
}
