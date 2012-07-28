#include "GHolders.h"
#include "GError.h"

namespace GClasses {

#ifdef _DEBUG
GTempBufSentinel::GTempBufSentinel(void* pBuf)
: m_pBuf(pBuf)
{
	*(char*)pBuf = 'S';
}

GTempBufSentinel::~GTempBufSentinel()
{
	GAssert(*(char*)m_pBuf == 'S'); // buffer overrun!
}
#endif // _DEBUG

void GOverrunSentinel::Check()
{
	if(m_sentinel != 0x5e47143a)
		throw Ex("buffer overrun!");
}


void verboten()
{
	throw Ex("Tried to copy a holder. (The method that facilitates this should have been private to catch this at compile time.)");
}

void FileHolder::reset(FILE* pFile)
{
	if(m_pFile && pFile != m_pFile)
	{
		if(fclose(m_pFile) != 0)
			GAssert(false);
	}
	m_pFile = pFile;
}

} // namespace GClasses
