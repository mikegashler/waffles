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

#ifndef __GTHREAD_H__
#define __GTHREAD_H__

#include "GError.h"
#ifndef WINDOWS
#	include <unistd.h>
#	include <sched.h>
#endif

namespace GClasses {

#ifdef WINDOWS
#	define BAD_HANDLE (void*)1
	typedef void* THREAD_HANDLE;
#else
#	ifdef DARWIN
#		define BAD_HANDLE (_opaque_pthread_t*)1
		typedef _opaque_pthread_t* THREAD_HANDLE;
#	else
#   ifdef __FreeBSD__
#		  define BAD_HANDLE (pthread_t)-1
		  typedef pthread_t THREAD_HANDLE;
#   else
		  typedef unsigned long int THREAD_HANDLE;
#		  define BAD_HANDLE (unsigned long int)-2
#   endif
#	endif
#endif

/// A wrapper for PThreads on Linux and for some corresponding WIN32 api on Windows
class GThread
{
public:
	static THREAD_HANDLE spawnThread(unsigned int (*pFunc)(void*), void* pData);

	/// it may be an error to sleep more than 976ms (1,000,000 / 1024) on Unix
	static void sleep(unsigned int nMiliseconds);
};

} // namespace GClasses

#endif // __GTHREAD_H__
