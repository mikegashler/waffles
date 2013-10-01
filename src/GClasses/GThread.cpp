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

#include "GThread.h"
#include "GError.h"
#ifdef WINDOWS
#	include <windows.h>
#else
#	include <pthread.h>
#endif

namespace GClasses {

// static
void GThread::sleep(unsigned int nMiliseconds)
{
#ifdef WINDOWS
	MSG aMsg;
	while(PeekMessage(&aMsg, NULL, WM_NULL, WM_NULL, PM_REMOVE))
	{
		TranslateMessage(&aMsg);
		DispatchMessage(&aMsg);
	}
	SleepEx(nMiliseconds, 1);
#else
	nMiliseconds ? usleep(nMiliseconds*1024) : sched_yield();		// it is an error to sleep for more than 1,000,000
#endif
}

THREAD_HANDLE GThread::spawnThread(unsigned int (*pFunc)(void*), void* pData)
{
#ifdef WINDOWS
	unsigned int nID;
	THREAD_HANDLE hThread = (void*)CreateThread/*_beginthreadex*/(
							NULL,
							0,
							(LPTHREAD_START_ROUTINE)pFunc,
							pData,
							0,
							(unsigned long*)&nID
							);
	if(hThread == BAD_HANDLE)
		throw Ex("Failed to create thread");
	return hThread;
#else
	pthread_t thread;
	if(pthread_create(&thread, NULL, (void*(*)(void*))pFunc, pData) != 0)
		throw Ex("Failed to create thread");
	pthread_detach(thread);
	return thread;
#endif
}

} // namespace GClasses

