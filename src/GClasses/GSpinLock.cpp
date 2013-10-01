/*
  The contents of this file are dedicated by all of its authors, including

    Michael S. Gashler,
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

#include "GSpinLock.h"
#include <time.h>
#include "GError.h"
#include "GThread.h"
#ifdef WINDOWS
#	include <windows.h>
#endif

namespace GClasses {

GSpinLock::GSpinLock()
{
	m_dwLocked = 0;
#ifndef WINDOWS
	pthread_mutex_init(&m_mutex, NULL);
#endif
#ifdef _DEBUG
	m_szWhoHoldsTheLock = "<Never Been Locked>";
#endif
}

GSpinLock::~GSpinLock()
{
#ifndef WINDOWS
	pthread_mutex_destroy(&m_mutex);
#endif
}

#ifdef WINDOWS
static inline unsigned int testAndSet(volatile long* pDWord)
{
	return InterlockedExchange(pDWord, 1);
}
#endif // WINDOWS

void GSpinLock::lock(const char* szWhoHoldsTheLock)
{
#ifdef _DEBUG
	time_t t;
	time_t tStartTime = time(&t);
	time_t tCurrentTime;
#endif // _DEBUG

#ifdef WINDOWS
	while(testAndSet(&m_dwLocked))
#else
	while(0!=pthread_mutex_trylock(&m_mutex))
#endif
	{
#ifdef _DEBUG
		tCurrentTime = time(&t);
		GAssert(tCurrentTime - tStartTime < 10); // Blocked for 10 seconds!
#endif // _DEBUG
		GThread::sleep(0);
	}
#ifndef WINDOWS
	m_dwLocked = 1;
#endif
#ifdef _DEBUG
	m_szWhoHoldsTheLock = szWhoHoldsTheLock;
#endif // _DEBUG
}

void GSpinLock::unlock()
{
#ifdef _DEBUG
	m_szWhoHoldsTheLock = "<Not Locked>";
#endif // _DEBUG
	m_dwLocked = 0;
#ifndef WINDOWS
	pthread_mutex_unlock(&m_mutex);
#endif
}


#ifndef NO_TEST_CODE

#define THREAD_COUNT 3 // 100
#define THREAD_ITERATIONS 500 // 2000

struct TestSpinLockThreadStruct
{
	int* pBalance;
	bool* pExitFlag;
	GSpinLock* pSpinLock;
	int nOne;
};

// This thread increments the balance a bunch of times.  We use a dilly-dally loop
// instead of just calling Sleep because we want our results to reflect
// random context-switches that can happen at any point whereas Sleep causes the
// context switch to happen immediately which may result it one never happening
// at any other point.
unsigned int TestSpinLockThread(void* pParameter)
{
	struct TestSpinLockThreadStruct* pThreadStruct = (struct TestSpinLockThreadStruct*)pParameter;
	int n, i;
	for(n = 0; n < THREAD_ITERATIONS; n++)
	{
		// Take the lock
		pThreadStruct->pSpinLock->lock("TestSpinLockThread");

		// read the balance
		int nBalance = *pThreadStruct->pBalance;

		// We increment nBalance in this funny way so that a smart optimizer won't
		// figure out that it can remove the nBalance variable from this logic.
		nBalance += pThreadStruct->nOne;

		// Dilly-dally
		for(i = 0; i < 10; i++)
			nBalance++;
		for(i = 0; i < 10; i++)
			nBalance--;

		// update the balance
		*pThreadStruct->pBalance = nBalance;

		// Release the lock
		pThreadStruct->pSpinLock->unlock();
	}

	// Clean up and exit
	GAssert(*pThreadStruct->pExitFlag == false); // expected this to be false
	*pThreadStruct->pExitFlag = true;
	delete(pThreadStruct);
	return 1;
}

// static
void GSpinLock::test()
{
	bool exitFlags[THREAD_COUNT];
	int n;
	for(n = 0; n < THREAD_COUNT; n++)
		exitFlags[n] = false;
	int nBalance = 0;
	GSpinLock sl;

	// spawn a bunch of threads
	for(n = 0; n < THREAD_COUNT; n++)
	{
		TestSpinLockThreadStruct* pThreadStruct = new struct TestSpinLockThreadStruct;
		pThreadStruct->pBalance = &nBalance;
		pThreadStruct->pExitFlag = &exitFlags[n];
		pThreadStruct->pSpinLock = &sl;
		pThreadStruct->nOne = 1;
		THREAD_HANDLE hThread = GThread::spawnThread(TestSpinLockThread, pThreadStruct);
		if(hThread == BAD_HANDLE)
			throw "failed";
	}

	// wait until all the threads are done
	while(true)
	{
		bool bDone = true;
		for(n = 0; n < THREAD_COUNT; n++)
		{
			if(!exitFlags[n])
			{
				bDone = false;
				GThread::sleep(0);
				break;
			}
		}
		if(bDone)
			break;
	}

	// Check the final balance
	if(nBalance != THREAD_COUNT * THREAD_ITERATIONS)
		throw "failed";
}

#endif // !NO_TEST_CODE

} // namespace GClasses

