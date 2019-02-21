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
#include <time.h>
#ifdef WINDOWS
#	include <windows.h>
#else
#	include <pthread.h>
#endif

using std::vector;

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

unsigned int GThread::startThread(void* pObj)
{
	GThread* pThread = (GThread*)pObj;
	pThread->m_finished = false;
	pThread->m_started = true;
	pThread->run();
	pThread->m_finished = true;
	return 0;
}

THREAD_HANDLE GThread::spawn()
{
	m_finished = false;
	m_started = false;
	return GThread::spawnThread(GThread::startThread, this);
}

void GThread::join(size_t miliseconds)
{
	while(!m_finished)
		GThread::sleep(miliseconds);
}






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
	while(pthread_mutex_trylock(&m_mutex) != 0)
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
			throw Ex("failed");
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
		throw Ex("failed");
}










void GWorkerThread::pump()
{
	while(m_keepAlive)
	{
		size_t jobId = m_master.nextJob(this);
		if(jobId == INVALID_INDEX)
			GThread::sleep(m_boredom & (~3)); // The "&(~3)" makes it so values < 4 will evaluate to 0, which just yields
		else
		{
#ifdef _DEBUG
			m_working = true;
#endif
			doJob(jobId);
#ifdef _DEBUG
			m_working = false;
#endif
		}
	}
	delete(this);
}






GMasterThread::GMasterThread()
: m_job(0), m_jobCount(0), m_pMasterLock(NULL)
{
}

GMasterThread::~GMasterThread()
{
	for(vector<GWorkerThread*>::iterator it = m_workers.begin(); it != m_workers.end(); it++)
	{
		if(m_pMasterLock)
		{
			GAssert(!(*it)->m_working); // A worker is still working on a job!
			(*it)->m_keepAlive = false; // The worker thread will delete the worker object
		}
		else
			delete(*it);
	}
	delete(m_pMasterLock);
}

void GMasterThread::addWorker(GWorkerThread* pWorker)
{
	if(m_pMasterLock)
		throw Ex("Sorry, workers may not be added after jobs have already been performed.");
	m_workers.push_back(pWorker);
}

void GMasterThread::doJobs(size_t jobCount)
{
	// Update the number of available jobs
	{
		GSpinLockHolder lockHolder(m_pMasterLock, "GMasterThread::reset");
		m_job = 0;
		m_jobCount = jobCount;
		m_activeWorkers = m_workers.size();
		for(vector<GWorkerThread*>::iterator it = m_workers.begin(); it != m_workers.end(); it++)
			(*it)->m_boredom = 0;
	}

	// Make sure there are workers to perform the jobs
	if(!m_pMasterLock)
	{
		if(m_workers.size() < 2)
		{
			// Just do the jobs now
			if(m_workers.size() == 0)
				throw Ex("There are no worker threads. addWorker must be called at least once before you call doJobs.");
			GWorkerThread* pWorker = m_workers[0];
			while(m_job < m_jobCount)
				pWorker->doJob(m_job++);
			m_activeWorkers = 0;
		}
		else
		{
			// Spawn a thread for each worker
			m_pMasterLock = new GSpinLock();
			for(vector<GWorkerThread*>::iterator it = m_workers.begin(); it != m_workers.end(); it++)
				GThread::spawnThread(spawnWorker, *it);
		}
	}

	// Sleep until all jobs are done
	while(m_activeWorkers > 0)
	{
		if(m_jobCount > 20)
			GThread::sleep(0); // todo: maybe sleep a little longer if there are a lot of jobs
	}
	GAssert(m_job >= m_jobCount);
}

size_t GMasterThread::nextJob(GWorkerThread* pWorker)
{
	GSpinLockHolder lockHolder(m_pMasterLock, "GMasterThread::nextJob");
	if(m_job < m_jobCount)
	{
		pWorker->m_boredom = 0;
		return m_job++;
	}
	else
	{
		if(pWorker->m_boredom == 0 && m_activeWorkers > 0)
			m_activeWorkers--;
		pWorker->m_boredom = std::min((size_t)100, (pWorker->m_boredom + 1) * 3 / 2);
		return INVALID_INDEX;
	}
}

// static
unsigned int GMasterThread::spawnWorker(void* pWorkerObject)
{
	((GWorkerThread*)pWorkerObject)->pump();
	return 0;
}


} // namespace GClasses

