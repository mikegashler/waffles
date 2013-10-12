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
#	include <pthread.h>
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



/// On Windows, this implements a spin-lock. On Linux, this wraps pthread_mutex.
class GSpinLock
{
protected:
#ifdef _DEBUG
	const char* m_szWhoHoldsTheLock;
#endif
	volatile long m_dwLocked; /// maintaned on all platform as posix mutexes don't have a way to get current state.
                                 /// when not Win32 be aware that this value is shadowing the real mutex, and cannot be
                                 /// depended on especially in a MP enviroment.
#ifndef WINDOWS
	pthread_mutex_t m_mutex;
#endif

public:
	GSpinLock();
	virtual ~GSpinLock();

#ifndef NO_TEST_CODE
	/// Performs unit tests for this class. Throws an exception if there is a failure.
	static void test();
#endif // !NO_TEST_CODE

	void lock(const char* szWhoHoldsTheLock);
	void unlock();
	bool isLocked() { return m_dwLocked != 0; } /// see note above about m_dwLocked.
};


/// This holder takes a lock (if it is non-NULL) when you construct it.
/// It guarantees to release the lock when it is destroyed.
class GSpinLockHolder
{
protected:
	GSpinLock* m_pLock;

public:
	GSpinLockHolder(GSpinLock* pLock, const char* szWhoHoldsTheLock)
	: m_pLock(pLock)
	{
		if(pLock)
			pLock->lock(szWhoHoldsTheLock);
	}

	~GSpinLockHolder()
	{
		if(m_pLock)
			m_pLock->unlock();
	}
};




class GMasterThread;

/// An abstract class for performing jobs.
/// The idea is that you should be able to write the code to perform the jobs,
/// then use it in either a serial or parallel manner.
/// The class you write, that inherits from this one, will typically
/// have additional constructor parameters that pass in any values or data
/// necessary to define the jobs.
class GWorkerThread
{
public:
	volatile bool m_keepAlive;
	size_t m_boredom; // don't change this value without taking the master lock
	GMasterThread& m_master;
#ifdef _DEBUG
	volatile bool m_working; // this exists only to support a debug assert
#endif

	GWorkerThread(GMasterThread& master) : m_keepAlive(true), m_boredom(0), m_master(master) {}
	virtual ~GWorkerThread() {}

	/// This method is called by the master thread. Users should not need to call it. It pulls jobs
	/// from the master thread (by calling nextJob()) and does them as long as m_keepAlive is true.
	/// If the master returns INVALID_INDEX for the job id, it goes to sleep for a short time.
	/// (Each time it receives INVALID_INDEX, it becomes more bored, and so sleeps a little longer,
	/// up to a maximum nap of 100ms. When it receives a job, the boredom counter resets to 0.)
	/// When m_keepAlive is set to false, this method will delete this object and exit.
	void pump();

	/// This method should be implemented to perform the job indicated
	/// by the specified id. (The job ids range from 0 to jobCount-1.)
	/// The implementing class should be designed such that jobId is sufficient
	/// for it to obtain any information that it needs to do the job.
	/// Also, this method is also reponsible to report the results in a
	/// thread-safe manner. Here is an example of how to take a lock in order to do
	/// something critical:
	/// GSpinLockHolder lockHolder(m_master.getLock(), "MyWorkerThread::doJob");
	/// (Note that this assumes the master will never be deleted while a worker is doing a job.)
	virtual void doJob(size_t jobId) = 0;
};


/// Manages a pool of GWorkerThread objects. To use this class,
/// first call addWorker one or more times. Then, call doJobs.
class GMasterThread
{
protected:
	size_t m_job;
	size_t m_jobCount;
	std::vector<GWorkerThread*> m_workers;
	GSpinLock* m_pMasterLock;
	volatile size_t m_activeWorkers;

public:
	GMasterThread();

	/// Note that all worker threads reference the master, so it is not okay to delete the
	/// master while a worker is still doing a job.
	~GMasterThread();

	/// Adds a worker to the pool. Takes ownership of the worker object.
	void addWorker(GWorkerThread* pWorker);

	/// Perform some jobs. The job ids will range from 0 to jobCount-1.
	/// If no workers have been added, throws an exception.
	/// If only one worker has been added, that worker performs all of the
	/// jobs in the same thread as the master (so no new threads are spawned).
	/// If two or more workers have been added to the pool, the jobs will be
	/// performed by those workers in separate threads.
	/// This method does not return until all the jobs are done.
	void doJobs(size_t jobCount);

	/// This method is called by worker threads to obtain the next available job.
	/// Calling it is a contract to complete the job. Returns INVALID_INDEX if
	/// there are no more jobs to do. In that case, the worker thread should terminate.
	/// (This method is already synchronized, so you do not need to take a lock
	/// before calling it.)
	size_t nextJob(GWorkerThread* pWorkerWhoWantsAJob);

	/// Returns a pointer to the master lock. (If there is only one worker, then there
	/// are no worker threads, and this method will return NULL. Note that GSpinLockHolder
	/// checks for NULL, so it provides a good way to take the lock.)
	GSpinLock* getLock() { return m_pMasterLock; }

protected:
	/// This is a helper function that spawns a worker thread
	static unsigned int spawnWorker(void* pWorkerObject);
};


} // namespace GClasses

#endif // __GTHREAD_H__
