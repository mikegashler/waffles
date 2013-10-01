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

#ifndef __GSPINLOCK_H__
#define __GSPINLOCK_H__

#ifndef WINDOWS
#	include <pthread.h>
#endif

namespace GClasses {

/// A spin-lock for synchronization purposes
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


class GSpinLockHolder
{
protected:
	GSpinLock* m_pLock;

public:
	GSpinLockHolder(GSpinLock* pLock, const char* szWhoHoldsTheLock)
	{
		m_pLock = pLock;
		pLock->lock(szWhoHoldsTheLock);
	}

	~GSpinLockHolder()
	{
		m_pLock->unlock();
	}
};

} // namespace GClasses

#endif // __GSPINLOCK_H__
