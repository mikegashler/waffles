/*
	Copyright (C) 2008, Mike Gashler

	This program is free software; you can redistribute it and/or
	modify it under the terms of the GNU General Public
	License as published by the Free Software Foundation; either
	version 2 of the License, or (at your option) any later version.

	see http://www.gnu.org/copyleft/gpl.html
*/

#ifndef __GLOADBALANCE__
#define __GLOADBALANCE__

#include <string>
#include <deque>
#include "GBlob.h"
#include "../GClasses/GPriorityQueue.h"

namespace GClasses {

class GSocketServer;
class GSocketClient;
class GWindowJob;

typedef void (*DoJobFunc)(void* pThis, GBlobIncoming& blobIn, GBlobOutgoing& blobOut);

/// The client for a load-balancing problem. Clients pull tasks from the server
/// when they are ready to do more work, then push the results back to the
/// master.
class GLoadBalanceSlave
{
protected:
	std::string m_addr;
	int m_port;
	GSocketClient* m_pSocket;
	DoJobFunc m_pDoJobFunc;
	int m_patience;
	int m_sleepMiliSecs;
	int m_timeOutSecs;
	bool m_stayConnected;

public:
	GLoadBalanceSlave(const char* szAddr, int port, DoJobFunc pDoJobFunc);
	~GLoadBalanceSlave();

	/// specify the number of time-outs before it bails.
	void setPatience(int patience) { m_patience = patience; }

	/// specify how long to sleep when there are no available jobs.
	void setSleepMiliSecs(int ms) { m_sleepMiliSecs = ms; }

	/// specify how long to wait without receiving a reply before it will
	/// re-send a request for a job to the master.
	void setTimeOutSecs(int secs) { m_timeOutSecs = secs; }

	/// specify whether or not to stay connected while doing the job.
	void setStayConnected(bool b) { m_stayConnected = b; }

	/// Returns 0 if the master said to exit. Returns 1 if it exited due to excessive time-outs.
	int go(void* pThis);

protected:
	/// The client calls this to request another task from the server. If
	/// stayConnected is true, the socket connection will remain open. If
	/// allDone is set to true by this call, the client should exit. The
	/// caller is responsible to delete the blob that is returned. NULL
	/// is returned if there are no tasks ready. (It will wait "timeoutSeconds"
	/// for a response from the server before setting "timedOut" to true and
	/// returning NULL.) The first uint in the blob will be the value 102,
	/// followed by the task blob.
	unsigned char* requestTask(int* pBlobSize, bool stayConnected, int timeoutSeconds, bool* timedOut, bool* allDone);

	/// Call this to send results back to the server
	void reportResults(unsigned char* pBlob, int blobSize, bool stayConnected);
};




/// The server for a load-balancing problem. Clients pull tasks from the server
/// when they are ready to do more work, then push the results back to the
/// master.
class GLoadBalanceMaster
{
protected:
	GSocketServer* m_pSocket;
	bool m_allDone;
	GBlobOutgoing m_blobOut;

public:
	GLoadBalanceMaster(int port);
	~GLoadBalanceMaster();

	/// Returns false if there are no ready results, otherwise
	/// returns a blob.
	/// (You should call this in a tight loop because other
	/// communication, including handing out tasks, relies on
	/// it to pump messages.)
	/// (Note that results may not come back in the same order, so you are
	/// responsible to encode enough information in your tasks and
	/// responses that you will recognize which responses correspond
	/// to which tasks.)
	bool results(GBlobIncoming* pBlobIn);

	/// This is called when a slave needs a task to do. If there are no
	/// tasks ready, just return false.
	virtual bool getTask(GBlobOutgoing* pBlobOut) = 0;

	void setAllDone() { m_allDone = true; }
};



/// If you have a constant number of task chains, this master will balance
/// between keeping clients busy on different tasks, and keeping all of the
/// chains moving at approximately the same rate.
class GLoadBalanceWindowMaster : public GLoadBalanceMaster
{
protected:
	GBlobIncoming m_blobIn;
	GPriorityQueue m_jobs;
	std::deque<GWindowJob*> m_recentSubmissions;
	GWindowJob* m_pJobArray;
	int m_dims;
	double m_patienceSeconds;

public:
	GLoadBalanceWindowMaster(int port, int dims);
	virtual ~GLoadBalanceWindowMaster();

	/// This value specifies how long the master will wait after
	/// submitting a job to one client before it will give the same
	/// job to another client if there is no other work to do of the
	/// same level of importance. If jobs have varying levels of
	/// urgency, patience encourages clients to work on less-urgent
	/// jobs if someone else is already working on more-urgent jobs.
	/// This can benefit the system overall. But after the patience
	/// runs out for the most urgent job, another client will start
	/// working on it.
	void setPatienceSeconds(double d) { m_patienceSeconds = d; }

	virtual void giveTaskToSlave(GBlobOutgoing* pBlobOut, int dim) = 0;
	virtual void receiveResultsFromSlave(GBlobIncoming* pBlobIn, int dim) = 0;

	static int compareByTimeStamp(void* pThis, void* pA, void* pB);

	virtual bool getTask(GBlobOutgoing* pBlobOut);

	/// You may want to overload this method for logging purposes
	virtual void onAdvanceDim(int iter);

	/// Call this in main pump. Returns true if it did some work. If
	/// it returns false, you should probably sleep for a bit.
	bool process();
};

} // namespace GClasses

#endif // __GLOADBALANCE__
