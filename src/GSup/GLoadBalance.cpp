/*
	Copyright (C) 2008, Mike Gashler

	This program is free software; you can redistribute it and/or
	modify it under the terms of the GNU General Public
	License as published by the Free Software Foundation; either
	version 2 of the License, or (at your option) any later version.

	see http://www.gnu.org/copyleft/gpl.html
*/

#include "GLoadBalance.h"
#include "../GClasses/GError.h"
#include "../GClasses/GHolders.h"
#include "GSocket.h"
#include "GBlob.h"
#include "../GClasses/GThread.h"
#include "../GClasses/GBits.h"
#include "../GClasses/GTime.h"
#include <time.h>
#include <string>

using namespace GClasses;
using std::string;


GLoadBalanceSlave::GLoadBalanceSlave(const char* szAddr, int port, DoJobFunc pDoJobFunc)
: m_addr(szAddr), m_port(port)
{
	m_pSocket = NULL;
	m_pDoJobFunc = pDoJobFunc;
	m_patience = 10;
	m_sleepMiliSecs = 1000;
	m_timeOutSecs = 20;
	m_stayConnected = true;
}

GLoadBalanceSlave::~GLoadBalanceSlave()
{
	delete(m_pSocket);
}

unsigned char* GLoadBalanceSlave::requestTask(size_t* pBlobSize, bool stayConnected, int timeoutSeconds, bool* timedOut, bool* allDone)
{
	*pBlobSize = 0;
	*timedOut = false;
	*allDone = false;
	unsigned char* pBlob = NULL;
	if(!m_pSocket)
	{
		m_pSocket = new GSocketClient(false, 65536);
		if(!m_pSocket->Connect(m_addr.c_str(), m_port, 20))
			ThrowError("Failed to connect to server");
	}
	unsigned int packetID = 101; // client requests a task
	if(!m_pSocket->Send(&packetID, sizeof(unsigned int)))
		ThrowError("Failed to send packet");
	time_t start = time(NULL);
	while(true)
	{
		if(m_pSocket->GetMessageCount() > 0)
		{
			size_t blobSize;
			pBlob = m_pSocket->GetNextMessage(&blobSize);
			ArrayHolder<unsigned char> hBlob(pBlob);
			if(blobSize < (int)sizeof(unsigned int))
				ThrowError("Client received tiny blob of size: ", to_str(blobSize));
			unsigned int packetID = *(unsigned int*)pBlob;
			if(packetID == 102) // server delivers a task to client
			{
				*pBlobSize = blobSize;
				pBlob = hBlob.release();
				break;
			}
			else if(packetID == 103) // server has no idle tasks left, please try again later
			{
				break;
			}
			else if(packetID == 104) // all done, please exit
			{
				*allDone = true;
				break;
			}
			else
				ThrowError("Client received unexpected packet with id: ", to_str(packetID));
		}
		else if(time(NULL) - start > timeoutSeconds)
		{
			*timedOut = true;
			break;
		}
	}
	if(!stayConnected)
	{
		delete(m_pSocket);
		m_pSocket = NULL;
	}
	return pBlob;
}

void GLoadBalanceSlave::reportResults(unsigned char* pBlob, size_t blobSize, bool stayConnected)
{
	if(!m_pSocket)
	{
		m_pSocket = new GSocketClient(false, 65536);
		if(!m_pSocket->Connect(m_addr.c_str(), m_port, 20))
			ThrowError("Failed to connect to server");
	}
	unsigned int packetID = 100; // client reports results for a task
	if(!m_pSocket->Send2(&packetID, sizeof(unsigned int), pBlob, blobSize))
		ThrowError("Client failed to send response packet");
	if(!stayConnected)
	{
		delete(m_pSocket);
		m_pSocket = NULL;
	}
}

int GLoadBalanceSlave::go(void* pThis)
{
	int ret = 0;
	int patience = m_patience;
	while(true)
	{
		bool timedOut;
		bool allDone;
		size_t blobSize;
		unsigned char* pBlob = requestTask(&blobSize, m_stayConnected, m_timeOutSecs, &timedOut, &allDone);
		if(pBlob)
		{
			GBlobIncoming blobIn(pBlob, blobSize, true);
			unsigned int packetID;
			blobIn.get(&packetID);
			GBlobOutgoing blobOut(16, true);
			m_pDoJobFunc(pThis, blobIn, blobOut);
			reportResults(blobOut.getBlob(), blobOut.getBlobSize(), m_stayConnected);
		}
		else
		{
			if(allDone)
				break;
			if(timedOut)
			{
				if(--patience <= 0)
				{
					ret = 1;
					break;
				}
			}
			GThread::sleep(m_sleepMiliSecs);
		}
	}
	return ret;
}

// -----------------------------------------------------------

GLoadBalanceMaster::GLoadBalanceMaster(int port) : m_blobOut(256, true)
{
	m_pSocket = new GSocketServer(false, 65536, port, 4096);
	m_allDone = false;
}

GLoadBalanceMaster::~GLoadBalanceMaster()
{
	delete(m_pSocket);
}

bool GLoadBalanceMaster::results(GBlobIncoming* pBlobIn)
{
	size_t blobSize;
	int connection;
	while(true)
	{
		if(m_pSocket->GetMessageCount() <= 0)
			return false;
		unsigned char* pBlob = m_pSocket->GetNextMessage(&blobSize, &connection);
		pBlobIn->setBlob(pBlob, blobSize, true);
		unsigned int packetID;
		pBlobIn->get(&packetID);
		switch(packetID)
		{
			case 100: // client reports results for a task
				return true;

			case 101: // client requests a task
				m_blobOut.setPos(0);
				if(getTask(&m_blobOut))
				{
					unsigned int newPacketID = 102; // server delivers a task to client
					if(!m_pSocket->Send2(&newPacketID, sizeof(unsigned int), m_blobOut.getBlob(), m_blobOut.getBlobSize(), connection))
						ThrowError("Failed to send response packet");
				}
				else
				{
					unsigned int newPacketID;
					if(m_allDone)
						newPacketID = 103; // server has no idle tasks left, please try again later
					else
						newPacketID = 104; // all done, please exit
					if(!m_pSocket->Send(&newPacketID, sizeof(unsigned int), connection))
						ThrowError("Failed to send response packet");
				}
				break;

			default:
				ThrowError("Server received unexpected packet with id: ", to_str(packetID));
				break;
		}
	}
}

// -----------------------------------------------------------

namespace GClasses {
class GWindowJob
{
public:
	int m_dim;
	int m_timeStamp;
	double m_lastSubmittedTime;

	GWindowJob() : m_timeStamp(0), m_lastSubmittedTime(0)
	{
	}
};
}


GLoadBalanceWindowMaster::GLoadBalanceWindowMaster(int port, int dims) : GLoadBalanceMaster(port), m_jobs(compareByTimeStamp, NULL), m_dims(dims)
{
	m_pJobArray = new GWindowJob[m_dims];
	for(int i = 0; i < m_dims; i++)
	{
		m_pJobArray[i].m_dim = i;
		m_jobs.insert(&m_pJobArray[i]);
	}
	m_patienceSeconds = 3;
}

// virtual
GLoadBalanceWindowMaster::~GLoadBalanceWindowMaster()
{
	delete[] m_pJobArray;
}

// static
int GLoadBalanceWindowMaster::compareByTimeStamp(void* pThis, void* pA, void* pB)
{
	return GBits::compareInts(((GWindowJob*)pA)->m_timeStamp, ((GWindowJob*)pB)->m_timeStamp);
}

// virtual
bool GLoadBalanceWindowMaster::getTask(GBlobOutgoing* pBlobOut)
{
	// Move jobs back into the priority queue
	double curtime = GTime::seconds();
	while(m_recentSubmissions.size() > 0)
	{
		GWindowJob* pJob = m_recentSubmissions.front();
		if(curtime - pJob->m_lastSubmittedTime < m_patienceSeconds)
			break;
		m_jobs.insert(m_recentSubmissions.front());
		m_recentSubmissions.pop_front();
	}
	if(m_jobs.size() <= 0)
		return false;

	// Transfer to the recently-submitted queue
	GWindowJob* pJob = (GWindowJob*)m_jobs.minimum();
	m_jobs.removeMin();
	pJob->m_lastSubmittedTime = curtime;
	m_recentSubmissions.push_back(pJob);

	// Make the blob
	pBlobOut->add(pJob->m_dim);
	pBlobOut->add(pJob->m_timeStamp);
	giveTaskToSlave(pBlobOut, pJob->m_dim);
	return true;
}

bool GLoadBalanceWindowMaster::process()
{
	if(results(&m_blobIn))
	{
		int dim; m_blobIn.get(&dim);
		if(dim < 0 || dim >= m_dims)
			ThrowError("dim from client out of range");
		int dimTimestamp; m_blobIn.get(&dimTimestamp);
		if(dim == 0)
			onAdvanceDim(dimTimestamp);
		if(dimTimestamp == m_pJobArray[dim].m_timeStamp)
		{
			receiveResultsFromSlave(&m_blobIn, dim);
			m_pJobArray[dim].m_timeStamp++;
			m_pJobArray[dim].m_lastSubmittedTime = 0;
		}
		else
		{
			// Received outdated results. Discard them.
		}
		return true;
	}
	else
		return false;
}
