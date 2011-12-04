// -------------------------------------------------------------
// The contents of this file may be distributed under the CC0
// license (http://creativecommons.org/publicdomain/zero/1.0/),
// or any compatible license, including (but not limited to) all
// OSI-approved licenses (http://www.opensource.org/licenses).
// -------------------------------------------------------------

#ifndef SATELLITE_H
#define SATELLITE_H

#include <time.h>
#include <GClasses/GBlob.h>

#define MAX_PACKET_SIZE 16384
#define BUF_SIZE 4096

namespace GClasses {
	class GPackageClient;
}

class Satellite
{
protected:
	bool m_keepRunning;
	int m_timeoutSecs;
	time_t m_lastPacketTime;
	GClasses::GPackageClient* m_pSocket;
	GClasses::GBlobIncoming m_blobIn;
	GClasses::GBlobOutgoing m_blobOut;

	// File transfer variables
	FILE* m_pFile;
	std::string m_filename;
	unsigned long long m_fileLen;
	size_t m_filePos;
	unsigned char m_fileHash[20];
	unsigned char* m_pBuf;
	bool m_uploading;

public:
	Satellite();
	~Satellite();

	void Connect(const char* szAddr, int port, int timeoutSecs);
	void Disconnect();

	// Returns true if there were packets to process
	bool Process();

	void Go(const char* szAddr, int port, int connectGapMins, int timeoutSecs);

protected:
	void HandlePacket(GClasses::GBlobIncoming* pBlob);
	void OnServerWantsFileList(GClasses::GBlobIncoming* pBlob);
	void OnServerWantsToDisconnect(GClasses::GBlobIncoming* pBlobIn);
	void OnServerWantsPing(GClasses::GBlobIncoming* pBlobIn);
	void OnServerWantsDirectoryChange(GClasses::GBlobIncoming* pBlobIn);
	void OnServerWantsToSendFile(GClasses::GBlobIncoming* pBlobIn);
	void OnServerSendsFileChunk(GClasses::GBlobIncoming* pBlobIn);
	void OnServerDoneSendingFile(GClasses::GBlobIncoming* pBlobIn);
	void OnServerWantsToDownloadFile(GClasses::GBlobIncoming* pBlobIn);
	void TransferFileChunk();
	void CompleteFileTransfer();
	void ExecuteCommand(GClasses::GBlobIncoming* pBlobIn);
	void Delete_File(GClasses::GBlobIncoming* pBlobIn);
};



void Sha1DigestFile(unsigned char* pOut20ByteHash, const char* filename);

void doShellCommandCenter(int port);

#endif // SATELLITE_H
