#include "GSmtp.h"
#include "../GClasses/GError.h"
#include "GSocket.h"
#include "../GClasses/GThread.h"
#include <time.h>
#include <string>

using std::string;

namespace GClasses {

GSmtp::GSmtp(const char* szTo, const char* szFrom, const char* szSubject, const char* szMessage, const char* szSMPTServer)
{
	m_pSocket = NULL;
	m_szFrom = szFrom;
	m_szTo = szTo;
	m_szSubject = szSubject;
	m_szMessage = szMessage;
	m_szSMPTServer = szSMPTServer;
	m_eState = SS_Init;
}

GSmtp::~GSmtp()
{
	delete m_pSocket;
}

void GSmtp::send()
{
	m_pSocket = new GSocketClient(false, 0);
	if(!m_pSocket->Connect(m_szSMPTServer, 25))
		ThrowError("Failed to connect to ", m_szSMPTServer, " on port ", to_str(25));
	size_t nMessageSize;
	unsigned char* pMessage;
	time_t tStart, t;
	time(&tStart);
	while(true)
	{
		while(m_pSocket->GetMessageCount() > 0)
		{
			pMessage = m_pSocket->GetNextMessage(&nMessageSize);
			receive((const char*)pMessage, nMessageSize);
		}
		if(m_eState == SS_Close)
			break;
		time(&t);
		if(t - tStart > 30)
			ThrowError("Timed out");
		GThread::sleep(100);
	}
}

/*static*/ void GSmtp::sendEmail(const char* szTo, const char* szFrom, const char* szSubject, const char* szMessage, const char* szSMPTServer)
{
	GSmtp* pSmtp = new GSmtp(szTo, szFrom, szSubject, szMessage, szSMPTServer);
	pSmtp->send();
}

// SMTP is a line-based protocol, so receive until we have a full line
void GSmtp::receive(const char* pBuff, size_t nLen)
{
	int i = -1;
	for(size_t n = 0; n < nLen; n++)
	{
		if(pBuff[n] == '\n')
			i = (int)n;
	}
	if(i >= 0)
	{
		for(int n = 0; n <= i; n++)
			m_receiveBuffer << pBuff[n];
		string s = m_receiveBuffer.str();
		m_receiveBuffer.str("");
		m_receiveBuffer.clear();
		receiveLine(s.c_str());
		receive(pBuff + i + 1, nLen - (i + 1));
	}
	else
	{
		for(size_t n = 0; n < nLen; n++)
			m_receiveBuffer << pBuff[n];
	}
}

void GSmtp::receiveLine(const char* szLine)
{
	char szBuff[1024];
	if (m_eState == SS_Init && szLine[0] == '2')
	{
		strcpy(szBuff, "HELO there\r\n");
		m_pSocket->Send((unsigned char*)szBuff, (int)strlen(szBuff));
        	m_eState = SS_Mail;
	}
	else if (m_eState == SS_Mail && szLine[0] == '2')
	{
		strcpy(szBuff, "MAIL FROM:");
		m_pSocket->Send((unsigned char*)szBuff, (int)strlen(szBuff));
		strcpy(szBuff, m_szFrom);
		m_pSocket->Send((unsigned char*)szBuff, (int)strlen(szBuff));
		strcpy(szBuff, "\r\n");
		m_pSocket->Send((unsigned char*)szBuff, (int)strlen(szBuff));
        	m_eState = SS_Rcpt;
	}
	else if (m_eState == SS_Rcpt && szLine[0] == '2')
	{
		strcpy(szBuff, "RCPT TO:");
		m_pSocket->Send((unsigned char*)szBuff, (int)strlen(szBuff));
		strcpy(szBuff, m_szTo);
		m_pSocket->Send((unsigned char*)szBuff, (int)strlen(szBuff));
		strcpy(szBuff, "\r\n");
		m_pSocket->Send((unsigned char*)szBuff, (int)strlen(szBuff));
		m_eState = SS_Data;
	}
	else if (m_eState == SS_Data && szLine[0] == '2')
	{
		strcpy(szBuff, "DATA\r\n");
		m_pSocket->Send((unsigned char*)szBuff, (int)strlen(szBuff));
		m_eState = SS_Body;
	}
	else if (m_eState == SS_Body && szLine[0] == '3')
	{
		strcpy(szBuff, "From: ");
		m_pSocket->Send((unsigned char*)szBuff, (int)strlen(szBuff));
		strcpy(szBuff, m_szFrom);
		m_pSocket->Send((unsigned char*)szBuff, (int)strlen(szBuff));
		strcpy(szBuff, "\nTo: ");
		m_pSocket->Send((unsigned char*)szBuff, (int)strlen(szBuff));
		strcpy(szBuff, m_szTo);
		m_pSocket->Send((unsigned char*)szBuff, (int)strlen(szBuff));
		strcpy(szBuff, "\nSubject: ");
		m_pSocket->Send((unsigned char*)szBuff, (int)strlen(szBuff));
		strcpy(szBuff, m_szSubject);
		m_pSocket->Send((unsigned char*)szBuff, (int)strlen(szBuff));
		strcpy(szBuff, "\n\n");
		m_pSocket->Send((unsigned char*)szBuff, (int)strlen(szBuff));
		m_pSocket->Send((unsigned char*)m_szMessage, (int)strlen(m_szMessage));
		strcpy(szBuff, "\n.\n");
		m_pSocket->Send((unsigned char*)szBuff, (int)strlen(szBuff));
		m_eState = SS_Quit;
	}
	else if (m_eState == SS_Quit && szLine[0] == '2')
	{
		strcpy(szBuff, "QUIT\r\n");
		m_pSocket->Send((unsigned char*)szBuff, (int)strlen(szBuff));
		m_eState = SS_Close;
	}
	else
	{
		ThrowError("Something's broken.  The SMTP server said: ", szLine);
	}
}

} // namespace GClasses

