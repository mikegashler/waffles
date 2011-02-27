#ifndef __GSMTP_H__
#define __GSMTP_H__

#include <sstream>

namespace GClasses {

class GQueue;
class GSocketClient;

/// For sending email to an SMTP server
class GSmtp
{
protected:
	enum SmtpState
	{
		SS_Init,
		SS_Mail,
		SS_Rcpt,
		SS_Data,
		SS_Body,
		SS_Quit,
		SS_Close,
	};

	SmtpState m_eState;
	const char* m_szFrom;
	const char* m_szTo;
	const char* m_szSubject;
	const char* m_szMessage;
	const char* m_szSMPTServer;
	GSocketClient* m_pSocket;
	std::ostringstream m_receiveBuffer;

	GSmtp(const char* szTo, const char* szFrom, const char* szSubject, const char* szMessage, const char* szSMPTServer);
public:
	virtual ~GSmtp();

	static void sendEmail(const char* szTo, const char* szFrom, const char* szSubject, const char* szMessage, const char* szSMPTServer);

	void send();

protected:
	void receive(const char* pBuff, int nLen);
	void receiveLine(const char* szLine);
};


} // namespace GClasses

#endif // __GSMTP_H__
