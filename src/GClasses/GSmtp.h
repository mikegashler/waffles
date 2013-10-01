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

#ifndef __GSMTP_H__
#define __GSMTP_H__

#include <sstream>

namespace GClasses {

class GTCPClient;

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
	GTCPClient* m_pSocket;
	std::ostringstream m_receiveBuffer;

	GSmtp(const char* szTo, const char* szFrom, const char* szSubject, const char* szMessage, const char* szSMPTServer);
public:
	virtual ~GSmtp();

	static void sendEmail(const char* szTo, const char* szFrom, const char* szSubject, const char* szMessage, const char* szSMPTServer);

	void send();

protected:
	void receive(const char* pBuff, size_t nLen);
	void receiveLine(const char* szLine);
};


} // namespace GClasses

#endif // __GSMTP_H__
