/*
  The contents of this file are dedicated by all of its authors, including

    Michael S. Gashler,
    anonymous contributors,

  to the public domain (http://creativecommons.org/publicdomain/zero/1.0/).

  Note that some moral obligations still exist in the absence of legal ones.
  For example, it would still be dishonest to deliberately misrepresent the
  origin of a work. Although we impose no legal requirements to obtain a
  license, it is beseeming for those who build on the works of others to
  give back useful improvements, or pay it forward in their own field. If
  you would like to cite us, a published paper about Waffles can be found
  at http://jmlr.org/papers/volume12/gashler11a/gashler11a.pdf. If you find
  our code to be useful, the Waffles team would love to hear how you use it.
*/


#include "login.h"
#include "server.h"

using std::ostream;
using std::cout;
using std::string;


void Login::pageAdmin(Server* pServer, GDynamicPageSession* pSession, ostream& response)
{
	// Check access privileges
	Account* pAccount = getAccount(pSession);
	if(!pAccount || !pAccount->isAdmin())
	{
		response << "Sorry, you must be an admin to access this page.";
		return;
	}

	// Process params
	std::vector<Account*>& accounts = pServer->accounts();
	if(pSession->paramsLen() > 0)
	{
		GHttpParamParser params(pSession->params());
		const char* szAction = params.find("action");
		if(strcmp(szAction, "adjust_permissions") == 0)
		{
			// Set assistants
			pServer->log("Clearing all admins, assistants, and bans");
			for(size_t i = 0; i < accounts.size(); i++)
			{
				Account* pUserAccount = accounts[i];
				if(i > 0)
					pUserAccount->makeAdmin(false);
				pUserAccount->makeAssistant(false);
				pUserAccount->makeBanned(false);
			}
			auto map = params.map();
			for(std::map<const char*, const char*, GClasses::strComp>::iterator it = map.begin(); it != map.end(); it++)
			{
				if(strncmp(it->first, "admin_", 6) == 0)
				{
					const char* szUsername = it->first + 6;
					Account* pUserAccount = pServer->findAccount(szUsername);
					if(strcmp(it->second, "true") == 0)
					{
						string s = "Making admin: ";
						s += pUserAccount->username();
						pServer->log(s.c_str());
						pUserAccount->makeAdmin(true);
					}
				}
				else if(strncmp(it->first, "assist_", 7) == 0)
				{
					const char* szUsername = it->first + 7;
					Account* pUserAccount = pServer->findAccount(szUsername);
					if(strcmp(it->second, "true") == 0)
					{
						string s = "Adding assistant: ";
						s += pUserAccount->username();
						pServer->log(s.c_str());
						pUserAccount->makeAssistant(true);
					}
				}
				else if(strncmp(it->first, "ban_", 4) == 0)
				{
					const char* szUsername = it->first + 4;
					Account* pUserAccount = pServer->findAccount(szUsername);
					if(strcmp(it->second, "true") == 0)
					{
						string s = "Banning user: ";
						s += pUserAccount->username();
						pServer->log(s.c_str());
						pUserAccount->makeBanned(true);
					}
				}
			}
		}
		else if(strcmp(szAction, "shutdown") == 0)
		{
			pServer->log("Shutting down server as directed by admin");
			pServer->m_keepGoing = false;
		}
		else
		{
			response << "Sorry, unrecognized action: " << szAction << "\n";
		}
	}

	// Form to give users assistant privileges
	response << "<h2>Grant user permissions</h2>\n\n";
	response << "<form method=\"post\">";
	response << "<input type=\"hidden\" name=\"action\" value=\"adjust_permissions\" />\n";
	response << "<table><tr>";
	response << "<td><b><i><u>Username</u></i></b></td>";
	response << "<td><b><i><u>Admin</u></i></b></td>";
	response << "<td><b><i><u>Assistant</u></i></b></td>";
	response << "<td><b><i><u>Banned</u></i></b></td>";
	response << "</tr>\n";
	for(size_t i = 0; i < accounts.size(); i++)
	{
		Account* pUserAccount = accounts[i];
		response << "<tr>";
		response << "<td>" << pUserAccount->username() << "</td>";
		response << "<td><input type=\"checkbox\" name=\"admin_" << pUserAccount->username() << "\" value=\"true\"" << (pUserAccount->isAdmin() ? " checked" : "") << "></td>";
		response << "<td><input type=\"checkbox\" name=\"assist_" << pUserAccount->username() << "\" value=\"true\"" << (pUserAccount->isAssistant() ? " checked" : "") << "></td>";
		response << "<td><input type=\"checkbox\" name=\"ban_" << pUserAccount->username() << "\" value=\"true\"" << (pUserAccount->isBanned() ? " checked" : "") << "></td>";
		response << "</tr>";
	}
	response << "</table>\n";
	response << "<br>\n";
	response << "<input type=\"submit\" value=\"Submit\"></form>";
	response << "</form><br><br>\n\n";

	// Form to train the recommender system a bit more
	response << "<form name=\"trainform\" method=\"get\">\n";
	response << "	Refine the recommender system:<br>\n";
	response << "	<input type=\"hidden\" name=\"action\" value=\"train\" />\n";
	response << "	<input type=\"submit\" value=\"Train\">\n";
	response << "</form><br><br>\n\n";

	// Form to shut down the server
	response << "<form name=\"shutdownform\" method=\"get\">\n";
	response << "	Shut down the daemon:<br>\n";
	response << "	<input type=\"hidden\" name=\"action\" value=\"shutdown\" />\n";
	response << "	<input type=\"submit\" value=\"Shut down now\">\n";
	response << "</form><br><br>\n\n";
}

void Login::pageAccount(Server* pServer, GDynamicPageSession* pSession, ostream& response)
{
	Terminal* pTerminal = getTerminal(pSession);
	Account* pAccount = pTerminal->currentAccount();
	if(pAccount)
	{
		response << "<h3>Manage your account</h3>\n";

		// Change username
		response << "<form><input type=\"hidden\" name=\"action\" value=\"changename\">";
		response << "Change your username to: <input type=\"text\" name=\"newname\" value=\"" << pAccount->username() << "\">";
		response << "<input type=\"submit\" value=\"Change username\">";
		response << "</form>";
		response << "<br><br>";

		// Add a password
		bool havePassword = true;
		if(pAccount->passwordHash() == nullptr || strlen(pAccount->passwordHash()) < 1)
			havePassword = false;
		if(!havePassword)
		{
			response << "If you would like, you can add a password so you can access this account from other devices.<br>";
		}
		response << "<form><input type=\"hidden\" name=\"action\" value=\"changepassword\">";
		response << "<table><tr><td align=right>" << (havePassword ? "Change" : "Set") << " your password to:</td><td><input type=\"password\" name=\"newpw\"></td><td></td></tr>";
		response << "<tr><td align=right>Again:</td><td><input type=\"password\" name=\"pwagain\"></td><td><input type=\"submit\" value=\"Change password\"></td></tr></table>";
		response << "</form>";
		response << "<br><br>";

		// Check box to require password on this account
		if(havePassword)
		{
			response << "<form><input type=\"hidden\" name=\"action\" value=\"requirepw\">";
			response << "<input type=\"checkbox\" name=\"cb\"" << (pTerminal->requirePassword() ? " checked" : "") << " onChange=\"this.form.submit()\">Require a password when I visit using this device.";
			response << "</form>";
			response << "<br><br>";
		}
/*
		// Button to nuke my account
		response << "Warning: Don't push this button unless you really mean it! ";
		response << "<form name=\"nukeself\" action=\"/account\" method=\"get\">\n";
		response << "<input type=\"hidden\" name=\"action\" value=\"nukeself\">\n";
		response << "<input type=\"submit\" value=\"Nuke My Account\">\n";
		response << "</form>\n";
*/
	}
	else
	{
		// Not logged in
		response << "Click on an account to log in:<br>(You can change the username or password after you log in.)<br><br>\n";
		response << "<table>\n";
		for(size_t i = 0; i < pTerminal->accountCount(); i++)
		{
			bool reqpw = pTerminal->requirePassword(i);
			response << "<tr><td>";
			Account* pAcc = pTerminal->account(i);
			if(!reqpw)
				response << "<a href=\"/b/account?action=login&name=" << pAcc->username() << "\">";
			response << "<div style=\"";
			response << "background-color:#203050;width:220px;height:60px;";
			response << "border:1px solid #000000;border-radius:15px;";
			response << "text-align:center;color:white";
			response << "\">";
			if(!pTerminal->requirePassword(i))
				response << "<br>";
			response << pAcc->username();
			if(pAcc->isAdmin())
				response << " (Admin)";
			if(reqpw)
			{
				response << "<form>";
				response << "<input type=\"hidden\" name=\"action\" value=\"login\">";
				response << "<input type=\"hidden\" name=\"name\" value=\"" << pAcc->username() << "\">";
				response << "<input type=\"password\" name=\"password\">";
				response << "</form>";
			}
			response << "</div>";
			if(!reqpw)
				response << "</a>";
			response << "</td><td>";
			response << "<a href=\"/account?action=forget&name=" << pAcc->username() << "\">Forget account</a>";
			response << "</td></tr>\n";
		}
		response << "</table>\n";
	}
/*
	Account* pAccount = getAccount(pSession);
	if(pSession->paramsLen() > 0)
	{
		// Check the password
		const char* szUsername = params.find("username");
		const char* szPasswordHash = params.find("password");
		if(szUsername)
		{
			Account* pLoadedAccount = pServer->loadAccount(szUsername, szPasswordHash);
			if(pLoadedAccount)
				getTerminal(pSession)->logIn(pLoadedAccount);
			else
				response << "<big><big>Incorrect Password! Please try again</big></big><br><br>\n";
		}
	}

	response << "<br><br>\n";
	response << "<SCRIPT language=\"JavaScript\" src=\"/b/sha1.js\" type=\"text/javascript\">\n</SCRIPT>\n";
	if(pAccount)
	{
		response << "Your current username is: ";
		const char* szUsername = pAccount->username();
		if(*szUsername == '_')
			response << "anonymous";
		else
			response << szUsername;
		response << ".<br>\n";
		response << "Switch user:<br>\n";
	}
	else
		response << "Please log in:<br><br>\n";
	response << "<form name=\"loginform\" action=\"/account\" method=\"get\" onsubmit=\"return HashPassword('";
	response << pServer->passwordSalt();
	response << "')\">\n";
	response << "	Username:<input type=\"text\" name=\"username\" ><br>\n";
	response << "	Password:<input type=\"password\" name=\"password\" ><br>\n";
	response << "	<input type=\"submit\" value=\"Log In\">\n";
	response << "</form><br>\n\n";

	response << "or <a href=\"/newaccount\">create a new account</a><br><br><br>\n";
*/
}

void Login::pageNewAccount(Server* pServer, GDynamicPageSession* pSession, ostream& response)
{
	const char* szUsername = "";
	const char* szPassword = "";
	const char* szPWAgain = "";
	GHttpParamParser params(pSession->params());
	if(pSession->paramsLen() > 0)
	{
		// Get the action
		const char* szError = NULL;
		const char* szAction = params.find("action");
		if(!szAction)
			szError = "Expected an action param";
		if(!szError && _stricmp(szAction, "newaccount") != 0)
			szError = "Unrecognized action";

		szUsername = params.find("username");
		szPassword = params.find("password");
		szPWAgain = params.find("pwagain");

		// Check the parameters
		if(!szUsername || strlen(szUsername) < 1)
			szError = "The username is not valid";
		if(!szPassword || strlen(szPassword) < 1)
			szError = "The password is not valid";
		if(!szPWAgain || strcmp(szPassword, szPWAgain) != 0)
			szError = "The passwords don't match";
		if(!szError)
		{
			// Create the account
			Account* pAccount = pServer->newAccount(szUsername, szPassword);
			if(!pAccount)
				szError = "That username is already taken.";
			else
			{
				pServer->saveState();
				response << "<big>An account has been successfully created.</big><br><br> Click here to <a href=\"/account\">log in</a><br>\n";
				return;
			}
		}
		if(szError)
		{
			response << "<center>";
			response << szError;
			response << "</center><br><br>\n\n";
			szPassword = "";
			szPWAgain = "";
		}
	}

	response << "<br><center><table width=\"400\" border=\"0\" cellpadding=\"0\" cellspacing=\"0\"><tr><td>\n";
	response << "<SCRIPT language=\"JavaScript\" src=\"/b/sha1.js\" type=\"text/javascript\">\n</SCRIPT>\n";
	response << "	<big><big><b>Create a new account</b></big></big><br><br>\n";
	response << "	<form name=\"newaccountform\" action=\"/newaccount\" method=\"post\" onsubmit=\"return HashNewAccount('";
	response << pServer->passwordSalt();
	response << "')\">\n";
	response << "		<input type=\"hidden\" name=\"action\" value=\"newaccount\" />\n";
	response << "		Username: <input type=\"text\" size=\"15\" name=\"username\" value=\"";
	response << szUsername;
	response << "\"><br><br>\n";
	response << "		Password: <input type=\"password\" name=\"password\" size=\"15\" value=\"";
	response << szPassword;
	response << "\"><br>\n";
	response << "		PW Again: <input type=\"password\" name=\"pwagain\" size=\"15\" value=\"";
	response << szPWAgain;
	response << "\"><br><br>\n";
	response << "		<input type=\"submit\" value=\"Submit\">\n";
	response << "	</form><br>\n\n";
	response << "</tr></td></table></center>\n";
}

void Login::pageTools(Server* pServer, GDynamicPageSession* pSession, std::ostream& response)
{
	Terminal* pTerminal = getTerminal(pSession);
	Account* pAccount = pTerminal->currentAccount();
	response << "let _username = " << pAccount->username() << ";\n\n";
	response << pServer->cache("tools.js");
}
