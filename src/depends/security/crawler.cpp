#include "crawler.h"
#include <string>
#include <queue>
#include <iostream>
#include <set>
#include "../../GClasses/GHtml.h"
#include "../../GClasses/GApp.h"
#include "../../GClasses/GHttp.h"
#include "../../GClasses/GThread.h"
#include "../../GClasses/GFile.h"

using namespace GClasses;
using std::string;
using std::queue;
using std::cout;
using std::set;

class BrokenLinkFinder;


unsigned char* downloadFromWeb(const char* szAddr, size_t timeout, size_t* pOutSize)
{
	if(*szAddr == '/')
	{
		return (unsigned char*)GFile::loadFile(szAddr, pOutSize);
	}
	else
	{
		GHttpClient client;
		if(!client.sendGetRequest(szAddr))
			throw Ex("Error connecting");
		float fProgress;
		time_t start = time(NULL);
		while(client.status(&fProgress) == GHttpClient::Downloading)
		{
			if((size_t)(time(NULL) - start) > timeout)
				break;
			GThread::sleep(50);
		}
		if(client.status(&fProgress) != GHttpClient::Done)
			throw Ex("Error downloading page");
		return client.releaseData(pOutSize);
	}
}
/*
class CrawlEntry
{
public:
	string origin;
	string destination;

	CrawlEntry(const char* szOrigin, const char* szDestination)
	: origin(szOrigin), destination(szDestination)
	{
	}
};

class BrokenLinkFinder
{
public:
	string m_base;
	queue<CrawlEntry*> m_q;
	set<string> m_beenThere;


	BrokenLinkFinder(string& sInitialUrl)
	{
		m_base = sInitialUrl;
		size_t last_slash = m_base.find_last_of("/");
		if(last_slash != string::npos)
			m_base.erase(last_slash + 1);
	}

	static string expandInitialUrl(const char* szInitialUrl)
	{
		string url(szInitialUrl);
		if(url.compare(0, 1, "/") != 0 && url.compare(0, 5, "http:") != 0 && url.compare(0, 6, "https:") != 0)
		{
			char buf[256];
			char* cwd = getcwd(buf, 256);
			if(!cwd)
				throw Ex("getcwd failed");
			string scwd(cwd);
			if(scwd[scwd.length() - 1] != '/')
				scwd += "/";
			url.insert(0, scwd);
		}
		return url;
	}

	void process_hyperlink(GHtmlElement* pEl)
	{
		for(size_t i = 0; i < pEl->attrNames.size(); i++)
		{
			if(_stricmp(pEl->attrNames[i].c_str(), "href") == 0)
			{
				string& url = pEl->attrValues[i];
				//cout << "	Links to: " << url << "\n";
				if(url.compare(0, 1, "/") != 0 && url.compare(0, 5, "http:") != 0 && url.compare(0, 6, "https:") != 0)
				{
					// Expand relative URLs
					string location = m_pEntry->destination;
					size_t last_slash = location.find_last_of("/");
					if(last_slash != string::npos)
						location.erase(last_slash + 1);
					url.insert(0, location);
				}
				if(url.compare(0, 1, "/") != 0 || url.compare(0, 5, "http:"))
				{
					CrawlEntry* pEntry = new CrawlEntry(m_pEntry->destination.c_str(), url.c_str());
					m_pCrawler->m_q.push(pEntry);
				}
				else
				{
					cout << "Skipping " << url << "\n";
				}
			}
		}
	}

	void process_html_file(GHtmlElement* pEl)
	{
		if(pEl->name.compare("a") == 0 || pEl->name.equals("A") == 0)
			process_hyperlink(pEl);
		for(size_t i = 0; i < pEl->children.size(); i++)
		{
			GHtmlElement* pChild = pEl->children[i];
			process_html_file(pEntry, pEl);
		}
	}

	void process_file(CrawlEntry* pEntry, unsigned char* pFile, size_t size)
	{
		if(!pFile)
		{
			cout << "Failed to download " << pEntry->destination << "\n";
			cout << "	Referenced from " << pEntry->origin << "\n";
			cout << "\n";
			return;
		}
		PathData pd;
		GFile::parsePath(pEntry->destination.c_str(), &pd);
		if(_stricmp(pEntry->destination.c_str() + pd.extStart, ".htm") == 0 || _stricmp(pEntry->destination.c_str() + pd.extStart, ".html") == 0)
		{
			GHtmlDoc doc((char*)pFile, size);
			process_html_file(pEntry, doc.document());
		}
		else
		{
			// Just ignore unknown file types
		}
	}

	void crawl(string& sInitialUrl)
	{
		// Process the files
		CrawlEntry* pSeed = new CrawlEntry("Initial URL", sInitialUrl.c_str());
		m_q.push(pSeed);
		while(m_q.size() > 0)
		{
			CrawlEntry* pEntry = m_q.front();
			set<string>::iterator it = m_beenThere.find(pEntry->destination);
			if(it == m_beenThere.end())
			{
				size_t size;
				unsigned char* pFile = NULL;
				try
				{
					pFile = downloadFromWeb(pEntry->destination.c_str(), 20, &size);
					m_beenThere.insert(pEntry->destination);
				}
				catch(std::exception& e)
				{
				}
				process_file(pEntry, pFile, size);
				delete[] pFile;
			}
			m_q.pop();
			delete(pEntry);
		}
	}
};


void findbrokenlinks(GArgReader& args)
{
	// Find URL base
	const char* szInitialUrl = args.pop_string();
	string s = BrokenLinkFinder::expandInitialUrl(szInitialUrl);
	BrokenLinkFinder blf(s);
	blf.crawl(s);
}
*/
