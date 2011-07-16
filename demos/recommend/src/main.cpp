// -------------------------------------------------------------
// The contents of this file may be distributed under the CC0
// license (http://creativecommons.org/publicdomain/zero/1.0/).
// Of course you may also distribute it under a more restrictive
// license, such as any of the OSI-approved licenses
// (http://www.opensource.org/licenses).
// -------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#ifdef WINDOWS
#	include <windows.h>
#	include <process.h>
#	include <direct.h>
#else
#	include <unistd.h>
#endif
#include <GClasses/GDynamicPage.h>
#include <GClasses/GImage.h>
#include <GClasses/GMatrix.h>
#include <GClasses/GDirList.h>
#include <GClasses/GApp.h>
#include <GClasses/GDom.h>
#include <GClasses/GString.h>
#include <GClasses/GHeap.h>
#include <GClasses/GHttp.h>
#include <GClasses/GFile.h>
#include <GClasses/GTime.h>
#include <GClasses/GRand.h>
#include <GClasses/GHashTable.h>
#include <GClasses/sha1.h>
#include <wchar.h>
#include <math.h>
#include <string>
#include <vector>
#include <exception>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <cmath>

using namespace GClasses;
using std::cout;
using std::cerr;
using std::vector;
using std::string;
using std::ostream;
using std::map;
using std::set;
using std::pair;
using std::make_pair;
using std::multimap;

class ViewFile;
class View;
class Account;

#define PERSONALITY_DIMS 6
#define ON_RECOMMEND_REFINE_PERSONALITY_ITERS 1000
#define ON_RATE_TRAINING_ITERS 10000
#define ON_STARTUP_TRAINING_ITERS 500000
#define CANDIDATE_ITEM_POOL_SIZE 350
#define MAX_RECOMMENDATIONS 10
#define MIN_REC_THRESHOLD 0.5f
//#define SHOW_PREDICTIONS


class Item
{
protected:
	std::string m_url;
	std::string m_title;
	std::string m_submitter;
	time_t m_date; // the date this item was submitted
	std::vector<double> m_weights; // used to predict the rating from a user's personality vector

public:
	Item(const char* szUrl, const char* szTitle, const char* szSubmitter, time_t date, GRand* pRand)
	{
		m_url = szUrl;
		m_title = szTitle;
		m_submitter = szSubmitter;
		m_date = date;
		m_weights.resize(PERSONALITY_DIMS + 1);
		for(size_t i = 0; i < PERSONALITY_DIMS + 1; i++)
			m_weights[i] = 0.1 * pRand->normal();
	}

	Item(GDomNode* pNode, GRand* pRand)
	{
		m_url = pNode->field("url")->asString();
		m_title = pNode->field("title")->asString();
		m_submitter = pNode->field("subm")->asString();
		m_date = (time_t)pNode->field("date")->asInt();
		m_weights.resize(PERSONALITY_DIMS + 1);
		GDomNode* pWeights = pNode->field("weights");
		GDomListIterator it(pWeights);
		size_t i;
		for(i = 0; i < (size_t)PERSONALITY_DIMS + 1 && it.current(); i++)
		{
			m_weights[i] = it.current()->asDouble();
			it.advance();
		}
		for( ; i < PERSONALITY_DIMS + 1; i++)
			m_weights[i] = 0.1 * pRand->normal();
	}

	const char* url() { return m_url.c_str(); }
	const char* title() { return m_title.c_str(); }

	GDomNode* toDom(GDom* pDoc)
	{
		GDomNode* pNode = pDoc->newObj();
		pNode->addField(pDoc, "url", pDoc->newString(m_url.c_str()));
		pNode->addField(pDoc, "title", pDoc->newString(m_title.c_str()));
		pNode->addField(pDoc, "subm", pDoc->newString(m_submitter.c_str()));
		pNode->addField(pDoc, "date", pDoc->newInt(m_date));
		GDomNode* pWeights = pNode->addField(pDoc, "weights", pDoc->newList());
		for(size_t i = 0; i < PERSONALITY_DIMS + 1; i++)
			pWeights->addItem(pDoc, pDoc->newDouble(m_weights[i]));
		return pNode;
	}

	double predictRating(const vector<double>& personality) const
	{
		vector<double>::const_iterator itW = m_weights.begin();
		vector<double>::const_iterator itP = personality.begin();

		// Add the bias weight
		double d = *(itW++);

		// Multiply the weight vector by the personality vector
		while(itW != m_weights.end())
			d += *(itW++) * *(itP++);

		// Squash with the logistic function
		return 1.0 / (1.0 + exp(-d));
	}

	// This method adjusts the weights in the opposite direction of the gradient of
	// the squared-error with respect to the weights.
	void trainWeights(double target, double learningRate, const vector<double>& personality)
	{
		GAssert(target >= 0.0 && target <= 1.0);
		double prediction = predictRating(personality);
		double err = learningRate * (target - prediction) * prediction * (1.0 - prediction);
		vector<double>::iterator itW = m_weights.begin();
		vector<double>::const_iterator itP = personality.begin();

		// Update the bias weight
		*(itW++) += err;

		// Update the other weights
		while(itW != m_weights.end())
		{
			*itW = std::max(-16.0, std::min(16.0, *itW + err * *(itP++))); // Clip the weights to prevent weight saturation
			itW++;
		}
#ifdef _DEBUG
		double postpred = predictRating(personality);
		GAssert(std::abs(target - postpred) < std::abs(target - prediction) || std::abs(target - postpred) < 0.001);
#endif
	}

	// This method adjusts the personality vector in the opposite direction of the gradient of
	// the squared-error with respect to the personality vector.
	void trainPersonality(double target, double learningRate, vector<double>& personality) const
	{
		GAssert(target >= 0.0 && target <= 1.0);
		double prediction = predictRating(personality);
		double err = learningRate * (target - prediction) * prediction * (1.0 - prediction);
		vector<double>::const_iterator itW = m_weights.begin();
		vector<double>::iterator itP = personality.begin();

		// Skip the bias weight (since it does not influence the gradient with respect to the personality vector)
		itW++;

		// Update the personality vector
		while(itW != m_weights.end())
		{
			*itP = std::max(0.0, std::min(1.0, *itP + err * *(itW++))); // Clip personality vector to stay within the unit hyper-cube
			itP++;
		}
#ifdef _DEBUG
		double postpred = predictRating(personality);
		GAssert(std::abs(target - postpred) <= std::abs(target - prediction) || std::abs(target - postpred) < 0.001);
#endif
	}
};


// This class contains all of the items that have been submitted to the recommender system
class Topic
{
protected:
	std::string m_descr;
	std::vector<Item*> m_items;

public:
	Topic(const char* szDescr)
	: m_descr(szDescr)
	{
	}

	~Topic()
	{
		for(vector<Item*>::iterator it = m_items.begin(); it != m_items.end(); it++)
			delete(*it);
	}

	size_t size() { return m_items.size(); }

	Item& item(size_t id)
	{
		GAssert(id < m_items.size());
		GAssert(m_items[id] != NULL);
		return *m_items[id];
	}

	const char* descr() { return m_descr.c_str(); }

	bool addItem(const char* szUrl, const char* szTitle, const char* szUsername, time_t date, GRand* pRand)
	{
		// Check for a dupe
		for(vector<Item*>::iterator it = m_items.begin(); it != m_items.end(); it++)
		{
			Item* pItem = *it;
			if(strcmp(pItem->url(), szUrl) == 0)
				return false;
		}

		m_items.push_back(new Item(szUrl, szTitle, szUsername, date, pRand));
		return true;
	}

	GDomNode* toDom(GDom* pDoc)
	{
		GDomNode* pNode = pDoc->newObj();
		pNode->addField(pDoc, "descr", pDoc->newString(m_descr.c_str()));
		GDomNode* pItems = pNode->addField(pDoc, "items", pDoc->newList());
		for(size_t i = 0; i < m_items.size(); i++)
			pItems->addItem(pDoc, m_items[i]->toDom(pDoc));
		return pNode;
	}

	void fromDom(GDomNode* pNode, GRand* pRand)
	{
		m_descr = pNode->field("descr")->asString();
		GDomNode* pItems = pNode->field("items");
		GAssert(m_items.size() == 0);
		GDomListIterator it(pItems);
		m_items.reserve(it.remaining());
		for( ; it.current(); it.advance())
			m_items.push_back(new Item(it.current(), pRand));
	}
};


class Server : public GDynamicPageServer
{
protected:
	std::string m_basePath;
	View* m_pViewRecommender;
	View* m_pViewSubmit;
	View* m_pViewUpdate;
	View* m_pViewAdmin;
	View* m_pViewLogin;
	View* m_pViewNewAccount;

	std::vector<Topic*> m_topics;

	// Typically the accounts would be stored in a database, but since this is
	// just a demo, we'll keep them all in memory for simplicity.
	std::map<std::string,Account*> m_accountsMap;
	std::vector<Account*> m_accountsVec;

public:
	Server(int port, GRand* pRand);
	virtual ~Server();
	void loadState();
	void saveState();
	virtual void handleRequest(const char* szUrl, const char* szParams, int nParamsLen, GDynamicPageSession* pSession, std::ostream& response);
	void getStatePath(char* buf);
	virtual void onEverySixHours();
	virtual void onStateChange();
	virtual void onShutDown();
	bool addItem(size_t topic, const char* szUrl, const char* szTitle, const char* szUsername);
	std::vector<Topic*>& topics() { return m_topics; }
	Account* loadAccount(const char* szUsername, const char* szPasswordHash);
	Account* newAccount(const char* szUsername, const char* szPasswordHash);
	GDomNode* serializeState(GDom* pDoc);
	void deserializeState(GDomNode* pNode);
	void proposeTopic(Account* pAccount, const char* szDescr);
	void newTopic(const char* szDescr);
	Account* randomAccount() { return m_accountsVec[(size_t)prng()->next(m_accountsVec.size())]; }
	void trainModel(size_t topic, size_t iters);
	void trainPersonality(Account* pAccount, size_t iters);
};

class Ratings
{
public:
	std::map<size_t, float> m_map;
	std::vector<pair<size_t, float> > m_vec;

	void addRating(size_t itemId, float rating)
	{
		m_map[itemId] = rating;
		m_vec.push_back(make_pair(itemId, rating));
	}

	void updateRating(size_t itemId, float rating)
	{
		if(m_map.find(itemId) == m_map.end())
			addRating(itemId, rating);
		else
		{
			m_map[itemId] = rating;
			for(vector<pair<size_t, float> >::iterator it = m_vec.begin(); it != m_vec.end(); it++)
			{
				if(it->first == itemId)
				{
					it->second = rating;
					break;
				}
			}
		}
	}
};

class Account : public GDynamicPageSessionExtension
{
protected:
	string m_afterLoginUrl;
	string m_afterLoginParams;
	string m_username;
	string m_passwordHash;
	std::vector<Ratings> m_ratings; // This is the training data for learning the user's personality vector.
	std::vector<double> m_personality; // This vector represents the user with respect to our model. That is, given the user's personality vector, our model should be able to predict the ratings of this user with some accuracy.
	size_t m_currentTopic;

	Account()
	: GDynamicPageSessionExtension(), m_currentTopic(-1)
	{
		m_personality.resize(PERSONALITY_DIMS);
	}

public:
	Account(const char* szUsername, const char* szPasswordHash)
	: GDynamicPageSessionExtension(), m_username(szUsername), m_passwordHash(szPasswordHash), m_currentTopic(-1)
	{
		m_personality.resize(PERSONALITY_DIMS);
		for(size_t i = 0; i < PERSONALITY_DIMS; i++)
			m_personality[i] = 0.5;
	}

	virtual ~Account()
	{
	}

	virtual void onDisown()
	{
	}

	std::vector<Ratings>& ratings() { return m_ratings; }

	void setAfterLoginUrlAndParams(const char* szUrl, const char* szParams)
	{
		m_afterLoginUrl = szUrl;
		m_afterLoginParams = szParams;
	}

	void clearAfterLoginStuff()
	{
		m_afterLoginUrl.clear();
		m_afterLoginParams.clear();
	}

	const char* afterLoginUrl()
	{
		return m_afterLoginUrl.c_str();
	}

	const char* afterLoginParams()
	{
		return m_afterLoginParams.c_str();
	}

	static Account* fromDom(GDomNode* pNode)
	{
		Account* pAccount = new Account();
		pAccount->m_username = pNode->field("username")->asString();
		pAccount->m_passwordHash = pNode->field("password")->asString();

		// Deserialize the personality vector
		GDomNode* pPersonality = pNode->field("pers");
		{
			GDomListIterator it(pPersonality);
			size_t i;
			for(i = 0; i < (size_t)PERSONALITY_DIMS && it.current(); i++)
				pAccount->m_personality[i] = it.current()->asDouble();
			for( ; i < PERSONALITY_DIMS; i++)
				pAccount->m_personality[i] = 0.5;
		}

		// Deserialize the ratings
		GDomNode* pRatings = pNode->field("ratings");
		size_t topic = 0;
		for(GDomListIterator it(pRatings); it.current(); it.advance())
		{
			ptrdiff_t j = (ptrdiff_t)it.current()->asInt();
			if(j < 0)
				topic = (size_t)(-j - 1);
			else
			{
				it.advance();
				if(it.current())
					pAccount->addRating(topic, (size_t)j, (float)it.current()->asDouble());
			}
		}
		return pAccount;
	}

	GDomNode* toDom(GDom* pDoc)
	{
		GDomNode* pAccount = pDoc->newObj();
		pAccount->addField(pDoc, "username", pDoc->newString(m_username.c_str()));
		pAccount->addField(pDoc, "password", pDoc->newString(m_passwordHash.c_str()));

		// Serialize the personality vector
		GDomNode* pPersonality = pAccount->addField(pDoc, "pers", pDoc->newList());
		for(size_t i = 0; i < PERSONALITY_DIMS; i++)
			pPersonality->addItem(pDoc, pDoc->newDouble(m_personality[i]));

		// Serialize the ratings
		size_t count = 0;
		for(vector<Ratings>::iterator i = m_ratings.begin(); i != m_ratings.end(); i++)
		{
			map<size_t, float>& map = i->m_map;
			if(map.size() > 0)
				count += (1 + 2 * map.size());
		}
		GDomNode* pRatings = pAccount->addField(pDoc, "ratings", pDoc->newList());
		size_t j = 0;
		for(vector<Ratings>::iterator i = m_ratings.begin(); i != m_ratings.end(); i++)
		{
			map<size_t, float>& m = i->m_map;
			if(m.size() > 0)
			{
				ptrdiff_t r = -1;
				r -= (j++);
				GAssert(r < 0);
				pRatings->addItem(pDoc, pDoc->newInt(r));
				for(map<size_t,float>::iterator it = m.begin(); it != m.end(); it++)
				{
					pRatings->addItem(pDoc, pDoc->newInt(it->first));
					double clipped = 0.01 * (double)floor(it->second * 100 + 0.5f);
					pRatings->addItem(pDoc, pDoc->newDouble(clipped));
				}
			}
		}

		return pAccount;
	}

	const char* username() { return m_username.c_str(); }
	const char* passwordHash() { return m_passwordHash.c_str(); }
	size_t currentTopic() { return m_currentTopic; }
	void setCurrentTopic(size_t topic) { m_currentTopic = topic; }
	vector<double>& personality() { return m_personality; }

	bool doesHavePassword()
	{
		return m_passwordHash.length() > 0;
	}

	void addRating(size_t topic, size_t itemId, float rating)
	{
		if(topic >= m_ratings.size())
			m_ratings.resize(topic + 1);
		m_ratings[topic].addRating(itemId, rating);
	}

	void updateRating(size_t topic, size_t itemId, float rating)
	{
		if(topic >= m_ratings.size())
			m_ratings.resize(topic + 1);
		m_ratings[topic].updateRating(itemId, rating);
	}
};

Account* getAccount(GDynamicPageSession* pSession)
{
	Account* pAccount = (Account*)pSession->extension();
	if(!pAccount)
	{
		Server* pServer = (Server*)pSession->server();
		char szGenericUsername[32];
		sprintf(szGenericUsername, "_%llu", pSession->id());
		pAccount = pServer->loadAccount(szGenericUsername, NULL);
		if(!pAccount)
		{
			pAccount = pServer->newAccount(szGenericUsername, NULL);
			if(!pAccount)
				ThrowError("Failed to create account");
		}
		pSession->setExtension(pAccount);
	}
	return pAccount;
}

void makeHeader(GDynamicPageSession* pSession, ostream& response)
{
	Account* pAccount = getAccount(pSession);
	response << "<html><head>\n";
	response << "	<title>Generic Recommender System</title>\n";
	response << "	<link rel=\"stylesheet\" type=\"text/css\" href=\"/style/style.css\" />\n";
	response << "</head><body><div id=\"wholepage\">\n";
	response << "\n\n\n\n\n<!-- Header Area --><div id=\"header\">\n";
	response << "	Generic Recommender System\n";
	response << "</div>\n\n\n\n\n<!-- Left Bar Area --><div id=\"sidebar\">\n";
	response << "	<center><img src=\"style/logo.png\"><br>\n";
	if(pAccount)
	{
		response << "Current user: ";
		const char* szUsername = pAccount->username();
		if(*szUsername == '_')
			response << "anonymous";
		else
			response << szUsername;
		response << "<br><br>\n";
	}
	response << "	</center>\n";
	response << "	<a href=\"/rec?nc=" << to_str((size_t)pSession->server()->prng()->next()) << "\">Recommendations</a><br>\n";
	response << "	<a href=\"/login\">Switch user</a><br>\n";
	response << "	<a href=\"/main.hbody\">Overview</a><br>\n";
	response << "	<a href=\"/admin\">Administration</a><br>\n";
	response << "	<br><br><br>\n";
	response << "</div>\n\n\n\n\n<!-- Main Body Area --><div id=\"mainbody\">\n";
}

void makeFooter(GDynamicPageSession* pSession, ostream& response)
{
	response << "</div>\n\n\n\n\n<!-- Footer Area --><div id=\"footer\">\n";
	response << "	The contents of this page are distributed under the <a href=\"http://creativecommons.org/publicdomain/zero/1.0/\">CC0 license</a>. <img src=\"http://i.creativecommons.org/l/zero/1.0/80x15.png\" border=\"0\" alt=\"CC0\" />\n";
	response << "</div>\n\n\n\n\n";
	response << "</div></body></html>\n";
}

class View
{
protected:
	Server* m_pServer;

public:
	View(Server* pServer) : m_pServer(pServer) {}
	virtual ~View() {}

	virtual void makePage(GDynamicPageSession* pSession, ostream& response)
	{
		makeHeader(pSession, response);
		makeBody(pSession, response);
		makeFooter(pSession, response);
	}

	virtual void makeBody(GDynamicPageSession* pSession, ostream& response) = 0;
};


// ------------------------------------------------------

class ViewRecommender : public View
{
public:
	ViewRecommender(Server* pServer) : View(pServer) {}
	virtual ~ViewRecommender() {}

	static void makeUrlSlider(Account* pAccount, ostream& response, Item& item, size_t id)
	{
		float score = 50.0f;
		size_t currentTopic = pAccount->currentTopic();
		if(currentTopic < pAccount->ratings().size())
		{
			map<size_t, float>& m = pAccount->ratings()[currentTopic].m_map;
			map<size_t, float>::iterator it = m.find(id);
			if(it != m.end())
				score = it->second;
		}
		response << "<table cellpadding=0 cellspacing=0><tr><td width=330>\n	";
#ifdef SHOW_PREDICTIONS
		response << "(" << 0.001 * floor(item.predictRating(pAccount->personality()) * 100000) << ") ";
#endif
		response << "<a href=\"" << item.url() << "\">";
		response << item.title() << "</a>\n";
		response << "</td><td>\n";
		response << "	<input type=checkbox name=\"check_slider" << id << "\" id=\"check_slider" << id << "\">\n";
		response << "	<input name=\"slider" << id << "\" id=\"slider" << id << "\" type=\"Text\" size=\"3\">\n";
		response << "</td><td>\n";
		response << "<script language=\"JavaScript\">\n";
		response << "	var A_INIT1 = { 's_checkname': 'check_slider" << id << "', 's_name': 'slider" << id << "', 'n_minValue' : 0, 'n_maxValue' : 100, 'n_value' : " << score << ", 'n_step' : 0.1 }\n";
		response << "	new slider(A_INIT1, A_TPL);\n";
		response << "</script>\n";
		response << "</td></tr></table>\n";
	}

	size_t pickRecommendations(multimap<float, size_t>& outResults, Account* pAccount)
	{
		// Train the personality a little bit
		size_t topic = pAccount->currentTopic();
		if(topic >= pAccount->ratings().size() || topic >= m_pServer->topics().size())
			return 0;
		Topic* pCurrentTopic = m_pServer->topics()[topic];
		size_t itemCount = pCurrentTopic->size();
		m_pServer->trainPersonality(pAccount, ON_RECOMMEND_REFINE_PERSONALITY_ITERS);

		// Pick some recommendations
		set<size_t> usedSet;
		size_t cands = 0;
		map<size_t, float>& m = pAccount->ratings()[topic].m_map;
		for(size_t i = 0; i < CANDIDATE_ITEM_POOL_SIZE; i++) // try a bunch of random items from this topic
		{
			size_t itemId = (size_t)m_pServer->prng()->next(itemCount);
			if(m.find(itemId) == m.end()) // if the user has not yet rated this item...
			{
				if(usedSet.find(itemId) == usedSet.end()) // if we have not tried this item already...
				{
					usedSet.insert(itemId);
					Item& item = pCurrentTopic->item(itemId);
					float f = (float)item.predictRating(pAccount->personality());
					if(f >= MIN_REC_THRESHOLD) // if we predict the user will have a positive opinion of this item...
					{
						cands++;
						outResults.insert(make_pair(f, itemId));
						if(outResults.size() > MAX_RECOMMENDATIONS)
							outResults.erase(outResults.begin()); // drop the weakest item
					}
				}
			}
		}
		return cands;
	}

	virtual void makeBody(GDynamicPageSession* pSession, ostream& response)
	{
		// Get the session extension
		Account* pAccount = getAccount(pSession);
		size_t currentTopic = pAccount->currentTopic();
		if(pSession->paramsLen() > 0)
		{
			// Get the topic
			GHttpParamParser params(pSession->params());
			const char* szTopic = params.find("topic");
			if(szTopic)
			{
				vector<Topic*>& topics = m_pServer->topics();
#ifdef WINDOWS
				size_t i = (size_t)_strtoui64(szTopic, NULL, 10);
#else
				size_t i = (size_t)strtoull(szTopic, NULL, 10);
#endif
				if(i < topics.size())
					pAccount->setCurrentTopic(i);
				else
					pAccount->setCurrentTopic((size_t)-1);
				currentTopic = pAccount->currentTopic();
			}

			// Check for topic proposals
			const char* szProposal = NULL;
			if(pAccount->doesHavePassword())
				szProposal = params.find("proposal");
			if(szProposal)
				m_pServer->proposeTopic(pAccount, szProposal);

			// Do the action
			if(currentTopic < m_pServer->topics().size())
			{
				const char* szAction = params.find("action");
				if(!szAction)
				{
				}
				else if(_stricmp(szAction, "add") == 0)
				{
					const char* szUrl = params.find("url");
					if(!szUrl)
						response << "[invalid params]<br>\n";
					else
					{
						const char* szTitle = params.find("title");
						if(!szTitle)
							response << "[invalid params]<br>\n";
						else
						{
							if(m_pServer->addItem(currentTopic, szUrl, szTitle, pAccount->username()))
							{
								response << "[The new url has been added. Thank you.]<br>\n";
								cout << "added " << szUrl << ", " << szTitle << "\n";
							}
							else
								response << "[That url is already in the system.]<br>\n";
						}
					}
				}
				else if(_stricmp(szAction, "rate") == 0)
				{
					// Make an set of all the checked ids
					set<size_t> checks;
					map<const char*, const char*, strComp>& paramMap = params.map();
					for(map<const char*, const char*, strComp>::iterator it = paramMap.begin(); it != paramMap.end(); it++)
					{
						const char* szName = it->first;
						if(_strnicmp(szName, "check_slider", 12) == 0)
						{
#ifdef WINDOWS
							size_t itemId = (size_t)_strtoui64(szName + 12, NULL, 10);
#else
							size_t itemId = (size_t)strtoull(szName + 12, NULL, 10);
#endif
							checks.insert(itemId);
						}
					}

					// find the corresponding scores for each topic id, and add the rating
					for(map<const char*, const char*, strComp>::iterator it = paramMap.begin(); it != paramMap.end(); it++)
					{
						const char* szName = it->first;
						if(_strnicmp(szName, "slider", 6) == 0)
						{
#ifdef WINDOWS
							size_t itemId = (size_t)_strtoui64(szName + 6, NULL, 10);
#else
							size_t itemId = (size_t)strtoull(szName + 6, NULL, 10);
#endif
							if(itemId < 0 && itemId >= m_pServer->topics().size())
							{
								response << "[url id " << itemId << " out of range.]<br>\n";
								continue;
							}
							set<size_t>::iterator tmp = checks.find(itemId);
							if(tmp != checks.end())
							{
								float score = (float)atof(it->second);
								if(score >= 0.0f && score <= 100.0f)
								{
									pAccount->updateRating(currentTopic, itemId, score);
									response << "[Rating recorded. Thank you.]<br>\n";
								}
								else
									response << "[the rating of " << score << " is out of range.]<br>\n";
							}
						}
					}

					// Do some training
					m_pServer->trainModel(currentTopic, ON_RATE_TRAINING_ITERS);
				}
			}
		}

		if(currentTopic < m_pServer->topics().size()) // if a topic has been selected...
		{
			// Make a set of recommendations
			Topic* pCurrentTopic = m_pServer->topics()[currentTopic];
			
			multimap<float, size_t> recommendations;
			size_t cands = pickRecommendations(recommendations, pAccount);

			// The slider-bar script
			response << "<script language=\"JavaScript\" src=\"style/slider.js\"></script>\n";
			response << "<script language=\"JavaScript\">\n";
			response << "	var A_TPL = { 'b_vertical' : false, 'b_watch': true, 'n_controlWidth': 321, 'n_controlHeight': 22, 'n_sliderWidth': 19, 'n_sliderHeight': 20, 'n_pathLeft' : 1, 'n_pathTop' : 1, 'n_pathLength' : 300, 's_imgControl': 'style/slider_bg.png', 's_imgSlider': 'style/slider_tab.png', 'n_zIndex': 1 }\n";
			response << "</script>\n";

			// Display the topic
			response << "<h2>" << pCurrentTopic->descr() << "</h2>\n";

			// Recommended picks
			response << "<form name=\"formname\" action=\"/rec\" method=\"post\">\n";
			response << "	<input type=\"hidden\" name=\"action\" value=\"rate\" />\n";
			response << "	<input type=\"hidden\" name=\"nc\" value=\"" << to_str((size_t)m_pServer->prng()->next()) << "\" />\n";
			response << "<h3>Recommended Items:</h3>\n";
			set<size_t> usedIds;
			size_t sliderCount = 0;
			if(recommendations.size() > 0)
			{
				for(multimap<float, size_t>::const_iterator it = recommendations.end(); it != recommendations.begin(); )
				{
					it--;
					if(usedIds.find(it->second) == usedIds.end())
					{
						Item& item = pCurrentTopic->item(it->second);
						usedIds.insert(it->second);
						makeUrlSlider(pAccount, response, item, it->second);
						sliderCount++;
					}
				}
			}
			else
			{
				if(pCurrentTopic->size() < 1)
					response << "Sorry, there are currently no items in this topic, so there is nothing for us to recommend.<br><br>\n";
				else if(cands < 1)
					response << "Sorry, we did not find any items that have been rated by others that you have not already rated.<br><br>\n";
				else if(pAccount->ratings().size() <= currentTopic || pAccount->ratings()[currentTopic].m_map.size() < 3)
					response << "Sorry, you have not yet rated enough items for us to predict what sort of things you will like.<br><br>\n";
				else
					response << "Sorry, we did not find any items that we predict you would like.<br><br>\n";
			}

			// Random picks
			if(pCurrentTopic->size() > 0)
			{
				bool gotOne = false;
				for(size_t i = 0; i < 10; i++)
				{
					size_t itemId = (size_t)m_pServer->prng()->next(pCurrentTopic->size());
					if(usedIds.find(itemId) == usedIds.end())
					{
						if(!gotOne)
						{
							response << "<h3>Random Picks:</h3>\n";
							gotOne = true;
						}
						usedIds.insert(itemId);
						Item& item = pCurrentTopic->item(itemId);
						makeUrlSlider(pAccount, response, item, itemId);
						sliderCount++;
					}
				}
			}

			// The update ratings button
			if(sliderCount > 0)
			{
				response << "<br><table><tr><td width=330></td><td>";
				response << "<input type=\"submit\" value=\"Update ratings\">";
				response << "</td></tr><tr><td></td><td>";
				response << "(Only checked items will be updated.)";
				response << "</td></tr></table>\n";
			}
			response << "</form><br><br>\n\n";

			// The choices links at the bottom of the page
			response << "<a href=\"/submit\">Submit a new item</a>";
			response << "&nbsp;&nbsp;&nbsp;&nbsp;";
			response << "<a href=\"/rec?topic=-1&nc=" << to_str((size_t)m_pServer->prng()->next()) << "\">Change topic</a>\n";
			response << "&nbsp;&nbsp;&nbsp;&nbsp;";
			response << "<a href=\"/update\">My ratings</a>\n";
			response << "&nbsp;&nbsp;&nbsp;&nbsp;";
			response << "<a href=\"/rec?nc=" << to_str((size_t)m_pServer->prng()->next()) << "\">" << "Pick again</a>\n";
/*
			response << "Stats:<br>\n";
			response << "Total Number of users: " << m_pServer->accounts().size() << "<br>\n";
			response << "Number of items in this topic: " << pCurrentTopic->size() << "<br>\n";
			std::map<size_t, float>* pMap = currentTopic < pAccount->ratings().size() ? pAccount->ratings()[currentTopic] : NULL;
			response << "Number of items you have rated in this topic: " << (pMap ? pMap->size() : (size_t)0) << "<br>\n<br>\n";
*/
		}
		else
		{
			vector<Topic*>& topics = m_pServer->topics();
			response << "<h3>Choose a topic:</h3>\n";
			if(topics.size() > 0)
			{
				response << "<ul>\n";
				size_t i = 0;
				for(vector<Topic*>::iterator it = topics.begin(); it != topics.end(); it++)
				{
					response << "	<li><a href=\"/rec?topic=" << i << "&nc=" << to_str((size_t)m_pServer->prng()->next()) << "\">" << (*it)->descr() << "</a></li>\n";
					i++;
				}
				response << "</ul><br><br><br>\n";
			}
			else
			{
				response << "There are currently no topics. Please ";
				if(_stricmp(pAccount->username(), "root") != 0)
					response << "ask the administrator to ";
				response << "go to the <a href=\"/admin\">admin</a> page and add at least one topic.<br><br><br>";
			}
			response << "<br><br>\n";

			// Make the form to propose new topics
			if(pAccount->doesHavePassword() && _stricmp(pAccount->username(), "root") != 0)
			{
				response << "<form name=\"propose\" action=\"/rec\" method=\"get\">\n";
				response << "	<h3>Propose a new topic:</h3>\n";
				response << "	<input type=\"text\" name=\"proposal\" size=\"55\"><input type=\"submit\" value=\"Submit\"><br>\n";
				response << "	(Your proposed topic will be added to a log file. Hopefully, someone actually reads the log file.)\n";
				response << "</form><br>\n\n";
			}
		}
	}
};

// ------------------------------------------------------

class ViewSubmit : public View
{
public:
	ViewSubmit(Server* pServer) : View(pServer) {}
	virtual ~ViewSubmit() {}

	virtual void makeBody(GDynamicPageSession* pSession, ostream& response)
	{
		Account* pAccount = getAccount(pSession);
		size_t currentTopic = pAccount->currentTopic();
		if(currentTopic >= m_pServer->topics().size())
		{
			string s = "/rec?nc=";
			s += to_str((size_t)m_pServer->prng()->next());
			m_pServer->redirect(response, s.c_str());
		}
		else
		{
			// Display the topic
			Topic* pCurrentTopic = m_pServer->topics()[currentTopic];
			response << "<h2>" << pCurrentTopic->descr() << "</h2>\n";

			// Make the form to submit a new item
			response << "<h3>Submit a new item to this topic</h3>\n";
			response << "<form name=\"formname\" action=\"/rec\" method=\"post\">\n";
			response << "	<input type=\"hidden\" name=\"action\" value=\"add\" />\n";
			response << "	<input type=\"hidden\" name=\"nc\" value=\"" << to_str((size_t)m_pServer->prng()->next()) << "\" />\n";
			response << "New URL: <input type=\"text\" name=\"url\" size=\"55\"><br>\n";
			response << "<small>Example: http://example.com/ttls.html</small><br>\n";
			response << "Title: <input type=\"text\" name=\"title\" size=\"55\"><br>\n";
			response << "<small>Example: Twinkle Twinkle Little Star, Wolfgang A. Mozart, recorded live, Salzburg, 1785</small><br>\n";
			response << "	<input type=\"submit\" value=\"Submit new item\">";
			response << "</form><br><br>\n\n";

			response << "Please keep the following rules in mind:\n";
			response << "<ul>\n";
			response << "	<li>Each URL that you submit should be for a web page that describes one or more files. (Please do not submit a direct link to the file itself.)</li>\n";
			response << "	<li>The page should clearly describe the file and state its license.</li>\n";
			response << "	<li>The page must have an obvious link to download the file (without requiring login, payment, survey, etc.).</li>\n";
			response << "	<li>The file should be on-topic.</li>\n";
			response << "	<li>The page may have ads, but they may not delay or interfere with an attempt to download the file.</li>\n";
			response << "	<li>Please give a score of 0 to any URL that does not follow the rules. This will ensure that the link is not recommended to others whose preferences are correlated with yours.</li>\n";
			response << "	<li>Items that receive mostly ratings of 0 may (or may not) be automatically removed, and the submitter's IP address may (or may not) be blacklisted.</li>\n";
			response << "</ul>\n";

			// The choices links at the bottom of the page
			response << "<br>\n";
			response << "<a href=\"/rec?topic=-1&nc=" << to_str((size_t)m_pServer->prng()->next()) << "\">Change topic</a>\n";
			response << "&nbsp;&nbsp;&nbsp;&nbsp;";
			response << "<a href=\"/update\">Update old ratings</a>\n";
			response << "&nbsp;&nbsp;&nbsp;&nbsp;";
			response << "<a href=\"/rec?nc=" << to_str((size_t)m_pServer->prng()->next()) << "\">" << "Get recommendations</a>\n";
		}
	}
};


// ------------------------------------------------------

class UpdateComparer
{
public:
	UpdateComparer()
	{
	}

	bool operator() (const pair<size_t,float>& a, const pair<size_t,float>& b) const
	{
		return a.second > b.second;
	}
};

class ViewUpdate : public View
{
public:
	ViewUpdate(Server* pServer) : View(pServer) {}
	virtual ~ViewUpdate() {}

	virtual void makeBody(GDynamicPageSession* pSession, ostream& response)
	{
		Account* pAccount = getAccount(pSession);
		size_t currentTopic = pAccount->currentTopic();
		if(currentTopic >= m_pServer->topics().size())
		{
			string s = "/rec?nc=";
			s += to_str((size_t)m_pServer->prng()->next());
			m_pServer->redirect(response, s.c_str());
		}
		else
		{
			// The slider-bar script
			response << "<script language=\"JavaScript\" src=\"style/slider.js\"></script>\n";
			response << "<script language=\"JavaScript\">\n";
			response << "	var A_TPL = { 'b_vertical' : false, 'b_watch': true, 'n_controlWidth': 321, 'n_controlHeight': 22, 'n_sliderWidth': 19, 'n_sliderHeight': 20, 'n_pathLeft' : 1, 'n_pathTop' : 1, 'n_pathLength' : 300, 's_imgControl': 'style/slider_bg.png', 's_imgSlider': 'style/slider_tab.png', 'n_zIndex': 1 }\n";
			response << "</script>\n";

			// Display the topic
			Topic* pCurrentTopic = m_pServer->topics()[currentTopic];
			response << "<h2>" << pCurrentTopic->descr() << "</h2>\n";

			// Display the items you have rated
			if(pAccount->ratings().size() > currentTopic)
			{
				vector<pair<size_t, float> >& v = pAccount->ratings()[currentTopic].m_vec;
				if(v.size() > 0)
				{
					response << "<h3>Your ratings</h3>\n";
					response << "<form name=\"formname\" action=\"/rec\" method=\"post\">\n";
					response << "	<input type=\"hidden\" name=\"action\" value=\"rate\" />\n";
					response << "	<input type=\"hidden\" name=\"nc\" value=\"" << to_str((size_t)m_pServer->prng()->next()) << "\" />\n";
					UpdateComparer comparer;
					std::sort(v.begin(), v.end(), comparer);
					for(vector<pair<size_t, float> >::iterator it = v.begin(); it != v.end(); it++)
					{
						Item& item = pCurrentTopic->item(it->first);
						ViewRecommender::makeUrlSlider(pAccount, response, item, it->first);
					}
					response << "<br><table><tr><td width=330></td><td>";
					response << "<input type=\"submit\" value=\"Update ratings\">";
					response << "</td></tr><tr><td></td><td>";
					response << "(Only checked items will be updated.)";
					response << "</td></tr></table>\n";
					response << "</form><br><br>\n\n";
				}
				else
					response << "You have not yet rated anything in this topic<br><br>\n";
			}
			else
				response << "You have not yet rated anything in this topic<br><br>\n";

			// The choices links at the bottom of the page
			response << "<a href=\"/submit\">Submit a new item</a>";
			response << "&nbsp;&nbsp;&nbsp;&nbsp;";
			response << "<a href=\"/rec?topic=-1&nc=" << to_str((size_t)m_pServer->prng()->next()) << "\">Change topic</a>\n";
			response << "&nbsp;&nbsp;&nbsp;&nbsp;";
			response << "<a href=\"/rec?nc=" << to_str((size_t)m_pServer->prng()->next()) << "\">" << "Get recommendations</a>\n";
		}
	}
};


// ------------------------------------------------------

class ViewAdmin : public View
{
public:
	ViewAdmin(Server* pServer) : View(pServer) {}
	virtual ~ViewAdmin() {}

	virtual void makeBody(GDynamicPageSession* pSession, ostream& response)
	{
		Account* pAccount = getAccount(pSession);
		if(_stricmp(pAccount->username(), "root") != 0)
		{
			response << "Sorry, only the user with username <i>root</i> can access the admin page (and if you are not the administrator, then this username has probably already been taken).<br><br><br><br><br><br><br><br>\n";
			return;
		}
		if(pSession->paramsLen() > 0)
		{
			GHttpParamParser params(pSession->params());
			const char* szAction = params.find("action");
			if(szAction)
			{
				// Do the action
				if(_stricmp(szAction, "shutdown") == 0)
				{
					cout << "root has told the server to shut down.\n";
					cout.flush();
					cerr.flush();
					m_pServer->shutDown();
				}
				else if(_stricmp(szAction, "newtopic") == 0)
				{
					const char* szDescr = params.find("descr");
					if(szDescr && strlen(szDescr) > 0)
					{
						m_pServer->newTopic(szDescr);
						response << "[The new topic has been added]<br>\n";
					}
					else
						response << "[You must enter a topic description]<br>\n";
				}
				else
					response << "[Unknown action! No action taken]<br>\n";
			}
		}

		response << "<h2>Admin controls</h2>\n\n";

		// Form to shut down the server
		response << "<form name=\"shutdownform\" action=\"/admin\" method=\"get\">\n";
		response << "	Shut down the daemon:<br>\n";
		response << "	<input type=\"hidden\" name=\"action\" value=\"shutdown\" />\n";
		response << "	<input type=\"submit\" value=\"Shut down now\">\n";
		response << "</form><br><br>\n\n";

		// Form to add a new topic
		response << "<form name=\"shutdownform\" action=\"/admin\" method=\"get\">\n";
		response << "	Add a new topic:<br>\n";
		response << "	<input type=\"hidden\" name=\"action\" value=\"newtopic\" />\n";
		response << "	<input type=\"text\" name=\"descr\" size=\"55\"><input type=\"submit\" value=\"Add\"><br>\n";
		response << "</form><br><br>\n\n";
	}
};

// ------------------------------------------------------

class ViewLogin : public View
{
public:
	ViewLogin(Server* pServer) : View(pServer) {}
	virtual ~ViewLogin() {}

	virtual void makeBody(GDynamicPageSession* pSession, ostream& response)
	{
		Account* pAccount = getAccount(pSession);
		if(pSession->paramsLen() >= 0)
		{
			// See if the user wants to log out
			GHttpParamParser params(pSession->params());
			const char* szAction = params.find("action");
			if(szAction)
			{
				if(_stricmp(szAction, "logout") == 0)
				{
					string s = "/rec?nc=";
					s += to_str((size_t)m_pServer->prng()->next());
					m_pServer->redirect(response, s.c_str());
					pSession->setExtension(NULL); // disconnect the account from this session
					return;
				}
				else
					response << "Unrecognized action: " << szAction << "<br><br>\n\n";
			}

			// Check the password
			const char* szUsername = params.find("username");
			const char* szPasswordHash = params.find("password");
			if(szUsername)
			{
				Account* pNewAccount = m_pServer->loadAccount(szUsername, szPasswordHash);
				if(pNewAccount)
				{
					string s;
					if(pAccount)
					{
						if(strlen(pAccount->afterLoginUrl()) > 0)
							s = pAccount->afterLoginUrl();
						if(strlen(pAccount->afterLoginParams()) > 0)
						{
							s += "?";
							s += pAccount->afterLoginParams();
						}
						if(s.length() < 1)
						{
							s = "/rec?nc=";
							s += to_str((size_t)m_pServer->prng()->next());
						}
					}
					else
					{
						s = "/rec?nc=";
						s += to_str((size_t)m_pServer->prng()->next());
					}

					// Log in with the new account
					pSession->setExtension(pNewAccount);
					m_pServer->redirect(response, s.c_str());
				}
				else
					response << "<big><big>Incorrect Password! Please try again</big></big><br><br>\n";
			}
		}

		response << "<br><br>\n";
		response << "<SCRIPT language=\"JavaScript\" src=\"/sha1.js\" type=\"text/javascript\">\n</SCRIPT>\n";
		if(pAccount)
		{
			response << "Your current username is: ";
			const char* szUsername = pAccount->username();
			if(*szUsername == '_')
				response << "anonymous";
			else
				response << szUsername;
			response << ".<br>\n";
			if(pAccount->doesHavePassword())
				response << "Click here to <a href=\"/login?action=logout\">log out</a>.<br>\n";
			response << "Log in as a different user:<br>\n";
		}
		else
			response << "Please enter credentials to log in:<br>\n";
		response << "<form name=\"loginform\" action=\"/login\" method=\"get\" onsubmit=\"return HashPassword('";
		response << m_pServer->passwordSalt();
		response << "')\">\n";
		response << "	Username:<input type=\"text\" name=\"username\" ><br>\n";
		response << "	Password:<input type=\"password\" name=\"password\" ><br>\n";
		response << "	<input type=\"submit\" value=\"Log In\">\n";
		response << "</form><br><br><br><br>\n\n";

		response << "or click here to <a href=\"/newaccount\">create a new account</a><br>\n";
		response << "(It's a good idea to have multiple accounts for multiple purposes.)<br>\n";
	}
};

// ------------------------------------------------------

class ViewNewAccount : public View
{
public:
	ViewNewAccount(Server* pServer) : View(pServer) {}
	virtual ~ViewNewAccount() {}

	virtual void makeBody(GDynamicPageSession* pSession, ostream& response);

protected:
	void GetCaptchaText(char* szOut, const char* szID);
	void MakeCaptcha(const char* szID, ostream& response);
	void SendCaptcha(const char* szText, ostream& response);
	bool CheckCaptchaText(const char* a, const char* b);
};

bool ViewNewAccount::CheckCaptchaText(const char* pA, const char* pB)
{
	while(*pA != '\0')
	{
		char a = (*pA | 32);
		char b = (*pB | 32);

		// Close enough
		if(a == 'O') a = '0';	if(b == 'O') b = '0';
		if(a == 'S') a = '5';	if(b == 'S') b = '5';
		if(a == 'G') a = '6';	if(b == 'G') b = '6';

		if(a != b)
			return false;
		pA++;
		pB++;
	}
	if(*pB != '\0')
		return false;
	return true;
}

void ViewNewAccount::GetCaptchaText(char* szOut, const char* szID)
{
	unsigned char digest[20];
	SHA_CTX ctx;
	memset(&ctx, '\0', sizeof(SHA_CTX));
	SHA1_Init(&ctx);
	const char* daemonSalt = m_pServer->daemonSalt();
	SHA1_Update(&ctx, (unsigned char*)szID, strlen(szID));
	SHA1_Update(&ctx, (unsigned char*)daemonSalt, strlen(daemonSalt));
	SHA1_Final(digest, &ctx);
	int i;
	for(i = 0; i < 6; i++)
	{
		szOut[i] = digest[i] % 36;
		if(szOut[i] >= 10)
			szOut[i] += 'A' - 10;
		else
			szOut[i] += '0';
	}
	szOut[i] = '\0';
}

#ifndef WINDOWS
void DeleteFile(const char* szFilename)
{
	char szBuf[64];
	strcpy(szBuf, "rm ");
	strcat(szBuf, szFilename);
	strcat(szBuf, " &");
	if(system(szBuf) == -1)
		ThrowError("Failed to delete file");
}
#endif

void ViewNewAccount::MakeCaptcha(const char* szID, ostream& response)
{
	char szText[32];
	GetCaptchaText(szText, szID);
	std::ostringstream& r = reinterpret_cast<std::ostringstream&>(response);
	r.str("");
	r.clear();

	// Make the filename
	char szTemp[512];
	GFile::tempFilename(szTemp);

	// Make the captcha
	GImage image;
	image.captcha(szText, m_pServer->prng());
	image.savePng(szTemp);
	m_pServer->sendFile("image/png", szTemp, response);
	DeleteFile(szTemp);
}

/*virtual*/ void ViewNewAccount::makeBody(GDynamicPageSession* pSession, ostream& response)
{
	if(_strnicmp(pSession->url(), "/captcha", 8) == 0)
	{
		char szID[9];
		memcpy(szID, pSession->url() + 8, 8);
		szID[8] = '\0';
		MakeCaptcha(szID, response);
		return;
	}

	const char* szUsername = "";
	const char* szPassword = "";
	const char* szPWAgain = "";
	const char* szCaptchaId = "";
	const char* szCaptcha = "";
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
		szCaptchaId = params.find("captchaid");

		// Check the parameters
		if(!szUsername || strlen(szUsername) < 1)
			szError = "The username is not valid";
		if(!szPassword || strlen(szPassword) < 1)
			szError = "The password is not valid";
		if(!szPWAgain || strcmp(szPassword, szPWAgain) != 0)
			szError = "The passwords don't match";
		if(!szCaptchaId)
			szError = "Expected a hidden captcha id";
		char szExpectedCaptchaText[32];
		GetCaptchaText(szExpectedCaptchaText, szCaptchaId);
		szCaptcha = params.find("captcha");
		if(!szCaptcha || strlen(szCaptcha) < 1)
			szError = "You must enter the captcha text as shown in the image";
		if(!CheckCaptchaText(szCaptcha, szExpectedCaptchaText))
			szError = "The captcha text is incorrect";
		if(!szError)
		{
			// Create the account
			Account* pAccount = m_pServer->newAccount(szUsername, szPassword);
			if(!pAccount)
				szError = "That username is already taken.";
			else
			{
				m_pServer->saveState();
				response << "<big>An account has been successfully created.</big><br><br> Click here to <a href=\"/login\">log in</a><br>\n";
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

	// Make a captcha ID
	char szCaptchaID[32];
	int i;
	for(i = 0; i < 8; i++)
		szCaptchaID[i] = (char)m_pServer->prng()->next(26) + 'a';
	szCaptchaID[i] = '\0';

	response << "<br><center><table width=\"400\" border=\"0\" cellpadding=\"0\" cellspacing=\"0\"><tr><td>\n";
	response << "<SCRIPT language=\"JavaScript\" src=\"/sha1.js\" type=\"text/javascript\">\n</SCRIPT>\n";
	response << "	<big><big><b>Create a new account</b></big></big><br><br>\n";
	response << "	<form name=\"newaccountform\" action=\"/newaccount\" method=\"post\" onsubmit=\"return HashNewAccount('";
	response << m_pServer->passwordSalt();
	response << "')\">\n";
	response << "		<input type=\"hidden\" name=\"action\" value=\"newaccount\" />\n";
	response << "		<input type=\"hidden\" name=\"captchaid\" value=\"";
	response << szCaptchaID;
	response << "\" />\n";
	response << "		Username: <input type=\"text\" size=\"15\" name=\"username\" value=\"";
	response << szUsername;
	response << "\"><br><br>\n";
	response << "		Password: <input type=\"password\" name=\"password\" size=\"15\" value=\"";
	response << szPassword;
	response << "\"><br>\n";
	response << "		PW Again: <input type=\"password\" name=\"pwagain\" size=\"15\" value=\"";
	response << szPWAgain;
	response << "\"><br><br>\n";
	response << "		<img src=\"captcha";
	response << szCaptchaID;
	response << ".png\"><br>\n";
	response << "		If you can't read the captcha, click here to <a href=\"/newaccount\">get another one</a><br>\n";
	response << "		Captcha: <input type=\"text\" size=\"15\" name=\"captcha\"><br>\n";
	response << "		<br><input type=\"submit\" value=\"Submit\">\n";
	response << "	</form><br>\n\n";
	response << "</tr></td></table></center>\n";
}

// ------------------------------------------------------

Server::Server(int port, GRand* pRand) : GDynamicPageServer(port, pRand)
{
	char buf[300];
	GTime::asciiTime(buf, 256, false);
	cout << "Server starting at: " << buf << "\n";
	GApp::appPath(buf, 256, true);
	strcat(buf, "web/");
	GFile::condensePath(buf);
	m_basePath = buf;
	m_pViewRecommender = new ViewRecommender(this);
	m_pViewSubmit = new ViewSubmit(this);
	m_pViewUpdate = new ViewUpdate(this);
	m_pViewAdmin = new ViewAdmin(this);
	m_pViewLogin = new ViewLogin(this);
	m_pViewNewAccount = new ViewNewAccount(this);
	loadState();
}

// virtual
Server::~Server()
{
	saveState();
	delete(m_pViewRecommender);
	delete(m_pViewSubmit);
	delete(m_pViewUpdate);
	delete(m_pViewAdmin);
	delete(m_pViewLogin);
	delete(m_pViewNewAccount);

	// Delete all the accounts
	flushSessions(); // ensure that there are no sessions referencing the accounts
	for(vector<Account*>::iterator it = m_accountsVec.begin(); it != m_accountsVec.end(); it++)
		delete(*it);

	// Delete all the topics
	for(vector<Topic*>::iterator it = m_topics.begin(); it != m_topics.end(); it++)
		delete(*it);
}

void Server::loadState()
{
	char statePath[300];
	getStatePath(statePath);
	if(GFile::doesFileExist(statePath))
	{
		GDom doc;
		doc.loadJson(statePath);
		deserializeState(doc.root());
		cout << "State loaded from: " << statePath << "\n";

		// Do some training to make sure the model is in good shape
		cout << "doing some training...\n";
		for(size_t i = 0; i < m_topics.size(); i++)
			trainModel(i, ON_STARTUP_TRAINING_ITERS);
		cout << "done.\n";
	}
	else
		cout << "No state file (" << statePath << ") found. Creating new state.\n";
}

void Server::saveState()
{
	GDom doc;
	doc.setRoot(serializeState(&doc));
	char szStoragePath[300];
	getStatePath(szStoragePath);
	doc.saveJson(szStoragePath);
	char szTime[256];
	GTime::asciiTime(szTime, 256, false);
	cout << "Server state saved at: " << szTime << "\n";
}

// virtual
void Server::handleRequest(const char* szUrl, const char* szParams, int nParamsLen, GDynamicPageSession* pSession, ostream& response)
{
	View* pView = NULL;
	if(strcmp(szUrl, "/") == 0)
		szUrl = "/rec";
	else if(strcmp(szUrl, "/favicon.ico") == 0)
		return;
	else if(strncmp(szUrl, "/login", 6) == 0)
		pView = m_pViewLogin;
	else if(strncmp(szUrl, "/rec", 4) == 0)
		pView = m_pViewRecommender;
	else if(strncmp(szUrl, "/submit", 7) == 0)
		pView = m_pViewSubmit;
	else if(strncmp(szUrl, "/update", 7) == 0)
		pView = m_pViewUpdate;
	else if(strncmp(szUrl, "/admin", 6) == 0)
		pView = m_pViewAdmin;
	else if(strncmp(szUrl, "/newaccount", 11) == 0)
		pView = m_pViewNewAccount;
	else if(strncmp(szUrl, "/captcha", 7) == 0)
		pView = m_pViewNewAccount;
	if(pView)
		pView->makePage(pSession, response);
	else
	{
		size_t len = strlen(szUrl);
		if(len > 6 && strcmp(szUrl + len - 6, ".hbody") == 0)
		{
			makeHeader(pSession, response);
			sendFileSafe(m_basePath.c_str(), szUrl + 1, response);
			makeFooter(pSession, response);
		}
		else
			sendFileSafe(m_basePath.c_str(), szUrl + 1, response);
	}
}

bool Server::addItem(size_t topic, const char* szUrl, const char* szTitle, const char* szUsername)
{
	if(topic < m_topics.size())
		return m_topics[topic]->addItem(szUrl, szTitle, szUsername, time(NULL), prng());
	else
		return false;
}

void getLocalStorageFolder(char* buf)
{
	if(!GFile::localStorageDirectory(buf))
		ThrowError("Failed to find local storage folder");
	strcat(buf, "/.recommend/");
	GFile::makeDir(buf);
	if(!GFile::doesDirExist(buf))
		ThrowError("Failed to create folder in storage area");
}

void Server::getStatePath(char* buf)
{
	getLocalStorageFolder(buf);
	strcat(buf, "state.json");
}

// virtual
void Server::onEverySixHours()
{
	saveState();
	fflush(stdout);
}

// virtual
void Server::onStateChange()
{
}

// virtual
void Server::onShutDown()
{
}


Account* Server::loadAccount(const char* szUsername, const char* szPasswordHash)
{
	if(!szPasswordHash)
		szPasswordHash = "";

	// Find the account
	map<string,Account*>::iterator it = m_accountsMap.find(szUsername);
	if(it == m_accountsMap.end())
		return NULL;
	Account* pAccount = it->second;

	// Check the password hash
	if(_stricmp(pAccount->passwordHash(), szPasswordHash) != 0)
		return NULL;
	return pAccount;
}

Account* Server::newAccount(const char* szUsername, const char* szPasswordHash)
{
	if(!szPasswordHash)
		szPasswordHash = "";

	// See if that username already exists
	map<string,Account*>::iterator it = m_accountsMap.find(szUsername);
	if(it != m_accountsMap.end())
		return NULL;

	// Make the account
	Account* pAccount = new Account(szUsername, szPasswordHash);
	m_accountsVec.push_back(pAccount);
	m_accountsMap.insert(make_pair(string(szUsername), pAccount));
	return pAccount;
}

void Server::proposeTopic(Account* pAccount, const char* szDescr)
{
	cout << "The following new topic was proposed by " << pAccount->username() << "\n";
	cout << "	" << szDescr << "\n";
}

void Server::newTopic(const char* szDescr)
{
	m_topics.push_back(new Topic(szDescr));
}

void Server::trainModel(size_t topic, size_t iters)
{
	// Do some training
	Topic* pCurrentTopic = m_topics[topic];
	for(size_t i = 0; i < iters; i++)
	{
		Account* pSomeAccount = randomAccount();
		if(pSomeAccount->ratings().size() > topic)
		{
			std::vector<pair<size_t, float> >& v = pSomeAccount->ratings()[topic].m_vec;
			if(v.size() > 0)
			{
				size_t index = (size_t)prng()->next(v.size());
				Item& item = pCurrentTopic->item(v[index].first);
				double target = 0.01 * (double)v[index].second;
				GAssert(target >= 0.0 && target <= 1.0);
				item.trainWeights(target, 0.1, pSomeAccount->personality());
				item.trainPersonality(target, 0.1, pSomeAccount->personality());
			}
		}
	}
}

void Server::trainPersonality(Account* pAccount, size_t iters)
{
	// Train the personality a little bit
	size_t topic = pAccount->currentTopic();
	if(topic >= pAccount->ratings().size() || topic >= m_topics.size())
		return;
	Topic* pCurrentTopic = m_topics[topic];
	std::vector<pair<size_t, float> >& v = pAccount->ratings()[topic].m_vec;
	if(v.size() > 0)
	{
		for(size_t iters = 0; iters < iters; iters++)
		{
			size_t index = (size_t)prng()->next(v.size());
			Item& item = pCurrentTopic->item(v[index].first);
			item.trainPersonality((double)v[index].second, 0.1, pAccount->personality());
		}
	}
}

GDomNode* Server::serializeState(GDom* pDoc)
{
	GDomNode* pNode = pDoc->newObj();

	// Captcha salt
	pNode->addField(pDoc, "daemonSalt", pDoc->newString(daemonSalt()));

	// Save the topcs
	GDomNode* pTopics = pNode->addField(pDoc, "topics", pDoc->newList());
	for(size_t i = 0; i < m_topics.size(); i++)
		pTopics->addItem(pDoc, m_topics[i]->toDom(pDoc));

	// Save the accounts
	GDomNode* pAccounts = pNode->addField(pDoc, "accounts", pDoc->newList());
	for(map<string,Account*>::iterator it = m_accountsMap.begin(); it != m_accountsMap.end(); it++)
	{
		Account* pAccount = it->second;
		pAccounts->addItem(pDoc, pAccount->toDom(pDoc));
	}

	return pNode;
}

void Server::deserializeState(GDomNode* pNode)
{
	// Captcha salt
	const char* daemonSalt = pNode->fieldIfExists("daemonSalt")->asString();
	if(daemonSalt)
		setDaemonSalt(daemonSalt);

	// Load the topics
	GAssert(m_topics.size() == 0);
	GDomNode* pTopics = pNode->field("topics");
	for(GDomListIterator it(pTopics); it.current(); it.advance())
	{
		Topic* pTopic = new Topic("");
		m_topics.push_back(pTopic);
		pTopic->fromDom(it.current(), prng());
	}

	// Load the accounts
	GAssert(m_accountsVec.size() == 0 && m_accountsMap.size() == 0);
	GDomNode* pAccounts = pNode->field("accounts");
	for(GDomListIterator it(pAccounts); it.current(); it.advance())
	{
		Account* pAccount = Account::fromDom(it.current());
		m_accountsVec.push_back(pAccount);
		m_accountsMap.insert(make_pair(string(pAccount->username()), pAccount));
	}
}







void LaunchBrowser(const char* szAddress, GRand* pRand)
{
	string s = szAddress;
	s += "/rec?nc=";
	s += to_str((size_t)pRand->next());
	if(!GApp::openUrlInBrowser(s.c_str()))
	{
		cout << "Failed to open the URL: " << s.c_str() << "\nPlease open this URL manually.\n";
		cout.flush();
	}
}

void redirectStandardStreams(const char* pPath)
{
	string s1(pPath);
	s1 += "stdout.log";
	if(!freopen(s1.c_str(), "a", stdout))
	{
		cout << "Error redirecting stdout\n";
		cerr << "Error redirecting stdout\n";
		ThrowError("Error redirecting stdout");
	}
	string s2(pPath);
	s2 += "stderr.log";
	if(!freopen(s2.c_str(), "a", stderr))
	{
		cout << "Error redirecting stderr\n";
		cerr << "Error redirecting stderr\n";
		ThrowError("Error redirecting stderr");
	}
}

void doit(void* pArg)
{
	{
#ifdef _DEBUG
		int port = 8987;
#else
		int port = 8988;
#endif
		size_t seed = getpid() * (size_t)time(NULL);
		GRand prng(seed);
		Server server(port, &prng);
		LaunchBrowser(server.myAddress(), &prng);
		server.go();
	}
	cout << "Goodbye.\n";
}

void doItAsDaemon()
{
	char path[300];
	getLocalStorageFolder(path);
	string s1 = path;
	s1 += "stdout.log";
	string s2 = path;
	s2 += "stderr.log";
	GApp::launchDaemon(doit, path, s1.c_str(), s2.c_str());
	cout << "Daemon running.\n	stdout >> " << s1.c_str() << "\n	stderr >> " << s2.c_str() << "\n";
}

int main(int nArgs, char* pArgs[])
{
	int nRet = 1;
	try
	{
		doit(NULL);
		//doItAsDaemon();
	}
	catch(std::exception& e)
	{
		cerr << e.what() << "\n";
	}
	return nRet;
}
