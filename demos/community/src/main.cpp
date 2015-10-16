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
#include <GClasses/GPlot.h>
#include <GClasses/GThread.h>
#include <GClasses/GRand.h>
#include <GClasses/GHashTable.h>
#include <GClasses/sha1.h>
#include <GClasses/GVec.h>
#include <GClasses/GHolders.h>
#include <GClasses/GBitTable.h>
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

class View;
class Account;
class ViewStats;

#define PERSONALITY_DIMS 3 // (one of these is used for the bias)
#define ON_RATE_TRAINING_ITERS 5000
#define ON_STARTUP_TRAINING_ITERS 250000
#define LEARNING_RATE 0.01


class Item
{
protected:
	std::string m_title;
	std::string m_submitter;
	time_t m_date; // the date this item was submitted
	std::vector<double> m_weights; // used to predict the rating from a user's personality vector

public:
	Item(const char* szTitle, const char* szSubmitter, time_t date, GRand* pRand)
	{
		m_title = szTitle;
		m_submitter = szSubmitter;
		m_date = date;
		m_weights.resize(PERSONALITY_DIMS);
		for(size_t i = 0; i < PERSONALITY_DIMS; i++)
			m_weights[i] = 0.01 * pRand->normal();
	}

	Item(GDomNode* pNode, GRand* pRand)
	{
		m_title = pNode->field("title")->asString();
		m_submitter = pNode->field("subm")->asString();
		m_date = (time_t)pNode->field("date")->asInt();
		m_weights.resize(PERSONALITY_DIMS);
		GDomNode* pWeights = pNode->field("weights");
		GDomListIterator it(pWeights);
		size_t i;
		for(i = 0; i < (size_t)PERSONALITY_DIMS && it.current(); i++)
		{
			m_weights[i] = it.current()->asDouble();
			it.advance();
		}
		for( ; i < PERSONALITY_DIMS; i++)
			m_weights[i] = LEARNING_RATE * pRand->normal();
	}

	const char* title() { return m_title.c_str(); }
	std::vector<double>& weights() { return m_weights; }

	GDomNode* toDom(GDom* pDoc)
	{
		GDomNode* pNode = pDoc->newObj();
		pNode->addField(pDoc, "title", pDoc->newString(m_title.c_str()));
		pNode->addField(pDoc, "subm", pDoc->newString(m_submitter.c_str()));
		pNode->addField(pDoc, "date", pDoc->newInt(m_date));
		GDomNode* pWeights = pNode->addField(pDoc, "weights", pDoc->newList());
		for(size_t i = 0; i < PERSONALITY_DIMS; i++)
			pWeights->addItem(pDoc, pDoc->newDouble(m_weights[i]));
		return pNode;
	}

	double predictRating(const vector<double>& personality) const
	{
		vector<double>::const_iterator itW = m_weights.begin();
		vector<double>::const_iterator itP = personality.begin();

		// Add the bias weights
		double d = *(itW++) + *(itP++);

		// Multiply the weight vector by the personality vector
		while(itW != m_weights.end())
			d += *(itW++) * *(itP++);

		// Squash with the logistic function
		//return 1.0 / (1.0 + exp(-d));
		return d;
	}

	// This method adjusts the weights in the opposite direction of the gradient of
	// the squared-error with respect to the weights.
	void trainWeights(double target, double learningRate, const vector<double>& personality)
	{
		GAssert(target >= 0.0 && target <= 1.0);
		double prediction = predictRating(personality);
		double err = learningRate * (target - prediction);// * prediction * (1.0 - prediction);
		vector<double>::iterator itW = m_weights.begin();
		vector<double>::const_iterator itP = personality.begin();

		// Update the bias weight
		itP++;
		*(itW++) += err;

		// Update the other weights
		while(itW != m_weights.end())
		{
			//*itW = std::max(-8.0, std::min(8.0, *itW + err * *(itP++)));
			*itW += err * *(itP++);
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
		double err = learningRate * (target - prediction);// * prediction * (1.0 - prediction);
		vector<double>::const_iterator itW = m_weights.begin();
		vector<double>::iterator itP = personality.begin();

		// Update the bias
		itW++;
		*(itP) += err;
		itP++;

		// Update the personality vector
		while(itW != m_weights.end())
		{
			//*itP = std::max(-1.0, std::min(1.0, *itP + err * *itW));
			*itP += err * *itW;
			itW++;
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

	void addItem(const char* szTitle, const char* szUsername, time_t date, GRand* pRand)
	{
		m_items.push_back(new Item(szTitle, szUsername, date, pRand));
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

	void deleteItemAndSwapInLast(size_t itemId)
	{
		delete(m_items[itemId]);
		m_items[itemId] = m_items[m_items.size() - 1];
		m_items.pop_back();
	}
};

class Server : public GDynamicPageServer
{
protected:
	std::vector<Topic*> m_topics;
	std::map<std::string,Account*> m_accountsMap;
	std::vector<Account*> m_accountsVec;

public:
	std::string m_basePath;

	Server(int port, GRand* pRand);
	virtual ~Server();
	void loadState();
	void saveState();
	void getStatePath(char* buf);
	virtual void onEverySixHours();
	virtual void onStateChange();
	virtual void onShutDown();
	void addItem(size_t topic, const char* szTitle, const char* szUsername);
	std::vector<Topic*>& topics() { return m_topics; }
	Account* loadAccount(const char* szUsername, const char* szPasswordHash);
	Account* newAccount(const char* szUsername, const char* szPasswordHash);
	void deleteAccount(Account* pAccount);
	GDomNode* serializeState(GDom* pDoc);
	void deserializeState(GDomNode* pNode);
	void proposeTopic(Account* pAccount, const char* szDescr);
	void newTopic(const char* szDescr);
	Account* randomAccount() { return m_accountsVec[(size_t)prng()->next(m_accountsVec.size())]; }
	Account* findAccount(const char* szName);
	void trainModel(size_t topic, size_t iters);
	void trainPersonality(Account* pAccount, size_t iters);
	std::vector<Account*>& accounts() { return m_accountsVec; }

	virtual GDynamicPageConnection* makeConnection(SOCKET sock);
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

	void withdrawRating(size_t itemId)
	{
		if(m_map.find(itemId) != m_map.end())
		{
			m_map.erase(itemId);
			for(vector<pair<size_t, float> >::iterator it = m_vec.begin(); it != m_vec.end(); it++)
			{
				if(it->first == itemId)
				{
					size_t index = it - m_vec.begin();
					std::swap(m_vec[index], m_vec[m_vec.size() - 1]);
					m_vec.erase(m_vec.end() - 1);
					break;
				}
			}
		}
	}

	void swapItems(size_t a, size_t b)
	{
		std::map<size_t, float>::iterator itA = m_map.find(a);
		std::map<size_t, float>::iterator itB = m_map.find(b);
		float ratingA = 0.0f;
		float ratingB = 0.0f;
		bool gotA = false;
		bool gotB = false;
		if(itA != m_map.end())
		{
			gotA = true;
			ratingA = itA->second;
		}
		if(itB != m_map.end())
		{
			gotB = true;
			ratingB = itB->second;
		}
		if(gotA)
			withdrawRating(a);
		if(gotB)
			withdrawRating(b);
		if(gotA)
			addRating(b, ratingA);
		if(gotB)
			addRating(a, ratingB);
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
	Account(const char* szUsername, const char* szPasswordHash, GRand& rand)
	: GDynamicPageSessionExtension(), m_username(szUsername), m_passwordHash(szPasswordHash), m_currentTopic(-1)
	{
		m_personality.resize(PERSONALITY_DIMS);
		for(size_t i = 0; i < PERSONALITY_DIMS; i++)
			m_personality[i] = LEARNING_RATE * rand.normal();
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

	static Account* fromDom(GDomNode* pNode, GRand& rand)
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
			{
				pAccount->m_personality[i] = std::max(-1.0, std::min(1.0, it.current()->asDouble()));
				it.advance();
			}
			for( ; i < PERSONALITY_DIMS; i++)
				pAccount->m_personality[i] = 0.02 * rand.normal();
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
					double clipped = 0.001 * (double)floor(it->second * 1000 + 0.5f);
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
		GAssert(rating >= 0.0f && rating <= 1.0f);
		if(topic >= m_ratings.size())
			m_ratings.resize(topic + 1);
		m_ratings[topic].addRating(itemId, rating);
	}

	void updateRating(size_t topic, size_t itemId, float rating)
	{
		GAssert(rating >= -1.0f && rating <= 1.0f);
		if(topic >= m_ratings.size())
			m_ratings.resize(topic + 1);
		m_ratings[topic].updateRating(itemId, rating);
	}

	void withdrawRating(size_t topic, size_t itemId)
	{
		if(topic < m_ratings.size())
			m_ratings[topic].withdrawRating(itemId);
	}

	void swapItems(size_t topic, size_t a, size_t b)
	{
		if(topic < m_ratings.size())
			m_ratings[topic].swapItems(a, b);
	}

	float predictRating(Item& item)
	{
		return item.predictRating(m_personality);
	}

	bool getRating(size_t topic, size_t itemId, float* pOutRating)
	{
		if(topic >= m_ratings.size())
			return false;
		map<size_t, float>& m = m_ratings[topic].m_map;
		map<size_t, float>::iterator it = m.find(itemId);
		if(it == m.end())
			return false;
		*pOutRating = it->second;
		return true;
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
				throw Ex("Failed to create account");
		}
		pSession->setExtension(pAccount);
	}
	return pAccount;
}


class ItemStats
{
protected:
	Item& m_item;
	size_t m_id;
	unsigned int m_agree, m_uncertain, m_disagree;
	unsigned int m_agg, m_dis;
	double m_deviation;

public:
	ItemStats(size_t topicId, Item& itm, size_t itemId, Account** pAccs, size_t accCount)
	: m_item(itm), m_id(itemId), m_agree(0), m_uncertain(0), m_disagree(0), m_agg(0), m_dis(0)
	{
		// Compute the mean
		Account** pAc = pAccs;
		float rating;
		double mean = 0.0;
		size_t count = 0;
		for(size_t i = 0; i < accCount; i++)
		{
			if((*pAc)->getRating(topicId, itemId, &rating))
			{
				mean += rating;
				count++;
				if(rating < 0.333334)
					m_disagree++;
				else if(rating > 0.666666)
					m_agree++;
				else
					m_uncertain++;
				if(rating < 0.5)
					m_dis++;
				else
					m_agg++;
			}
			pAc++;
		}
		mean /= count;

		// Compute the deviation
		pAc = pAccs;
		double var = 0.0;
		for(size_t i = 0; i < accCount; i++)
		{
			if((*pAc)->getRating(topicId, itemId, &rating))
			{
				double d = mean - rating;
				var += (d * d);
			}
			pAc++;
		}
		m_deviation = sqrt(var / count);
	}

	Item& item() { return m_item; }
	size_t id() { return m_id; }
	unsigned int disagree() { return m_disagree; }
	unsigned int uncertain() { return m_uncertain; }
	unsigned int agree() { return m_agree; }
	unsigned int split() { return std::min(m_agg, m_dis); }

	double controversy() const
	{
		return m_deviation * (m_agg + m_dis);
	}

	static bool comparer(const ItemStats* pA, const ItemStats* pB)
	{
		return pA->controversy() > pB->controversy();
	}
};

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


void makeHeader(GDynamicPageSession* pSession, ostream& response)
{
	Account* pAccount = getAccount(pSession);
	response << "<html><head>\n";
	response << "	<title>Community Modeler</title>\n";
	response << "	<link rel=\"stylesheet\" type=\"text/css\" href=\"/style/style.css\" />\n";
	response << "</head><body><div id=\"wholepage\">\n";
	response << "\n\n\n\n\n<!-- Header Area --><div id=\"header\">\n";
	response << "	Community Modeler\n";
	response << "</div>\n\n\n\n\n<!-- Left Bar Area --><div id=\"sidebar\">\n";
	response << "	<center>";//<img src=\"style/logo.png\"><br>\n";
	bool loggedin = false;
	if(pAccount)
	{
		response << "Welcome, ";
		const char* szUsername = pAccount->username();
		if(*szUsername == '_')
			response << "anonymous";
		else
		{
			response << szUsername;
			loggedin = true;
		}
		response << ".<br><br>\n";
	}
	response << "	</center>\n";
	if(loggedin)
		response << "	<a href=\"/login?action=logout\">log out</a><br>\n";
	response << "	<a href=\"/survey?nc=" << to_str((size_t)pSession->server()->prng()->next()) << "\">Survey</a><br>\n";
//	response << "	<a href=\"/login\">Switch user</a><br>\n";
//	response << "	<a href=\"/main.hbody\">Overview</a><br>\n";
	response << "	<a href=\"/admin\">Options</a><br>\n";
	response << "	<br><br><br>\n";
	response << "</div>\n\n\n\n\n<!-- Main Body Area --><div id=\"mainbody\">\n";
}

void makeFooter(GDynamicPageSession* pSession, ostream& response)
{
	response << "</div>\n\n\n\n\n<!-- Footer Area --><div id=\"footer\">\n";
//	response << "	The contents of this page are distributed under the <a href=\"http://creativecommons.org/publicdomain/zero/1.0/\">CC0 license</a>. <img src=\"http://i.creativecommons.org/l/zero/1.0/80x15.png\" border=\"0\" alt=\"CC0\" />\n";
	response << "<br>";
	response << "</div>\n\n\n\n\n";
	response << "</div></body></html>\n";
}

class Connection : public GDynamicPageConnection
{
public:
	Connection(SOCKET sock, GDynamicPageServer* pServer) : GDynamicPageConnection(sock, pServer)
	{
	}
	
	virtual ~Connection()
	{
	}

	virtual void handleRequest(GDynamicPageSession* pSession, std::ostream& response);

	void makeUrlSlider(Account* pAccount, size_t itemId, ostream& response)
	{
		// Compute the rating (or predicted rating if this item has not been rated)
		size_t currentTopic = pAccount->currentTopic();
		Topic* pCurrentTopic = ((Server*)m_pServer)->topics()[currentTopic];
		Item& item = pCurrentTopic->item(itemId);
		float score;
		if(!pAccount->getRating(currentTopic, itemId, &score))
			score = pAccount->predictRating(item);
		score *= 500.0;
		score += 500.0;
		score = 0.1 * floor(score);

		// Display the slider
		response << "<table cellpadding=0 cellspacing=0><tr><td width=430>\n	";
// uncomment the next line to always display the predicted score in brackets
//response << "[" << 0.1 * floor(pAccount->predictRating(item) * 1000) << "] ";
		response << item.title();
		response << "\n";
		response << "</td><td>\n";
		response << "	<input type=checkbox name=\"check_slider" << itemId << "\" id=\"check_slider" << itemId << "\">\n";
		response << "	<input name=\"slider" << itemId << "\" id=\"slider" << itemId << "\" type=\"Text\" size=\"3\">\n";
		response << "</td><td>\n";
		response << "<script language=\"JavaScript\">\n";
		response << "	var A_INIT1 = { 's_checkname': 'check_slider" << itemId << "', 's_name': 'slider" << itemId << "', 'n_minValue' : 0, 'n_maxValue' : 100, 'n_value' : " << score << ", 'n_step' : 0.1 }\n";
		response << "	new slider(A_INIT1, A_TPL);\n";
		response << "</script>\n";
		response << "</td></tr></table>\n";
	}


	void makeLoginBody(GDynamicPageSession* pSession, ostream& response, Account* pAccount)
	{
		if(pSession->paramsLen() > 0)
		{
			// Check the password
			GHttpParamParser params(pSession->params());
			const char* szUsername = params.find("username");
			const char* szPasswordHash = params.find("password");
			if(szUsername)
			{
				Account* pNewAccount = ((Server*)m_pServer)->loadAccount(szUsername, szPasswordHash);
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
							s = "/survey?nc=";
							s += to_str((size_t)m_pServer->prng()->next());
						}
					}
					else
					{
						s = "/survey?nc=";
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
		response << "Please log in:<br><br>\n";
		response << "<form name=\"loginform\" action=\"/login\" method=\"get\" onsubmit=\"return HashPassword('";
		response << ((Server*)m_pServer)->passwordSalt();
		response << "')\">\n";
		response << "	Username:<input type=\"text\" name=\"username\" ><br>\n";
		response << "	Password:<input type=\"password\" name=\"password\" ><br>\n";
		response << "	<input type=\"submit\" value=\"Log In\">\n";
		response << "</form><br>\n\n";

		response << "or <a href=\"/newaccount\">create a new account</a><br><br><br>\n";
	}

	virtual void surveyMakePage(GDynamicPageSession* pSession, ostream& response)
	{
		makeHeader(pSession, response);

		// Check whether the user is logged in
		Account* pAccount = getAccount(pSession);
		if(!pAccount)
		{
			makeLoginBody(pSession, response, pAccount);
			return;
		}
		const char* szUsername = pAccount->username();
		if(*szUsername == '_')
		{
			makeLoginBody(pSession, response, pAccount);
			return;
		}

		size_t currentTopic = pAccount->currentTopic();
		if(pSession->paramsLen() > 0)
		{
			// Get the topic
			GHttpParamParser params(pSession->params());
			const char* szTopic = params.find("topic");
			if(szTopic)
			{
				vector<Topic*>& topics = ((Server*)m_pServer)->topics();
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
				((Server*)m_pServer)->proposeTopic(pAccount, szProposal);

			// Do the action
			if(currentTopic < ((Server*)m_pServer)->topics().size())
			{
				const char* szAction = params.find("action");
				if(!szAction)
				{
				}
				else if(_stricmp(szAction, "add") == 0)
				{
					const char* szTitle = params.find("title");
					if(!szTitle)
						response << "[invalid params]<br>\n";
					else
					{
						((Server*)m_pServer)->addItem(currentTopic, szTitle, pAccount->username());
						response << "[The new statement has been added. Thank you.]<br>\n";
						cout << "added " << szTitle << "\n";
						((Server*)m_pServer)->saveState();
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
							if(itemId >= ((Server*)m_pServer)->topics().size())
							{
								response << "[statement id " << itemId << " out of range.]<br>\n";
								continue;
							}
							set<size_t>::iterator tmp = checks.find(itemId);
							if(tmp != checks.end())
							{
								float score = (float)atof(it->second);
								if(score >= 0.0f && score <= 100.0f)
								{
									pAccount->updateRating(currentTopic, itemId, 0.02 * score - 1.0);
									response << "[Rating recorded. Thank you.]<br>\n";
								}
								else
									response << "[the rating of " << score << " is out of range.]<br>\n";
							}
						}
					}

					// Do some training
					((Server*)m_pServer)->trainPersonality(pAccount, ON_RATE_TRAINING_ITERS);
					((Server*)m_pServer)->trainModel(currentTopic, ON_RATE_TRAINING_ITERS);
					((Server*)m_pServer)->saveState();
				}
			}
		}

		if(currentTopic < ((Server*)m_pServer)->topics().size()) // if a topic has been selected...
		{
			Topic* pCurrentTopic = ((Server*)m_pServer)->topics()[currentTopic];
			
			// The slider-bar script
			response << "<script language=\"JavaScript\" src=\"style/slider.js\"></script>\n";
			response << "<script language=\"JavaScript\">\n";
			response << "	var A_TPL = { 'b_vertical' : false, 'b_watch': true, 'n_controlWidth': 321, 'n_controlHeight': 22, 'n_sliderWidth': 19, 'n_sliderHeight': 20, 'n_pathLeft' : 1, 'n_pathTop' : 1, 'n_pathLength' : 300, 's_imgControl': 'style/slider_bg.png', 's_imgSlider': 'style/slider_tab.png', 'n_zIndex': 1 }\n";
			response << "</script>\n";

			// Display the topic
			response << "<h2>" << pCurrentTopic->descr() << "</h2>\n";
			response << "<form name=\"formname\" action=\"/survey\" method=\"post\">\n";
			response << "	<input type=\"hidden\" name=\"action\" value=\"rate\" />\n";
			response << "	<input type=\"hidden\" name=\"nc\" value=\"" << to_str((size_t)m_pServer->prng()->next()) << "\" />\n";

			// Random picks
			size_t* pIndexes = new size_t[pCurrentTopic->size()];
			Holder<size_t> hIndexes(pIndexes);
			GIndexVec::makeIndexVec(pIndexes, pCurrentTopic->size());
			GIndexVec::shuffle(pIndexes, pCurrentTopic->size(), m_pServer->prng());
			size_t sliderCount = 0;
			for(size_t i = 0; i < pCurrentTopic->size(); i++)
			{
				if(sliderCount >= 8)
					break;
				size_t itemId = pIndexes[i];
				float rating;
				if(pAccount->getRating(currentTopic, itemId, &rating))
					continue;
				if(sliderCount == 0)
				{
					response << "<h3>A few statements for your evaluation:</h3>\n";
					response << "<p>It is okay to skip statements you find ambiguous, invasive, or uninteresting. For your convenience, the sliders have been set to reflect predictions of your opinions. As you express more opinions, these predictions should improve.</p>\n";
				}
				makeUrlSlider(pAccount, itemId, response);
				sliderCount++;
			}

			// The update ratings button
			if(sliderCount > 0)
			{
				response << "<br><table><tr><td width=330></td><td>";
				response << "<input type=\"submit\" value=\"Update opinions\">";
				response << "</td></tr><tr><td></td><td>";
				response << "(Only checked items will be updated.)";
				response << "</td></tr></table>\n";
			}
			else
			{
				response << "Thank you. You have expressed your opinion about all ";
				response << to_str(pCurrentTopic->size());
				response << " survey statements in this topic.<br><br>\n";
			}

			response << "</form><br><br>\n\n";

			// The choices links at the bottom of the page
			response << "<a href=\"/submit\">Submit a new statement</a>";
			response << "&nbsp;&nbsp;&nbsp;&nbsp;";
			response << "<a href=\"/survey?topic=-1&nc=" << to_str((size_t)m_pServer->prng()->next()) << "\">Change topic</a>\n";
			response << "&nbsp;&nbsp;&nbsp;&nbsp;";
			response << "<a href=\"/update\">My opinions</a>\n";
			response << "&nbsp;&nbsp;&nbsp;&nbsp;";
			response << "<a href=\"/stats?nc=" << to_str((size_t)m_pServer->prng()->next()) << "\">Vizualize</a>\n";

/*
			response << "Stats:<br>\n";
			response << "Total Number of users: " << ((Server*)m_pServer)->accounts().size() << "<br>\n";
			response << "Number of items in this topic: " << pCurrentTopic->size() << "<br>\n";
			std::map<size_t, float>* pMap = currentTopic < pAccount->ratings().size() ? pAccount->ratings()[currentTopic] : NULL;
			response << "Number of items you have rated in this topic: " << (pMap ? pMap->size() : (size_t)0) << "<br>\n<br>\n";
*/
		}
		else
		{
			vector<Topic*>& topics = ((Server*)m_pServer)->topics();
			response << "<h3>Choose a topic:</h3>\n";
			if(topics.size() > 0)
			{
				response << "<ul>\n";
				size_t i = 0;
				for(vector<Topic*>::iterator it = topics.begin(); it != topics.end(); it++)
				{
					response << "	<li><a href=\"/survey?topic=" << i << "&nc=" << to_str((size_t)m_pServer->prng()->next()) << "\">" << (*it)->descr() << "</a></li>\n";
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
/*
			// Make the form to propose new topics
			if(pAccount->doesHavePassword() && _stricmp(pAccount->username(), "root") != 0)
			{
				response << "<form name=\"propose\" action=\"/survey\" method=\"get\">\n";
				response << "	<h3>Propose a new topic:</h3>\n";
				response << "	<input type=\"text\" name=\"proposal\" size=\"55\"><input type=\"submit\" value=\"Submit\"><br>\n";
				response << "	(Your proposed topic will be added to a log file. Hopefully, someone actually reads the log file.)\n";
				response << "</form><br>\n\n";
			}
*/
		}
		makeFooter(pSession, response);
	}

	virtual void submitMakePage(GDynamicPageSession* pSession, ostream& response)
	{
		makeHeader(pSession, response);
		Account* pAccount = getAccount(pSession);
		size_t currentTopic = pAccount->currentTopic();
		if(currentTopic >= ((Server*)m_pServer)->topics().size())
		{
			string s = "/survey?nc=";
			s += to_str((size_t)m_pServer->prng()->next());
			m_pServer->redirect(response, s.c_str());
		}
		else
		{
			// Display the topic
			Topic* pCurrentTopic = ((Server*)m_pServer)->topics()[currentTopic];
			response << "<h2>" << pCurrentTopic->descr() << "</h2>\n";

			// Make the form to submit a new item
			response << "<h3>Submit a new statement to this topic</h3>\n";
			response << "<form name=\"formname\" action=\"/survey\" method=\"post\">\n";
			response << "	<input type=\"hidden\" name=\"action\" value=\"add\" />\n";
			response << "	<input type=\"hidden\" name=\"nc\" value=\"" << to_str((size_t)m_pServer->prng()->next()) << "\" />\n";
			response << "Statement: <input type=\"text\" name=\"title\" size=\"55\"><br>\n";
			response << "	<input type=\"submit\" value=\"Submit\">";
			response << "</form><br><br>\n\n";

			// The choices links at the bottom of the page
			response << "<br>\n";
			response << "<a href=\"/survey?topic=-1&nc=" << to_str((size_t)m_pServer->prng()->next()) << "\">Change topic</a>\n";
			response << "&nbsp;&nbsp;&nbsp;&nbsp;";
			response << "<a href=\"/update\">My opinions</a>\n";
			response << "&nbsp;&nbsp;&nbsp;&nbsp;";
			response << "<a href=\"/survey?nc=" << to_str((size_t)m_pServer->prng()->next()) << "\">" << "Survey</a>\n";
			response << "&nbsp;&nbsp;&nbsp;&nbsp;";
			response << "<a href=\"/stats?nc=" << to_str((size_t)m_pServer->prng()->next()) << "\">Vizualize</a>\n";
		}
		makeFooter(pSession, response);
	}

	double computeVariance(double* pCentroid, Topic& topic, size_t topicId, Account** pAccs, size_t accCount)
	{
		// Compute the centroid
		GVec::setAll(pCentroid, 0.0, topic.size());
		Account** pAc = pAccs;
		for(size_t j = 0; j < accCount; j++)
		{
			double* pC = pCentroid;
			for(size_t i = 0; i < topic.size(); i++)
			{
				float rating;
				if(!(*pAc)->getRating(topicId, i, &rating))
					rating = (*pAc)->predictRating(topic.item(i));
				(*pC) += rating;
				pC++;
			}
			pAc++;
		}
		GVec::multiply(pCentroid, 1.0 / accCount, topic.size());

		// Measure the sum-squared error with the centroid
		double sse = 0.0;
		pAc = pAccs;
		for(size_t j = 0; j < accCount; j++)
		{
			double* pC = pCentroid;
			for(size_t i = 0; i < topic.size(); i++)
			{
				float rating;
				if(!(*pAc)->getRating(topicId, i, &rating))
					rating = (*pAc)->predictRating(topic.item(i));
				double d = *pC - rating;
				sse += (d * d);
				pC++;
			}
			pAc++;
		}
		return sse;
	}

	size_t divideAccounts(Topic& topic, size_t topicId, Account** pAccs, size_t accCount, size_t itm)
	{
		size_t head = 0;
		size_t tail = accCount;
		while(tail > head)
		{
			float rating;
			if(!pAccs[head]->getRating(topicId, itm, &rating))
				rating = pAccs[head]->predictRating(topic.item(itm));
			if(rating < 0.0)
			{
				tail--;
				std::swap(pAccs[head], pAccs[tail]);
			}
			else
				head++;
		}
		GAssert(head == tail);
		return head;
	}

	void makeTree(Topic& topic, size_t topicId, GBitTable& bt, Account** pAccs, size_t accCount, ostream& response, vector<char>& prefix, int type)
	{
		// Try splitting on each of the remaining statements
		size_t best = (size_t)-1;
		double mostCont = 0.0;
		double* pCentroid = new double[topic.size()];
		ArrayHolder<double> hCentroid(pCentroid);
		size_t tieCount = 0;
		for(size_t i = 0; i < topic.size(); i++)
		{
			if(bt.bit(i))
				continue;
			ItemStats is(topicId, topic.item(i), i, pAccs, accCount);
			double c = is.controversy();
			if(is.split() > 0)
			{
				if(c > mostCont)
				{
					mostCont = c;
					best = i;
					tieCount = 0;
				}
				else if(c == mostCont)
				{
					tieCount++;
					if(m_pServer->prng()->next(tieCount + 1) == 0)
						best = i;
				}
			}
		}

		if(best != (size_t)-1)
		{
			// Divide on the best statement
			size_t firstHalfSize = divideAccounts(topic, topicId, pAccs, accCount, best);
			bt.set(best);

			// Recurse
			prefix.push_back(' ');
			if(type >= 0) prefix.push_back(' '); else prefix.push_back('|');
			prefix.push_back(' ');
			prefix.push_back(' ');
			makeTree(topic, topicId, bt, pAccs, firstHalfSize, response, prefix, 1);

			for(vector<char>::iterator it = prefix.begin(); it != prefix.end(); it++)
				response << *it;
			response << " |\n";
			prefix.pop_back(); prefix.pop_back(); prefix.pop_back(); prefix.pop_back();
			for(vector<char>::iterator it = prefix.begin(); it != prefix.end(); it++)
				response << *it;
			if(type == 0)
				response << "---&gt;";
			else
				response << " +-&gt;";
			response << topic.item(best).title() << "\n";
			prefix.push_back(' ');
			if(type <= 0) prefix.push_back(' '); else prefix.push_back('|');
			prefix.push_back(' ');
			prefix.push_back(' ');
			for(vector<char>::iterator it = prefix.begin(); it != prefix.end(); it++)
				response << *it;
			response << " |\n";

			makeTree(topic, topicId, bt, pAccs + firstHalfSize, accCount - firstHalfSize, response, prefix, -1);
			prefix.pop_back(); prefix.pop_back(); prefix.pop_back(); prefix.pop_back();

			bt.unset(best);
		}
		else
		{
			for(size_t j = 0; j < accCount; j++)
			{
				for(vector<char>::iterator it = prefix.begin(); it != prefix.end(); it++)
					response << *it;
				response << " +-&gt;<a href=\"/stats?user=" << pAccs[j]->username() << "\">";
				response << pAccs[j]->username() << "</a>\n";
			}
		}
	}

	virtual void makeItemBody(GDynamicPageSession* pSession, ostream& response, size_t topicId, size_t itemId, Item& item, Account** pAccs, size_t accCount)
	{
		response << "<h2>" << item.title() << "</h2>\n";
		std::multimap<double,Account*> mm;
		while(accCount > 0)
		{
			float rating;
			if((*pAccs)->getRating(topicId, itemId, &rating))
			{
				double r = rating * 50000 + 50000;
				double clipped = 0.001 * (double)floor(r + 0.5f);
				mm.insert(std::pair<double,Account*>(clipped,*pAccs));
			}
			accCount--;
			pAccs++;
		}
		response << "<h3>Disagree</h3>\n";
		response << "<table>\n";
		size_t hh = 0;
		for(std::multimap<double,Account*>::iterator it = mm.begin(); it != mm.end(); it++)
		{
			if(hh == 0 && it->first > 33.3333)
			{
				response << "</table>\n<h3>Uncertain</h3>\n<table>\n";
				hh++;
			}
			if(hh == 1 && it->first > 66.6666)
			{
				response << "</table>\n<h3>Agree</h3>\n<table>\n";
				hh++;
			}
			response << "<tr><td>" << it->second->username() << "</td><td>" << to_str(it->first) << "</td></tr>\n";
		}
		if(hh == 0)
		{
			response << "</table>\n<h3>Uncertain</h3>\n<table>\n";
			hh++;
		}
		if(hh == 1)
		{
			response << "</table>\n<h3>Agree</h3>\n<table>\n";
			hh++;
		}
		response << "</table>\n";
	}

	virtual void makeUserBody(GDynamicPageSession* pSession, ostream& response, Account* pA, Account* pB, size_t topicId, Topic& topic)
	{
		std::multimap<float,size_t> m;
		float rA = 0.0f;
		float rB = 0.0f;
		for(size_t i = 0; i < topic.size(); i++)
		{
			if(pA->getRating(topicId, i, &rA))
			{
				if(pB->getRating(topicId, i, &rB))
					m.insert(std::pair<float,size_t>(-std::abs(rB - rA), i));
			}
		}
		if(m.size() == 0)
		{
			response << "You have no ratings in common.<br><br>\n";
			return;
		}
		response << "<table><tr><td><u>" << pA->username() << "</u></td><td><u>" << pB->username() << "</u></td><td><u>delta</u></td><td><u>Statement</u></td></tr>\n";
		for(std::multimap<float,size_t>::iterator it = m.begin(); it != m.end(); it++)
		{
			pA->getRating(topicId, it->second, &rA);
			pB->getRating(topicId, it->second, &rB);
			response << "<tr><td>" << to_str(0.1 * floor(rA * 500 + 500)) << "</td><td>" << to_str(0.1 * floor(rB * 500 + 500)) << "</td><td>" << to_str(0.1 * floor(std::abs(rA - rB) * 500)) << "</td><td>" << topic.item(it->second).title() << "</td></tr>\n";
		}
		response << "</table>\n";
	}

	void statsMakePage(GDynamicPageSession* pSession, ostream& response)
	{
		makeHeader(pSession, response);

		// Get the topic
		Account* pAccount = getAccount(pSession);
		size_t currentTopic = pAccount->currentTopic();
		if(currentTopic >= ((Server*)m_pServer)->topics().size())
		{
			response << "Unrecognized topic.";
			return;
		}
		Topic& topic = *((Server*)m_pServer)->topics()[currentTopic];

		// Copy the account pointers into an array
		std::vector<Account*>& accs = ((Server*)m_pServer)->accounts();
		Account** pAccs = new Account*[accs.size()];
		ArrayHolder<Account*> hAccs(pAccs);
		Account** pAc = pAccs;
		size_t accountCount = 0;
		for(std::vector<Account*>::iterator it = accs.begin(); it != accs.end(); it++)
		{
			if((*it)->username()[0] != '_' && strcmp((*it)->username(), "root") != 0)
			{
				*(pAc++) = *it;
				accountCount++;
			}
		}

		// Check the params
		GHttpParamParser params(pSession->params());
		const char* szItemId = params.find("item");
		if(szItemId)
		{
			size_t itemId = atoi(szItemId);
			makeItemBody(pSession, response, currentTopic, itemId, topic.item(itemId), pAccs, accountCount);
			return;
		}
		const char* szOtherUser = params.find("user");
		if(szOtherUser)
		{
			Account* pOther = ((Server*)m_pServer)->findAccount(szOtherUser);
			if(!pOther)
				response << "[No such user]<br><br>\n";
			else
				makeUserBody(pSession, response, pAccount, pOther, currentTopic, topic);
			return;
		}

		GBitTable bt(topic.size());
		vector<char> prefix;
		response << "This ascii-art tree was constructed by dividing on the most controversial statements within each branch.\n";
		response << "This tree is arranged such that the ascending branches lead to the usernames of people who agree with the statement, and the descending branches lead to the usernames of people who disagree with the statement.\n";
		response << "(In cases of uncertainty or lack of response, predictions were used to make any judgement calls necessary to construct this tree.)\n";
		response << "<br><br>\n";
		response << "<pre>\n";
		makeTree(topic, currentTopic, bt, pAccs, accountCount, response, prefix, 0);
		response << "</pre>\n";
		response << "<br><br>\n";

		// Make a table of items sorted by controversy
		std::vector<ItemStats*> items;
		for(size_t i = 0; i < topic.size(); i++)
			items.push_back(new ItemStats(currentTopic, topic.item(i), i, pAccs, accountCount));
		sort(items.begin(), items.end(), ItemStats::comparer);
		response << "<table><tr><td><b><i><u>Statement</u></i></b></td><td><b><i><u>Disagree</u></i></b></td><td><b><i><u>Uncertain</u></i></b></td><td><b><i><u>Agree</u></i></b></td><td><b><i><u>Controversy</u></i></b></td></tr>\n";
		for(vector<ItemStats*>::iterator it = items.begin(); it != items.end(); it++)
		{
			response << "<tr><td>";
			response << "<a href=\"/stats?item=" << to_str((*it)->id()) << "\">" << (*it)->item().title() << "</a>";
			response << "</td><td>";
			response << to_str((*it)->disagree());
			response << "</td><td>";
			response << to_str((*it)->uncertain());
			response << "</td><td>";
			response << to_str((*it)->agree());
			response << "</td><td>";
			response << to_str((*it)->controversy());
			response << "</td></tr>\n";
			delete(*it);
		}
		response << "</table><br><br>\n";

		response << "<h3>A vizualization of the users in this community:</h3>\n";
		response << "<img src=\"users.svg\"><br><br>\n";
		response << "<h3>A vizualization of the items in this community:</h3>\n";
		response << "<img src=\"items.svg\"><br><br>\n";

		// The choices links at the bottom of the page
		response << "<a href=\"/submit\">Submit a new statement</a>";
		response << "&nbsp;&nbsp;&nbsp;&nbsp;";
		response << "<a href=\"/survey?topic=-1&nc=" << to_str((size_t)m_pServer->prng()->next()) << "\">Change topic</a>\n";
		response << "&nbsp;&nbsp;&nbsp;&nbsp;";
		response << "<a href=\"/update\">My opinions</a>\n";
		response << "&nbsp;&nbsp;&nbsp;&nbsp;";
		response << "<a href=\"/survey?nc=" << to_str((size_t)m_pServer->prng()->next()) << "\">" << "Survey</a>\n";

		makeFooter(pSession, response);
	}

	void plotUsers(GDynamicPageSession* pSession, ostream& response)
	{
		setContentType("image/svg+xml");
		GSVG svg(800, 800);

		vector<Account*>& accounts = ((Server*)m_pServer)->accounts();
		double xmin = 0;
		double ymin = 0;
		double xmax = 0;
		double ymax = 0;
		for(size_t i = 0; i < accounts.size(); i++)
		{
			Account* pAcc = accounts[i];
			const char* szUsername = pAcc->username();
			if(*szUsername == '_')
				continue;
			vector<double>& profile = pAcc->personality();
			xmin = std::min(xmin, profile[1]);
			xmax = std::max(xmax, profile[1]);
			ymin = std::min(ymin, profile[2]);
			ymax = std::max(ymax, profile[2]);
		}
		double wid = xmax - xmin;
		double hgt = ymax - ymin;
		xmin -= 0.1 * wid;
		xmax += 0.1 * wid;
		ymin -= 0.1 * hgt;
		ymax += 0.1 * hgt;
		if(xmax - xmin < 1e-4)
			xmax += 1e-4;
		if(ymax - ymin < 1e-4)
			ymax += 1e-4;
		svg.newChart(xmin, ymin, xmax, ymax, 0, 0, 0);
		for(size_t i = 0; i < accounts.size(); i++)
		{
			Account* pAcc = accounts[i];
			const char* szUsername = pAcc->username();
			if(*szUsername == '_')
				continue;
			vector<double>& profile = pAcc->personality();
			svg.dot(profile[1], profile[2], 0.75, 0x008080);
			svg.text(profile[1], profile[2], szUsername, 0.75);
		}
		svg.print(response);
	}

	void plotItems(GDynamicPageSession* pSession, ostream& response)
	{
		// Get the topic
		Account* pAccount = getAccount(pSession);
		size_t currentTopic = pAccount->currentTopic();
		if(currentTopic >= ((Server*)m_pServer)->topics().size())
		{
			response << "Unrecognized topic.";
			return;
		}
		Topic& topic = *((Server*)m_pServer)->topics()[currentTopic];

		setContentType("image/svg+xml");
		GSVG svg(800, 800);

		double xmin = 0;
		double ymin = 0;
		double xmax = 0;
		double ymax = 0;
		for(size_t i = 0; i < topic.size(); i++)
		{
			Item& item = topic.item(i);
			vector<double>& weights = item.weights();
			xmin = std::min(xmin, weights[1]);
			xmax = std::max(xmax, weights[1]);
			ymin = std::min(ymin, weights[2]);
			ymax = std::max(ymax, weights[2]);
		}
		double wid = xmax - xmin;
		double hgt = ymax - ymin;
		xmin -= 0.1 * wid;
		xmax += 0.1 * wid;
		ymin -= 0.1 * hgt;
		ymax += 0.1 * hgt;
		if(xmax - xmin < 1e-4)
			xmax += 1e-4;
		if(ymax - ymin < 1e-4)
			ymax += 1e-4;
		svg.newChart(xmin, ymin, xmax, ymax, 0, 0, 0);
		for(size_t i = 0; i < topic.size(); i++)
		{
			Item& item = topic.item(i);
			const char* szTitle = item.title();
			vector<double>& weights = item.weights();
			svg.dot(weights[1], weights[2], 0.75, 0x008080);
			svg.text(weights[1], weights[2], szTitle, 0.75);
		}
		svg.print(response);
	}

	virtual void updateMakePage(GDynamicPageSession* pSession, ostream& response)
	{
		makeHeader(pSession, response);

		Account* pAccount = getAccount(pSession);
		size_t currentTopic = pAccount->currentTopic();
		if(currentTopic >= ((Server*)m_pServer)->topics().size())
		{
			string s = "/survey?nc=";
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
			Topic* pCurrentTopic = ((Server*)m_pServer)->topics()[currentTopic];
			response << "<h2>" << pCurrentTopic->descr() << "</h2>\n";

			// Display the items you have rated
			if(pAccount->ratings().size() > currentTopic)
			{
				vector<pair<size_t, float> >& v = pAccount->ratings()[currentTopic].m_vec;
				if(v.size() > 0)
				{
					response << "<h3>Your opinions</h3>\n";
					response << "<form name=\"formname\" action=\"/survey\" method=\"post\">\n";
					response << "	<input type=\"hidden\" name=\"action\" value=\"rate\" />\n";
					response << "	<input type=\"hidden\" name=\"nc\" value=\"" << to_str((size_t)m_pServer->prng()->next()) << "\" />\n";
					UpdateComparer comparer;
					std::sort(v.begin(), v.end(), comparer);
					for(vector<pair<size_t, float> >::iterator it = v.begin(); it != v.end(); it++)
						makeUrlSlider(pAccount, it->first, response);
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
			response << "<a href=\"/survey?topic=-1&nc=" << to_str((size_t)m_pServer->prng()->next()) << "\">Change topic</a>\n";
			response << "&nbsp;&nbsp;&nbsp;&nbsp;";
			response << "<a href=\"/survey?nc=" << to_str((size_t)m_pServer->prng()->next()) << "\">" << "Survey</a>\n";
			response << "&nbsp;&nbsp;&nbsp;&nbsp;";
			response << "<a href=\"/stats?nc=" << to_str((size_t)m_pServer->prng()->next()) << "\">Vizualize</a>\n";
		}
		makeFooter(pSession, response);
	}

	virtual void adminMakePage(GDynamicPageSession* pSession, ostream& response)
	{
		makeHeader(pSession, response);
		Account* pAccount = getAccount(pSession);
		if(pSession->paramsLen() > 0)
		{
			GHttpParamParser params(pSession->params());
			const char* szAction = params.find("action");
			if(szAction)
			{
				// Do the action
				if(_stricmp(szAction, "shutdown") == 0)
				{
					if(_stricmp(pAccount->username(), "root") == 0)
					{
						cout << "root has told the server to shut down.\n";
						cout.flush();
						cerr.flush();
						m_pServer->shutDown();
					}
				}
				else if(_stricmp(szAction, "newtopic") == 0)
				{
					if(_stricmp(pAccount->username(), "root") == 0)
					{
						const char* szDescr = params.find("descr");
						if(szDescr && strlen(szDescr) > 0)
						{
							((Server*)m_pServer)->newTopic(szDescr);
							response << "[The new topic has been added]<br>\n";
						}
						else
							response << "[You must enter a topic description]<br>\n";
					}
				}
				else if(_stricmp(szAction, "nukeself") == 0)
				{
					((Server*)m_pServer)->deleteAccount(pAccount);
					string s = "/survey?nc=";
					s += to_str((size_t)m_pServer->prng()->next());
					m_pServer->redirect(response, s.c_str());
					pSession->setExtension(NULL); // disconnect the account from this session
					return;
				}
				else
					response << "[Unknown action! No action taken]<br>\n";
			}
			const char* szDel = params.find("del");
			if(szDel)
			{
				size_t currentTopic = pAccount->currentTopic();
				if(currentTopic >= ((Server*)m_pServer)->topics().size())
					response << "[invalid topic id]<br><br>\n";
				else
				{
					size_t index = atoi(szDel);
					std::vector<Account*>& accs = ((Server*)m_pServer)->accounts();
					Topic& topic = *((Server*)m_pServer)->topics()[currentTopic];
					if(index >= topic.size())
						response << "[invalid item index]<br><br>\n";
					else
					{
						cout << "Deleted item " << topic.item(index).title() << "\n";
						for(vector<Account*>::iterator it = accs.begin(); it != accs.end(); it++)
						{
							(*it)->withdrawRating(currentTopic, index);
							(*it)->swapItems(currentTopic, index, topic.size() - 1);
						}
						topic.deleteItemAndSwapInLast(index);
						response << "[Item successfully deleted]<br><br>\n";
					}
				}
			}
		}

		// Root controls
		if(_stricmp(pAccount->username(), "root") == 0)
		{
			response << "<h2>Root controls</h2>\n\n";

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
		else
			response << "(Note: Additional controls are available to the user with username \"root\".)<br><br>\n";
		
		// Form to delete a statement
		response << "<h2>Delete Statements</h2>\n\n";
		size_t currentTopic = pAccount->currentTopic();
		if(currentTopic >= ((Server*)m_pServer)->topics().size())
			response << "<p>No topic has been selected. If you want to delete one or more statements, please click on \"Survey\", choose a topic, then return to here.</p>\n";
		else
		{
			response << "<p>If a statement can be corrected, it is courteous to submit a corrected version after you delete it. Valid reasons to delete a statement include: not controversial enough, too long-winded, confusing, difficult to negate, ambiguous, off-topic, etc.</p>";
			Topic& topic = *((Server*)m_pServer)->topics()[currentTopic];
			response << "<table>\n";
			for(size_t i = 0; i < topic.size(); i++)
			{
				Item& itm = topic.item(i);
				response << "<tr><td>";
				response << "<form name=\"delitem\" action=\"/admin\" method=\"get\"><input type=\"hidden\" name=\"del\" value=\"" << to_str(i) << "\"><input type=\"submit\" value=\"Delete\"></form>";
				response << "</td><td>" << itm.title() << "</td></tr>\n";
			}
			response << "</table>\n";
			response << "<br>\n";
			response << "</form><br><br>\n\n";
		}

		// Button to nuke my account
		response << "Warning: Don't push this button unless you really mean it! ";
		response << "<form name=\"nukeself\" action=\"/admin\" method=\"get\">\n";
		response << "<input type=\"hidden\" name=\"action\" value=\"nukeself\">\n";
		response << "<input type=\"submit\" value=\"Nuke My Account\">\n";
		response << "</form>\n";
		
		makeFooter(pSession, response);
	}

	virtual void loginMakePage(GDynamicPageSession* pSession, ostream& response)
	{
		makeHeader(pSession, response);
		Account* pAccount = getAccount(pSession);
		if(pSession->paramsLen() > 0)
		{
			// See if the user wants to log out
			GHttpParamParser params(pSession->params());
			const char* szAction = params.find("action");
			if(szAction)
			{
				if(_stricmp(szAction, "logout") == 0)
				{
					string s = "/survey?nc=";
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
				Account* pNewAccount = ((Server*)m_pServer)->loadAccount(szUsername, szPasswordHash);
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
							s = "/survey?nc=";
							s += to_str((size_t)m_pServer->prng()->next());
						}
					}
					else
					{
						s = "/survey?nc=";
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
			response << "Switch user:<br>\n";
		}
		else
			response << "Please log in:<br><br>\n";
		response << "<form name=\"loginform\" action=\"/login\" method=\"get\" onsubmit=\"return HashPassword('";
		response << ((Server*)m_pServer)->passwordSalt();
		response << "')\">\n";
		response << "	Username:<input type=\"text\" name=\"username\" ><br>\n";
		response << "	Password:<input type=\"password\" name=\"password\" ><br>\n";
		response << "	<input type=\"submit\" value=\"Log In\">\n";
		response << "</form><br>\n\n";

		response << "or <a href=\"/newaccount\">create a new account</a><br><br><br>\n";
		makeFooter(pSession, response);
	}

	void newAccountMakePage(GDynamicPageSession* pSession, ostream& response);
};

void Connection::newAccountMakePage(GDynamicPageSession* pSession, ostream& response)
{
	makeHeader(pSession, response);
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
			Account* pAccount = ((Server*)m_pServer)->newAccount(szUsername, szPassword);
			if(!pAccount)
				szError = "That username is already taken.";
			else
			{
				((Server*)m_pServer)->saveState();
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

	response << "<br><center><table width=\"400\" border=\"0\" cellpadding=\"0\" cellspacing=\"0\"><tr><td>\n";
	response << "<SCRIPT language=\"JavaScript\" src=\"/sha1.js\" type=\"text/javascript\">\n</SCRIPT>\n";
	response << "	<big><big><b>Create a new account</b></big></big><br><br>\n";
	response << "	<form name=\"newaccountform\" action=\"/newaccount\" method=\"post\" onsubmit=\"return HashNewAccount('";
	response << ((Server*)m_pServer)->passwordSalt();
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
	makeFooter(pSession, response);
}

// virtual
void Connection::handleRequest(GDynamicPageSession* pSession, ostream& response)
{
	if(strcmp(m_szUrl, "/favicon.ico") == 0)
		return;
	if(strncmp(m_szUrl, "/login", 6) == 0)
		loginMakePage(pSession, response);
	else if(strcmp(m_szUrl, "/") == 0 || strncmp(m_szUrl, "/survey", 4) == 0)
		surveyMakePage(pSession, response);
	else if(strncmp(m_szUrl, "/submit", 7) == 0)
		submitMakePage(pSession, response);
	else if(strncmp(m_szUrl, "/stats", 6) == 0)
		statsMakePage(pSession, response);
	else if(strncmp(m_szUrl, "/update", 7) == 0)
		updateMakePage(pSession, response);
	else if(strncmp(m_szUrl, "/admin", 6) == 0)
		adminMakePage(pSession, response);
	else if(strncmp(m_szUrl, "/newaccount", 11) == 0)
		newAccountMakePage(pSession, response);
	else if(strncmp(m_szUrl, "/users.svg", 10) == 0)
		plotUsers(pSession, response);
	else if(strncmp(m_szUrl, "/items.svg", 10) == 0)
		plotItems(pSession, response);
	else
	{
		size_t len = strlen(m_szUrl);
		if(len > 6 && strcmp(m_szUrl + len - 6, ".hbody") == 0)
		{
			makeHeader(pSession, response);
			sendFileSafe(((Server*)m_pServer)->m_basePath.c_str(), m_szUrl + 1, response);
			makeFooter(pSession, response);
		}
		else
			sendFileSafe(((Server*)m_pServer)->m_basePath.c_str(), m_szUrl + 1, response);
	}
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
	cout << "Base path: " << m_basePath << "\n";
	loadState();
}

// virtual
Server::~Server()
{
	saveState();

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

void Server::addItem(size_t topic, const char* szTitle, const char* szUsername)
{
	if(topic >= m_topics.size())
	{
		cout << "Topic ID out of range\n";
		return;
	}
	m_topics[topic]->addItem(szTitle, szUsername, time(NULL), prng());
}

void getLocalStorageFolder(char* buf)
{
	if(!GFile::localStorageDirectory(buf))
		throw Ex("Failed to find local storage folder");
	strcat(buf, "/.community/");
	GFile::makeDir(buf);
	if(!GFile::doesDirExist(buf))
		throw Ex("Failed to create folder in storage area");
}

void Server::getStatePath(char* buf)
{
	getLocalStorageFolder(buf);
	strcat(buf, "state.json");
}

// virtual
void Server::onEverySixHours()
{
	for(size_t i = 0; i < 3; i++)
	{
		size_t topicId = m_pRand->next(m_topics.size());
		trainModel(topicId, ON_RATE_TRAINING_ITERS);
	}
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
	Account* pAccount = new Account(szUsername, szPasswordHash, *m_pRand);
	m_accountsVec.push_back(pAccount);
	m_accountsMap.insert(make_pair(string(szUsername), pAccount));
	cout << "Made new account for " << szUsername << "\n";
	cout.flush();
	return pAccount;
}

void Server::deleteAccount(Account* pAccount)
{
	string s;
	for(std::map<std::string,Account*>::iterator it = m_accountsMap.begin(); it != m_accountsMap.end(); it++)
	{
		if(it->second == pAccount)
		{
			s = it->first;
			break;
		}
	}
	m_accountsMap.erase(s);
	std::vector<Account*>::iterator it;
	for(it = m_accountsVec.begin(); it != m_accountsVec.end(); it++)
	{
		if(*it == pAccount)
			break;
	}
	if(*it == pAccount)
		m_accountsVec.erase(it);
	cout << "Account " << pAccount->username() << " deleted.\n";
	saveState();
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
				double target = (double)v[index].second;
				GAssert(target >= 0.0 && target <= 1.0);
				item.trainWeights(target, LEARNING_RATE, pSomeAccount->personality());
				item.trainPersonality(target, LEARNING_RATE, pSomeAccount->personality());
			}
		}
	}
}

Account* Server::findAccount(const char* szName)
{
	std::map<std::string,Account*>::iterator it = m_accountsMap.find(szName);
	if(it == m_accountsMap.end())
		return NULL;
	else
		return it->second;
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
		for(size_t i = 0; i < iters; i++)
		{
			size_t index = (size_t)prng()->next(v.size());
			Item& item = pCurrentTopic->item(v[index].first);
			item.trainPersonality((double)v[index].second, LEARNING_RATE, pAccount->personality());
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
		if(pAccount->username()[0] != '_') // Don't bother persisting anonymous accounts
			pAccounts->addItem(pDoc, pAccount->toDom(pDoc));
	}

	return pNode;
}

void Server::deserializeState(GDomNode* pNode)
{
	// Captcha salt
	const char* daemon_Salt = pNode->fieldIfExists("daemonSalt")->asString();
	if(daemon_Salt)
		setDaemonSalt(daemon_Salt);

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
		Account* pAccount = Account::fromDom(it.current(), *m_pRand);
		m_accountsVec.push_back(pAccount);
		m_accountsMap.insert(make_pair(string(pAccount->username()), pAccount));
	}
}

// virtual
GDynamicPageConnection* Server::makeConnection(SOCKET sock)
{
	return new Connection(sock, this);
}






void LaunchBrowser(const char* szAddress, GRand* pRand)
{
	string s = szAddress;
	s += "/survey?nc=";
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
		throw Ex("Error redirecting stdout");
	}
	string s2(pPath);
	s2 += "stderr.log";
	if(!freopen(s2.c_str(), "a", stderr))
	{
		cout << "Error redirecting stderr\n";
		cerr << "Error redirecting stderr\n";
		throw Ex("Error redirecting stderr");
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
	if(chdir(path) != 0)
		throw Ex("Failed to change dir to ", path);
	cout << "Launching daemon...\n";
	GApp::launchDaemon(doit, path, s1.c_str(), s2.c_str());
	if(!getcwd(path, 300))
	{
	}
	cout << "Daemon running in " << path << ".\n	stdout >> " << s1.c_str() << "\n	stderr >> " << s2.c_str() << "\n";
}

int main(int nArgs, char* pArgs[])
{
	int nRet = 1;
	try
	{
		if(nArgs > 1 && strcmp(pArgs[1], "daemon") == 0)
			doItAsDaemon();
		else
			doit(NULL);
	}
	catch(std::exception& e)
	{
		cerr << e.what() << "\n";
	}
	return nRet;
}
