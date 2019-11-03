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


#include "recommender.h"
#include <GClasses/GRand.h>
#include <GClasses/GDom.h>
#include <GClasses/GHolders.h>
#include <GClasses/GVec.h>
#include <GClasses/GPlot.h>
#include <map>
#include <cmath>
#include <algorithm>
#include "server.h"


using std::set;
using std::map;
using std::string;
using std::vector;
using std::cerr;
using std::cout;
using std::ostream;
using std::pair;


User::User(string username, GRand& rand)
: m_username(username)
{
	m_personality.resize(USER_PROFILE_SIZE);
	for(size_t i = 0; i < USER_PROFILE_SIZE; i++)
		m_personality[i] = LEARNING_RATE * rand.normal();
}

User::~User()
{
}

// static
User* User::fromDom(GDomNode* pNode, GRand& rand)
{
	const char* username = pNode->getString("username");
	User* pUser = new User(username, rand);

	// Deserialize the personality vector
	GDomNode* pPersonality = pNode->get("pers");
	{
		GDomListIterator it(pPersonality);
		size_t i;
		for(i = 0; i < (size_t)USER_PROFILE_SIZE && it.current(); i++)
		{
			pUser->m_personality[i] = std::max(-1.0, std::min(1.0, it.currentDouble()));
			it.advance();
		}
		for( ; i < USER_PROFILE_SIZE; i++)
			pUser->m_personality[i] = 0.02 * rand.normal();
	}

	// Deserialize the ratings
	GDomNode* pRatings = pNode->get("ratings");
	for(GDomListIterator it(pRatings); it.current(); it.advance())
	{
		ptrdiff_t j = (ptrdiff_t)it.currentInt();
		it.advance();
		if(it.current())
			pUser->addRating((size_t)j, (float)it.currentDouble());
	}
	return pUser;
}

GDomNode* User::toDom(GDom* pDoc)
{
	GDomNode* pUser = pDoc->newObj();
	pUser->add(pDoc, "username", m_username.c_str());

	// Serialize the personality vector
	GDomNode* pPersonality = pUser->add(pDoc, "pers", pDoc->newList());
	for(size_t i = 0; i < USER_PROFILE_SIZE; i++)
		pPersonality->add(pDoc, m_personality[i]);

	// Serialize the ratings
	size_t count = 0;
	count += 2 * m_map.size();
	GDomNode* pRatings = pUser->add(pDoc, "ratings", pDoc->newList());
	for(std::map<size_t, float>::iterator it = m_map.begin(); it != m_map.end(); it++)
	{
		pRatings->add(pDoc, it->first);
		pRatings->add(pDoc, it->second);
	}

	return pUser;
}

bool User::getRating(size_t itemId, float* pOutRating)
{
	map<size_t, float>& m = m_map;
	map<size_t, float>::iterator it = m.find(itemId);
	if(it == m.end())
		return false;
	*pOutRating = it->second;
	return true;
}

void User::addRating(size_t itemId, float rating)
{
	GAssert(rating >= -1.0f && rating <= 1.0f);
	m_map[itemId] = rating;
	m_vec.push_back(std::make_pair(itemId, rating));
}

void User::updateRating(size_t itemId, float rating)
{
	GAssert(rating >= -1.0f && rating <= 1.0f);
	if(m_map.find(itemId) == m_map.end())
		addRating(itemId, rating);
	else
	{
		m_map[itemId] = rating;
		for(vector<std::pair<size_t, float> >::iterator it = m_vec.begin(); it != m_vec.end(); it++)
		{
			if(it->first == itemId)
			{
				it->second = rating;
				break;
			}
		}
	}
}

void User::withdrawRating(size_t itemId)
{
	if(m_map.find(itemId) != m_map.end())
	{
		m_map.erase(itemId);
		for(vector<std::pair<size_t, float> >::iterator it = m_vec.begin(); it != m_vec.end(); it++)
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

void User::swapItems(size_t a, size_t b)
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









Item::Item(const char* szLeft, const char* szRight, const char* szSubmitter, time_t date, GRand* pRand)
{
	m_left = szLeft;
	m_right = szRight;
	m_submitter = szSubmitter;
	m_date = date;
	m_weights.resize(ITEM_PROFILE_SIZE);
	for(size_t i = 0; i < ITEM_PROFILE_SIZE; i++)
		m_weights[i] = 0.01 * pRand->normal();
}

Item::Item(GDomNode* pNode, GRand* pRand)
{
	m_left = pNode->getString("left");
	m_right = pNode->getString("right");
	m_submitter = pNode->getString("subm");
	m_date = (time_t)pNode->getInt("date");
	m_weights.resize(ITEM_PROFILE_SIZE);
	GDomNode* pWeights = pNode->get("weights");
	GDomListIterator it(pWeights);
	size_t i;
	for(i = 0; i < (size_t)ITEM_PROFILE_SIZE && it.current(); i++)
	{
		m_weights[i] = it.currentDouble();
		it.advance();
	}
	for( ; i < ITEM_PROFILE_SIZE; i++)
		m_weights[i] = LEARNING_RATE * pRand->normal();
}

GDomNode* Item::toDom(GDom* pDoc)
{
	GDomNode* pNode = pDoc->newObj();
	pNode->add(pDoc, "left", m_left.c_str());
	pNode->add(pDoc, "right", m_right.c_str());
	pNode->add(pDoc, "subm", m_submitter.c_str());
	pNode->add(pDoc, "date", (long long)m_date);
	GDomNode* pWeights = pNode->add(pDoc, "weights", pDoc->newList());
	for(size_t i = 0; i < ITEM_PROFILE_SIZE; i++)
		pWeights->add(pDoc, m_weights[i]);
	return pNode;
}








class Sample
{
public:
	size_t user;
	size_t item;
	float rating;

	Sample(size_t _user, size_t _item, float _rating)
	: user(_user), item(_item), rating(_rating)
	{
	}

	Sample(const Sample& other)
	: user(other.user), item(other.item), rating(other.rating)
	{
	}

	virtual ~Sample()
	{
	}

};



NeuralRecommender::NeuralRecommender(GRand& _rand)
: rand(_rand), input_buf(USER_PROFILE_SIZE + ITEM_PROFILE_SIZE)
{
	nn.add(new GBlockLinear(USER_PROFILE_SIZE, COMMON_SIZE));
	nn.concat(new GBlockLinear(ITEM_PROFILE_SIZE, COMMON_SIZE), USER_PROFILE_SIZE);
	nn.add(new GBlockEl(2 * COMMON_SIZE));
	nn.add(new GBlockBiasProduct(COMMON_SIZE));
	nn.add(new GBlockLinear(COMMON_SIZE, PREDICT_SIZE));
	nn.init(rand);
}

NeuralRecommender::~NeuralRecommender()
{
	flush();
}

void NeuralRecommender::flush()
{
	for(vector<User*>::iterator it = m_users.begin(); it != m_users.end(); it++)
		delete(*it);
	m_users.clear();
	m_userMap.clear();
	for(size_t i = 0; i < m_items.size(); i++)
		delete(m_items[i]);
	m_items.clear();
}

GDomNode* NeuralRecommender::serialize(GDom* pDoc)
{
	GDomNode* pNode = pDoc->newObj();

	// Save the users
	GDomNode* pUsers = pNode->add(pDoc, "users", pDoc->newList());
	for(vector<User*>::iterator it = m_users.begin(); it != m_users.end(); it++)
		pUsers->add(pDoc, (*it)->toDom(pDoc));

	// Save the items
	GDomNode* pItems = pNode->add(pDoc, "items", pDoc->newList());
	for(size_t i = 0; i < m_items.size(); i++)
		pItems->add(pDoc, m_items[i]->toDom(pDoc));

	// Save the model
	pNode->add(pDoc, "model", nn.serialize(pDoc));

	return pNode;
}

void NeuralRecommender::deserialize(const GDomNode* pNode)
{
	flush();

	// Load the users
	GDomNode* pUsers = pNode->get("users");
	for(GDomListIterator it(pUsers); it.current(); it.advance())
	{
		User* pUser = User::fromDom(it.current(), rand);
		addUser(pUser);
	}

	// Load the items
	GDomNode* pItems = pNode->get("items");
	GDomListIterator it(pItems);
	m_items.reserve(it.remaining());
	for( ; it.current(); it.advance())
		m_items.push_back(new Item(it.current(), &rand));

	// Load the model
	nn.deserialize(pNode->get("model"), rand);
}

void NeuralRecommender::refine(size_t iters)
{
	// Gather samples from all the users
	std::vector<Sample> samples;
	samples.reserve(1000);
	for(size_t i = 0; i < m_users.size(); i++)
	{
		User* pUser = m_users[i];
		for(size_t j = 0; j < pUser->m_vec.size(); j++)
		{
			std::pair<size_t, float>& p = pUser->m_vec[j];
			samples.push_back(Sample(i, p.first, p.second));
		}
	}

	// Train
	if(samples.size() < 1)
		return;
	GVec target(1);
	GVec inputBlame(USER_PROFILE_SIZE + ITEM_PROFILE_SIZE);
	for(size_t i = 0; i < iters; i++)
	{
		size_t index = rand.next(samples.size());
		Sample& samp = samples[index];
		User* pUser = m_users[samp.user];
		Item& itm = item(samp.item);
		input_buf.copy(0, pUser->personality());
		input_buf.copy(USER_PROFILE_SIZE, itm.weights());
		nn.forwardProp(input_buf);
		target[0] = samp.rating;
		nn.computeBlame(target);
		nn.backpropagate(&inputBlame);
		nn.updateGradient();
		nn.step(LEARNING_RATE, 0.0);
		pUser->personality().addScaled(LEARNING_RATE, inputBlame, 0, USER_PROFILE_SIZE);
		itm.weights().addScaled(LEARNING_RATE, inputBlame, USER_PROFILE_SIZE, ITEM_PROFILE_SIZE);
	}
}

void NeuralRecommender::refineUserOnly(User* pUser, size_t iters)
{
	// Gather samples from the user
	std::vector<Sample> samples;
	samples.reserve(pUser->m_vec.size());
	for(size_t j = 0; j < pUser->m_vec.size(); j++)
	{
		std::pair<size_t, float>& p = pUser->m_vec[j];
		samples.push_back(Sample(0, p.first, p.second));
	}

	// Train
	GVec target(1);
	GVec inputBlame(USER_PROFILE_SIZE + ITEM_PROFILE_SIZE);
	for(size_t i = 0; i < iters; i++)
	{
		size_t index = rand.next(samples.size());
		Sample& samp = samples[index];
		Item& itm = item(samp.item);
		input_buf.copy(0, pUser->personality());
		input_buf.copy(USER_PROFILE_SIZE, itm.weights());
		nn.forwardProp(input_buf);
		target[0] = samp.rating;
		nn.computeBlame(target);
		nn.backpropagate(&inputBlame);
		//nn.updateGradient();
		//nn.step(LEARNING_RATE, 0.0);
		pUser->personality().addScaled(LEARNING_RATE, inputBlame, 0, USER_PROFILE_SIZE);
		//itm.weights().addScaled(LEARNING_RATE, inputBlame, USER_PROFILE_SIZE, ITEM_PROFILE_SIZE);
	}
}

double NeuralRecommender::predictRating(User& user, Item& item)
{
	input_buf.copy(0, user.personality());
	input_buf.copy(USER_PROFILE_SIZE, item.weights());
	GVec& pred = nn.forwardProp(input_buf);
	return pred[0];
}

User* NeuralRecommender::findUser(const char* szUsername)
{
	map<string,User*>::iterator it = m_userMap.find(szUsername);
	if(it == m_userMap.end())
		return nullptr;
	User* pUser = it->second;
	return pUser;
}

User* NeuralRecommender::findOrMakeUser(const char* szUsername)
{
	User* pUser = findUser(szUsername);
	if(!pUser)
	{
		pUser = new User(szUsername, rand);
		addUser(pUser);
	}
	return pUser;
}

void NeuralRecommender::addUser(User* pUser)
{
	m_users.push_back(pUser);
	m_userMap.insert(std::pair<string,User*>(pUser->username(),pUser));
}

void NeuralRecommender::addItem(const char* szLeft, const char* szRight, const char* szUsername)
{
	m_items.push_back(new Item(szLeft, szRight, szUsername, time(NULL), &rand));
}

Item& NeuralRecommender::item(size_t id)
{
	GAssert(id < m_items.size());
	GAssert(m_items[id] != NULL);
	return *m_items[id];
}

void NeuralRecommender::deleteItemAndSwapInLast(size_t itemId)
{
	delete(m_items[itemId]);
	m_items[itemId] = m_items[m_items.size() - 1];
	m_items.pop_back();
}











void Submit::makeSliderScript(ostream& response)
{
	response << "<script language=\"JavaScript\" src=\"/b/style/slider.js\"></script>\n";
	response << "<script language=\"JavaScript\">\n";
	response << "	var A_TPL = { 'b_vertical' : false, 'b_watch': true, 'n_controlWidth': 321, 'n_controlHeight': 22, 'n_sliderWidth': 19, 'n_sliderHeight': 20, 'n_pathLeft' : 1, 'n_pathTop' : 1, 'n_pathLength' : 300, 's_imgControl': 'style/slider_bg.png', 's_imgSlider': 'style/slider_tab.png', 'n_zIndex': 1 }\n";
	response << "</script>\n";
}

void Submit::makeUrlSlider(Server* pServer, Account* pAccount, size_t itemId, ostream& response)
{
	// Compute the rating (or predicted rating if this item has not been rated)
	NeuralRecommender& rec = pServer->recommender();
	Item& item = rec.item(itemId);
	float score;
	User* pUser = pAccount->getUser(pServer->recommender());
	if(!pUser->getRating(itemId, &score))
		score = rec.predictRating(*pUser, item);

	// Display the slider
	response << "<table cellpadding=0 cellspacing=0><tr><td width=300>\n	";
	response << item.left() << "\n";
	response << "</td><td>\n";
	response << "	<input type=checkbox name=\"check_slider" << itemId << "\" id=\"check_slider" << itemId << "\">\n";
	response << "	<input name=\"slider" << itemId << "\" id=\"slider" << itemId << "\" type=\"Text\" size=\"3\">\n";
	response << "</td><td>\n";
	response << "<script language=\"JavaScript\">\n";
	response << "	var A_INIT1 = { 's_checkname': 'check_slider" << itemId << "', 's_name': 'slider" << itemId << "', 'n_minValue' : -1, 'n_maxValue' : 1, 'n_value' : " << score << ", 'n_step' : 0.01 }\n";
	response << "	new slider(A_INIT1, A_TPL);\n";
	response << "</script>\n";
	response << "</td><td width=300>\n";
	response << item.right() << "\n";
	response << "</td></tr></table>\n";
}

void Submit::pageFaq(Server* pServer, GDynamicPageSession* pSession, ostream& response)
{
	response << pServer->cache("faq.html");
}

void Submit::pageSurvey(Server* pServer, GDynamicPageSession* pSession, ostream& response)
{
	Account* pAccount = getAccount(pSession);
	NeuralRecommender& rec = pServer->recommender();
	User* pUser = pAccount->getUser(rec);
	if(pSession->paramsLen() > 0)
	{
		// Do the action
		GHttpParamParser params(pSession->params());
		const char* szAction = params.find("action");
		if(!szAction)
		{
		}
		else if(_stricmp(szAction, "add") == 0)
		{
			const char* szLeft = params.find("pearl");
			if(!szLeft)
			{
				response << "[You're supposed to type something before you press 'Submit'.]<br>\n";
				response << params.to_str(true);
			}
			else
			{
				rec.addItem(szLeft, "", pAccount->username());
				cout << pAccount->username() << " added this pearl: " << szLeft << "\n";
				pServer->saveState();
				response << "Thank you for the pearl! I'll try to rebut it as soon as I find the time. If you'd like, you may evaluate these pearls and my rebuttals.<br>\n";
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
					set<size_t>::iterator tmp = checks.find(itemId);
					if(tmp != checks.end())
					{
						float score = (float)atof(it->second);
						if(score >= -1.0f && score <= 1.0f)
						{
							pUser->updateRating(itemId, score);
							response << "[Rating recorded. Thank you.]<br>\n";
						}
						else
							response << "[the rating of " << score << " is out of range.]<br>\n";
					}
				}
			}

			// Do some training
			rec.refineUserOnly(pUser, ON_RATE_TRAINING_ITERS); // trains just personalities
			rec.refine(ON_RATE_TRAINING_ITERS); // trains both personalities and weights
			pServer->saveState();
		}
	}

	// Display the topic
	makeSliderScript(response);
	response << "<h2>Pearls</h2>\n";
	response << "<form name=\"formname\" method=\"post\">\n";
	response << "	<input type=\"hidden\" name=\"action\" value=\"rate\" />\n";

	// Random picks
	size_t* pIndexes = new size_t[rec.items().size()];
	Holder<size_t> hIndexes(pIndexes);
	GIndexVec::makeIndexVec(pIndexes, rec.items().size());
	GIndexVec::shuffle(pIndexes, rec.items().size(), pServer->prng());
	size_t sliderCount = 0;
	for(size_t i = 0; i < rec.items().size(); i++)
	{
		if(sliderCount >= 8)
			break;
		size_t itemId = pIndexes[i];
		float rating;
		if(pUser->getRating(itemId, &rating))
			continue;
		if(sliderCount == 0)
		{
			response << "<h3>A few statements for your evaluation:</h3>\n";
			response << "<p>It is okay to skip statements you find ambiguous, invasive, or uninteresting. For your convenience, the sliders have been set to reflect predictions of your opinions. As you express more opinions, these predictions should improve.</p>\n";
		}
		makeUrlSlider(pServer, pAccount, itemId, response);
		response << "<br><br>\n";
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
		response << "Thank you. You have rated all ";
		response << to_str(rec.items().size());
		response << " pearls that have been submitted so far.<br><br>\n";
	}

	response << "</form><br><br>\n\n";

	// The choices links at the bottom of the page
	response << "<a href=\"/b/submit\">Submit a new statement</a>";
	response << "&nbsp;&nbsp;&nbsp;&nbsp;";
	response << "<a href=\"/b/survey?topic=-1\">Change topic</a>\n";
	response << "&nbsp;&nbsp;&nbsp;&nbsp;";
	response << "<a href=\"/b/update\">My opinions</a>\n";
	response << "&nbsp;&nbsp;&nbsp;&nbsp;";
	response << "<a href=\"/b/stats\">Vizualize</a>\n";

/*
	response << "Stats:<br>\n";
	response << "Total Number of users: " << pServer->accounts().size() << "<br>\n";
	response << "Number of items in this topic: " << pCurrentTopic->size() << "<br>\n";
	std::map<size_t, float>* pMap = currentTopic < pAccount->ratings().size() ? pAccount->ratings()[currentTopic] : NULL;
	response << "Number of items you have rated in this topic: " << (pMap ? pMap->size() : (size_t)0) << "<br>\n<br>\n";
*/
}

void Submit::pageRespond(Server* pServer, GDynamicPageSession* pSession, ostream& response)
{
	// Check access privileges
	Account* pAccount = getAccount(pSession);
	NeuralRecommender& rec = pServer->recommender();
	if(!pAccount->isAdmin() && !pAccount->isAssistant())
	{
		response << "Sorry, you must be an admin or assistant to access this page.";
		return;
	}

	// Process params
	if(pSession->paramsLen() > 0)
	{
		GHttpParamParser params(pSession->params());
		auto map = params.map();
		std::map<const char*, const char*, GClasses::strComp>::iterator it = map.end();
		while(it != map.begin())
		{
			it--;
			const char* key = it->first;
			const char* val = it->second;
			if(strncmp(key, "del_", 4) == 0)
			{
				size_t index = atoi(key + 4);
				const std::vector<User*>& users = pServer->recommender().users();
				if(index >= rec.items().size())
					response << "[invalid item index]<br><br>\n";
				else
				{
					cout << "Deleted pearl: " << rec.item(index).left() << " / " << rec.item(index).right() << "\n";
					for(std::vector<User*>::const_iterator it2 = users.begin(); it2 != users.end(); it2++)
					{
						(*it2)->withdrawRating(index);
						(*it2)->swapItems(index, rec.items().size() - 1);
					}
					rec.deleteItemAndSwapInLast(index);
					//response << "[Item successfully deleted]<br><br>\n";
				}
			}
			else if(strncmp(key, "re_", 3) == 0)
			{
				size_t index = atoi(key + 3);
				if(index >= rec.items().size())
					response << "[invalid item index]<br><br>\n";
				else
				{
					rec.item(index).setRight(val);
					cout << "Added response: " << rec.item(index).left() << " / " << rec.item(index).right() << "\n";
				}
			}
			else
			{
				response << "[unrecognized action: " << key << "]<br><br>\n";
			}
		}
	}

	// Form to delete statements or respond to them
	response << "<h2>Delete Statements</h2>\n\n";
	response << "<form name=\"responses\" id=\"responses\" method=\"post\">";
	response << "<table>\n";
	for(size_t i = 0; i < rec.items().size(); i++)
	{
		Item& itm = rec.item(i);
		std::string s = itm.right();
		if(s.length() > 0)
			continue;
		response << "<tr><td>";
		response << "Delete?<br><input type=\"checkbox\" name=\"del_" << to_str(i) << "\" value=\"true\">";
		response << "</td><td>" << itm.left() << "</td><td>";
		response << "<textarea rows=\"12\" cols=\"40\" maxlength=\"800\" id=\"re_" << to_str(i) << "\" name=\"re_" << to_str(i) << "\" form=\"responses\"></textarea>";
		response << "</td></tr>\n";
	}
	response << "</table>\n";
	response << "<br>\n";
	response << "<input type=\"submit\" value=\"Submit\"></form>";
	response << "</form><br><br>\n\n";
}

void Submit::pageNewSurveyItem(Server* pServer, GDynamicPageSession* pSession, ostream& response)
{
	response << pServer->cache("submit.html");
}
/*
double Submit::computeVariance(double* pCentroid, Topic& topic, User** pUsers, size_t accCount)
{
	// Compute the centroid
	GVecWrapper vw(pCentroid, topic.size());
	vw.fill(0.0);
	User** pUs = pUsers;
	for(size_t j = 0; j < accCount; j++)
	{
		double* pC = pCentroid;
		for(size_t i = 0; i < topic.size(); i++)
		{
			float rating;
			if(!(*pUs)->getRating(i, &rating))
				rating = (*pUs)->predictRating(topic.item(i));
			(*pC) += rating;
			pC++;
		}
		pUs++;
	}
	double t = 1.0 / std::max((size_t)1, accCount);
	for(size_t i = 0; i < topic.size(); i++)
		pCentroid[i] *= t;

	// Measure the sum-squared error with the centroid
	double sse = 0.0;
	pUs = pUsers;
	for(size_t j = 0; j < accCount; j++)
	{
		double* pC = pCentroid;
		for(size_t i = 0; i < topic.size(); i++)
		{
			float rating;
			if(!(*pUs)->getRating(i, &rating))
				rating = (*pUs)->predictRating(topic.item(i));
			double d = *pC - rating;
			sse += (d * d);
			pC++;
		}
		pUs++;
	}
	return sse;
}
*/
size_t Submit::divideAccounts(Server* pServer, User** pUsers, size_t accCount, size_t itm)
{
	NeuralRecommender& rec = pServer->recommender();
	size_t head = 0;
	size_t tail = accCount;
	while(tail > head)
	{
		float rating;
		if(!pUsers[head]->getRating(itm, &rating))
			rating = rec.predictRating(*pUsers[head], rec.item(itm));
		if(rating > 0.0)
		{
			tail--;
			std::swap(pUsers[head], pUsers[tail]);
		}
		else
			head++;
	}
	GAssert(head == tail);
	return head;
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
	ItemStats(Item& itm, size_t itemId, User** pUsers, size_t accCount)
	: m_item(itm), m_id(itemId), m_agree(0), m_uncertain(0), m_disagree(0), m_agg(0), m_dis(0)
	{
		// Compute the mean
		User** Us = pUsers;
		float rating;
		double mean = 0.0;
		size_t count = 0;
		for(size_t i = 0; i < accCount; i++)
		{
			if((*Us)->getRating(itemId, &rating))
			{
				mean += rating;
				count++;
				if(rating < -0.333333)
					m_disagree++;
				else if(rating > 0.333333)
					m_agree++;
				else
					m_uncertain++;
				if(rating < 0.5)
					m_dis++;
				else
					m_agg++;
			}
			Us++;
		}
		mean /= count;

		// Compute the deviation
		Us = pUsers;
		double var = 0.0;
		for(size_t i = 0; i < accCount; i++)
		{
			if((*Us)->getRating(itemId, &rating))
			{
				double d = mean - rating;
				var += (d * d);
			}
			Us++;
		}
		m_deviation = sqrt(var / std::max((size_t)1, count));
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




void Submit::makeTree(Server* pServer, GBitTable& bt, User** pUsers, size_t accCount, ostream& response, vector<char>& prefix, int type)
{
	// Try splitting on each of the remaining statements
	size_t best = (size_t)-1;
	double mostCont = 0.0;
	NeuralRecommender& rec = pServer->recommender();
	double* pCentroid = new double[rec.items().size()];
	ArrayHolder<double> hCentroid(pCentroid);
	size_t tieCount = 0;
	for(size_t i = 0; i < rec.items().size(); i++)
	{
		if(bt.bit(i))
			continue;
		ItemStats is(rec.item(i), i, pUsers, accCount);
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
				if(pServer->prng()->next(tieCount + 1) == 0)
					best = i;
			}
		}
	}

	if(best != (size_t)-1)
	{
		// Divide on the best statement
		size_t firstHalfSize = divideAccounts(pServer, pUsers, accCount, best);
		bt.set(best);

		// Recurse
		prefix.push_back(' ');
		if(type >= 0) prefix.push_back(' '); else prefix.push_back('|');
		prefix.push_back(' ');
		prefix.push_back(' ');
		makeTree(pServer, bt, pUsers, firstHalfSize, response, prefix, 1);

		for(vector<char>::iterator it = prefix.begin(); it != prefix.end(); it++)
			response << *it;
		response << "/ (" << rec.item(best).left() << ")\n";
		prefix.pop_back(); prefix.pop_back(); prefix.pop_back(); prefix.pop_back();
		for(vector<char>::iterator it = prefix.begin(); it != prefix.end(); it++)
			response << *it;
		if(type == 0)
			response << "---(\n";
		else
			response << " +-(\n";
		prefix.push_back(' ');
		if(type <= 0) prefix.push_back(' '); else prefix.push_back('|');
		prefix.push_back(' ');
		prefix.push_back(' ');
		for(vector<char>::iterator it = prefix.begin(); it != prefix.end(); it++)
			response << *it;
		response << "\\ (" << rec.item(best).right() << ")\n";

		makeTree(pServer, bt, pUsers + firstHalfSize, accCount - firstHalfSize, response, prefix, -1);
		prefix.pop_back(); prefix.pop_back(); prefix.pop_back(); prefix.pop_back();

		bt.unset(best);
	}
	else
	{
		for(size_t j = 0; j < accCount; j++)
		{
			for(vector<char>::iterator it = prefix.begin(); it != prefix.end(); it++)
				response << *it;
			response << " +-&gt;<a href=\"/stats?user=" << pUsers[j]->username() << "\">";
			response << pUsers[j]->username() << "</a>\n";
		}
	}
}

string to_string(double d, int decimal_places)
{
	double p = pow(10, decimal_places);
	d *= p;
	d += 0.5;
	d = floor(d);
	d /= p;
	return to_str(d);
}


void Submit::makeItemBody(GDynamicPageSession* pSession, ostream& response, size_t itemId, Item& item, User** pUsers, size_t accCount)
{
	std::multimap<double,User*> mm;
	while(accCount > 0)
	{
		float rating;
		if((*pUsers)->getRating(itemId, &rating))
			mm.insert(std::pair<double,User*>(rating,*pUsers));
		accCount--;
		pUsers++;
	}

	// First show all the left-leaning answers from the sorted map
	response << "<h3>" << item.left() << "</h3>\n";
	response << "<table>\n";
	size_t hh = 0;
	for(std::multimap<double,User*>::iterator it = mm.begin(); it != mm.end(); it++)
	{
		if(hh == 0 && it->first > -0.3333) // When they cross into being uncertain
		{
			response << "</table>\n<h3>Uncertain</h3>\n<table>\n";
			hh++;
		}
		if(hh == 1 && it->first > 0.3333) // when they cross into being right-learning
		{
			response << "</table>\n<h3>" << item.right() << "</h3>\n<table>\n";
			hh++;
		}
		response << "<tr><td>" << it->second->username() << "</td><td>" << to_string(it->first, 2) << "</td></tr>\n";
	}

	if(hh == 0) // in the obscure case where we haven't found any uncertain people yet...
	{
		response << "</table>\n<h3>Uncertain</h3>\n<table>\n";
		hh++;
	}
	if(hh == 1) // in the obscure case where we haven't found any right-leaning people yet...
	{
		response << "</table>\n<h3>" << item.right() << "</h3>\n<table>\n";
		hh++;
	}
	response << "</table>\n";
}

void Submit::makeUserBody(GDynamicPageSession* pSession, ostream& response, User* pA, User* pB, NeuralRecommender& rec)
{
	std::multimap<float,size_t> m;
	float rA = 0.0f;
	float rB = 0.0f;
	for(size_t i = 0; i < rec.items().size(); i++)
	{
		if(pA->getRating(i, &rA))
		{
			if(pB->getRating(i, &rB))
				m.insert(std::pair<float,size_t>(-std::abs(rB - rA), i));
		}
	}
	if(m.size() == 0)
	{
		response << "You have no ratings in common.<br><br>\n";
		return;
	}
	response << "<table><tr><td><u>" << pA->username() << "</u></td><td><u>" << pB->username() << "</u></td><td><u>product</u></td><td><u>Left</u></td><td><u>Right</u></td></tr>\n";
	for(std::multimap<float,size_t>::iterator it = m.begin(); it != m.end(); it++)
	{
		pA->getRating(it->second, &rA);
		pB->getRating(it->second, &rB);
		response << "<tr><td>" << to_str(0.1 * floor(10 * rA)) << "</td><td>";
		response << to_str(0.1 * floor(10 * rB)) << "</td><td>";
		response << to_str(0.1 * floor(10 * rA * rB)) << "</td><td>";
		response << rec.item(it->second).left() << "</td><td>";
		response << rec.item(it->second).right() << "</td></tr>\n";
	}
	response << "</table>\n";
}

void Submit::pageStats(Server* pServer, GDynamicPageSession* pSession, ostream& response)
{
	// Get the topic
	Account* pAccount = getAccount(pSession);
	User* pUser = pAccount->getUser(pServer->recommender());
	NeuralRecommender& rec = pServer->recommender();

	// Copy the account pointers into an array
	const std::vector<User*>& users = pServer->recommender().users();
	User** pAccs = new User*[users.size()];
	ArrayHolder<User*> hAccs(pAccs);
	User** pAc = pAccs;
	size_t accountCount = 0;
	for(std::vector<User*>::const_iterator it = users.begin(); it != users.end(); it++)
	{
		*(pAc++) = *it;
		accountCount++;
	}

	// Check the params
	GHttpParamParser params(pSession->params());
	const char* szItemId = params.find("item");
	if(szItemId)
	{
		size_t itemId = atoi(szItemId);
		makeItemBody(pSession, response, itemId, rec.item(itemId), pAccs, accountCount);
		return;
	}
	const char* szOtherUser = params.find("user");
	if(szOtherUser)
	{
		User* pOther = rec.findUser(szOtherUser);
		if(!pOther)
			response << "[No such user]<br><br>\n";
		else
			makeUserBody(pSession, response, pUser, pOther, rec);
		return;
	}

	GBitTable bt(rec.items().size());
	vector<char> prefix;
	response << "This ascii-art tree was constructed by dividing on the most controversial statements within each branch.\n";
	response << "This tree is arranged such that the ascending branches lead to the usernames of people who support the left statement, and the descending branches lead to the usernames of people who support the right statement.\n";
	response << "(In cases lacking response, predictions were used to make any judgement calls necessary to construct this tree, so some placements may be estimated.)\n";
	response << "<br><br>\n";
	response << "<pre>\n";
	makeTree(pServer, bt, pAccs, accountCount, response, prefix, 0);
	response << "</pre>\n";
	response << "<br><br>\n";

	// Make a table of items sorted by controversy
	std::vector<ItemStats*> items;
	for(size_t i = 0; i < rec.items().size(); i++)
		items.push_back(new ItemStats(rec.item(i), i, pAccs, accountCount));
	std::sort(items.begin(), items.end(), ItemStats::comparer);
	response << "<table><tr><td><b><i><u>Statement</u></i></b></td><td><b><i><u>Lean Left</u></i></b></td><td><b><i><u>Uncertain</u></i></b></td><td><b><i><u>Lean Right</u></i></b></td><td><b><i><u>Controversy</u></i></b></td></tr>\n";
	for(vector<ItemStats*>::iterator it = items.begin(); it != items.end(); it++)
	{
		response << "<tr><td>";
		response << "<a href=\"/stats?item=" << to_str((*it)->id()) << "\">" << (*it)->item().left() << " / " << (*it)->item().right() << "</a>";
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
	response << "<a href=\"/b/submit\">Submit a new question</a>";
	response << "&nbsp;&nbsp;&nbsp;&nbsp;";
	response << "<a href=\"/b/update\">My opinions</a>\n";
	response << "&nbsp;&nbsp;&nbsp;&nbsp;";
	response << "<a href=\"/b/survey\">" << "Survey</a>\n";
}

void Submit::plotUsers(Server* pServer, GDynamicPageSession* pSession, ostream& response)
{
	pSession->connection()->setContentType("image/svg+xml");
	GSVG svg(800, 800);

	const std::vector<User*>& users = pServer->recommender().users();
	double xmin = 0;
	double ymin = 0;
	double xmax = 0;
	double ymax = 0;
	for(std::vector<User*>::const_iterator it = users.begin(); it != users.end(); it++)
	{
		User* pUser = *it;
		GVec& profile = pUser->personality();
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
	for(std::vector<User*>::const_iterator it = users.begin(); it != users.end(); it++)
	{
		User* pUser = *it;
		GVec& profile = pUser->personality();
		svg.dot(profile[1], profile[2], 0.75, 0x008080);
		svg.text(profile[1], profile[2], (*it)->username(), 0.75);
	}
	svg.print(response);
}

void Submit::plotItems(Server* pServer, GDynamicPageSession* pSession, ostream& response)
{
	// Get the topic
	NeuralRecommender& rec = pServer->recommender();

	pSession->connection()->setContentType("image/svg+xml");
	GSVG svg(800, 800);

	double xmin = 0;
	double ymin = 0;
	double xmax = 0;
	double ymax = 0;
	for(size_t i = 0; i < rec.items().size(); i++)
	{
		Item& item = rec.item(i);
		GVec& weights = item.weights();
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
	for(size_t i = 0; i < rec.items().size(); i++)
	{
		Item& item = rec.item(i);
		const char* szTitle = item.left();
		GVec& weights = item.weights();
		svg.dot(weights[1], weights[2], 0.75, 0x008080);
		svg.text(weights[1], weights[2], szTitle, 0.75);
	}
	svg.print(response);
}

void Submit::pageUpdateResponses(Server* pServer, GDynamicPageSession* pSession, ostream& response)
{
	Account* pAccount = getAccount(pSession);
	User* pUser = pAccount->getUser(pServer->recommender());
	makeSliderScript(response);

	// Display the topic
	response << "<h2>Pearls</h2>\n";

	// Display the items you have rated
	vector<pair<size_t, float> >& v = pUser->m_vec;
	if(v.size() > 0)
	{
		response << "<h3>Your opinions</h3>\n";
		response << "<form name=\"formname\" action=\"/survey\" method=\"post\">\n";
		response << "	<input type=\"hidden\" name=\"action\" value=\"rate\" />\n";
		UpdateComparer comparer;
		std::sort(v.begin(), v.end(), comparer);
		for(vector<pair<size_t, float> >::iterator it = v.begin(); it != v.end(); it++)
			makeUrlSlider(pServer, pAccount, it->first, response);
		response << "<br><table><tr><td width=330></td><td>";
		response << "<input type=\"submit\" value=\"Update ratings\">";
		response << "</td></tr><tr><td></td><td>";
		response << "(Only checked items will be updated.)";
		response << "</td></tr></table>\n";
		response << "</form><br><br>\n\n";
	}
	else
		response << "You have not yet rated anything<br><br>\n";

	// The choices links at the bottom of the page
	response << "<a href=\"/submit\">Submit a new item</a>";
	response << "&nbsp;&nbsp;&nbsp;&nbsp;";
	response << "<a href=\"/survey?topic=-1\">Change topic</a>\n";
	response << "&nbsp;&nbsp;&nbsp;&nbsp;";
	response << "<a href=\"/survey\">" << "Survey</a>\n";
	response << "&nbsp;&nbsp;&nbsp;&nbsp;";
	response << "<a href=\"/stats\">Vizualize</a>\n";
}
