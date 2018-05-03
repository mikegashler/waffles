#include "recommender.h"
#include <GClasses/GRand.h>
#include <GClasses/GDom.h>
#include <map>
#include <cmath>

using std::map;
using std::string;
using std::vector;
using std::cerr;
using std::cout;

User::User(string username, GRand& rand)
: m_username(username)
{
	m_personality.resize(PERSONALITY_DIMS);
	for(size_t i = 0; i < PERSONALITY_DIMS; i++)
		m_personality[i] = LEARNING_RATE * rand.normal();
}

User::~User()
{
	for(size_t i = 0; i < m_ratings.size(); i++)
		delete(m_ratings[i]);
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
		for(i = 0; i < (size_t)PERSONALITY_DIMS && it.current(); i++)
		{
			pUser->m_personality[i] = std::max(-1.0, std::min(1.0, it.currentDouble()));
			it.advance();
		}
		for( ; i < PERSONALITY_DIMS; i++)
			pUser->m_personality[i] = 0.02 * rand.normal();
	}

	// Deserialize the ratings
	GDomNode* pRatings = pNode->get("ratings");
	size_t topic = 0;
	for(GDomListIterator it(pRatings); it.current(); it.advance())
	{
		ptrdiff_t j = (ptrdiff_t)it.currentInt();
		if(j < 0)
			topic = (size_t)(-j - 1);
		else
		{
			it.advance();
			if(it.current())
				pUser->addRating(topic, (size_t)j, (float)it.currentDouble());
		}
	}
	return pUser;
}

GDomNode* User::toDom(GDom* pDoc)
{
	GDomNode* pUser = pDoc->newObj();
	pUser->add(pDoc, "username", m_username.c_str());

	// Serialize the personality vector
	GDomNode* pPersonality = pUser->add(pDoc, "pers", pDoc->newList());
	for(size_t i = 0; i < PERSONALITY_DIMS; i++)
		pPersonality->add(pDoc, m_personality[i]);

	// Serialize the ratings
	size_t count = 0;
	for(vector<Ratings*>::iterator i = m_ratings.begin(); i != m_ratings.end(); i++)
	{
		map<size_t, float>& map = (*i)->m_map;
		if(map.size() > 0)
			count += (1 + 2 * map.size());
	}
	GDomNode* pRatings = pUser->add(pDoc, "ratings", pDoc->newList());
	size_t j = 0;
	for(vector<Ratings*>::iterator i = m_ratings.begin(); i != m_ratings.end(); i++)
	{
		map<size_t, float>& m = (*i)->m_map;
		if(m.size() > 0)
		{
			ptrdiff_t r = -1;
			r -= (j++);
			GAssert(r < 0);
			pRatings->add(pDoc, (long long)r);
			for(map<size_t,float>::iterator it = m.begin(); it != m.end(); it++)
			{
				pRatings->add(pDoc, it->first);
				pRatings->add(pDoc, it->second);
			}
		}
	}

	return pUser;
}

void User::addRating(size_t topic, size_t itemId, float rating)
{
	GAssert(rating >= -1.0f && rating <= 1.0f);
	while(topic >= m_ratings.size())
		m_ratings.push_back(new Ratings());
	m_ratings[topic]->addRating(itemId, rating);
}

void User::updateRating(size_t topic, size_t itemId, float rating)
{
	GAssert(rating >= -1.0f && rating <= 1.0f);
	while(topic >= m_ratings.size())
		m_ratings.push_back(new Ratings());
	m_ratings[topic]->updateRating(itemId, rating);
}

void User::withdrawRating(size_t topic, size_t itemId)
{
	if(topic < m_ratings.size())
		m_ratings[topic]->withdrawRating(itemId);
}

void User::swapItems(size_t topic, size_t a, size_t b)
{
	if(topic < m_ratings.size())
		m_ratings[topic]->swapItems(a, b);
}

float User::predictRating(Item& item)
{
	return item.predictRating(m_personality);
}

bool User::getRating(size_t topic, size_t itemId, float* pOutRating)
{
	if(topic >= m_ratings.size())
		return false;
	map<size_t, float>& m = m_ratings[topic]->m_map;
	map<size_t, float>::iterator it = m.find(itemId);
	if(it == m.end())
		return false;
	*pOutRating = it->second;
	return true;
}











Item::Item(const char* szLeft, const char* szRight, const char* szSubmitter, time_t date, GRand* pRand)
{
	m_left = szLeft;
	m_right = szRight;
	m_submitter = szSubmitter;
	m_date = date;
	m_weights.resize(PERSONALITY_DIMS);
	for(size_t i = 0; i < PERSONALITY_DIMS; i++)
		m_weights[i] = 0.01 * pRand->normal();
}

Item::Item(GDomNode* pNode, GRand* pRand)
{
	m_left = pNode->getString("left");
	m_right = pNode->getString("right");
	m_submitter = pNode->getString("subm");
	m_date = (time_t)pNode->getInt("date");
	m_weights.resize(PERSONALITY_DIMS);
	GDomNode* pWeights = pNode->get("weights");
	GDomListIterator it(pWeights);
	size_t i;
	for(i = 0; i < (size_t)PERSONALITY_DIMS && it.current(); i++)
	{
		m_weights[i] = it.currentDouble();
		it.advance();
	}
	for( ; i < PERSONALITY_DIMS; i++)
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
	for(size_t i = 0; i < PERSONALITY_DIMS; i++)
		pWeights->add(pDoc, m_weights[i]);
	return pNode;
}

double Item::predictRating(const vector<double>& personality) const
{
	vector<double>::const_iterator itW = m_weights.begin();
	vector<double>::const_iterator itP = personality.begin();

	// Add the bias weights
	double d = *(itW++) + *(itP++);

	// Multiply the weight vector by the personality vector
	while(itW != m_weights.end())
		d += *(itW++) * *(itP++);

	return d;
}

// This method adjusts the weights in the opposite direction of the gradient of
// the squared-error with respect to the weights.
void Item::trainWeights(double target, double learningRate, const vector<double>& personality)
{
	GAssert(target >= -1.0 && target <= 1.0);
	double prediction = predictRating(personality);
	double err = learningRate * (target - prediction);
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
void Item::trainPersonality(double target, double learningRate, vector<double>& personality) const
{
	GAssert(target >= -1.0 && target <= 1.0);
	double prediction = predictRating(personality);
	double err = learningRate * (target - prediction);
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













Topic::Topic(const char* szDescr)
: m_descr(szDescr)
{
}

Topic::~Topic()
{
	flushItems();
}

void Topic::flushItems()
{
	for(size_t i = 0; i < m_items.size(); i++)
		delete(m_items[i]);
	m_items.clear();
}

Item& Topic::item(size_t id)
{
	GAssert(id < m_items.size());
	GAssert(m_items[id] != NULL);
	return *m_items[id];
}

void Topic::addItem(const char* szLeft, const char* szRight, const char* szUsername, time_t date, GRand* pRand)
{
	m_items.push_back(new Item(szLeft, szRight, szUsername, date, pRand));
}

GDomNode* Topic::toDom(GDom* pDoc)
{
	GDomNode* pNode = pDoc->newObj();
	pNode->add(pDoc, "descr", m_descr.c_str());
	GDomNode* pItems = pNode->add(pDoc, "items", pDoc->newList());
	for(size_t i = 0; i < m_items.size(); i++)
		pItems->add(pDoc, m_items[i]->toDom(pDoc));
	return pNode;
}

// static
Topic* Topic::fromDom(GDomNode* pNode, GRand* pRand)
{
	const char* szDescr = pNode->getString("descr");
	Topic* pNewTopic = new Topic(szDescr);
	GDomNode* pItems = pNode->get("items");
	GDomListIterator it(pItems);
	pNewTopic->m_items.reserve(it.remaining());
	for( ; it.current(); it.advance())
		pNewTopic->m_items.push_back(new Item(it.current(), pRand));
	return pNewTopic;
}

void Topic::deleteItemAndSwapInLast(size_t itemId)
{
	delete(m_items[itemId]);
	m_items[itemId] = m_items[m_items.size() - 1];
	m_items.pop_back();
}
















void Ratings::addRating(size_t itemId, float rating)
{
	m_map[itemId] = rating;
	m_vec.push_back(std::make_pair(itemId, rating));
}

void Ratings::updateRating(size_t itemId, float rating)
{
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

void Ratings::withdrawRating(size_t itemId)
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

void Ratings::swapItems(size_t a, size_t b)
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












Recommender::Recommender(GRand& rand)
: m_rand(rand)
{
}

Recommender::~Recommender()
{
	flush();
}

void Recommender::flush()
{
	for(vector<User*>::iterator it = m_users.begin(); it != m_users.end(); it++)
		delete(*it);
	m_users.clear();
	m_userMap.clear();
	for(size_t i = 0; i < m_topics.size(); i++)
		delete(m_topics[i]);
	m_topics.clear();
}

GDomNode* Recommender::serialize(GDom* pDoc)
{
	GDomNode* pNode = pDoc->newObj();

	// Save the topcs
	GDomNode* pTopics = pNode->add(pDoc, "topics", pDoc->newList());
	for(size_t i = 0; i < m_topics.size(); i++)
		pTopics->add(pDoc, m_topics[i]->toDom(pDoc));

	// Save the users
	GDomNode* pUsers = pNode->add(pDoc, "users", pDoc->newList());
	for(vector<User*>::iterator it = m_users.begin(); it != m_users.end(); it++)
		pUsers->add(pDoc, (*it)->toDom(pDoc));

	return pNode;
}

void Recommender::deserialize(const GDomNode* pNode)
{
	flush();

	// Load the topics
	GDomNode* pTopics = pNode->get("topics");
	for(GDomListIterator it(pTopics); it.current(); it.advance())
	{
		Topic* pTopic = Topic::fromDom(it.current(), &m_rand);
		m_topics.push_back(pTopic);
	}

	// Load the users
	GDomNode* pUsers = pNode->get("users");
	for(GDomListIterator it(pUsers); it.current(); it.advance())
	{
		User* pUser = User::fromDom(it.current(), m_rand);
		addUser(pUser);
	}
}

User* Recommender::findUser(const char* szUsername)
{
	map<string,User*>::iterator it = m_userMap.find(szUsername);
	if(it == m_userMap.end())
		return nullptr;
	User* pUser = it->second;
	return pUser;
}

User* Recommender::findOrMakeUser(const char* szUsername)
{
	User* pUser = findUser(szUsername);
	if(!pUser)
	{
		pUser = new User(szUsername, m_rand);
		addUser(pUser);
	}
	return pUser;
}

void Recommender::addUser(User* pUser)
{
	m_users.push_back(pUser);
	m_userMap.insert(std::pair<string,User*>(pUser->username(),pUser));
}

void Recommender::addItem(size_t topic, const char* szLeft, const char* szRight, const char* szUsername)
{
	if(topic >= m_topics.size())
	{
		cerr << "Topic ID out of range\n";
		return;
	}
	m_topics[topic]->addItem(szLeft, szRight, szUsername, time(NULL), &m_rand);
}

void Recommender::proposeTopic(const char* username, const char* szDescr)
{
	cout << "The following new topic was proposed by " << username << "\n";
	cout << "	" << szDescr << "\n";
}

void Recommender::newTopic(const char* szDescr)
{
	m_topics.push_back(new Topic(szDescr));
}

void Recommender::refineModel(size_t topic, size_t iters)
{
	if(m_users.size() < 1)
		return;

	// Do some training
	Topic* pCurrentTopic = m_topics[topic];
	for(size_t i = 0; i < iters; i++)
	{
		User* pRandomUser = m_users[m_rand.next(m_users.size())];
		if(pRandomUser->ratings().size() > topic)
		{
			std::vector<std::pair<size_t, float> >& v = pRandomUser->ratings()[topic]->m_vec;
			if(v.size() > 0)
			{
				vector<double>& personality = pRandomUser->personality();
				size_t index = (size_t)m_rand.next(v.size());
				Item& item = pCurrentTopic->item(v[index].first);
				double target = (double)v[index].second;
				GAssert(target >= -1.0 && target <= 1.0);
				for(size_t j = 0; j < item.weights().size(); j++)
					item.weights()[j] *= (1.0 - LEARNING_RATE * REGULARIZATION_TERM);
				for(size_t j = 0; j < personality.size(); j++)
					personality[j] *= (1.0 - LEARNING_RATE * REGULARIZATION_TERM);
				item.trainWeights(target, LEARNING_RATE, personality);
				item.trainPersonality(target, LEARNING_RATE, personality);
			}
		}
	}
}

void Recommender::refinePersonality(User* pUser, size_t topic, size_t iters)
{
	// Train the personality a little bit
	if(topic >= pUser->ratings().size() || topic >= m_topics.size())
		return;
	Topic* pCurrentTopic = m_topics[topic];
	std::vector<std::pair<size_t, float> >& v = pUser->ratings()[topic]->m_vec;
	if(v.size() > 0)
	{
		for(size_t i = 0; i < iters; i++)
		{
			size_t index = (size_t)m_rand.next(v.size());
			Item& item = pCurrentTopic->item(v[index].first);
			item.trainPersonality((double)v[index].second, LEARNING_RATE, pUser->personality());
		}
	}
}



