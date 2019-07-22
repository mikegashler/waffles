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

#ifndef RECOMMENDER
#define RECOMMENDER


#include <GClasses/GBitTable.h>
#include <GClasses/GDynamicPage.h>
#include <GClasses/GVec.h>
#include <GClasses/GNeuralNet.h>
#include <string>
#include <vector>
#include <map>


namespace GClasses
{
	class GRand;
	class GDom;
	class GDomNode;
}
class UserRatings;
class Item;
class Account;
class Server;
using namespace GClasses;


#define USER_PROFILE_SIZE 6
#define ITEM_PROFILE_SIZE 6
#define COMMON_SIZE 12
#define PREDICT_SIZE 1 //3
#define ON_RATE_TRAINING_ITERS 50000
#define ON_TRAIN_TRAINING_ITERS 1000000
#define ON_STARTUP_TRAINING_ITERS 250000
#define LEARNING_RATE 0.001
#define REGULARIZATION_TERM 0.00001


class User
{
protected:
	std::string m_username;
	GVec m_personality; // This vector represents the user with respect to our model. That is, given the user's personality vector, our model should be able to predict the ratings of this user with some accuracy.

public:
	std::map<size_t, float> m_map; // Map from item id to rating
	std::vector<std::pair<size_t, float> > m_vec; // List of item id and rating pairs

	User(std::string username, GRand& rand);
	~User();

	GVec& personality() { return m_personality; }

	static User* fromDom(GDomNode* pNode, GRand& rand);

	GDomNode* toDom(GDom* pDoc);

	const char* username() { return m_username.c_str(); }
	bool getRating(size_t itemId, float* pOutRating);

	void addRating(size_t itemId, float rating);
	void updateRating(size_t itemId, float rating);
	void withdrawRating(size_t itemId);
	void swapItems(size_t a, size_t b);
};


class Item
{
protected:
	std::string m_left;
	std::string m_right;
	std::string m_submitter;
	time_t m_date; // the date this item was submitted
	GVec m_weights; // used to predict the rating from a user's personality vector

public:
	Item(const char* szLeft, const char* szRight, const char* szSubmitter, time_t date, GRand* pRand);
	Item(GDomNode* pNode, GRand* pRand);

	const char* left() { return m_left.c_str(); }
	void setLeft(const char* str) { m_left = str; }
	const char* right() { return m_right.c_str(); }
	void setRight(const char* str) { m_right = str; }
	GVec& weights() { return m_weights; }

	GDomNode* toDom(GDom* pDoc);
};




// A grouping of items
class Topic
{
public:
	GDomNode* toDom(GDom* pDoc);
	static Topic* fromDom(GDomNode* pNode, GRand* pRand);
};






class NeuralRecommender
{
protected:
	GRand& rand;
	GVec input_buf;
	GNeuralNet nn;
	std::vector<User*> m_users;
	std::map<std::string,User*> m_userMap;
	std::vector<Item*> m_items;

public:
	NeuralRecommender(GRand& _rand);
	virtual ~NeuralRecommender();

	void flush();
	GDomNode* serialize(GDom* pDoc);
	void deserialize(const GDomNode* pNode);
	void refine(size_t iters);
	void refineUserOnly(User* pUser, size_t iters);
	double predictRating(User& user, Item& item);
	const std::vector<User*>& users() { return m_users; }
	const std::vector<Item*>& items() { return m_items; }
	User* findUser(const char* szUsername);
	User* findOrMakeUser(const char* szUsername);

	// Takes ownership of pUser
	void addUser(User* pUser);

	void addItem(const char* szLeft, const char* szRight, const char* szUsername, time_t date, GRand* pRand);
	void addItem(const char* szLeft, const char* szRight, const char* szUsername);
	Item& item(size_t id);
	void deleteItemAndSwapInLast(size_t itemId);
};




/// Makes pages that collect survey responses and visualize the results
class Submit
{
public:

	/// Display the FAQ page
	static void pageFaq(Server* pServer, GDynamicPageSession* pSession, std::ostream& response);

	/// Generates a page of slider bars for users to take a survey
	static void pageSurvey(Server* pServer, GDynamicPageSession* pSession, std::ostream& response);

	/// Generates a page that lets users submit new survey questions
	static void pageNewSurveyItem(Server* pServer, GDynamicPageSession* pSession, std::ostream& response);

	/// Generates a page for submitting responses
	static void pageRespond(Server* pServer, GDynamicPageSession* pSession, std::ostream& response);

	/// Generates a page that summarizes survey responses
	static void pageStats(Server* pServer, GDynamicPageSession* pSession, std::ostream& response);

	/// Generates an SVG plot of users
	static void plotUsers(Server* pServer, GDynamicPageSession* pSession, std::ostream& response);

	/// Generates an SVG plot of items
	static void plotItems(Server* pServer, GDynamicPageSession* pSession, std::ostream& response);

	/// A page that lets users update their survey responses
	static void pageUpdateResponses(Server* pServer, GDynamicPageSession* pSession, std::ostream& response);

protected:
	static void makeSliderScript(std::ostream& response);
	static void makeUrlSlider(Server* pServer, Account* pAccount, size_t itemId, std::ostream& response);
	static size_t divideAccounts(Server* pServer, User** pUsers, size_t accCount, size_t itm);
	static void makeTree(Server* pServer, GBitTable& bt, User** pUsers, size_t accCount, std::ostream& response, std::vector<char>& prefix, int type);
	static void makeItemBody(GDynamicPageSession* pSession, std::ostream& response, size_t itemId, Item& item, User** pUsers, size_t accCount);
	static void makeUserBody(GDynamicPageSession* pSession, std::ostream& response, User* pA, User* pB, NeuralRecommender& rec);
};



#endif // RECOMMENDER
