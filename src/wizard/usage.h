#ifndef _USAGE_H_
#define _USAGE_H_

#include <string>
#include <vector>
#include <stdlib.h>
#include "../GClasses/GError.h"

//#define DEBUG_HELPERS

class UsageNode
{
protected:
	std::vector<std::string> m_parts;
	std::string m_default_value;
	std::string m_description;
	std::vector<UsageNode*> m_choices;
#ifdef DEBUG_HELPERS
	const char* p0;
	const char* p1;
	const char* p2;
	const char* p3;
#endif

public:
	UsageNode(const char* templ, const char* descr);
	~UsageNode();
	UsageNode* add(const char* templ, const char* descr = "");
	void print(int depth, int tabSize, int maxWidth, int maxDepth, bool descriptions);

	const char* tok() { return m_parts[0].c_str(); }
	void setTok(const char* tok)
	{
		if(m_parts.size() < 1)
			m_parts.push_back(tok);
		else
			m_parts[0] = tok;
	}

	const char* descr() { return m_description.c_str(); }
	int findPart(const char* name);
	std::vector<std::string>& parts() { return m_parts; }
	std::string& default_value() { return m_default_value; }
	std::vector<UsageNode*>& choices() { return m_choices; }
	UsageNode* choice(const char* name);
	void sig(std::string* pS);
};

// Master tree
UsageNode* makeMasterUsageTree();
UsageNode* makeLearnUsageTree();
UsageNode* makeGenerateUsageTree();
UsageNode* makePlotUsageTree();
UsageNode* makeTransformUsageTree();
UsageNode* makeRecommendUsageTree();
UsageNode* makeSparseUsageTree();

// Global nodes
UsageNode* makeAlgorithmUsageTree();
UsageNode* makeNeighborUsageTree();
UsageNode* makeCollaborativeFilterUsageTree();


#endif // _USAGE_H_
