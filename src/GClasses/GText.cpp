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

#include "GText.h"
#include "GHashTable.h"
#include "GHeap.h"
#include "GStemmer.h"
#include <math.h>

namespace GClasses {

using std::vector;

GWordIterator::GWordIterator(const char* text, size_t len)
: m_pText(text), m_len(len)
{
}

GWordIterator::~GWordIterator()
{
}

bool GWordIterator_isWordChar(char c)
{
	if(c >= 'a' && c <= 'z')
		return true;
	if(c >= 'A' && c <= 'Z')
		return true;
	return false;
}

bool GWordIterator::next(const char** ppWord, size_t* pLen)
{
	while(m_len > 0 && !GWordIterator_isWordChar(*m_pText))
	{
		m_pText++;
		m_len--;
	}
	*ppWord = m_pText;
	while(m_len > 0 && GWordIterator_isWordChar(*m_pText))
	{
		m_pText++;
		m_len--;
	}
	*pLen = m_pText - *ppWord;
	return *pLen > 0;
}




const char* g_szStopWords[] = 
{
	"a","about","all","also","although","an","and","any","are","as","at",
	"be","but","by",
	"can",
	"did",
	"each","every",
	"for","from",
	"had","have","he","her","him","his","how",
	"i","if","in","is","it","its",
	"my",
	"nbsp","next","no","not",
	"of","on","one","or","our","out",
	"quite",
	"really",
	"so","some",
	"that","the","them","then","there","this","to","too",
	"use",
	"very",
	"was","we","what","when","where","who","will","with",
	"you",
};

GVocabulary::GVocabulary(bool stemWords)
: m_minWordSize(4), m_vocabSize(0)
{
	if(stemWords)
		m_pStemmer = new GStemmer();
	else
		m_pStemmer = NULL;
	m_pStopWords = new GConstStringHashTable(113, false);
	m_pVocabulary = new GConstStringToIndexHashTable(3571, false);
	m_pHeap = new GHeap(1024);
	m_docNumber = 0;
	m_pWordStats = NULL;
}

GVocabulary::~GVocabulary()
{
	delete(m_pStemmer);
	delete(m_pStopWords);
	delete(m_pVocabulary);
	delete(m_pHeap);
	delete(m_pWordStats);
}

void GVocabulary::addStopWord(const char* szWord)
{
	const char* pWord = szWord;
	if(m_pStemmer)
		pWord = m_pStemmer->getStem(szWord, strlen(szWord));
	m_pStopWords->add(m_pHeap->add(pWord), NULL);
}

void GVocabulary::addTypicalStopWords()
{
	for(size_t i = 0; i < (sizeof(g_szStopWords) / sizeof(const char*)); i++)
		m_pStopWords->add(g_szStopWords[i], NULL);
}

void GVocabulary::addWord(const char* szWord, size_t nLen)
{
	if(nLen < m_minWordSize)
		return;

	// Find the stem
	const char* szStem;
	if(m_pStemmer)
		szStem = m_pStemmer->getStem(szWord, nLen);
	else
	{
		nLen = std::min((size_t)63, nLen);
		memcpy(wordBuf, szWord, nLen); // todo: make lowercase
		wordBuf[nLen] = '\0';
		szStem = wordBuf;
	}

	// Don't add stop words
	void* pValue;
	if(m_pStopWords->get(szStem, &pValue))
		return; // it's a stop word

	// Check for existing words
	size_t nIndex;
	if(m_pVocabulary->get(szStem, &nIndex))
	{
		// Track word stats
		if(m_pWordStats)
		{
			GAssert(nIndex < m_pWordStats->size());
			GWordStats& wordStats = (*m_pWordStats)[nIndex];
			if(wordStats.m_lastDocContainingWord != m_docNumber)
			{
				wordStats.m_lastDocContainingWord = m_docNumber;
				wordStats.m_docsContainingWord++;
				wordStats.m_curDocFreq = 0;
			}
			wordStats.m_curDocFreq++;
			wordStats.m_maxWordFreq = std::max(wordStats.m_curDocFreq, wordStats.m_maxWordFreq);
		}
		return;
	}

	// Add the word to the vocabulary
	nIndex = m_vocabSize;
	char* pStoredWord = m_pHeap->add(szStem);
	m_pVocabulary->add(pStoredWord, m_vocabSize++);

	// Track word stats
	if(m_pWordStats)
	{
		m_pWordStats->resize(m_vocabSize);
		GWordStats& wordStats = (*m_pWordStats)[nIndex];
		wordStats.m_lastDocContainingWord = m_docNumber;
		wordStats.m_szWord = pStoredWord;
	}
}

GWordStats& GVocabulary::stats(size_t word)
{
	if(!m_pWordStats)
		throw Ex("You didn't call newDoc before adding vocabulary words, so stats weren't tracked");
	return (*m_pWordStats)[word];
}

void GVocabulary::addWordsFromTextBlock(const char* text, size_t len)
{
	GWordIterator it(text, len);
	const char* pWord;
	size_t wordLen;
	while(true)
	{
		if(!it.next(&pWord, &wordLen))
			break;
		addWord(pWord, wordLen);
	}
}

size_t GVocabulary::wordIndex(const char* szWord, size_t len)
{
	// Find the stem
	const char* szStem;
	if(m_pStemmer)
		szStem = m_pStemmer->getStem(szWord, len);
	else
	{
		len = std::min((size_t)63, len);
		memcpy(wordBuf, szWord, len); // todo: make lowercase
		wordBuf[len] = '\0';
		szStem = wordBuf;
	}

	// Look up the stem
	size_t val;
	if(!m_pVocabulary->get(szStem, &val))
		return INVALID_INDEX;
	return val;
}

void GVocabulary::newDoc()
{
	if(!m_pWordStats)
	{
		if(m_vocabSize > 0)
			throw Ex("If you call newDoc, then you must call it before the first word is added");
		m_pWordStats = new vector<GWordStats>();
		m_docNumber = 0;
		return;
	}
	m_docNumber++;
}

const char* GVocabulary::word(size_t index)
{
	if(!m_pWordStats)
		throw Ex("You didn't call newDoc before adding vocabulary words, so the indexes weren't tracked");
	return (*m_pWordStats)[index].m_szWord;
}

double GVocabulary::weight(size_t word)
{
	GWordStats& ws = stats(word);
	return log((double)docCount() / ws.m_docsContainingWord) / ws.m_maxWordFreq;
}










GDiffer::GDiffer()
{
}

GDiffer::~GDiffer()
{
	for(size_t i = 0; i < m_chunks.size(); i++)
		delete(m_chunks[i]);
}

void GDiffer::simplify(size_t min_match_size)
{
	std::vector<GDiffChunk*> old_chunks;
	old_chunks.swap(m_chunks);
	for(size_t i = 0; i < old_chunks.size(); i++)
	{
		GDiffChunk* pChunk = old_chunks[i];
		if(pChunk->left != INVALID_INDEX && pChunk->right != INVALID_INDEX && pChunk->len < min_match_size) // If it is a match that is too small
		{
			addChunk(pChunk->left, INVALID_INDEX, pChunk->len);
			addChunk(INVALID_INDEX, pChunk->right, pChunk->len);
		}
		else
			addChunk(pChunk->left, pChunk->right, pChunk->len);
		delete(pChunk);
	}
}

/*static*/
void GDiffer::first_matching_segment(const char* file1, size_t len1, size_t pos1, const char* file2, size_t len2, size_t pos2, size_t* match1, size_t* match2, size_t* matchLen)
{
	*matchLen = 0;
	if(pos1 >= len1 || pos2 >= len2)
		return;
	size_t chars1[256];
	size_t chars2[256];
	for(size_t i = 0; i < 256; i++)
	{
		chars1[i] = INVALID_INDEX;
		chars2[i] = INVALID_INDEX;
	}
	for(size_t i = 0; pos1 + i < len1 || pos2 + i < len2; i++)
	{
		// Store these characters
		unsigned char c1 = 0;
		unsigned char c2 = 0;
		if(pos1 + i < len1)
		{
			c1 = (unsigned char)file1[pos1 + i];
			if(chars1[c1] == INVALID_INDEX)
				chars1[c1] = pos1 + i;
		}
		if(pos2 + i < len2)
		{
			c2 = (unsigned char)file2[pos2 + i];
			if(chars2[c2] == INVALID_INDEX)
				chars2[c2] = pos2 + i;
		}

		// Check for matches
		if(pos1 + i < len1)
		{
			if(chars2[c1] != INVALID_INDEX)
			{
				*match1 = chars1[c1];
				*match2 = chars2[c1];
				for((*matchLen)++; *match1 + *matchLen < len1 && *match2 + (*matchLen) < len2 && file1[*match1 + *matchLen] == file2[*match2 + *matchLen]; (*matchLen)++)
				{
				}
				return;
			}
		}
		if(pos2 + i < len2)
		{
			if(chars1[c2] != INVALID_INDEX)
			{
				*match1 = chars1[c2];
				*match2 = chars2[c2];
				for((*matchLen)++; *match1 + *matchLen < len1 && *match2 + *matchLen < len2 && file1[*match1 + *matchLen] == file2[*match2 + *matchLen]; (*matchLen)++)
				{
				}
				return;
			}
		}
	}
}

/*static*/
void GDiffer::biggest_matching_segment_one_way(const char* file1, size_t len1, size_t pos1, size_t spot, const char* file2, size_t len2, size_t pos2, size_t* match1, size_t* match2, size_t* matchLen)
{
	*matchLen = 0;
	char c = file1[spot];
	for(size_t i = pos2; i < len2; i++)
	{
		if(file2[i] == c)
		{
			size_t bef = 0;
			size_t aft = 1;
			while(pos1 + bef < spot && pos2 + bef < i && file1[spot - bef - 1] == file2[i - bef - 1])
				bef++;
			while(spot + aft < len1 && i + aft < len2 && file1[spot + aft] == file2[i + aft])
				aft++;
			if(bef + aft > *matchLen)
			{
				*matchLen = bef + aft;
				*match1 = spot - bef;
				*match2 = i - bef;
			}
		}
	}
}

/*static*/
void GDiffer::biggest_matching_segment(const char* file1, size_t len1, size_t pos1, const char* file2, size_t len2, size_t pos2, size_t* match1, size_t* match2, size_t* matchLen)
{
	//std::cout << "BMS(" << to_str(pos1) << " - " << to_str(len1) << ")(" << to_str(pos2) << " - " << to_str(len2) << ")";
	*matchLen = 0;
	size_t segments = 2;
	while(true)
	{
		size_t size = (len1 - pos1) / segments;
		if(size < 12)
			break;
		for(size_t i = 1; i < segments; i += 2)
		{
			size_t index = pos1 + i * size;
			size_t m1 = 0;
			size_t m2 = 0;
			size_t len = 0;
			//std::cout << ", " << to_str(index);
			biggest_matching_segment_one_way(file1, len1, pos1, index, file2, len2, pos2, &m1, &m2, &len);
			if(len > *matchLen)
			{
				*matchLen = len;
				*match1 = m1;
				*match2 = m2;
			}
		}
		if(*matchLen > size)
			break;
		segments *= 2;
	}
	//std::cout << " -> (" << to_str(*match1) << ", " << to_str(*match2) << ", " << to_str(*matchLen) << ")\n";
}

void GDiffer::addChunk(size_t left, size_t right, size_t len)
{
	//std::cout << "Adding chunk (" << (left == INVALID_INDEX ? "--" : to_str(left)) << ", " << (right == INVALID_INDEX ? "--" : to_str(right)) << ", " << to_str(len) << ")\n";

	// See if we can just extend the previous chunk
	if(m_chunks.size() > 0)
	{
		GDiffChunk* pPrev = m_chunks[m_chunks.size() - 1];
		if(left != INVALID_INDEX && right != INVALID_INDEX) // match
		{
			if(pPrev->left != INVALID_INDEX && pPrev->right != INVALID_INDEX) // prev match
			{
				GAssert(pPrev->left + pPrev->len == left);
				GAssert(pPrev->right + pPrev->len == right);
				pPrev->len += len;
				return;
			}
		}
		else if(left != INVALID_INDEX) // left only
		{
			if(pPrev->left != INVALID_INDEX && pPrev->right == INVALID_INDEX) // prev left only
			{
				GAssert(pPrev->left + pPrev->len == left);
				pPrev->len += len;
				return;
			}
			if(pPrev->left == INVALID_INDEX && pPrev->right != INVALID_INDEX && m_chunks.size() > 1) // prev right only
			{
				if(m_chunks.size() > 1)
				{
					pPrev = m_chunks[m_chunks.size() - 2];
					if(pPrev->left != INVALID_INDEX && pPrev->right == INVALID_INDEX) // prev-prev left only
					{
						GAssert(pPrev->left + pPrev->len == left);
						pPrev->len += len;
						return;
					}
				}
			}
		}
		else
		{
			GAssert(right != INVALID_INDEX); // right only
			if(pPrev->right != INVALID_INDEX && pPrev->left == INVALID_INDEX) // prev right only
			{
				GAssert(pPrev->right + pPrev->len == right);
				pPrev->len += len;
				return;
			}
			if(pPrev->right == INVALID_INDEX && pPrev->left != INVALID_INDEX) // prev left only
			{
				if(m_chunks.size() > 1)
				{
					pPrev = m_chunks[m_chunks.size() - 2];
					if(pPrev->right != INVALID_INDEX && pPrev->left == INVALID_INDEX) // prev-prev right only
					{
						GAssert(pPrev->right + pPrev->len == right);
						pPrev->len += len;
						return;
					}
				}
			}
		}
	}

	// Make a new chunk
	GDiffChunk* pChunk = new GDiffChunk();
	pChunk->left = left;
	pChunk->right = right;
	pChunk->len = len;
	m_chunks.push_back(pChunk);
}

void GDiffer::compare(const char* file1, size_t len1, size_t pos1, const char* file2, size_t len2, size_t pos2)
{
	// Do head matching
	size_t headMatchLen = 0;
	while(pos1 + headMatchLen < len1 && pos2 + headMatchLen < len2 && file1[pos1 + headMatchLen] == file2[pos2 + headMatchLen])
		headMatchLen++;
	if(headMatchLen > 0)
	{
		addChunk(pos1, pos2, headMatchLen);
		pos1 += headMatchLen;
		pos2 += headMatchLen;
	}

	// Do tail matching
	size_t tailMatchLen = 0;
	while(pos1 + tailMatchLen < len1 && pos2 + tailMatchLen < len2 && file1[len1 - 1 - tailMatchLen] == file2[len2 - 1 - tailMatchLen])
		tailMatchLen++;
	GAssert(tailMatchLen || (tailMatchLen < len1 && tailMatchLen < len2));
	len1 -= tailMatchLen;
	len2 -= tailMatchLen;

	// If there is more to process
	if(len1 > pos1 && len2 > pos2)
	{
		// Try to find a big matching segment
		size_t match1;
		size_t match2;
		size_t matchLen;
		biggest_matching_segment(file1, len1, pos1, file2, len2, pos2, &match1, &match2, &matchLen);
		if(matchLen > 0)
		{
			// Recursively process everything before the big matching segment
			if(match1 > pos1 || match2 > pos2)
				compare(file1, match1, pos1, file2, match2, pos2);

			// Add the big matching segment
			addChunk(match1, match2, matchLen);

			// Recursively process everything after the big matching segment
			if(match1 + matchLen < len1 || match2 + matchLen < len2)
				compare(file1, len1, match1 + matchLen, file2, len2, match2 + matchLen);
		}
		else
		{
			// Systematically find the first matching segment
			first_matching_segment(file1, len1, pos1, file2, len2, pos2, &match1, &match2, &matchLen);
			if(matchLen > 0)
			{
				// Whatever precedes the first match must be different
				if(match1 > pos1)
					addChunk(pos1, INVALID_INDEX, match1 - pos1);
				if(match2 > pos2)
					addChunk(INVALID_INDEX, pos2, match2 - pos2);

				// Add the first matching segment
				addChunk(match1, match2, matchLen);

				// Recursively process everything after the first matching segment
				if(match1 + matchLen < len1 || match2 + matchLen < len2)
					compare(file1, len1, match1 + matchLen, file2, len2, match2 + matchLen);
			}
			else
			{
				// Whatever is left is different
				if(len1 > pos1)
					addChunk(pos1, INVALID_INDEX, len1 - pos1);
				if(len2 > pos2)
					addChunk(INVALID_INDEX, pos2, len2 - pos2);
			}
		}
	}
	else
	{
		// Any left-over segments are differences
		if(len1 > pos1)
			addChunk(pos1, INVALID_INDEX, len1 - pos1);
		if(len2 > pos2)
			addChunk(INVALID_INDEX, pos2, len2 - pos2);
	}

	// Add the matching tail
	if(tailMatchLen > 0)
		addChunk(len1, len2, tailMatchLen);
}




} // namespace GClasses

