/*
	Copyright (C) 2006, Mike Gashler

	This library is free software; you can redistribute it and/or
	modify it under the terms of the GNU Lesser General Public
	License as published by the Free Software Foundation; either
	version 2.1 of the License, or (at your option) any later version.

	see http://www.gnu.org/copyleft/lesser.html
*/

#ifndef __GHASHTABLE_H__
#define __GHASHTABLE_H__

#include <stdio.h>
#include "GError.h"
#ifndef WINDOWS
#	include <stdint.h>
#endif
#include <vector>
#include <string.h>

namespace GClasses {

class GQueue;
class HashTableNode;
struct HashBucket;

/// The base class of hash tables
class GHashTableBase
{
friend class GHashTableEnumerator;
protected:
	struct HashBucket* m_pBuckets;
	struct HashBucket* m_pFirstEmpty;
	int m_nBucketCount;
	int m_nCount;
	int m_nModCount;

	GHashTableBase(int nInitialBucketCount);

public:
	virtual ~GHashTableBase();

	/// Returns the number of items in this hash table
	int size() { return m_nCount; }

	/// Returns a number that changes when the contents of this table are modified
	/// (This is useful for detecting invalidated iterators)
	int revisionNumber() { return m_nModCount; }

protected:
	/// Returns a hash of the key
	virtual unsigned int hash(const char* pKey, int nBucketCount) = 0;

	/// Returns true iff the keys compare equal
	virtual bool areKeysEqual(const char* pKey1, const char* pKey2) = 0;
	void _Resize(int nNewSize);

	/// Adds a key/value pair to the hash table
	void _Add(const char* pKey, const void* pValue);

	/// Returns true and the first occurrence of a value with the specified key if one exists
        template<class T>
	bool _Get(const char* pKey, T** pOutValue);

	/// Removes the first found occurrence of the specified key
	void _Remove(const char* pKey);

	/// Returns the number of values with the specified key
	int _Count(const char* pKey);
};




#define UNCOMMON_INT 0x80000000

/// This class iterates over the values in a hash table
class GHashTableEnumerator
{
protected:
	GHashTableBase* m_pHashTable;
	int m_nPos;
	int m_nModCount;

public:
	GHashTableEnumerator(GHashTableBase* pHashTable)
	{
		m_pHashTable = pHashTable;
		m_nModCount = m_pHashTable->revisionNumber();
		m_nPos = 0;
	}

	/// Gets the next element in the hash table. ppValue is set to
	/// the value and the return value is the key. Returns NULL when
	/// it reaches the end of the collection. (The first time it is
	/// called, it returns the first item in the collection.)
	const char* next(void** ppOutValue);

	/// Returns the value associated with the current key
	void* currentValue();
};




/// Implements a typical hash table. (It doesn't take ownership
/// of the objects you add, so you must still delete them yourself.)
class GHashTable : public GHashTableBase
{
public:
	GHashTable(int nInitialBucketCount)
		: GHashTableBase(nInitialBucketCount)
	{
	}

	virtual ~GHashTable()
	{
	}

#ifndef NO_TEST_CODE
	/// Performs unit tests for this class. Throws an exception if there is a failure.
	static void test();
#endif // !NO_TEST_CODE

	/// Computes a hash of the key
	virtual unsigned int hash(const char* pKey, int nBucketCount)
	{
		return (unsigned int)(((uintptr_t)pKey) % nBucketCount);
	}

	/// Returns true iff the two keys are equal
	virtual bool areKeysEqual(const char* pKey1, const char* pKey2)
	{
		return pKey1 == pKey2;
	}

	/// Adds a pointer key and value pair to the hash table.  The key can not be NULL
	void add(const void* pKey, const void* pValue)
	{
		_Add((const char*)pKey, pValue);
	}

	/// Gets a value based on the key
	bool get(const void* pKey, void** ppOutValue)
	{
		return _Get((const char*)pKey, ppOutValue);
	}

	/// Removes an entry from the hash table
	void remove(const void* pKey)
	{
		_Remove((const char*)pKey);
	}
};





/// Hash table based on keys of constant strings (or at least strings
/// that won't change during the lifetime of the hash table).  It's a
/// good idea to use a GHeap in connection with this class.
class GConstStringHashTable : public GHashTableBase
{
protected:
	bool m_bCaseSensitive;

public:
	GConstStringHashTable(int nInitialBucketCount, bool bCaseSensitive)
		: GHashTableBase(nInitialBucketCount)
	{
		m_bCaseSensitive = bCaseSensitive;
	}

	virtual ~GConstStringHashTable()
	{
	}

	/// Computes a hash of the key
	virtual unsigned int hash(const char* pKey, int nBucketCount)
	{
		unsigned int n = 0;
		if(m_bCaseSensitive)
		{
			while(*pKey != '\0')
			{
				n += (*pKey);
				pKey++;
			}
		}
		else
		{
			while(*pKey != '\0')
			{
				n += ((*pKey) & ~0x20);
				pKey++;
			}
		}
		return n % nBucketCount;
	}

	/// Returns true iff the two keys are equal
	virtual bool areKeysEqual(const char* pKey1, const char* pKey2)
	{
		if(m_bCaseSensitive)
			return(strcmp(pKey1, pKey2) == 0);
		else
			return(_stricmp(pKey1, pKey2) == 0);
	}

	/// Adds a key and value pair to the hash table.  The key should be a constant
	/// string (or at least a string that won't change over the lifetime of the
	/// hash table).  The GHeap class provides a good place to store such a
	/// string.
	void add(const char* pKey, const void* pValue)
	{
		_Add(pKey, pValue);
	}

	/// Gets the value for the specified key
        template<class T>
	bool get(const char* pKey, T** ppOutValue)
	{
		return _Get(pKey, ppOutValue);
	}

	/// Gets the value for the specified key
	bool get(const char* pKey, int nLen, void** ppOutValue);

	/// Removes an entry from the hash table
	void remove(const char* pKey)
	{
		_Remove(pKey);
	}
};



/// Hash table based on keys of constant strings (or at least strings
/// that won't change during the lifetime of the hash table).  It's a
/// good idea to use a GHeap in connection with this class.
class GConstStringToIntsHashTable : public GHashTableBase
{
protected:
	bool m_bCaseSensitive;

public:
	GConstStringToIntsHashTable(int nInitialBucketCount, bool bCaseSensitive)
		: GHashTableBase(nInitialBucketCount)
	{
		m_bCaseSensitive = bCaseSensitive;
	}

	virtual ~GConstStringToIntsHashTable()
	{
	}

	/// Computes a hash of the key
	virtual unsigned int hash(const char* pKey, int nBucketCount)
	{
		unsigned int n = 0;
		if(m_bCaseSensitive)
		{
			while(*pKey != '\0')
			{
				n += (*pKey);
				pKey++;
			}
		}
		else
		{
			while(*pKey != '\0')
			{
				n += ((*pKey) & ~0x20);
				pKey++;
			}
		}
		return n % nBucketCount;
	}

	/// Returns true iff the two keys are equal
	virtual bool areKeysEqual(const char* pKey1, const char* pKey2)
	{
		if(m_bCaseSensitive)
			return(strcmp(pKey1, pKey2) == 0);
		else
			return(_stricmp(pKey1, pKey2) == 0);
	}

	/// Adds a key and value pair to the hash table.  The key should be a constant
	/// string (or at least a string that won't change over the lifetime of the
	/// hash table).  The GHeap class provides a good place to store such a
	/// string.
	void add(const char* pKey, int nValue)
	{
		uintptr_t tmp = nValue;
		_Add(pKey, (const void*)tmp);
	}

	/// Gets the value for the specified key
	bool get(const char* pKey, int* pValue)
	{
	  void* tmp = NULL;
	  bool bRet = _Get(pKey, &tmp);
	  *pValue = (int)reinterpret_cast<uintptr_t>(tmp);
	  return bRet;
	}

	/// Gets the value for the specified key
	bool get(const char* pKey, int nLen, int* pValue);

	/// Removes an entry from the hash table
	void remove(const char* pKey)
	{
		_Remove(pKey);
	}
};

/// This is an internal structure used by GHashTable
struct HashBucket
{
	HashBucket* pPrev;
	HashBucket* pNext;
	const char* pKey;
	const void* pValue;
};

template<class T>
bool GHashTableBase::_Get(const char* pKey, T** pOutValue)
{
	GAssert(pKey != NULL);
	unsigned int nPos = hash(pKey, m_nBucketCount);
	GAssert(nPos < (unsigned int)m_nBucketCount); // Out of range
	if(!m_pBuckets[nPos].pKey || m_pBuckets[nPos].pPrev)
		return false;
	struct HashBucket* pBucket;
	for(pBucket = &m_pBuckets[nPos]; pBucket; pBucket = pBucket->pNext)
	{
		if(areKeysEqual(pBucket->pKey, pKey))
		{
		  *pOutValue = const_cast<T*>(reinterpret_cast<const T*>(pBucket->pValue));
			return true;
		}
	}
	return false;
}



/*
/// Hash table based on keys of constant strings (or at least strings
/// that won't change during the lifetime of the hash table).  It's a
/// good idea to use a GHeap in connection with this class.
class GConstUnicodeHashTable : public GHashTableBase
{
protected:
	bool m_bCaseSensitive;

public:
	GConstUnicodeHashTable(int nInitialBucketCount, bool bCaseSensitive)
		: GHashTableBase(nInitialBucketCount)
	{
		m_bCaseSensitive = bCaseSensitive;
	}

	virtual ~GConstUnicodeHashTable()
	{
	}


	virtual unsigned int hash(const char* pKey, int nBucketCount)
	{
		const wchar_t* wszKey = (const wchar_t*)pKey;
		unsigned int n = 0;
		if(m_bCaseSensitive)
		{
			while(*wszKey != L'\0')
			{
				n += (*wszKey);
				wszKey++;
			}
		}
		else
		{
			while(*wszKey != L'\0')
			{
				if(*wszKey <= L'Z' && *wszKey >= L'A') // todo: use a better func
					n += *wszKey + 32;
				else
					n += *wszKey;
				wszKey++;
			}
		}
		return n % nBucketCount;
	}

	virtual bool areKeysEqual(const char* pKey1, const char* pKey2)
	{
		if(m_bCaseSensitive)
			return(wcscmp((const wchar_t*)pKey1, (const wchar_t*)pKey2) == 0);
		else
			return(wcsicmp((const wchar_t*)pKey1, (const wchar_t*)pKey2) == 0);
	}

	/// Adds a key and value pair to the hash table.  The key should be a constant
	/// string (or at least a string that won't change over the lifetime of the
	/// hash table).  The GHeap class provides a good place to store such a
	/// string.
	void add(const wchar_t* pKey, const void* pValue)
	{
		_Add((const char*)pKey, pValue);
	}

	/// Gets the value for the specified key
	bool get(const wchar_t* pKey, void** ppOutValue)
	{
		return _Get((const char*)pKey, ppOutValue);
	}

	/// Removes an entry from the hash table
	void remove(const wchar_t* pKey)
	{
		_Remove((const char*)pKey);
	}
};
*/


/// Objects used with GNodeHashTable should inherit from this class. They
/// must implement two methods (to hash and compare the nodes).
class HashTableNode
{
public:
	HashTableNode() {}
	virtual ~HashTableNode() {}

	/// Returns a hash value for this node
	virtual unsigned int hash(int nBucketCount) = 0;

	/// Returns true iff this compares equal to pThat
	virtual bool equals(HashTableNode* pThat) = 0;
};



/// This is a hash table that uses any object which inherits from
/// HashTableNode as the key
class GNodeHashTable : public GHashTableBase
{
protected:
	std::vector<HashTableNode*>* m_pNodes;

public:
	GNodeHashTable(bool bOwnNodes, int nInitialBucketCount);
	virtual ~GNodeHashTable();

	/// Computes a hash of the key
	virtual unsigned int hash(const char* pKey, int nBucketCount)
	{
		HashTableNode* pRec = (HashTableNode*)pKey;
		return pRec->hash(nBucketCount);
	}

	/// Returns true iff the two keys are equal
	virtual bool areKeysEqual(const char* pKey1, const char* pKey2)
	{
		HashTableNode* pRec1 = (HashTableNode*)pKey1;
		HashTableNode* pRec2 = (HashTableNode*)pKey2;
		return pRec1->equals(pRec2);
	}

	/// Adds an object to this hash table
	void add(HashTableNode* pRec);

	/// Gets the value for the specified key
	HashTableNode* get(HashTableNode* pLikeMe)
	{
		HashTableNode* pOut = NULL;
		_Get((const char*)pLikeMe, (void**)(void*)&pOut);
		return pOut;
	}
};



} // namespace GClasses

#endif // __GHASHTABLE_H__
