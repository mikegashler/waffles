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

#ifndef GTREE_H
#define GTREE_H

#include <GClasses/GError.h>

using namespace GClasses;

/// This is a helper class used by GIndexedMultiSet
template <typename T>
class GTreeNode
{
public:
	T key;
	GTreeNode* left;
	GTreeNode* right;
	size_t size;

	GTreeNode(T k) : key(k)
	{
		isolate();
	}

	~GTreeNode()
	{
		delete(left);
		delete(right);
	}

	void isolate()
	{
		left = NULL;
		right = NULL;
		size = 1;
	}

	void recount()
	{
		size = 1 + (left ? left->size : 0) + (right ? right->size : 0);
	}

	void print(size_t depth)
	{
		if(right)
			right->print(depth + 1);
		for(size_t i = 0; i < depth; i++)
			std::cout << "    ";
		std::cout << to_str(key) << " (" << to_str(size) << ")\n";
		if(left)
			left->print(depth + 1);
	}

	GTreeNode<T>* rotateLeft()
	{
		GTreeNode<T>* c = right;
		GTreeNode<T>* gc = right->left;
		right = gc;
		c->left = this;
		recount();
		c->recount();
		return c;
	}

	GTreeNode<T>* rotateRight()
	{
		GTreeNode<T>* c = left;
		GTreeNode<T>* gc = left->right;
		left = gc;
		c->right = this;
		recount();
		c->recount();
		return c;
	}

	GTreeNode<T>* balance()
	{
		size_t lcount = left ? left->size : 0;
		size_t rcount = right ? right->size : 0;
		if((lcount + 1) * 2 < (rcount + 1))
		{
			size_t lcount2 = right->left ? right->left->size : 0;
			size_t rcount2 = right->right ? right->right->size : 0;
			if(lcount2 > rcount2)
				right = right->rotateRight();
			return rotateLeft();
		}
		else if((rcount + 1) * 2 <= (lcount + 1))
		{
			size_t lcount2 = left->left ? left->left->size : 0;
			size_t rcount2 = left->right ? left->right->size : 0;
			if(lcount2 < rcount2)
				left = left->rotateLeft();
			return rotateRight();
		}
		else
		{
			recount();
			return this;
		}
	}

	GTreeNode<T>* insert(GTreeNode<T>* newNode)
	{
		if(newNode->key < key)
		{
			if(left)
				left = left->insert(newNode);
			else
				left = newNode;
		}
		else
		{
			if(right)
				right = right->insert(newNode);
			else
				right = newNode;
		}
		return balance();
	}

	GTreeNode<T>* get(size_t index)
	{
		size_t lcount = left ? left->size : 0;
		if(index < lcount)
			return left->get(index);
		else if(index > lcount)
			return right ? right->get(index - lcount - 1) : NULL;
		else
			return this;
	}

	GTreeNode<T>* find(T k, size_t start, size_t* outIndex)
	{
		if(k < key)
			return left ? left->find(k, start, outIndex) : NULL;
		else if(k > key)
			return right ? right->find(k, left ? start + left->size + 1 : start + 1, outIndex) : NULL;
		else
		{
			if(outIndex)
				*outIndex = start + (left ? left->size : 0);
			return this;
		}
	}

	GTreeNode<T>* remove_by_index(size_t index, GTreeNode<T>** outNode)
	{
		size_t lcount = left ? left->size : 0;
		if(index < lcount)
			left = left->remove_by_index(index, outNode);
		else if(index > lcount)
			right = right->remove_by_index(index - lcount - 1, outNode);
		else
		{
			*outNode = this;
			size_t rcount = right ? right->size : 0;
			if(lcount < rcount)
				return left ? right->insert(left) : right;
			else
				return right ? left->insert(right) : left;
		}
		return balance();
	}

	GTreeNode<T>* remove_by_value(T k, GTreeNode<T>** outNode)
	{
		if(k < key)
		{
			if(!left)
				throw Ex("Not found");
			left = left->remove_by_value(k, outNode);
		}
		else if(k > key)
		{
			if(!right)
				throw Ex("Not found");
			right = right->remove_by_value(k, outNode);
		}
		else
		{
			*outNode = this;
			size_t lcount = left ? left->size : 0;
			size_t rcount = right ? right->size : 0;
			if(lcount < rcount)
				return left ? right->insert(left) : right;
			else
				return right ? left->insert(right) : left;
		}
		return balance();
	}
};


/// A multiset class that can be queried by index.
/// It is implemented using a balanced tree structure, so
/// most operations take O(log(n)) time.
template <typename T>
class GIndexedMultiSet
{
private:
	GTreeNode<T>* root;
	GTreeNode<T>* spare;

public:
	/// General-purpose constructor
	GIndexedMultiSet() : root(NULL), spare(NULL)
	{
	}

	~GIndexedMultiSet()
	{
		delete(root);
		delete(spare);
	}

	/// Print a representation of the tree to stdout.
	void print()
	{
		if(root)
			root->print(0);
	}

	/// Insert a value into the multiset.
	void insert(T key)
	{
		if(spare)
			spare->key = key;
		else
			spare = new GTreeNode<T>(key);
		if(root)
			root = root->insert(spare);
		else
			root = spare;
		spare = NULL;
	}

	/// Drop the value at the specified index (in sorted order, beginning with 0).
	void drop_by_index(size_t index)
	{
		if(!root || index >= root->size)
			throw Ex("index ", to_str(index), " out of range [0-", to_str(root ? root->size : 0), "]");
		delete(spare);
		root = root->remove_by_index(index, &spare);
		spare->isolate();
	}

	/// Drop one occurrence of the specified value from this multiset.
	/// An exception is thrown if the value does not occur in the multiset.
	void drop_by_value(T key)
	{
		if(!root)
			throw Ex("Not found");
		delete(spare);
		root = root->remove_by_value(key, &spare);
		spare->isolate();
	}

	/// Return the value at the specified index in this multiset (in sorted order, beginning with 0).
	T get(size_t index)
	{
		if(!root || index >= root->size)
			throw Ex("index ", to_str(index), " out of range [0-", to_str(root ? root->size : 0), "]");
		return root->get(index)->key;
	}

	/// Returns the index of one occurrence of the specified value in this multiset.
	/// The the value occurs multiple times, the returned index is arbitrarily chosen from among them.
	size_t find(T key)
	{
		size_t outIndex;
		GTreeNode<T>* node = root ? root->find(key, 0, &outIndex) : NULL;
		if(node)
			return outIndex;
		else
			throw Ex("not found");
	}

	/// Return the total number of values in this multiset.
	size_t size()
	{
		return root ? root->size : 0;
	}
};

#endif // GTREE_H
