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

#include "GError.h"
#include <iostream>

namespace GClasses
{

/// This is a helper class used by GIndexedMultiSet
template <typename T>
class GTreeNode
{
public:
	T key;
	GTreeNode* left;
	GTreeNode* right;
	size_t size;

	GTreeNode(const T& k) : key(k)
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

	GTreeNode<T>* find(const T& k, size_t start, size_t* outIndex)
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

	GTreeNode<T>* remove_by_value(const T& k, GTreeNode<T>** outNode)
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
	void insert(const T& key)
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
	void drop_by_value(const T& key)
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
	/// If the value occurs multiple times, the returned index is arbitrarily chosen from among them.
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


template <typename T> class GRelationalRow;


template <typename T>
class GRelationalElement
{
public:
	GRelationalRow<T>* par;
	GRelationalRow<T>* left;
	GRelationalRow<T>* right;
	size_t size;

	GRelationalElement()
	{
	}

	~GRelationalElement()
	{
	}

	void isolate()
	{
		par = NULL;
		left = NULL;
		right = NULL;
		size = 1;
	}

};


template <typename T>
class GRelationalRow
{
public:
	T row;
	GRelationalElement<T>* el;

	GRelationalRow(const T& k, size_t columnCount) : row(k)
	{
		el = new GRelationalElement<T>[columnCount];
		isolate(columnCount);
	}

	~GRelationalRow()
	{
		delete(el[0].left);
		delete(el[0].right);
		delete[] el;
	}

	void isolate(size_t cols)
	{
		for(size_t i = 0; i < cols; i++)
			el[i].isolate();
	}

	void recount(size_t c)
	{
		el[c].size = 1 + (el[c].left ? el[c].left->el[c].size : 0) + (el[c].right ? el[c].right->el[c].size : 0);
	}

	GRelationalRow<T>* rotateLeft(size_t c)
	{
		GRelationalRow<T>* t = el[c].right;
		GRelationalRow<T>* g = t->el[c].left;
		GRelationalRow<T>* p = el[c].par;
		el[c].right = g;
		t->el[c].left = this;
		t->el[c].par = p;
		if(g)
			g->el[c].par = this;
		el[c].par = t;
		recount(c);
		t->recount(c);
		return t;
	}

	GRelationalRow<T>* rotateRight(size_t c)
	{
		GRelationalRow<T>* t = el[c].left;
		GRelationalRow<T>* g = t->el[c].right;
		GRelationalRow<T>* p = el[c].par;
		el[c].left = g;
		t->el[c].right = this;
		t->el[c].par = p;
		if(g)
			g->el[c].par = this;
		el[c].par = t;
		recount(c);
		t->recount(c);
		return t;
	}

	GRelationalRow<T>* balance(size_t c)
	{
		size_t lcount = el[c].left ? el[c].left->el[c].size : 0;
		size_t rcount = el[c].right ? el[c].right->el[c].size : 0;
		if((lcount + 1) * 2 < (rcount + 1))
		{
			size_t lcount2 = el[c].right->el[c].left ? el[c].right->el[c].left->el[c].size : 0;
			size_t rcount2 = el[c].right->el[c].right ? el[c].right->el[c].right->el[c].size : 0;
			if(lcount2 > rcount2)
				el[c].right = el[c].right->rotateRight(c);
			return rotateLeft(c);
		}
		else if((rcount + 1) * 2 <= (lcount + 1))
		{
			size_t lcount2 = el[c].left->el[c].left ? el[c].left->el[c].left->el[c].size : 0;
			size_t rcount2 = el[c].left->el[c].right ? el[c].left->el[c].right->el[c].size : 0;
			if(lcount2 < rcount2)
				el[c].left = el[c].left->rotateLeft(c);
			return rotateRight(c);
		}
		else
		{
			recount(c);
			return this;
		}
	}

	template<typename Comp>
	GRelationalRow<T>* insert(GRelationalRow<T>* newNode, size_t c, Comp& comp)
	{
		if(comp(newNode->row, row, c))
		{
			if(el[c].left)
				el[c].left = el[c].left->insert(newNode, c, comp);
			else
			{
				el[c].left = newNode;
				newNode->el[c].par = this;
			}
		}
		else
		{
			if(el[c].right)
				el[c].right = el[c].right->insert(newNode, c, comp);
			else
			{
				el[c].right = newNode;
				newNode->el[c].par = this;
			}
		}
		return balance(c);
	}

	GRelationalRow<T>* get(size_t index, size_t c)
	{
		size_t lcount = el[c].left ? el[c].left->el[c].size : 0;
		if(index < lcount)
			return el[c].left->get(index, c);
		else if(index > lcount)
			return el[c].right ? el[c].right->get(index - lcount - 1, c) : NULL;
		else
			return this;
	}

	/// This should typically only be called through one of the methods in GRelationalTable.
	/// Returns a row that matches in the specified column, or NULL if there is no match.
	template<typename Comp>
	GRelationalRow<T>* find(T r, size_t start, size_t* outIndex, size_t c, Comp& comp)
	{
		if(comp(r, row, c))
			return el[c].left ? el[c].left->find(r, start, outIndex, c, comp) : NULL;
		else if(comp(row, r, c))
			return el[c].right ? el[c].right->find(r, el[c].left ? start + el[c].left->el[c].size + 1 : start + 1, outIndex, c, comp) : NULL;
		else
		{
			if(outIndex)
				*outIndex = start + (el[c].left ? el[c].left->el[c].size : 0);
			return this;
		}
	}

	/// This should typically only be called through one of the methods in GRelationalTable.
	/// Returns a row that matches in the specified column, or an adjacent row if there is no exact match.
	template<typename Comp>
	GRelationalRow<T>* approximate(T r, size_t start, size_t* outIndex, size_t c, Comp& comp)
	{
		if(comp(r, row, c))
			return el[c].left ? el[c].left->approximate(r, start, outIndex, c, comp) : this;
		else if(comp(row, r, c))
			return el[c].right ? el[c].right->approximate(r, el[c].left ? start + el[c].left->el[c].size + 1 : start + 1, outIndex, c, comp) : this;
		else
		{
			if(outIndex)
				*outIndex = start + (el[c].left ? el[c].left->el[c].size : 0);
			return this;
		}
	}

	/// This should typically only be called through GRelationalTable::remove.
	template<typename Comp>
	GRelationalRow<T>* remove(size_t c, Comp& comp)
	{
		size_t lcount = el[c].left ? el[c].left->el[c].size : 0;
		size_t rcount = el[c].right ? el[c].right->el[c].size : 0;
		GRelationalRow<T>* child;
		if(lcount < rcount)
			child = el[c].left ? el[c].right->insert(el[c].left, c, comp) : el[c].right;
		else
			child = el[c].right ? el[c].left->insert(el[c].right, c, comp) : el[c].left;
		GRelationalRow<T>* par = el[c].par;
		if(child)
			child->el[c].par = par;
		if(par)
		{
			if(par->el[c].left == this)
				par->el[c].left = child ? child->balance(c) : NULL;
			else
			{
				GAssert(par->el[c].right == this);
				par->el[c].right = child ? child->balance(c) : NULL;
			}
			child = par;
			par = child->el[c].par;
			while(par)
			{
				if(par->el[c].left == child)
					par->el[c].left = child->balance(c);
				else
				{
					GAssert(par->el[c].right == child);
					par->el[c].right = child->balance(c);
				}
				child = par;
				par = child->el[c].par;
			}
		}
		if(child)
			return child->balance(c);
		else
			return NULL;
	}

	GRelationalRow<T>* next(size_t c)
	{
		GRelationalRow<T>* r = el[c].right;
		if(r)
		{
			while(r->el[c].left)
				r = r->el[c].left;
			return r;
		}
		else
		{
			r = this;
			GRelationalRow<T>* p = el[c].par;
			while(p)
			{
				if(p->el[c].left == r)
					return p;
				r = p;
				p = r->el[c].par;
			}
			return NULL;
		}
	}

	GRelationalRow<T>* prev(size_t c)
	{
		GRelationalRow<T>* l = el[c].left;
		if(l)
		{
			while(l->el[c].right)
				l = l->el[c].right;
			return l;
		}
		else
		{
			l = this;
			GRelationalRow<T>* p = el[c].par;
			while(p)
			{
				if(p->el[c].right == l)
					return p;
				l = p;
				p = l->el[c].par;
			}
			return NULL;
		}
	}

	template <typename Comp>
	void print(const Comp& comp, std::ostream& stream, size_t col, size_t columnCount, size_t depth)
	{
		if(el[col].left)
			el[col].left->print(comp, stream, col, columnCount, depth + 1);
		for(size_t i = 0; i < depth; i++)
			stream << "  ";
		comp.print(stream, row);
		stream << "\n";
		if(el[col].right)
			el[col].right->print(comp, stream, col, columnCount, depth + 1);
	}
};


/// See GTree.cpp for an example of how to use this class.
template <typename T, typename Comp>
class GRelationalTable
{
protected:
	const Comp& comp;
	GRelationalRow<T>** roots;
	GRelationalRow<T>* spare;

public:
	GRelationalTable(const Comp& c) : comp(c), spare(NULL)
	{
		roots = new GRelationalRow<T>*[c.cols()];
		for(size_t i = 0; i < c.cols(); i++)
			roots[i] = NULL;
	}

	~GRelationalTable()
	{
		delete(spare);
		delete(roots[0]);
		delete[] roots;
	}

	/// Drops all content from this table
	void clear()
	{
		delete(roots[0]);
		for(size_t i = 0; i < comp.cols(); i++)
			roots[i] = NULL;
	}

	/// Inserts a row into this relational table.
	void insert(const T& row)
	{
		if(spare)
			spare->row = row;
		else
			spare = new GRelationalRow<T>(row, comp.cols());
		if(roots[0])
		{
			for(size_t i = 0; i < comp.cols(); i++)
				roots[i] = roots[i]->insert(spare, i, comp);
		}
		else
		{
			for(size_t i = 0; i < comp.cols(); i++)
				roots[i] = spare;
		}
		spare = NULL;
	}

	/// Returns the number of rows in this relational table.
	size_t size()
	{
		if(roots[0])
			return roots[0]->el[0].size;
		else
			return 0;
	}

	/// Returns the nth row when the rows are sorted by column col.
	GRelationalRow<T>* get(size_t n, size_t col)
	{
		return roots[col]->get(n, col);
	}

	/// Returns a row where the element in column col matches the one in the provided row.
	/// Returns NULL if no matches exist. If outIndex is non-NULL, it will be made to point
	/// to the index of the returned row when sorted in column col.
	GRelationalRow<T>* find(const T& row, size_t col, size_t* outIndex = NULL)
	{
		if(roots[col])
			return roots[col]->find(row, 0, outIndex, col, comp);
		else
			return NULL;
	}

	/// Returns the first occurrence of a row where the element in column col is equal or
	/// greater than the one in the provided row. Returns NULL if no rows contain an element
	/// in column col greater than the one in row. If outIndex is non-NULL, it will be made to point
	/// to the index of the returned row when sorted in column col.
	GRelationalRow<T>* firstEqualOrGreater(const T& row, size_t col, size_t* outIndex = NULL)
	{
		if(!roots[col])
			return NULL;
		GRelationalRow<T>* node = roots[col]->approximate(row, 0, outIndex, col, comp);
		while(true)
		{
			GRelationalRow<T>* prev = node->prev(col);
			if(!prev)
				break;
			if(comp(prev->row, row, col))
				break;
			node = prev;
			if(*outIndex)
				(*outIndex)--;
		}
		while(true)
		{
			if(!comp(node->row, row, col))
				break;
			node = node->next(col);
			if(*outIndex)
				(*outIndex)++;
			if(!node)
				break;
		}
		return node;
	}

	/// Removes the specified row from this relational table.
	/// (Behavior is undefined if the specified row is not actually in the table.)
	void remove(GRelationalRow<T>* row)
	{
		for(size_t i = 0; i < comp.cols(); i++)
			roots[i] = row->remove(i, comp);
		delete(spare);
		spare = row;
		spare->isolate(comp.cols());
	}

	/// Prints a simple representation of the tree in the specified column.
	/// This method assumes that the Comp object has a method with the signature "void print(std::ostream& stream, T) const".
	void print(std::ostream& stream, size_t col)
	{
		if(roots[col])
			roots[col]->print(comp, stream, col, comp.cols(), 0);
		else
			stream << "[empty]\n";
	}
};

void GRelationalTable_test();

} // namespace GClasses

#endif // GTREE_H
