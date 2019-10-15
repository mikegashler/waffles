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

#include "GTree.h"
#include <string>

using std::string;

namespace GClasses {

class MyRelationalObject
{
public:
	string m_s;
	int m_n;

	MyRelationalObject(int n, string s) : m_s(s), m_n(n) {}
};

class MyRelationalObjectComparer
{
public:
	size_t cols() const { return 3; }

	bool operator ()(const MyRelationalObject* a, const MyRelationalObject* b, size_t col) const
	{
		if(col == 0)
			return a->m_n < b->m_n;
		else if(col == 1)
			return a->m_n * a->m_s.length() < b->m_n * b->m_s.length();
		else
			return a->m_s.compare(b->m_s) > 0;
	}
};

void GRelationalTable_test()
{
	MyRelationalObject ob1(9, "a"); // 9*1=9
	MyRelationalObject ob2(4, "yo"); // 4*2=8
	MyRelationalObject ob3(1, "abcdefghijklmnopqrstuvwxyz"); // 1*26=26
	MyRelationalObject ob4(3, "sasquatch"); // 3*9=27
	MyRelationalObjectComparer comp;
	GRelationalTable<MyRelationalObject*,MyRelationalObjectComparer> table(comp);
	table.insert(&ob1);
	table.insert(&ob2);
	table.insert(&ob3);
	table.insert(&ob4);

	// Check the size
	if(table.size() != 4)
		throw Ex("failed");

	// Check column 0 (sorted by m_n, smallest first)
	if(table.get(0, 0)->row->m_n != 1)
		throw Ex("failed");
	if(table.get(1, 0)->row->m_n != 3)
		throw Ex("failed");
	if(table.get(2, 0)->row->m_n != 4)
		throw Ex("failed");
	if(table.get(3, 0)->row->m_n != 9)
		throw Ex("failed");

	// Check column 1 (sorted by m_n*m_s.length(), smallest first)
	if(table.get(0, 1)->row->m_n != 4)
		throw Ex("failed");
	if(table.get(1, 1)->row->m_n != 9)
		throw Ex("failed");
	if(table.get(2, 1)->row->m_n != 1)
		throw Ex("failed");
	if(table.get(3, 1)->row->m_n != 3)
		throw Ex("failed");

	// Check column 2 (sorted by m_s, reverse-alphabetically)
	if(table.get(0, 2)->row->m_n != 4)
		throw Ex("failed");
	if(table.get(1, 2)->row->m_n != 3)
		throw Ex("failed");
	if(table.get(2, 2)->row->m_n != 1)
		throw Ex("failed");
	if(table.get(3, 2)->row->m_n != 9)
		throw Ex("failed");

	// Test find and findGreaterOrEqual
	MyRelationalObject t1(4, "banana");
	MyRelationalObject t2(5, "banana");
	GRelationalRow<MyRelationalObject*>* ta = table.find(&t1, 0);
	if(!ta)
		throw Ex("failed");
	if(ta->row->m_n != 4)
		throw Ex("failed");
	GRelationalRow<MyRelationalObject*>* tb = table.find(&t2, 0);
	if(tb)
		throw Ex("failed");
	GRelationalRow<MyRelationalObject*>* tc = table.firstEqualOrGreater(&t1, 0);
	if(!tc)
		throw Ex("failed");
	if(tc->row->m_n != 4)
		throw Ex("failed");
	GRelationalRow<MyRelationalObject*>* td = table.firstEqualOrGreater(&t2, 0);
	if(!td)
		throw Ex("failed");
	if(td->row->m_n != 9)
		throw Ex("failed");

	// Remove two items
	table.remove(table.get(0, 2)); // remove {4, "yo"}
	table.remove(table.get(1, 2)); // remove {1, "abcdefghijklmnopqrstuvwxyz"}

	// Check the size
	if(table.size() != 2)
		throw Ex("failed");

	// Check the remaining values
	if(table.get(0, 0)->row->m_n != 3)
		throw Ex("failed");
	if(table.get(1, 0)->row->m_n != 9)
		throw Ex("failed");

	// Check column 1 (sorted by m_n*m_s.length(), smallest first)
	if(table.get(0, 1)->row->m_n != 9)
		throw Ex("failed");
	if(table.get(1, 1)->row->m_n != 3)
		throw Ex("failed");

	// Check column 2 (sorted by m_s, reverse-alphabetically)
	if(table.get(0, 2)->row->m_n != 3)
		throw Ex("failed");
	if(table.get(1, 2)->row->m_n != 9)
		throw Ex("failed");

	// Remove the reest
	table.remove(table.get(0, 0));
	table.remove(table.get(0, 0));

	// Check the size
	if(table.size() != 0)
		throw Ex("failed");
}

} // namespace GClasses

