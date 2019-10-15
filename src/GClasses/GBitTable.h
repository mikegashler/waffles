/*
  The contents of this file are dedicated by all of its authors, including

    Michael S. Gashler,
    Eric Moyer,
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

#ifndef __GBITTABLE_H__
#define __GBITTABLE_H__

#include "GError.h"

namespace GClasses {

/// Represents a table of bits.
class GBitTable
{
protected:
	size_t m_size;
	size_t* m_pBits;

public:
	/// All bits are initialized to false
	GBitTable(size_t bitCount);
	
	///Copy Constructor
	GBitTable(const GBitTable& o);

	///Operator=
	GBitTable& operator=(const GBitTable& o);

	virtual ~GBitTable();

	/// Performs unit tests for this class. Throws an exception if there is a failure.
	static void test();

	/// Sets all bits to false
	void clearAll();

	/// Sets all bits to true
	void setAll();

	/// Returns the bit at index
	bool bit(size_t index) const;

	/// Sets the bit at index
	void set(size_t index);

	/// Clears the bit at index
	void unset(size_t index);

	/// Toggles the bit at index
	void toggle(size_t index);

	/// Returns true iff the bit tables are exactly equal.
	/// Returns false if the tables are not the same size.
	bool equals(const GBitTable& that) const;

	/// Returns true iff the first "count" bits are set. (Note that
	/// for most applications, it is more efficient to simply maintain
	/// a count of the number of bits that are set than to call this method.)
	bool areAllSet(size_t count);

	/// Returns true iff the first "count" bits are clear
	bool areAllClear(size_t count);
};

} // namespace GClasses

#endif // __GBITTABLE_H__
