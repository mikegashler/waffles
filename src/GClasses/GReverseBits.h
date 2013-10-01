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

#include <limits>
#include <cassert>
namespace GClasses{

  ///Template used for reversing numBits of type T.  You shouldn't
  ///need this, use the function reverseBits
  ///
  ///Taken from the post at 
  ///http://www.velocityreviews.com/forums/t457514-reverse-bit-order.html
  ///by
  ///Pete Becker, Author of "The Standard C++ Library Extensions: a Tutorial and
  ///Reference"
  template <class T, unsigned numBits> struct GBitReverser_imp {
    static inline T reverse(T val, T mask){
      mask >>= (numBits/2);
      return GBitReverser_imp<T, numBits/2>::reverse((val >> (numBits/2)) & mask, mask)
	| (GBitReverser_imp<T, numBits/2>::reverse(val & mask, mask) << (numBits/2));
    }
  };
  
  ///Base case of template used for reversing numBits of type T.  You
  ///shouldn't need this, use the function reverseBits
  ///
  ///Taken from the post at 
  ///http://www.velocityreviews.com/forums/t457514-reverse-bit-order.html
  ///by
  ///Pete Becker, Author of "The Standard C++ Library Extensions: a Tutorial and
  ///Reference"
  template <class T> struct GBitReverser_imp<T,1>{
    static inline T reverse(T val, T)
    {
      return val;
    }
  };

  
  ///Reverses the bits of value given that T is an unsigned integral
  ///type with binary representation and a number of bits that are a
  ///power of 2.
  ///
  ///Modified (added checks and some consts for readability) from the post at 
  ///http://www.velocityreviews.com/forums/t457514-reverse-bit-order.html
  ///by
  ///Pete Becker, Author of "The Standard C++ Library Extensions: a Tutorial and
  ///Reference"
  template<class T> 
  T reverseBits(T value){
    assert(!std::numeric_limits<T>::is_signed); //Is unsigned
    assert(std::numeric_limits<T>::is_integer); //Is integral type?
    assert(std::numeric_limits<T>::radix == 2); //Is binary
    const int bits = std::numeric_limits<T>::digits;
    const T max = std::numeric_limits<T>::max();
    return GBitReverser_imp<T,bits>::reverse(value, max);
  }

  void reverseBitsTest();

} //Namespace GClasses
