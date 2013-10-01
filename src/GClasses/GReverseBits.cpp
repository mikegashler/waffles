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

#include "GError.h"
#include "GReverseBits.h"
#include <bitset>
#ifdef WINDOWS
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
#else
#	include <stdint.h> //Change to cstdint when C++0x comes out
#endif
namespace GClasses{

  void reverseBitsTest(){
    using std::bitset;
    using std::string;
    uint8_t a = (uint8_t)bitset<8>(string("11100010")).to_ulong();
    uint8_t a_rev = (uint8_t)bitset<8>(string("01000111")).to_ulong();

    if(a != reverseBits(a_rev) && a_rev != reverseBits(a)){
      throw Ex("reverseBits failed to correctly reverse an 8 bit number.");
    }

    uint16_t b
      = (uint16_t)bitset<16>(string("1011010010000001")).to_ulong();
    uint16_t b_rev
      = (uint16_t)bitset<16>(string("1000000100101101")).to_ulong();

    if(b != reverseBits(b_rev) && b_rev != reverseBits(b)){
      throw Ex("reverseBits failed to correctly reverse a 16 bit number.");
    }

    uint32_t c
      =bitset<32>(string("01011011110111000011111001101010")).to_ulong();
    uint32_t c_rev
      =bitset<32>(string("01010110011111000011101111011010")).to_ulong();

    if(c != reverseBits(c_rev) && c_rev != reverseBits(c)){
      throw Ex("reverseBits failed to correctly reverse a 32 bit number.");
    }

  }

} //Namespace GClasses
