 /*
   -------------------------------------------------------------------------
   Copyright (c) 2001, Dr Brian Gladman <brg@gladman.me.uk>, Worcester, UK.
   All rights reserved.
   
   TERMS

   Redistribution and use in source and binary forms, with or without 
   modification, are permitted subject to the following conditions:

   1. Redistributions of source code must retain the above copyright 
      notice, this list of conditions and the following disclaimer. 

   2. Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the 
      documentation and/or other materials provided with the distribution. 

   3. The copyright holder's name must not be used to endorse or promote 
      any products derived from this software without his specific prior 
      written permission. 

   This software is provided 'as is' with no express or implied warranties 
   of correctness or fitness for purpose.
   -------------------------------------------------------------------------

   This is a byte oriented version of SHA256 that operates on arrays of bytes
   stored in memory. The operation uses a type 'sha256_ctx' to hold details of
   the current hash state and uses the following three calls:

   void sha256_begin(sha256_ctx ctx[])
   void sha256_hash(const unsigned char data[], unsigned long len, sha256_ctx ctx[])
   void sha256_end(unsigned char hval[], sha256_ctx ctx[])

   The first subroutine initialises a hash computation by setting up the 
   context in the sha256_ctx context.

   The second subroutine hashes 8-bit bytes from array data[] into the hash 
   state withinh sha256_ctx context, the number of bytes to be hashed being 
   given by the the unsigned long integer len.

   The third subroutine completes the hash calculation and places the
   resulting digest value in the array of 8-bit bytes hval[]

   This implementation of SHA256 also supports SHA384 and SHA512 but these
   hash functions depend on the use of 64-bit long integers and are not very 
   efficient on 32-bit machines. This code is NOT recommended for these hash
   functions.

   My thanks to Erik Andersen <andersen@codepoet-consulting.com> for testing 
   this code on big-endian systems and for his assistance with corrections
 */

/* define the hash functions that you need  */

#define SHA_2
#define SHA_256
#define SHA_384
#define SHA_512

#include <string.h>     /* for memcpy() etc.        */
#include <stdlib.h>     /* for _lrotr with VC++     */

/*  1. PLATFORM SPECIFIC INCLUDES */

#if defined(__GNU_LIBRARY__)
#  include <endian.h>
#  include <byteswap.h>
#elif defined(__CRYPTLIB__)
#  if defined( INC_ALL )
#    include "crypt.h"
#  elif defined( INC_CHILD )
#    include "../crypt.h"
#  else
#    include "crypt.h"
#  endif
#  if defined(DATA_LITTLEENDIAN)
#    define PLATFORM_BYTE_ORDER SHA_LITTLE_ENDIAN
#  else
#    define PLATFORM_BYTE_ORDER SHA_BIG_ENDIAN
#  endif
#elif defined(_MSC_VER)
#  include <stdlib.h>
#elif !defined(WIN32)
#  include <stdlib.h>
#  if !defined (_ENDIAN_H)
#    include <sys/param.h>
#  else
#    include _ENDIAN_H
#  endif
#endif

/*  2. BYTE ORDER IN 32-BIT WORDS

    To obtain the highest speed on processors with 32-bit words, this code 
    needs to determine the order in which bytes are packed into such words.
    The following block of code is an attempt to capture the most obvious 
    ways in which various environemnts specify their endian definitions. 
	It may well fail, in which case the definitions will need to be set by 
    editing at the points marked **** EDIT HERE IF NECESSARY **** below.
*/
#define SHA_LITTLE_ENDIAN   1234 /* byte 0 is least significant (i386) */
#define SHA_BIG_ENDIAN      4321 /* byte 0 is most significant (mc68k) */

#if !defined(PLATFORM_BYTE_ORDER)
#if defined(LITTLE_ENDIAN) || defined(BIG_ENDIAN)
#  if defined(LITTLE_ENDIAN) && defined(BIG_ENDIAN)
#    if defined(BYTE_ORDER)
#      if   (BYTE_ORDER == LITTLE_ENDIAN)
#        define PLATFORM_BYTE_ORDER SHA_LITTLE_ENDIAN
#      elif (BYTE_ORDER == BIG_ENDIAN)
#        define PLATFORM_BYTE_ORDER SHA_BIG_ENDIAN
#      endif
#    endif
#  elif defined(LITTLE_ENDIAN) && !defined(BIG_ENDIAN) 
#    define PLATFORM_BYTE_ORDER SHA_LITTLE_ENDIAN
#  elif !defined(LITTLE_ENDIAN) && defined(BIG_ENDIAN)
#    define PLATFORM_BYTE_ORDER SHA_BIG_ENDIAN
#  endif
#elif defined(_LITTLE_ENDIAN) || defined(_BIG_ENDIAN)
#  if defined(_LITTLE_ENDIAN) && defined(_BIG_ENDIAN)
#    if defined(_BYTE_ORDER)
#      if   (_BYTE_ORDER == _LITTLE_ENDIAN)
#        define PLATFORM_BYTE_ORDER SHA_LITTLE_ENDIAN
#      elif (_BYTE_ORDER == _BIG_ENDIAN)
#        define PLATFORM_BYTE_ORDER SHA_BIG_ENDIAN
#      endif
#    endif
#  elif defined(_LITTLE_ENDIAN) && !defined(_BIG_ENDIAN) 
#    define PLATFORM_BYTE_ORDER SHA_LITTLE_ENDIAN
#  elif !defined(_LITTLE_ENDIAN) && defined(_BIG_ENDIAN)
#    define PLATFORM_BYTE_ORDER SHA_BIG_ENDIAN
#  endif
#elif 0     /* **** EDIT HERE IF NECESSARY **** */
#define PLATFORM_BYTE_ORDER SHA_LITTLE_ENDIAN
#elif defined(BYTE_ORDER_BIG_ENDIAN)
#define PLATFORM_BYTE_ORDER SHA_BIG_ENDIAN
#else
#  define PLATFORM_BYTE_ORDER SHA_LITTLE_ENDIAN
#endif
#endif

#if !defined(PLATFORM_BYTE_ORDER)
#  error Please set undetermined byte order (lines 134 or 136 of sha2.c).
#endif

/* this Microsft VC++ intrinsic rotate makes a big difference to the speed of this code */

#if defined(_MSC_VER)
#define rotr32(x,n)   _lrotr(x,n)
#else
#define rotr32(x,n)   (((x) >> n) | ((x) << (32 - n)))
#endif

#define rotr64(x,n)   (((x) >> n) | ((x) << (64 - n)))

/* reverse byte order in 32-bit words   */

#if !defined(bswap_32)
#define bswap_32(x) (rotr32((x), 24) & 0x00ff00ff | rotr32((x), 8) & 0xff00ff00)
#endif

#if !defined(bswap_64)
#define bswap_64(x) (((uint64_t)(bswap_32((uint32_t)(x)))) << 32 | bswap_32((uint32_t)((x) >> 32)))
#endif

#include "sha2.h"

/*  Defining FAST_COPY will generally improve speed but it assumes that 
    arrays of 32-bit words can be addressed as arrays of bytes by 
    casting the array base address. Defining WORD_COPY avoids this problem 
    by assembling bytes into a word variable before copying to memory. If
    neither is defined a slow but safe byte oriented version is used.
*/
#if 1
#define FAST_COPY
#elif 0
#define WORD_COPY
#endif

#if defined(FAST_COPY) && (PLATFORM_BYTE_ORDER == SHA_LITTLE_ENDIAN)
#define SWAP_BYTES
#else
#undef  SWAP_BYTES
#endif

#if defined(SHA_2) || defined(SHA_256)

/* SHA256 mixing function definitions   */

#define ch(x,y,z)   (((x) & (y)) ^ (~(x) & (z)))
#define maj(x,y,z)  (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))

#define s256_0(x) (rotr32((x),  2) ^ rotr32((x), 13) ^ rotr32((x), 22)) 
#define s256_1(x) (rotr32((x),  6) ^ rotr32((x), 11) ^ rotr32((x), 25)) 
#define g256_0(x) (rotr32((x),  7) ^ rotr32((x), 18) ^ ((x) >>  3)) 
#define g256_1(x) (rotr32((x), 17) ^ rotr32((x), 19) ^ ((x) >> 10)) 

/* rotated SHA256 round definition. Rather than swapping variables as in    */
/* FIPS-180, different variables are 'rotated' on each round, returning     */
/* to their starting positions every eight rounds                           */

#define h2(i) ctx->wdat[i & 15] += \
    g256_1(ctx->wdat[(i + 14) & 15]) + ctx->wdat[(i + 9) & 15] + g256_0(ctx->wdat[(i + 1) & 15])

#define h2_cycle(i,j)  \
    v[(7 - i) & 7] += (j ? h2(i) : ctx->wdat[i & 15]) + k256[i + j] \
        + s256_1(v[(4 - i) & 7]) + ch(v[(4 - i) & 7], v[(5 - i) & 7], v[(6 - i) & 7]); \
    v[(3 - i) & 7] += v[(7 - i) & 7]; \
    v[(7 - i) & 7] += s256_0(v[(0 - i) & 7]) + maj(v[(0 - i) & 7], v[(1 - i) & 7], v[(2 - i) & 7])

/* SHA256 mixing data   */

const uint32_t k256[64] =
{   x_32(428a2f98), x_32(71374491), x_32(b5c0fbcf), x_32(e9b5dba5), 
    x_32(3956c25b), x_32(59f111f1), x_32(923f82a4), x_32(ab1c5ed5), 
    x_32(d807aa98), x_32(12835b01), x_32(243185be), x_32(550c7dc3), 
    x_32(72be5d74), x_32(80deb1fe), x_32(9bdc06a7), x_32(c19bf174), 
    x_32(e49b69c1), x_32(efbe4786), x_32(0fc19dc6), x_32(240ca1cc), 
    x_32(2de92c6f), x_32(4a7484aa), x_32(5cb0a9dc), x_32(76f988da), 
    x_32(983e5152), x_32(a831c66d), x_32(b00327c8), x_32(bf597fc7), 
    x_32(c6e00bf3), x_32(d5a79147), x_32(06ca6351), x_32(14292967), 
    x_32(27b70a85), x_32(2e1b2138), x_32(4d2c6dfc), x_32(53380d13), 
    x_32(650a7354), x_32(766a0abb), x_32(81c2c92e), x_32(92722c85),
    x_32(a2bfe8a1), x_32(a81a664b), x_32(c24b8b70), x_32(c76c51a3), 
    x_32(d192e819), x_32(d6990624), x_32(f40e3585), x_32(106aa070), 
    x_32(19a4c116), x_32(1e376c08), x_32(2748774c), x_32(34b0bcb5), 
    x_32(391c0cb3), x_32(4ed8aa4a), x_32(5b9cca4f), x_32(682e6ff3), 
    x_32(748f82ee), x_32(78a5636f), x_32(84c87814), x_32(8cc70208), 
    x_32(90befffa), x_32(a4506ceb), x_32(bef9a3f7), x_32(c67178f2),
};

/* Compile 64 bytes of hash data into SHA256 digest value  */

static void sha256_compile(sha256_ctx ctx[1])
{   uint32_t    v[8], j;

    memcpy(v, ctx->hash, 32);

    for(j = 0; j < 64; j += 16)
    {
        h2_cycle( 0, j); h2_cycle( 1, j); h2_cycle( 2, j); h2_cycle( 3, j);
        h2_cycle( 4, j); h2_cycle( 5, j); h2_cycle( 6, j); h2_cycle( 7, j);
        h2_cycle( 8, j); h2_cycle( 9, j); h2_cycle(10, j); h2_cycle(11, j);
        h2_cycle(12, j); h2_cycle(13, j); h2_cycle(14, j); h2_cycle(15, j);
    }

    ctx->hash[0] += v[0]; ctx->hash[1] += v[1]; ctx->hash[2] += v[2]; ctx->hash[3] += v[3];
    ctx->hash[4] += v[4]; ctx->hash[5] += v[5]; ctx->hash[6] += v[6]; ctx->hash[7] += v[7];
}

/* SHA256 hash data in an array of bytes into hash buffer and call the      */
/* hash_compile function as required.                                       */

/* If FAST_COPY is defined the data to be hashed is processed as an array   */
/* bytes and compiled into the buffer ctx->wdat[] of 32-bit words in the    */
/* native byte order.  On little endian machines a 32-bit word byte swap    */
/* is then performed before this data is compiled into the hash. But when   */
/* FAST_COPY is not defined the bytes are compiled into the buffer in the   */
/* big-endian format directly so no later byte order changes are needed.    */

#if defined(FAST_COPY)

void sha256_hash(const unsigned char data[], const unsigned long len, sha256_ctx ctx[1])
{   uint32_t            free = 64 - (uint32_t)(ctx->count[0] & 63), rlen = len, j;
    const unsigned char *sp = data;
    unsigned char       *p = ((unsigned char*)ctx->wdat) + (ctx->count[0] & 63);

    while(rlen >= free)     /* tranfer whole blocks while possible  */
    {
        memcpy(p, sp, free);
        if((ctx->count[0] += free) < free)
            ++(ctx->count[1]);
        sp += free; rlen -= free; free = 64; 
        p = (unsigned char*)ctx->wdat;
#if defined(SWAP_BYTES)
        for(j = 0; j < 16; ++j)
            ctx->wdat[j] = bswap_32(ctx->wdat[j]);
#endif
        sha256_compile(ctx);
    }

    memcpy(p, sp, rlen);    /* transfer partial block               */
    if((ctx->count[0] += rlen) < rlen)
        ++(ctx->count[1]);
    p = (unsigned char*)ctx->wdat;
    rlen = (ctx->count[0] & 63);
    while(rlen & 3)         /* ensure rest of 32-bit word is zero   */
        *(p + rlen++) = 0;	/* [could be done in sha256_end()]		*/
}

#elif defined(WORD_COPY)    /* alternative implementations          */

void sha256_hash(const unsigned char data[], unsigned long len, sha256_ctx ctx[1])
{   uint32_t i = 0, *ptr, cnt, val, j;

    ptr = ctx->wdat + ((ctx->count[0] >> 2) & 15);
    cnt = (ctx->count[0] << 3) & 24;
    val = (cnt ? *ptr : 0);
    while(i < len)
    {   /* assemble values in big-endian format */
        val |= ((uint32_t)data[i++]) << 24 - cnt;
        if(!++(ctx->count[0])) ++(ctx->count[1]);
        if(!(cnt = (cnt + 8) & 24))
        {   *ptr++ = val; val = 0;
            if(ptr - ctx->wdat == 16) 
            {
                sha256_compile(ctx); 
                ptr = ctx->wdat;
            }
        }
    }
    *ptr = val;
}

#else

void sha256_hash(const unsigned char data[], unsigned long len, sha256_ctx ctx[1])
{   uint32_t i = 0, cnt = ctx->count[0], j;

    while(i < len)
    {   /* assemble values in big-endian format */
        if((cnt & 3) == 0) ctx->wdat[(cnt >> 2) & 15] = 0;
        ctx->wdat[(cnt >> 2) & 15] |= (uint32_t)data[i++] << (24 - (8 * (cnt & 3)));
        if(!++(ctx->count[0])) ++(ctx->count[1]);
        if((++cnt & 63) == 0) 
        {
            sha256_compile(ctx);
        }
    }
}

#endif

/* SHA256 Final padding and digest calculation  */

void sha256_end(unsigned char hval[], sha256_ctx ctx[1])
{   uint32_t    i, j, cnt = (uint32_t)(ctx->count[0] & 63);

#if defined(SWAP_BYTES)
        for(i = 0, j = (cnt + 3) >> 2; i < j; ++i)
            ctx->wdat[i] = bswap_32(ctx->wdat[i]);
#endif

    /* we are now in big-endian order within the ctx->wdat[] buffer	*/
	/* we now need to add the padding which is a single 1 bit and	*/
	/* as many zero bits as necessary. Start by padding out the		*/
	/* last valid 32-bit word in the buffer							*/

    if(cnt & 3)
        ctx->wdat[(cnt >> 2) & 15] |= x_32(00000080) << (24  - (8 * (cnt & 3)));
    else
        ctx->wdat[(cnt >> 2) & 15] = x_32(80000000);

    if((cnt & 63) > 55)	/* there is not enough space in the buffer	*/
    {					/* for the length field - pad and empty it	*/
        if((cnt & 63) == 55) ctx->wdat[14] = 0;
        if((cnt & 63) <= 59) ctx->wdat[15] = 0;
        sha256_compile(ctx);
        cnt = 0;
    }
    else	/* compute a word index for the empty buffer positions	*/
        cnt = (cnt >> 2) + 1;

    while(cnt < 14)	/* and zero pad all but last two positions		*/ 
        ctx->wdat[cnt++] = 0;
    
    /* the following 32-bit length fields are assembled in the		*/
	/* wrong byte order on little endian machines but this is		*/
	/* corrected later since they are only ever used as 32-bit		*/
	/* word values.													*/

    ctx->wdat[14] = (ctx->count[1] << 3) | (ctx->count[0] >> 29);
    ctx->wdat[15] = ctx->count[0] << 3;

    sha256_compile(ctx);

	/* extract the hash value as bytes in case the hash buffer is	*/
	/* mislaigned for 32-bit words									*/

	for(i = 0; i < SHA256_DIGEST_LENGTH; ++i)
        hval[i] = (unsigned char)(ctx->hash[i >> 2] >> (24 - 8 * (i & 3)));
}

#endif

#if defined(SHA_2) || defined(SHA_384) || defined(SHA_512)

/* SHA512 mixing function definitions   */

#define s512_0(x) (rotr64((x), 28) ^ rotr64((x), 34) ^ rotr64((x), 39)) 
#define s512_1(x) (rotr64((x), 14) ^ rotr64((x), 18) ^ rotr64((x), 41)) 
#define g512_0(x) (rotr64((x),  1) ^ rotr64((x),  8) ^ ((x) >>  7)) 
#define g512_1(x) (rotr64((x), 19) ^ rotr64((x), 61) ^ ((x) >>  6)) 

/* rotated SHA512 round definition. Rather than swapping variables as in    */
/* FIPS-180, different variables are 'rotated' on each round, returning     */
/* to their starting positions every eight rounds                           */

#define h5(i) ctx->wdat[i & 15] += \
    g512_1(ctx->wdat[(i + 14) & 15]) + ctx->wdat[(i + 9) & 15] + g512_0(ctx->wdat[(i + 1) & 15])

#define h5_cycle(i,j)  \
    v[(7 - i) & 7] += (j ? h5(i) : ctx->wdat[i & 15]) + k512[i + j] \
        + s512_1(v[(4 - i) & 7]) + ch(v[(4 - i) & 7], v[(5 - i) & 7], v[(6 - i) & 7]); \
    v[(3 - i) & 7] += v[(7 - i) & 7]; \
    v[(7 - i) & 7] += s512_0(v[(0 - i) & 7]) + maj(v[(0 - i) & 7], v[(1 - i) & 7], v[(2 - i) & 7])

/* SHA384/SHA512 mixing data    */

const uint64_t  k512[80] = 
{
    x_64(428a2f98d728ae22), x_64(7137449123ef65cd), 
    x_64(b5c0fbcfec4d3b2f), x_64(e9b5dba58189dbbc),
    x_64(3956c25bf348b538), x_64(59f111f1b605d019),
    x_64(923f82a4af194f9b), x_64(ab1c5ed5da6d8118),
    x_64(d807aa98a3030242), x_64(12835b0145706fbe),
    x_64(243185be4ee4b28c), x_64(550c7dc3d5ffb4e2),
    x_64(72be5d74f27b896f), x_64(80deb1fe3b1696b1),
    x_64(9bdc06a725c71235), x_64(c19bf174cf692694),
    x_64(e49b69c19ef14ad2), x_64(efbe4786384f25e3),
    x_64(0fc19dc68b8cd5b5), x_64(240ca1cc77ac9c65),
    x_64(2de92c6f592b0275), x_64(4a7484aa6ea6e483),
    x_64(5cb0a9dcbd41fbd4), x_64(76f988da831153b5),
    x_64(983e5152ee66dfab), x_64(a831c66d2db43210),
    x_64(b00327c898fb213f), x_64(bf597fc7beef0ee4),
    x_64(c6e00bf33da88fc2), x_64(d5a79147930aa725),
    x_64(06ca6351e003826f), x_64(142929670a0e6e70),
    x_64(27b70a8546d22ffc), x_64(2e1b21385c26c926),
    x_64(4d2c6dfc5ac42aed), x_64(53380d139d95b3df),
    x_64(650a73548baf63de), x_64(766a0abb3c77b2a8),
    x_64(81c2c92e47edaee6), x_64(92722c851482353b),
    x_64(a2bfe8a14cf10364), x_64(a81a664bbc423001),
    x_64(c24b8b70d0f89791), x_64(c76c51a30654be30),
    x_64(d192e819d6ef5218), x_64(d69906245565a910),
    x_64(f40e35855771202a), x_64(106aa07032bbd1b8),
    x_64(19a4c116b8d2d0c8), x_64(1e376c085141ab53),
    x_64(2748774cdf8eeb99), x_64(34b0bcb5e19b48a8),
    x_64(391c0cb3c5c95a63), x_64(4ed8aa4ae3418acb),
    x_64(5b9cca4f7763e373), x_64(682e6ff3d6b2b8a3),
    x_64(748f82ee5defb2fc), x_64(78a5636f43172f60),
    x_64(84c87814a1f0ab72), x_64(8cc702081a6439ec),
    x_64(90befffa23631e28), x_64(a4506cebde82bde9),
    x_64(bef9a3f7b2c67915), x_64(c67178f2e372532b),
    x_64(ca273eceea26619c), x_64(d186b8c721c0c207),
    x_64(eada7dd6cde0eb1e), x_64(f57d4f7fee6ed178),
    x_64(06f067aa72176fba), x_64(0a637dc5a2c898a6),
    x_64(113f9804bef90dae), x_64(1b710b35131c471b),
    x_64(28db77f523047d84), x_64(32caab7b40c72493),
    x_64(3c9ebe0a15c9bebc), x_64(431d67c49c100d4c),
    x_64(4cc5d4becb3e42b6), x_64(597f299cfc657e2a),
    x_64(5fcb6fab3ad6faec), x_64(6c44198c4a475817)
};

/* Compile 64 bytes of hash data into SHA384/SHA512 digest value  */

static void sha512_compile(sha512_ctx ctx[1])
{   uint64_t    v[8];
    uint32_t    j;

    memcpy(v, ctx->hash, 64);

    for(j = 0; j < 80; j += 16)
    {
        h5_cycle( 0, j); h5_cycle( 1, j); h5_cycle( 2, j); h5_cycle( 3, j);
        h5_cycle( 4, j); h5_cycle( 5, j); h5_cycle( 6, j); h5_cycle( 7, j);
        h5_cycle( 8, j); h5_cycle( 9, j); h5_cycle(10, j); h5_cycle(11, j);
        h5_cycle(12, j); h5_cycle(13, j); h5_cycle(14, j); h5_cycle(15, j);
    }

    ctx->hash[0] += v[0]; ctx->hash[1] += v[1]; ctx->hash[2] += v[2]; ctx->hash[3] += v[3];
    ctx->hash[4] += v[4]; ctx->hash[5] += v[5]; ctx->hash[6] += v[6]; ctx->hash[7] += v[7];
}

/* SHA512 hash data in an array of bytes into hash buffer and call the      */
/* hash_compile function as required.                                       */

/* If FAST_COPY is defined the data to be hashed is processed as an array   */
/* bytes and compiled into the buffer ctx->wdat[] of 32-bit words in the    */
/* native byte order.  On little endian machines a 32-bit word byte swap    */
/* is then performed before this data is compiled into the hash. But when   */
/* FAST_COPY is not defined the bytes are compiled into the buffer in the   */
/* big-endian format directly so no later byte order changes are needed.    */

#if defined(FAST_COPY)

void sha512_hash(const unsigned char data[], const unsigned long len, sha512_ctx ctx[1])
{   uint32_t            free = 128 - (uint32_t)(ctx->count[0] & 127), rlen = len, j;
    const unsigned char *sp = data;
    unsigned char       *p = ((unsigned char*)ctx->wdat) + (ctx->count[0] & 127);

    while(rlen >= free)     /* tranfer whole blocks while possible  */
    {
        memcpy(p, sp, free);
        if((ctx->count[0] += free) < free)
            ++(ctx->count[1]);
        sp += free; rlen -= free; free = 128; 
        p = (unsigned char*)ctx->wdat;
#if defined(SWAP_BYTES)
        for(j = 0; j < 16; ++j)
            ctx->wdat[j] = bswap_64(ctx->wdat[j]);
#endif
        sha512_compile(ctx);
    }

    memcpy(p, sp, rlen);    /* transfer partial block               */
    if((ctx->count[0] += rlen) < rlen)
        ++(ctx->count[1]);
    p = (unsigned char*)ctx->wdat;
    rlen = (uint32_t)(ctx->count[0] & 127);
    while(rlen & 7)         /* ensure rest of 32-bit word is zero   */
        *(p + rlen++) = 0;
}

#elif defined(WORD_COPY)    /* alternative implementations          */

void sha512_hash(const unsigned char data[], unsigned long len, sha512_ctx ctx[1])
{   uint32_t    i = 0, cnt, j;
    uint64_t    *ptr, val;

    ptr = ctx->wdat + ((ctx->count[0] >> 3) & 15);
    cnt = (ctx->count[0] << 3) & 56;
    val = (cnt ? *ptr : 0);
    while(i < len)
    {   /* assemble values in big-endian format */
        val |= ((uint64_t)data[i++]) << 56 - cnt;
        if(!++(ctx->count[0])) ++(ctx->count[1]);
        if(!(cnt = (cnt + 8) & 56))
        {   *ptr++ = val; val = 0;
            if(ptr - ctx->wdat == 16) 
            {
                sha512_compile(ctx); 
                ptr = ctx->wdat;
            }
        }
    }
    *ptr = val;
}

#else

void sha512_hash(const unsigned char data[], unsigned long len, sha512_ctx ctx[1])
{   uint32_t i = 0, cnt = ctx->count[0], j;

    while(i < len)
    {   /* assemble values in big-endian format */
        if((cnt & 7) == 0) ctx->wdat[(cnt >> 3) & 15] = 0;
        ctx->wdat[(cnt >> 3) & 15] |= (uint64_t)data[i++] << (56 - (8 * (cnt & 7)));
        if(!++(ctx->count[0])) ++(ctx->count[1]);
        if((++cnt & 127) == 0) 
        {
            sha512_compile(ctx);
        }
    }
}

#endif

/* SHA384/512 Final padding and digest calculation  */

static void sha_end(unsigned char hval[], sha512_ctx ctx[1], const unsigned int hlen)
{   uint32_t    i, j, cnt = (uint32_t)(ctx->count[0] & 127);

#if defined(SWAP_BYTES)
        for(i = 0, j = (cnt + 7) >> 3; i < j; ++i)
            ctx->wdat[i] = bswap_64(ctx->wdat[i]);
#endif

    /* we are now in big-endian order within the ctx-wdat[] buffer  */

    if(cnt & 7)
        ctx->wdat[(cnt >> 3) & 15] 
            |= x_64(0000000000000080) << (56  - (8 * (cnt & 7)));
    else
        ctx->wdat[(cnt >> 3) & 15] = x_64(8000000000000000);

    if((cnt & 127) > 111)
    {
        if((cnt & 127) == 111) ctx->wdat[14] = 0;
        if((cnt & 127) <= 119) ctx->wdat[15] = 0;
        sha512_compile(ctx);
        cnt = 0;
    }
    else
        cnt = (cnt >> 3) + 1;

    while(cnt < 14)
        ctx->wdat[cnt++] = 0;
    
    /* the following 32-bit length fields are assembled in the wrong	*/
	/* byte	order on little endian machines but this is corrected later	*/
	/* since they are only picked up as 32-bit word values.				*/

    ctx->wdat[14] = (ctx->count[1] << 3) | (ctx->count[0] >> 29);
    ctx->wdat[15] = ctx->count[0] << 3;

    sha512_compile(ctx);

    for(i = 0; i < hlen; ++i)
        hval[i] = (unsigned char)(ctx->hash[i >> 3] >> (56 - 8 * (i & 7)));
}

#endif

#if defined(SHA_2) || defined(SHA_256)

/* SHA256 initialisation data */

const uint32_t i256[8] =
{
    x_32(6a09e667), x_32(bb67ae85), x_32(3c6ef372), x_32(a54ff53a),
    x_32(510e527f), x_32(9b05688c), x_32(1f83d9ab), x_32(5be0cd19)
};

#if defined(SHA_256)

void sha256_begin(sha256_ctx ctx[1])
{
    ctx->count[0] = ctx->count[1] = 0;
    memcpy(ctx->hash, i256, 32);
}

#endif
#endif

#if defined(SHA_2) || defined(SHA_384)

/* SHA384 initialisation data   */

const uint64_t  i384[80] = 
{
    x_64(cbbb9d5dc1059ed8), x_64(629a292a367cd507),
    x_64(9159015a3070dd17), x_64(152fecd8f70e5939),
    x_64(67332667ffc00b31), x_64(8eb44a8768581511),
    x_64(db0c2e0d64f98fa7), x_64(47b5481dbefa4fa4)
};

#if defined(SHA_384)

void sha384_begin(sha384_ctx ctx[1])
{
    ctx->count[0] = ctx->count[1] = 0;
    memcpy(ctx->hash, i384, 64);
}

void sha384_end(unsigned char hval[], sha384_ctx ctx[1])
{
    sha_end(hval, ctx, SHA384_DIGEST_LENGTH);
}

#endif
#endif

#if defined(SHA_2) || defined(SHA_512)

/* SHA512 initialisation data   */

const uint64_t  i512[80] = 
{
    x_64(6a09e667f3bcc908), x_64(bb67ae8584caa73b),
    x_64(3c6ef372fe94f82b), x_64(a54ff53a5f1d36f1),
    x_64(510e527fade682d1), x_64(9b05688c2b3e6c1f),
    x_64(1f83d9abfb41bd6b), x_64(5be0cd19137e2179)
};

#if defined(SHA_512)

void sha512_begin(sha512_ctx ctx[1])
{
    ctx->count[0] = ctx->count[1] = 0;
    memcpy(ctx->hash, i512, 64);
}

void sha512_end(unsigned char hval[], sha512_ctx ctx[1])
{
    sha_end(hval, ctx, SHA512_DIGEST_LENGTH);
}

#endif
#endif

#if defined(SHA_2)

#define CTX_256(x)  ((x)->uu->ctx256)
#define CTX_384(x)  ((x)->uu->ctx512)
#define CTX_512(x)  ((x)->uu->ctx512)

/* SHA2 initialisation */

int sha2_begin(sha2_ctx ctx[1], const unsigned long len)
{   unsigned long   l = len;
    switch(len)
    {
        case 256:   l = len >> 3;
        case  32:   CTX_256(ctx)->count[0] = CTX_256(ctx)->count[1] = 0;
                    memcpy(CTX_256(ctx)->hash, i256, 32); break;
        case 384:   l = len >> 3;
        case  48:   CTX_384(ctx)->count[0] = CTX_384(ctx)->count[1] = 0;
                    memcpy(CTX_384(ctx)->hash, i384, 64); break;
        case 512:   l = len >> 3;
        case  64:   CTX_512(ctx)->count[0] = CTX_512(ctx)->count[1] = 0;
                    memcpy(CTX_512(ctx)->hash, i512, 64); break;
        default:    return SHA2_BAD;
    }
    
    ctx->sha2_len = l; return SHA2_GOOD;
}

void sha2_hash(const unsigned char data[], const unsigned long len, sha2_ctx ctx[1])
{
    switch(ctx->sha2_len)
    {
        case 32: sha256_hash(data, len, CTX_256(ctx)); return;
        case 48: sha384_hash(data, len, CTX_384(ctx)); return;
        case 64: sha512_hash(data, len, CTX_512(ctx)); return;
    }
}

void sha2_end(unsigned char hval[], sha2_ctx ctx[1])
{
    switch(ctx->sha2_len)
    {
        case 32: sha256_end(hval, CTX_256(ctx)); return;
        case 48: sha_end(hval, CTX_384(ctx), SHA384_DIGEST_LENGTH); return;
        case 64: sha_end(hval, CTX_512(ctx), SHA512_DIGEST_LENGTH); return;
    }
}

#endif


