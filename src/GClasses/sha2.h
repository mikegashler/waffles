#ifndef _SHA256_H
#define _SHA256_H

#if defined(__cplusplus)
extern "C"
{
#endif

#include "uitypes.h"

#define SHA256_DIGEST_LENGTH    32
#define SHA384_DIGEST_LENGTH    48
#define SHA512_DIGEST_LENGTH    64

#define SHA2_GOOD   0
#define SHA2_BAD    1

/* type to hold the SHA256 context  */

#ifdef DARWIN
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
#endif // DARWIN

typedef struct
{   uint32_t count[2];
    uint32_t hash[8];
    uint32_t wdat[16];
} sha256_ctx;

typedef struct
{   uint64_t count[2];
    uint64_t hash[8];
    uint64_t wdat[16];
} sha512_ctx;

typedef sha512_ctx  sha384_ctx;

typedef struct
{   union
    {   sha256_ctx  ctx256[1];
        sha512_ctx  ctx512[1];
    } uu[1];
    uint32_t    sha2_len;
} sha2_ctx;

void sha256_begin(sha256_ctx ctx[1]);
void sha256_hash(const unsigned char data[], const unsigned long len, sha256_ctx ctx[1]);
void sha256_end(unsigned char hval[], sha256_ctx ctx[1]);

void sha384_begin(sha384_ctx ctx[1]);
#define sha384_hash sha512_hash
void sha384_end(unsigned char hval[], sha384_ctx ctx[1]);

void sha512_begin(sha512_ctx ctx[1]);
void sha512_hash(const unsigned char data[], const unsigned long len, sha512_ctx ctx[1]);
void sha512_end(unsigned char hval[], sha512_ctx ctx[1]);

int sha2_begin(sha2_ctx ctx[1], const unsigned long len);
void sha2_hash(const unsigned char data[], const unsigned long len, sha2_ctx ctx[1]);
void sha2_end(unsigned char hval[], sha2_ctx ctx[1]);

#if defined(__cplusplus)
}
#endif

#endif

