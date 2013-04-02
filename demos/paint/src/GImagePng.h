// -------------------------------------------------------------
// The contents of this file may be distributed under the CC0
// license (http://creativecommons.org/publicdomain/zero/1.0/).
// Of course you may also distribute it under a more restrictive
// license, such as any of the OSI-approved licenses
// (http://www.opensource.org/licenses).
// -------------------------------------------------------------

#ifndef GIMAGEPNG_H
#define GIMAGEPNG_H

namespace GClasses {

class GImage;

void loadPng(GImage* pImage, const unsigned char* pData, size_t nDataSize);
void loadPng(GImage* pImage, const char* szFilename);
void loadPngFromHex(GClasses::GImage* pImage, const char* szHex);

void savePng(GClasses::GImage* pImage, FILE* pFile, bool bIncludeAlphaChannel);
void savePng(GClasses::GImage* pImage, FILE* pFile);
void savePng(GClasses::GImage* pImage, const char* szFilename);

}

#endif
