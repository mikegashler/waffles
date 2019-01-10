#ifndef CRAWLER_H
#define CRAWLER_H

#include <stdlib.h>

namespace GClasses {
	class GArgReader;
}

unsigned char* downloadFromWeb(const char* szAddr, size_t timeout, size_t* pOutSize);
//void findbrokenlinks(GClasses::GArgReader& args);


#endif
