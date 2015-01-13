#ifndef __GIMAGEJPEG_H__
#define __GIMAGEJPEG_H__

#include <GClasses/GImage.h>
#include <stdlib.h>

namespace GClasses {

void loadJpeg(GImage* pImage, const char* filename);
void saveJpeg(GImage* pImage, const char* filename);  //DOEST WORK. LEE WORK ON IT LATER

}

#endif //__GIMAGEJPEG_H__
