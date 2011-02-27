// -------------------------------------------------------------
// The contents of this file may be distributed under the CC0
// license (http://creativecommons.org/publicdomain/zero/1.0/).
// Of course you may also distribute it under a more restrictive
// license, such as any of the OSI-approved licenses
// (http://www.opensource.org/licenses).
// -------------------------------------------------------------

#ifndef __RAYTRACE_H__
#define __RAYTRACE_H__

#include "Gui.h"
#include <GClasses/GRand.h>
#include <GClasses/GImage.h>
#include <GClasses/GRayTrace.h>

class RayTraceController;
class RayTraceDialog;


class RayTraceView : public ViewBase
{
friend class RayTraceController;
protected:
	RayTraceDialog* m_pDialog;
	GClasses::GImage* m_pImage;

public:
	RayTraceView(RayTraceController* pController);
	virtual ~RayTraceView();

	virtual void onChar(char c);
	virtual void onMouseDown(int nButton, int x, int y);
	virtual void onMouseUp(int nButton, int x, int y);
	virtual bool onMousePos(int x, int y);
	void SetImage(GClasses::GImage* pImage) { m_pImage = pImage; }

protected:
	virtual void draw(SDL_Surface *pScreen);
};










class RayTraceController : public ControllerBase
{
protected:
	GClasses::GRayTraceScene* m_pScene;
	bool m_bRendering;
	GClasses::GRand m_prng;

public:
	RayTraceController();
	virtual ~RayTraceController();

	void RunModal();
	void OnSelectFile(const char* szFilename);
	void SetHighQuality(bool b);
};

#endif // __RAYTRACE_H__

