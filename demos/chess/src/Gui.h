// -------------------------------------------------------------
// The contents of this file may be distributed under the CC0
// license (http://creativecommons.org/publicdomain/zero/1.0/).
// Of course you may also distribute it under a more restrictive
// license, such as any of the OSI-approved licenses
// (http://www.opensource.org/licenses).
// -------------------------------------------------------------

#ifndef __GUI_H__
#define __GUI_H__

#include <GClasses/GWidgets.h>

class ControllerBase;

#ifndef NOGUI
#include <SDL/SDL.h>


inline Uint32* getPixMem32(SDL_Surface *surface, int x, int y)
{
	return (Uint32*)((Uint8*)surface->pixels + y * surface->pitch + (x << 2));
}

inline Uint16* getPixMem16(SDL_Surface *pScreen, int x, int y)
{
	return (Uint16*)pScreen->pixels + y * pScreen->pitch / 2 + x;
}


class ViewBase
{
protected:
	static SDL_Surface* s_pScreen;
	SDL_Surface* m_pScreen;
	GClasses::GRect m_screenRect;

public:
	ViewBase();
	virtual ~ViewBase();

	void update();

	virtual void onChar(char c) {}
	virtual void onSpecialKey(int key) {}
	virtual void onMouseDown(int nButton, int x, int y) {}
	virtual void onMouseUp(int nButton, int x, int y) {}
	virtual bool onMousePos(int x, int y) { return false; }
	GClasses::GRect* screenRect() { return &m_screenRect; }
	void captureScreen(GClasses::GImage* pImage);

protected:
	static void blitImage(SDL_Surface* pScreen, int x, int y, GClasses::GImage* pImage);
	static void stretchClipAndBlitImage(SDL_Surface* pScreen, GClasses::GRect* pDestRect, GClasses::GRect* pClipRect, GClasses::GImage* pImage);

	virtual void draw(SDL_Surface* pScreen) = 0;
	void drawDot(SDL_Surface *pScreen, int x, int y, unsigned int col, int nSize);
	SDL_Surface* makeScreen(int x, int y);
};

// when *pKeepRunning is false, the popup will close
// x, and y are the left and top margins. (Typically they will both be zero)
// pBackgroundImage is a screen shot to show in the background
// pDialog is the dialog to pop up in the middle
void RunPopup(GClasses::GWidgetDialog* pDialog);


void OpenFile(const char* szFilename);
void OpenAppFile(const char* szRelativeFilename);

#endif // !NOGUI

class MessageBoxDialog : public GClasses::GWidgetDialog
{
protected:
	GClasses::GWidgetTextButton* m_pOK;

public:
	MessageBoxDialog(const char* szMessage);
	virtual ~MessageBoxDialog();

	virtual void onReleaseTextButton(GClasses::GWidgetTextButton* pButton);
};



class GetStringDialog : public GClasses::GWidgetDialog
{
public:
	GClasses::GWidgetTextBox* m_pTextBox;
	GClasses::GWidgetTextButton* m_pOK;
	GClasses::GWidgetTextButton* m_pCancel;

	GetStringDialog(const char* szMessage);
	virtual ~GetStringDialog();

	virtual void onTextBoxPressEnter(GClasses::GWidgetTextBox* pTextBox);
	virtual void onReleaseTextButton(GClasses::GWidgetTextButton* pButton);
};


class GetOpenFilenameDialog : public GClasses::GWidgetDialog
{
protected:
	GClasses::GWidgetFileSystemBrowser* m_pFileBrowser;
	GClasses::GWidgetTextButton* m_pCancel;
	char* m_szFilename;

public:
	GetOpenFilenameDialog(const char* szMessage, const char* szExt);
	virtual ~GetOpenFilenameDialog();

	virtual void onSelectFilename(GClasses::GWidgetFileSystemBrowser* pBrowser, const char* szFilename);
	virtual void onReleaseTextButton(GClasses::GWidgetTextButton* pButton);
	const char* filename();
	char* releaseFilename();
};


class GetSaveFilenameDialog : public GClasses::GWidgetDialog
{
protected:
	GClasses::GWidgetFileSystemBrowser* m_pFileBrowser;
	GClasses::GWidgetTextBox* m_pFilename;
	GClasses::GWidgetTextButton* m_pSave;
	GClasses::GWidgetTextButton* m_pCancel;
	char* m_szFilename;

public:
	GetSaveFilenameDialog(const char* szMessage, const char* initialFilename, const char* szExt);
	virtual ~GetSaveFilenameDialog();

	virtual void onSelectFilename(GClasses::GWidgetFileSystemBrowser* pBrowser, const char* szFilename);
	virtual void onReleaseTextButton(GClasses::GWidgetTextButton* pButton);
	const char* filename();
	char* releaseFilename();
};


class ControllerBase
{
protected:
	bool m_bKeepRunning;

	static GClasses::GImage* s_pManualImage;
	static GClasses::GImage* s_pCloseImage;

#ifndef NOGUI
	enum KeyState
	{
		Normal,
		Holding,
		Repeating,
	};

	int m_keyboard[SDLK_LAST];
	int m_mouse[4];
	int m_mouseX;
	int m_mouseY;
	KeyState m_eKeyState;
	SDLKey m_lastPressedKey;
	double m_dKeyRepeatTimer;
	ViewBase* m_pView;
#endif // !NOGUI
public:
	ControllerBase();
	virtual ~ControllerBase();

	void quit() { m_bKeepRunning = false; }

#ifndef NOGUI
	virtual void onChar(char c) { m_pView->onChar(c); }
	virtual void onSpecialKey(int key) { m_pView->onSpecialKey(key); }
	virtual void onMouseDown(int nButton, int x, int y) { m_pView->onMouseDown(nButton, x, y); }
	virtual void onMouseUp(int nButton, int x, int y) { m_pView->onMouseUp(nButton, x, y); }
	virtual bool onMousePos(int x, int y) { return m_pView->onMousePos(x, y); }

	bool handleEvents(double dTimeDelta); // HandleEvents returns true if it thinks the view needs to be updated
	void mousePos(int* pX, int* pY);
#endif // !NOGUI

protected:
#ifndef NOGUI
	void handleKeyPress(SDLKey eKey, SDLMod mod);
	void handleMouseClick(int nButton);
#endif // !NOGUI
};


#endif // __GUI_H__
