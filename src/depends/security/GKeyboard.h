// -------------------------------------------------------------
// The contents of this file may be distributed under the CC0
// license (http://creativecommons.org/publicdomain/zero/1.0/).
// Of course you may also distribute it under a more restrictive
// license, such as any of the OSI-approved licenses
// (http://www.opensource.org/licenses).
// -------------------------------------------------------------

#ifndef __GKEYBOARD_H__
#define __GKEYBOARD_H__

#ifdef WIN32
#	define DIRECTINPUT_VERSION 0x0700
#	include <dinput.h>
#else // WIN32
#	include <X11/Xlib.h>
#	define MAX_KEY_CODE_LEN 32
#endif // !WIN32

namespace GClasses {

typedef void (*LogKeyCallBack)(void* pThis, char c);

/// This is a cross-platform keystroke sniffer. You must only use it for doing good, and not evil.
class GKeyboard
{
protected:
	bool m_bKeepWatching;
#ifdef WIN32
	LPDIRECTINPUT m_lpDI;
	LPDIRECTINPUTDEVICE m_lpDIDKeyboard;
	BYTE m_keys[256];
	LogKeyCallBack m_pLogKeyFunc;
	void* m_pParam;
#else // WIN32
	LogKeyCallBack m_pLogKeyFunc;
	void* m_pParam;
	Display* m_pDisplay;
	char m_szKeyCode[MAX_KEY_CODE_LEN];
#endif // !WIN32

public:
	GKeyboard(LogKeyCallBack pLogKeyFunc, void* pParam);
	~GKeyboard();

	void Watch();
	void Stop() { m_bKeepWatching = false; }

protected:
#ifdef WIN32
	void Update();
#else // WIN32
	void GetKeyCode(XEvent* pEvent);
	void SelectAllChildWindows(Window root, unsigned long type);
#endif // !WIN32
};



} // namespace GClasses

#endif // __GKEYBOARD_H__
