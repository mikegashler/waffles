// -------------------------------------------------------------
// The contents of this file may be distributed under the CC0
// license (http://creativecommons.org/publicdomain/zero/1.0/).
// Of course you may also distribute it under a more restrictive
// license, such as any of the OSI-approved licenses
// (http://www.opensource.org/licenses).
// -------------------------------------------------------------

#ifdef WINDOWS

// ---------------------
//  DirectInput Version
// ---------------------
#include "GKeyboard.h"

using namespace GClasses;

// This table converts keyboard scan codes to ASCII (no shift)
static unsigned char SCAN_CODES_NO_SHIFT[] =
{
// +	0		1		2		3		4		5		6		7		8		9
//////////////////////////////////////////////////////////////////////////////////////
/* 0*/	0,		27,		49,		50,		51,		52,		53,		54,		55,		56,
/*10*/	57,		48,		45,		61,		8,		9,		113,	119,	101,	114,
/*20*/	116,	121,	117,	105,	111,	112,	91,		93,		13,		0,
/*30*/	97,		115,	100,	102,	103,	104,	106,	107,	108,	59,
/*40*/	39,		96,		0,		92,		122,	120,	99,		118,	98,		110,
/*50*/	109,	44,		46,		47,		0,		0,		0,		32,		0,		0,
/*60*/	0,		0,		0,		0,		0,		0,		0,		0,		0,		0,
/*70*/	0,		0,		0,		0,		0,		0,		0,		0,		0,		0,
/*80*/	0,		0,		0,		0,		0,		0,		0,		0,		0,		0,
};

static unsigned char SCAN_CODES_WITH_SHIFT[] =
{
// +	0		1		2		3		4		5		6		7		8		9
//////////////////////////////////////////////////////////////////////////////////////
/* 0*/	0,		27,		33,		64,		35,		36,		37,		94,		38,		42,
/*10*/	40,		41,		95,		43,		8,		15,		81,		87,		69,		82,
/*20*/	84,		89,		85,		73,		79,		80,		123,	125,	13,		0,
/*30*/	65,		83,		68,		70,		71,		72,		74,		75,		76,		58,
/*40*/	34,		126,	0,		124,	90,		88,		67,		86,		66,		78,
/*50*/	77,		60,		62,		63,		0,		0,		0,		32,		0,		0,
/*60*/	0,		0,		0,		0,		0,		0,		0,		0,		0,		0,
/*70*/	0,		0,		0,		0,		0,		0,		0,		0,		0,		0,
/*80*/	0,		0,		0,		0,		0,		0,		0,		0,		0,		0,
};

// Left Shift = 42
// Right Shift = 54
// Ctrl = 29
// Alt = 56
// F1-F10 = 59-68
// F11-F12 = 87-88
// Arrows R,L,U,D = 77, 75, 72, 80
// PgUp/Dn = 73, 81
// Home/End = 71, 79
// Ins/Del = 82, 83
// Caps = 58
// Scroll Lock = 70

GKeyboard::GKeyboard(LogKeyCallBack pLogKeyFunc, void* pParam)
{
	HINSTANCE hInst = GetModuleHandle(0);
	m_pLogKeyFunc = pLogKeyFunc;
	m_pParam = pParam;
	m_lpDI = NULL;
	m_lpDIDKeyboard = NULL;
	HRESULT rval;
	rval = DirectInputCreate(hInst, DIRECTINPUT_VERSION, &m_lpDI, NULL);
	if(rval != DI_OK)
		throw "failed to create direct input";
	rval = m_lpDI->CreateDevice(GUID_SysKeyboard, &m_lpDIDKeyboard, NULL);
	if(rval == DI_OK)
	{
		m_lpDIDKeyboard->SetDataFormat(&c_dfDIKeyboard);
		m_lpDIDKeyboard->SetCooperativeLevel(NULL, DISCL_NONEXCLUSIVE | DISCL_FOREGROUND);
		rval = m_lpDIDKeyboard->Acquire();
		if(rval != DI_OK)
			throw "failed to acquire keyboard";
	}
	m_bKeepWatching = true;
}

GKeyboard::~GKeyboard()
{
	if(m_lpDIDKeyboard)
		m_lpDIDKeyboard->Release();
	if(m_lpDI)
		m_lpDI->Release();
}

void GKeyboard::Update()
{
	if(m_lpDIDKeyboard->GetDeviceState(256, &m_keys) == (DIERR_INPUTLOST | DIERR_NOTACQUIRED))
		m_lpDIDKeyboard->Acquire();
}

void GKeyboard::Watch()
{
	// clear the old keys array
	unsigned char bOldKeys[90];
	int n;
	for(n = 0; n < 90; n++)
		bOldKeys[n] = 0;

	// Watch for key presses
	MSG msg;
	while(m_bKeepWatching)
	{
		// Process Windows messages (so you can't tell that this is running in the background)
		if(PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
		{
			if(msg.message == WM_QUIT)
				break; // Exit gracefully if Windows tells this program to quit
			else
			{
				TranslateMessage(&msg);
				DispatchMessage(&msg);
			}
		}
		else
		{
			// Check for key scans
			Update();
			for(n = 0; n < 90; n++)
			{
				if(m_keys[n] && !bOldKeys[n])
				{
					unsigned char nAscii;
					if(m_keys[42] | m_keys[54])
						nAscii = SCAN_CODES_WITH_SHIFT[n];
					else
						nAscii = SCAN_CODES_NO_SHIFT[n];
					if(nAscii > 0)
						m_pLogKeyFunc(m_pParam, nAscii);
				}
				bOldKeys[n] = m_keys[n];
			}
			Sleep(50);
		}
	}
}









#else // WINDOWS

// -------------------
//  X-Windows Version
// -------------------

#include "GKeyboard.h"
#include <GClasses/GError.h>
#include <stdio.h>
#include <X11/Intrinsic.h>
#include <X11/StringDefs.h>
#include <X11/Xutil.h>
#include <X11/Shell.h>

using namespace GClasses;

int XlibErrorHandler(Display* pDisplay, XErrorEvent* pEvent)
{
	//if(pEvent->error_code != BadWindow)
	//	ThrowError("Unexpected error\n");
	return 0;
}

GKeyboard::GKeyboard(LogKeyCallBack pLogKeyFunc, void* pParam)
{
	m_pLogKeyFunc = pLogKeyFunc;
	m_pParam = pParam;
	m_bKeepWatching = true;
	m_pDisplay = NULL;
	XSetErrorHandler(XlibErrorHandler);
}

GKeyboard::~GKeyboard()
{
}

void GKeyboard::SelectAllChildWindows(Window root, unsigned long type)
{
	Window parent;
	Window* pChildren;
	unsigned int nChildCount;

	if(!XQueryTree(m_pDisplay, root, &root, &parent, &pChildren, &nChildCount))
	{
		//GAssert(false); // failed to query tree
		return;
	}
	if(nChildCount <= 0)
		return;

	XSelectInput(m_pDisplay, root, type);
	int i;
	for(i = 0; i < (int)nChildCount; i++)
	{
		XSelectInput(m_pDisplay, pChildren[i], type);
		SelectAllChildWindows(pChildren[i], type);
	}
	XFree((char*)pChildren);
}

void GKeyboard::GetKeyCode(XEvent* pEvent)
{
	if(!pEvent)
	{
		m_szKeyCode[0] = '\0';
		return;
	}
	KeySym ks;
	int nLen = XLookupString((XKeyEvent*)pEvent, m_szKeyCode, MAX_KEY_CODE_LEN, &ks,NULL);
	m_szKeyCode[nLen] = '\0';
	if(nLen == 0)
	{
		char* szKeyString = XKeysymToString(ks);
		if(szKeyString)
			strcpy(m_szKeyCode, szKeyString);
		else
			m_szKeyCode[0] = '\0';
	}
}

void GKeyboard::Watch()
{
	XEvent event;
	m_pDisplay = XOpenDisplay(":0");
	if(!m_pDisplay)
		throw "failed to open display";
	SelectAllChildWindows(DefaultRootWindow(m_pDisplay),
			KeyPressMask | // receive KeyPress, KeyRelease, ButtonPress, and ButtonRelease events
			SubstructureNotifyMask // receive CreateNotify events
		);
	while(m_bKeepWatching)
	{
		XNextEvent(m_pDisplay, &event);
		if(event.type == KeyPress)
		{
			GetKeyCode(&event);
			if(m_szKeyCode[0] == '\0')
				continue;
			if(m_szKeyCode[1] == '\0')
				m_pLogKeyFunc(m_pParam, m_szKeyCode[0]);
			else
			{
				// todo: handle special keys better
				//m_pLogKeyFunc(m_pParam, m_szKeyCode[0]);
			}
		}
		else if(event.type == CreateNotify)
		{
			SelectAllChildWindows(event.xcreatewindow.parent/*window*/,
						KeyPressMask | // receive KeyPress, KeyRelease, ButtonPress, and ButtonRelease events
						SubstructureNotifyMask // receive CreateNotify events
					);
		}
		else
		{
			//ThrowError("unknown event type\n");
		}
	}
	XCloseDisplay(m_pDisplay);
	m_pDisplay = NULL;
}


#endif // !WINDOWS
