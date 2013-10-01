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

#ifndef __GWINDOWS_H__
#define __GWINDOWS_H__

namespace GClasses {

#ifdef WINDOWS

class GWindow;
class GInput;

/// This class is for Windows-only functions
class GWindows
{
public:
	/// This allows Windows to unload its message stack.  It is a
	/// good idea to call this frequently when you are in a big loop
	/// so Windows can multi-task properly and the user can still
	/// have control of his computer.
	static void yield();

	/// This calls the Windows standard dialog for getting a filename
	/// for opening or saving.
	//static int GetOpenFilename(HWND hWnd, char *message, char *mask, char *bufr);
	//static int GetSaveFilename(HWND hWnd, char *message, char *mask, char *bufr);

	/// This displays a little dialog to get an integer from the user.
	/// The GInput parameter must already be initialized and have keyboard
	/// enabled.
//	static int GetInt(GWindow* pWindow, GInput* pInput, char* szMsg);

	/// This destroys a file completely so it can't be recovered
//	static bool shredFile(const char* szFilename);

	/// This first checks to see if a directory exists and
	/// makes it if it doesn't.  This is better than most
	/// directory making commands because it will make as
	/// many nested directories as necessary without complaining.
//	static void MakeDir(const char* szPath);

	/// This sets the value *addr to true and returns whatever value
	/// it was before it was set to true.  It both tests and sets
	/// atomically so this function can beused to write syncronization
	/// primitives.
//	static bool TestAndSet(bool* addr);
};

#endif // WINDOWS


} // namespace GClasses

#endif // __GWINDOWS_H__
