/*
	Copyright (C) 2006, Mike Gashler

	This library is free software; you can redistribute it and/or
	modify it under the terms of the GNU Lesser General Public
	License as published by the Free Software Foundation; either
	version 2.1 of the License, or (at your option) any later version.

	see http://www.gnu.org/copyleft/lesser.html
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
};

#endif // WINDOWS


} // namespace GClasses

#endif // __GWINDOWS_H__
