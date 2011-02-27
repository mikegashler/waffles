// -------------------------------------------------------------
// The contents of this file may be distributed under the CC0
// license (http://creativecommons.org/publicdomain/zero/1.0/).
// Of course you may also distribute it under a more restrictive
// license, such as any of the OSI-approved licenses
// (http://www.opensource.org/licenses).
// -------------------------------------------------------------

#ifndef __CARONHILL_H__
#define __CARONHILL_H__

#include "Gui.h"


class CarOnHillController : public ControllerBase
{
protected:

public:
	CarOnHillController();
	virtual ~CarOnHillController();

	void RunModal();
};

#endif // __CARONHILL_H__
