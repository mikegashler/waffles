// -------------------------------------------------------------
// The contents of this file may be distributed under the CC0
// license (http://creativecommons.org/publicdomain/zero/1.0/).
// Of course you may also distribute it under a more restrictive
// license, such as any of the OSI-approved licenses
// (http://www.opensource.org/licenses).
// -------------------------------------------------------------

#ifndef __MANIFOLD_H__
#define __MANIFOLD_H__

#include "Gui.h"


class ManifoldMenuController : public ControllerBase
{
protected:

public:
	ManifoldMenuController();
	virtual ~ManifoldMenuController();

	void RunModal();
};

#endif // __MANIFOLD_H__
