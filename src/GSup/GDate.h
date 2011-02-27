/*
	Copyright (C) 2006, Mike Gashler

	This library is free software; you can redistribute it and/or
	modify it under the terms of the GNU Lesser General Public
	License as published by the Free Software Foundation; either
	version 2.1 of the License, or (at your option) any later version.

	see http://www.gnu.org/copyleft/lesser.html
*/

#ifndef __GDATE_H__
#define __GDATE_H__

namespace GClasses {

typedef unsigned int GDate;

#define INVALID_DATE ((GDate)-1)

inline int GetYear(GDate date)
{
	return ((int)(date >> 16)) - 0x8000;
}

inline int GetMonth(GDate date)
{
	return (int)((date >> 8) & 0xff);
}

inline int GetDay(GDate date)
{
	return (int)(date & 0xff);
}

inline GDate MakeDate(int year, int month, int day)
{
	return (GDate)(((year + 0x8000) << 16) | (month << 8) | day);
}

inline int CompareDates(GDate d1, GDate d2)
{
	if(d1 < d2)
		return -1;
	else if(d2 < d1)
		return 1;
	return 0;
}

GDate ParseDate(const char* szString);

#ifndef NO_TEST_CODE
void TestGDate();
#endif // !NO_TEST_CODE

} // namespace GClasses

#endif // __GDATE_H__
