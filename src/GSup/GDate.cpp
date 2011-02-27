/*
	Copyright (C) 2006, Mike Gashler

	This library is free software; you can redistribute it and/or
	modify it under the terms of the GNU Lesser General Public
	License as published by the Free Software Foundation; either
	version 2.1 of the License, or (at your option) any later version.

	see http://www.gnu.org/copyleft/lesser.html
*/

#include <string.h>
#include <stddef.h>
#include "GDate.h"
#include "../GClasses/GError.h"
#include <stdlib.h>

namespace GClasses {

const char* g_szMonths = "JanFebMarAprMayJunJulAugSepOctNovDec";
const char* g_szDelimeters = "/-.,\\";

int FindMonthString(const char* szString)
{
	int i, j;
	for(i = 0; szString[i] != '\0'; i++)
	{
		if(szString[i] < 'A' || szString[i] > 'z')
			continue;
		for(j = 0; j < 12; j++)
		{
			if(_strnicmp(szString + i, g_szMonths + 3 * j, 3) == 0)
				return j + 1;
		}
	}
	return -1;
}

int FindFirstNumber(const char* szString)
{
	int i;
	for(i = 0; szString[i] != '\0'; i++)
	{
		if(szString[i] >= '0' && szString[i] <= '9')
			return i;
	}
	return -1;
}

bool FindTwoNumbers(const char* szString, int* pIndex1, int* pIndex2)
{
	int nFirst = FindFirstNumber(szString);
	if(nFirst < 0)
		return false;
	int i;
	for(i = nFirst + 1; szString[i] >= '0' && szString[i] <= '9'; i++)
	{
	}
	int nSecond = FindFirstNumber(szString + i);
	if(nSecond < 0)
		return false;
	*pIndex1 = nFirst;
	*pIndex2 = nSecond + i;
	return true;
}

bool FindThreeNumbers(const char* szString, int* pIndex1, int* pIndex2, int* pIndex3)
{
	if(!FindTwoNumbers(szString, pIndex1, pIndex2))
		return false;
	int i;
	for(i = (*pIndex2) + 1; szString[i] >= '0' && szString[i] <= '9'; i++)
	{
	}
	int nThird = FindFirstNumber(szString + i);
	if(nThird < 0)
		return false;
	*pIndex3 = nThird + i;
	return true;
}

// returns 0 if it can't tell which one is the year
// returns 1 if a is obviously a year and b is not
// returns 2 if b is obviously a year and a is not
int ResolveYear(int a, int b)
{
	if(a > 31 && b <= 31)
		return 1;
	if(b > 31 && a <= 31)
		return 2;
	return 0;
}

// returns 0 if it can't tell which one is the day
// returns 1 if a is obviously a day and b is not
// returns 2 if b is obviously a day and a is not
int ResolveDay(int a, int b)
{
	if(a > 12 && b <= 12)
		return 1;
	if(b > 12 && a <= 12)
		return 2;
	return 0;
}

GDate ParseDate(const char* szString)
{
	int nMonth = FindMonthString(szString);
	if(nMonth >= 0)
	{
		int nIndex1, nIndex2;
		if(!FindTwoNumbers(szString, &nIndex1, &nIndex2))
		{
			nIndex1 = FindFirstNumber(szString);
			int n = atoi(&szString[nIndex1]);
			if(n > 31)
				return MakeDate(n, nMonth, 1);
			else
				return INVALID_DATE;
		}
		int n1 = atoi(&szString[nIndex1]);
		int n2 = atoi(&szString[nIndex2]);
		int nRes = ResolveYear(n1, n2);
		if(nRes > 0)
		{
			if(nRes == 1)
				return MakeDate(n1, nMonth, n2); // n1 is the year
			else
				return MakeDate(n2, nMonth, n1); // n2 is the year
		}
		else
			return MakeDate(n2, nMonth, n1);
	}
	else
	{
		int i1, i2, i3;
		if(FindThreeNumbers(szString, &i1, &i2, &i3))
		{
			int n1 = atoi(&szString[i1]);
			int n2 = atoi(&szString[i2]);
			int n3 = atoi(&szString[i3]);
			int nRes = ResolveYear(n1, n3);
			if(nRes == 1)
				return MakeDate(n1, n2, n3); // n1 is the year
			else if(nRes == 2)
			{
				// n3 is the year
				nRes = ResolveDay(n1, n2);
				if(nRes == 1)
					return MakeDate(n3, n2, n1);
				else if(nRes == 2)
					return MakeDate(n3, n1, n2);
			}

			// look for delimeters
			int i;
			for(i = i1; i < i2; i++)
			{
				if(szString[i] == '/' || szString[i] == '-')
					return MakeDate(n3, n1, n2); // American delimeters
				if(szString[i] == ',' || szString[i] == '.')
					return MakeDate(n3, n2, n1); // European delimeters
			}
			return INVALID_DATE; // can't tell
		}
		else
		{
			int index = FindFirstNumber(szString);
			if(index < 0)
				return INVALID_DATE;
			int n = atoi(&szString[index]);
			return MakeDate(n, 1, 1);
		}
	}
}

#ifndef NO_TEST_CODE
void TestDate(const char* szString, int year, int month, int day)
{
	GDate date = ParseDate(szString);
	if(GetYear(date) != year)
		throw "wrong answer";
	if(GetMonth(date) != month)
		throw "wrong answer";
	if(GetDay(date) != day)
		throw "wrong answer";
}

void TestGDate()
{
	TestDate("31 DEC 2000", 2000, 12, 31);
	TestDate("jan 02, 2006", 2006, 1, 2);
	TestDate("14 feb 1401", 1401, 2, 14);
	TestDate("07/13/1999", 1999, 7, 13);
	TestDate("13-7-1999", 1999, 7, 13);
	TestDate("6-7-1999", 1999, 6, 7);
	TestDate("6.7.1999", 1999, 7, 6);
	TestDate("1993-2-1", 1993, 2, 1);
}
#endif // !NO_TEST_CODE

} // namespace GClasses
