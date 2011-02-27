#include "GBits.h"

using namespace GClasses;

//static
bool GBits::isValidFloat(const char* pString, int len)
{
	if(len <= 0)
		return false;
	if(*pString == '-' || *pString == '+')
	{
		pString++;
		len--;
		if(len == 0)
			return false;
	}
	int digits = 0;
	int decimals = 0;
	while(len > 0)
	{
		if(*pString == '.')
			decimals++;
		else if(*pString >= '0' && *pString <= '9')
			digits++;
		else
			break;
		pString++;
		len--;
	}
	if(decimals > 1)
		return false;
	if(digits < 1)
		return false;
	if(len > 0 && (*pString == 'e' || *pString == 'E'))
	{
		pString++;
		len--;
		if(len == 0)
			return false;
		if(*pString == '-' || *pString == '+')
		{
			pString++;
			len--;
		}
		if(len == 0)
			return false;
		while(*pString >= '0' && *pString <= '9')
		{
			pString++;
			len--;
		}
	}
	if(len > 0)
		return false;
	return true;
}

