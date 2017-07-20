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

#include <stdio.h>
#include "GTime.h"
#include "GString.h"
#include <time.h>
#include <cmath>
#ifdef WINDOWS
#	include <windows.h>
#else
#	include <sys/time.h>
#endif
#include <sstream>

namespace GClasses {

/*static*/ double GTime::seconds()
{
#ifdef WINDOWS
	return (double)GetTickCount() * 1e-3;
#else
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);

// 	struct timespec ts;
// 	clock_gettime(CLOCK_MONOTONIC, &ts);
// 	return (double)ts.tv_sec + (double)tv_nsec * 1e-9;
#endif
}

/*static*/ const char* GTime::asciiTime(char* szBuf, int nSize, bool bGreenwichMeanTime)
{
	time_t t = time((time_t*)0);
#ifdef WINDOWS
	struct tm thetime;
	if(bGreenwichMeanTime)
		gmtime_s(&thetime, &t);
	else
		localtime_s(&thetime, &t);
	asctime_s(szBuf, nSize, &thetime);
	int len = (int)strlen(szBuf);
#else
	struct tm* pTime = bGreenwichMeanTime ? gmtime(&t) : localtime(&t);
	int len = safe_strcpy(szBuf, asctime(pTime), nSize);
#endif

	// Remove trailing whitespace (cuz asctime appends a '\n' at the end)
	while(len > 0 && szBuf[len - 1] <= ' ')
		szBuf[--len] = '\0';
	return szBuf;
}

void GTime_printTwoDigits(std::ostream& os, unsigned int n)
{
	if(n < 10)
		os << "0";
	os << n;
}

/*static*/ void GTime::appendTimeStampValue(std::string* pS, const char* sep1, const char* sep2, const char* sep3, bool bGreenwichMeanTime)
{
	time_t t = time((time_t*)0);
#ifdef WINDOWS
	struct tm thetime;
	if(bGreenwichMeanTime)
		gmtime_s(&thetime, &t);
	else
		localtime_s(&thetime, &t);
	struct tm* pTime = &thetime;
#else
	struct tm* pTime = bGreenwichMeanTime ? gmtime(&t) : localtime(&t);
#endif
	std::ostringstream os;
	unsigned int n = 1900 + pTime->tm_year;
	os << n;
	os << sep1;
	GTime_printTwoDigits(os, pTime->tm_mon + 1);
	os << sep1;
	GTime_printTwoDigits(os, pTime->tm_mday);
	os << sep2;
	GTime_printTwoDigits(os, pTime->tm_hour);
	os << sep3;
	GTime_printTwoDigits(os, pTime->tm_min);
	os << sep3;
	GTime_printTwoDigits(os, pTime->tm_sec);
	(*pS).append(os.str());
}

int GTime_parseIntVal(const char* buf, int min, int max, bool* pOk)
{
	for(size_t i = 0; buf[i] != '\0'; i++)
	{
		if(buf[i] < '0' || buf[i] > '9')
			*pOk = false;
	}
	int val = atoi(buf);
	if(val < min || val > max)
		*pOk = false;
	return val;
}

bool is_double_char(char c)
{
	if(c == '.')
		return true;
	if(c >= '0' && c <= '9')
		return true;
	return false;
}

double GTime_parseDoubleVal(const char* buf, double min, double max, bool* pOk)
{
	for(size_t i = 0; buf[i] != '\0'; i++)
	{
		if(!is_double_char(buf[i]))
			*pOk = false;
	}
	double val = atof(buf);
	if(val < min || val > max)
		*pOk = false;
	return val;
}

/*static*/ bool GTime::fromString(double* pOutTime, const char* szData, const char* szFormat)
{
	double fractional_seconds = 0.0;
	struct tm ts;
	ts.tm_sec = 0;
	ts.tm_min = 0;
	ts.tm_hour = 0;
	ts.tm_mday = 1;
	ts.tm_mon = 0;
	ts.tm_year = 70;
	ts.tm_wday = 0;
	ts.tm_yday = 0;
	ts.tm_isdst = 0;
	char buf[32];
	while(*szFormat)
	{
		size_t formatChars = 0;
		size_t dataChars = 0;
		while(formatChars < 30)
		{
			if(szFormat[formatChars] != szFormat[0])
				break;
			if(szData[formatChars] == '\0')
				return false;
			buf[dataChars] = szData[formatChars];
			
			// If we're reading integer values, and we already have at least one digit, but the numbers run out before the formatting changes, let that be okay.
			if(dataChars > 0 && is_double_char(szData[dataChars - 1]) && !is_double_char(szData[dataChars]))
				dataChars--;

			formatChars++;
			dataChars++;
		}
		buf[dataChars] = '\0';
		bool ok = true;
		switch(szFormat[0])
		{
			case 'y':
			case 'Y':
				ts.tm_year = GTime_parseIntVal(buf, 1000, 3000, &ok) - 1900;
				break;
			case 'M':
				ts.tm_mon = GTime_parseIntVal(buf, 1, 12, &ok) - 1;
				break;
			case 'd':
			case 'D':
				ts.tm_mday = GTime_parseIntVal(buf, 1, 31, &ok);
				break;
			case 'h':
			case 'H':
				ts.tm_hour = GTime_parseIntVal(buf, 0, 23, &ok);
				break;
			case 'm':
				ts.tm_min = GTime_parseIntVal(buf, 0, 59, &ok);
				break;
			case 's':
			case 'S':
			{
				double d = GTime_parseDoubleVal(buf, 0, 61, &ok);
				ts.tm_sec = (int)std::floor(d);
				fractional_seconds = d - std::floor(d);
			}
				break;
			default:
				for(size_t j = 0; j < formatChars; j++)
				{
					if(buf[j] != szFormat[0])
						return false;
				}
				break;
		}
		if(!ok)
			return false;
		szData += dataChars;
		szFormat += formatChars;
	}
	time_t tt = mktime(&ts);
	*pOutTime = (double)tt + fractional_seconds;
	return true;
}





GProgressEstimator::GProgressEstimator(size_t totalIters, size_t sampleSize)
: m_totalIters(totalIters), m_sampleSize(sampleSize)
{
	m_startTime = GTime::seconds();
	m_iterStartTime = m_startTime;
	m_queue.resize(sampleSize);
	m_queuePos = 0;
}

GProgressEstimator::~GProgressEstimator()
{
}

const char* GProgressEstimator::estimate(size_t iter)
{
	// Measure the duration of the last iteration
	double timeNow = GTime::seconds();
	double iterDuration = timeNow - m_iterStartTime;
	if(m_samples.size() >= m_sampleSize)
		m_samples.drop_by_value(m_queue[m_queuePos]);
	m_samples.insert(iterDuration);
	m_queue[m_queuePos] = iterDuration;
	m_queuePos++;
	if(m_queuePos >= m_sampleSize)
		m_queuePos = 0;
	m_iterStartTime = timeNow;

	// Generate an estimate string
	double median = m_samples.get(m_samples.size() / 2);
	m_prevMedian = median;
	double elapsed = timeNow - m_startTime;
	m_message = "Elapsed=";
	m_message += to_str(elapsed);
	m_message += ",	ETA=";
	m_message += to_str(m_totalIters - iter);
	m_message += " remaining iters * ";
	m_message += to_str(median);
	m_message += " s/iter = ";
	size_t secs = (size_t)(median * (m_totalIters - iter));
	if(secs > 60)
	{
		size_t mins = secs / 60;
		secs -= 60 * mins;
		if(mins > 60)
		{
			size_t hours = mins / 60;
			mins -= 60 * hours;
			if(hours > 24)
			{
				size_t days = hours / 24;
				hours -= 24 * hours;
				if(days > 365)
				{
					size_t years = days / 365;
					days -= years * 365;
					m_message += to_str(years);
					m_message += "y";
				}
				m_message += to_str(days);
				m_message += "d";
			}
			m_message += to_str(hours);
			m_message += "h";
		}
		m_message += to_str(mins);
		m_message += "m";
	}
	m_message += to_str(secs);
	m_message += "s";
	return m_message.c_str();
}


} // namespace GClasses

