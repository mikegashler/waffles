/*
	Copyright (C) 2006, Mike Gashler

	This library is free software; you can redistribute it and/or
	modify it under the terms of the GNU Lesser General Public
	License as published by the Free Software Foundation; either
	version 2.1 of the License, or (at your option) any later version.

	see http://www.gnu.org/copyleft/lesser.html
*/

#ifdef WINDOWS

#include "GWindows.h"
#include "../GClasses/GString.h"
#include "../GClasses/GError.h"
#include <windows.h>
#include <direct.h>
#include <stdio.h>
#include <errno.h>
#include <io.h>
#include <time.h>
#include <process.h>

namespace GClasses {

/*
void GWindows::MakeDir(const char* szPath)
{
   // Get Ready
   if(!szPath)
      return;
   if(szPath[0] == '\0')
      return;
   char szBuff[512];
   safe_strcpy(szBuff, szPath, 512);
   char* pSlash;

   // Move to first slash
   for(pSlash = szBuff + 1; *pSlash != '\\' && *pSlash != '/' && *pSlash != '\0'; pSlash++);

   // If the path starts with a '.' or a drive letter, move to next slash
   if(szBuff[0] == '.' || szBuff[1] == ':' || (szBuff[1] != '\0' && szBuff[2] == ':'))
   {
      if(*pSlash == '\0')
         return;
      for(pSlash++; *pSlash != '\\' && *pSlash != '/' && *pSlash != '\0'; pSlash++);
   }

   // Sequentially make each directory
   while(true)
   {
      char cTmp = *pSlash;
      *pSlash = '\0';
      if(mkdir(szBuff))
      {
         if(errno == ENOENT)
            GAssert(false); // Path Not Found in GWindows::MakeDir
      }
      *pSlash = cTmp;
      if(cTmp == '\0')
         break;
      for(pSlash++; *pSlash != '\\' && *pSlash != '/' && *pSlash != '\0'; pSlash++);
   }
}
*/
/*
bool GWindows::shredFile(const char* szOldFilename)
{
	// Turn off all file attributes
	srand((unsigned)time(NULL) + rand());

	if(!SetFileAttributes(szOldFilename, FILE_ATTRIBUTE_NORMAL))
	{
		GAssert(false); // Error resetting file attributes
		return false;
	}

	// Rename the file so the filename info will be lost
	char szDrive[256];
	char szPath[256];
	char szFilename[256];
	_splitpath(szOldFilename, szDrive, szPath, NULL, NULL);
	_makepath(szFilename, szDrive, szPath, "qxvzwjgh", ".dnb");
	if(rename(szOldFilename, szFilename))
	{
		switch(errno)
		{
			case EACCES:	GAssert(false, "Failed to rename file--bad new name\n");		break;
			case ENOENT:	GAssert(false, "Failed to rename file--not found\n");			break;
			case EINVAL:	GAssert(false, "Failed to rename file--invalid chars\n");		break;
			default:		GAssert(false, "Failed to rename file--unknown reason\n");	break;
		}
		return false;
	}

	// Overwrite the file 16 times
	int n;
	for(n = 0; n < 16; n++)
	{
		// Open the file
		FILE* pFile = fopen(szFilename, "rb+");
		if(!pFile)
		{
			GAssert(false, "Failed to open the file to shred\n");
			return false;
		}

		// Write junk over the entire file
		char szJunk[512];
		for(n = 0; n < 512; n++)
			szJunk[n] = rand() % 256;
		int nFileSize = _filelength(_fileno(pFile));
		int i = 0;
		for(n = 0; n < nFileSize; n++)
		{
			if(fwrite(&szJunk[i], 1, 1, pFile) != 1)
			{
				GAssert(false, "Error writing zeros over file to shred\n");
				return false;
			}
			i++;
			if(i >= 512)
				i = 0;
		}
		fflush(pFile);
		Sleep(20);

		// Reset the file access times (time created, last accessed, and last written)
// 		FILETIME ft;
// 		ft.dwHighDateTime = 0;
// 		ft.dwLowDateTime = 0;
// 		if(!SetFileTime((HANDLE)_fileno(pFile), &ft, &ft, &ft))
// 		{
// 			char szBuff[1025];
// 			FormatMessage(FORMAT_MESSAGE_FROM_SYSTEM, NULL, GetLastError(), 0, szBuff, 1024, NULL);
// 			GAssert(false, szBuff);
// 			return false;
// 		}
		fclose(pFile);
	}

	// Reset the filesize to zero
	FILE* pFile = fopen(szFilename, "wb");
	if(!pFile)
	{
		GAssert(false, "Error Resetting filesize to zero\n");
		return false;
	}
	fclose(pFile);

	// Delete the file
	if(!DeleteFile(szFilename))
	{
		GAssert(false, "Error deleting file after shredding\n");
		return false;
	}
	
	return true;
}
*/
/*
// ************ How to ensure Mutual Exclusion using GWindows::TestAndSet()
//  bool bInUse = false;
//  [ Normal code ]
//	while(GWindows::TestAndSet(&bInUse))
//		Sleep(0);
//  [ Critical Section (code that modifies data common to multiple threads) ]
//	bInUse = false;
//  [ Normal code ]
bool GWindows::TestAndSet(bool* addr)
{
	__asm mov eax, addr	// move the address into register eax
	__asm bts [eax], 0	// dereference eax, and test and set
	__asm jc  bitSet	// if the carry bit is set, jump
	return false;		// otherwise return 0
bitSet:
	return true;
}
*/


// This gives Windows a chance to process other messages.
// It is important to call this periodically when in a
// large loop so Windows can do multi-tasking.
void GWindows::yieldToWindows()
{
	 MSG   aMsg;

	 while(PeekMessage(&aMsg, NULL, WM_NULL, WM_NULL, PM_REMOVE))
//   while(PeekMessage(&aMsg, m_hWnd, WM_NULL, WM_NULL, PM_REMOVE))
	 {
			TranslateMessage(&aMsg);
			DispatchMessage(&aMsg);
	 }
}

/*
int GWindows::GetOpenFilename(HWND hWnd, char *message, char *mask, char *bufr)
{
	char szOldPath[512];
	getcwd(szOldPath, 512);
	OPENFILENAME t;
	TCHAR         szFile[MAX_PATH]      = "\0";
	strcpy(szFile, mask);
	t.lStructSize       = sizeof(OPENFILENAME);
	t.hwndOwner         = hWnd;
//	t.hInstance         = g_hInst;
	t.lpstrFilter       = NULL;
	t.lpstrCustomFilter = NULL;
	t.nMaxCustFilter    = 0;
	t.nFilterIndex      = 0;
	t.lpstrFile         = szFile;
	t.nMaxFile          = sizeof(szFile);
	t.lpstrFileTitle    = NULL;
	t.nMaxFileTitle     = 0;
	t.lpstrInitialDir   = NULL;
	t.lpstrTitle        = message;
	t.nFileOffset       = 0;
	t.nFileExtension    = 0;
	t.lpstrDefExt       = NULL;
//	t.lCustData         = (LPARAM)&sMyData;
//	t.lpfnHook 		      = (LPOFNHOOKPROC) ComDlg32DlgProc;
//	t.lpTemplateName    = MAKEINTRESOURCE(IDD_COMDLG32);
	t.Flags             = OFN_EXPLORER;
	if(GetOpenFileName(&t))
	{
		int i, j;
		j = 0;
		i = 0;
		for( ;t.lpstrFile[i] != 0 && i < MAX_PATH; i++)
		{
			bufr[j] = t.lpstrFile[i];
			j++;
		}
		bufr[j] = '\0';
	}
	else
		bufr[0] = '\0';
	chdir(szOldPath);
	if(bufr[0] == '\0')
		return(IDCANCEL);
	else
		return(IDOK);
}

int GWindows::GetSaveFilename(HWND hWnd, char *message, char *mask, char *bufr)
{
	char szOldPath[512];
	getcwd(szOldPath, 512);
	OPENFILENAME t;
	TCHAR         szFile[MAX_PATH]      = "\0";
	strcpy(szFile, mask);
	t.lStructSize       = sizeof(OPENFILENAME);
	t.hwndOwner         = hWnd;
//	t.hInstance         = g_hInst;
	t.lpstrFilter       = NULL;
	t.lpstrCustomFilter = NULL;
	t.nMaxCustFilter    = 0;
	t.nFilterIndex      = 0;
	t.lpstrFile         = szFile;
	t.nMaxFile          = sizeof(szFile);
	t.lpstrFileTitle    = NULL;
	t.nMaxFileTitle     = 0;
	t.lpstrInitialDir   = NULL;
	t.lpstrTitle        = message;
	t.nFileOffset       = 0;
	t.nFileExtension    = 0;
	t.lpstrDefExt       = NULL;
//	t.lCustData         = (LPARAM)&sMyData;
//	t.lpfnHook 		      = (LPOFNHOOKPROC) ComDlg32DlgProc;
//	t.lpTemplateName    = MAKEINTRESOURCE(IDD_COMDLG32);
	t.Flags             = OFN_EXPLORER;
	if(GetSaveFileName(&t))
	{
		int i, j;
		j = 0;
		i = 0;
		for( ;t.lpstrFile[i] != 0 && i < MAX_PATH; i++)
		{
			bufr[j] = t.lpstrFile[i];
			j++;
		}
		bufr[j] = '\0';
		chdir(szOldPath);
		return(IDOK);
	}
	chdir(szOldPath);
	return(IDCANCEL);
}
*/

#endif // WINDOWS

} // namespace GClasses

