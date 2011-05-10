/*
	Copyright (C) 2006, Mike Gashler

	This library is free software; you can redistribute it and/or
	modify it under the terms of the GNU Lesser General Public
	License as published by the Free Software Foundation; either
	version 2.1 of the License, or (at your option) any later version.

	see http://www.gnu.org/copyleft/lesser.html
*/

#ifndef __GFILE_H__
#define __GFILE_H__

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <istream>

namespace GClasses {

/// Helper struct to hold the results from GFile::ParsePath
struct PathData
{
	int dirStart;
	int fileStart;
	int extStart;
	int len;
};


/// Contains some useful routines for manipulating files
class GFile
{
public:
	/// returns true if the file exists
	static bool doesFileExist(const char *filename);

	/// returns true if the directory exists
	static bool doesDirExist(const char* szDir);

	/// Deletes the specified file. Returns true iff successful.
	static bool deleteFile(const char* szFilename);

	/// Removes the specified directory. Fails if it is not empty.
	/// Returns true iff successful.
	static bool removeDir(const char* szDir);

	/// This finds the last slash in szBuff and returns a
	/// pointer to the char past that.  (If there are no
	/// slashes or back-slashes, it returns szBuff)
	static const char* clipPath(const char* szBuff);

	/// This finds the last slash in szBuff and sets it
	/// to '\0' and returns szBuff.
	static char* clipFilename(char* szBuff);

	/// returns a user's home directory for the various OS's
	static bool localStorageDirectory(char *toHere);

	/// This copies a file.  It doesn't check to see if it is
	/// overwriting--it just does the copying.  On success it
	/// returns true.  On error it returns false.  It won't
	/// work with a file bigger than 2GB.  Both paths must
	/// include the filename.
	static bool copyFile(const char* szSrcPath, const char* szDestPath);

	/// Loads a file into memory and returns a pointer to the
	/// memory.  You must delete the buffer it returns.
	static char* loadFile(const char* szFilename, size_t* pnSize);

	/// Saves a buffer as a file.  Returns true on success
	static void saveFile(const char* pBuf, size_t nSize, const char* szFilename);

	/// This is a brute force way to make a directory.  It
	/// iterates through each subdir in szDir and calls mkdir
	/// until it has created the complete set of nested directories.
	static bool makeDir(const char* szDir);

	/// Remove extra ".." folders in the path
	static void condensePath(char* szPath);

	/// This returns the number of seconds since 1970 UTC
	static time_t modifiedTime(const char* szFilename);

	/// Set the last modified time of a file
	static void setModifiedTime(const char *filename, time_t t);
/*
	/// This only writes one pass of random numbers over the file, so it may still be
	/// possible for the file to be recovered with expensive hardware that takes
	/// advantage of the fact that the hard disk write head may drift slightly while
	/// writing in order to read older data that may still be encoded along the edge
	/// of the path on the platter.
	static bool shredFile(const char* szFilename);

	/// Delete a folder and recursively shred all it's contents. Returns true if successful
	/// This only writes one pass of random numbers over the file--see the warning on
	/// ShredFile.
	static bool shredFolder(const char* szPath);
*/
	/// Identifies the folder, file, extension, and total length from a path
	static void parsePath(const char* szPath, struct PathData* pData);

	/// returns a temporary filename
	static void tempFilename(char* pBuf);
};


/// This implements a simple compression/decompression algorithm
class GCompressor
{
public:
	/// Compress pIn. You are responsible to delete[] pOut. The new length is guaranteed to be at
	/// most len+5, and typically will be much smaller. Also, the first 4 bytes in the compressed
	/// data will be len (the size when uncompressed).
	static unsigned char* compress(unsigned char* pIn, unsigned int len, unsigned int* pOutNewLen);

	/// Uncompress pIn. You are responsible to delete[] pOut.
	static unsigned char* uncompress(unsigned char* pIn, unsigned int len, unsigned int* pOutUncompressedLen);

#ifndef NO_TEST_CODE
	static void test();
#endif
};



/// This is a simple tokenizer that reads a file, one token at-a-time.
class GTokenizer
{
protected:
	char* m_pBufStart;
	char* m_pBufPos;
	char* m_pBufEnd;
	std::istream* m_pStream;
	size_t m_len;
	size_t m_line;

public:
	/// Opens the specified filename.
	GTokenizer(const char* szFilename);

	/// Uses the provided buffer of data. (If len is 0, then it
	/// will read until a null-terminator is found.)
	GTokenizer(const char* pFile, size_t len);
	~GTokenizer();

	/// Returns the next character in the stream. Returns '\0' if there are
	/// no more characters in the stream. (This could theoretically be ambiguous if the
	/// the next character in the stream is '\0', but presumably this class
	/// is mostly used for parsing text files, and that character should not
	/// occur in a text file.)
	char peek();

	/// This method skips past any characters in the string szDelimeters, then it
	/// reads a token until the next character would be a delimeter. If szDelimeters
	/// is NULL, then any characters <= ' ' are considered to be delimeters.
	/// A pointer to the buffered and null-terminated token is returned. NULL is
	/// returned if the end-of-file is reached. (Calling this method is
	/// the same as a call to skip, followed by a call to next, passing szDelimeters
	/// to both calls.)
	const char* nextTok(const char* szDelimeters = NULL);

	/// Reads until the next character is one of the specified delimeters.
	/// If szDelimeters is NULL, then any characters <= ' ' are considered
	/// to be delimeters. (The delimeter character is not read. Hence, it is typical
	/// to call skip before or after calling this method to advance past the delimeter.
	/// Repeatedly calling next will just stay in the same place and return
	/// empty tokens because it will never move past the delimeter. The nextTok
	/// method is provided for convenience to call both of these methods.)
	/// The token returned by this method will have been copied into an
	/// internal buffer, null-terminated, and a pointer to that buffer is returned.
	/// This method returns NULL if there are no more tokens in the file.
	const char* next(const char* szDelimeters = NULL);

	/// Reads past any characters specified in the list of delimeters.
	/// If szDelimeters is NULL, then any characters <= ' ' are considered
	/// to be delimeters.
	void skip(const char* szDelimeters = NULL);

	/// Skips the next 'n' characters. (Stops if the end-of-file is reached.)
	void skip(size_t n);

	/// Skip until the next character is one of the delimeters.
	/// (This method is the same as next, except that it does not buffer what it reads.)
	void skipTo(const char* szDelimeters = NULL);

	/// Reads past the specified string of characters. If the characters
	/// that are read from the file do not exactly match those in the string,
	/// an exception is thrown.
	void expect(const char* szString);

	/// Returns the current line number. (Begins at 1. Each time a '\n' is encountered,
	/// the line number is incremented. Mac line-endings do not increment the
	/// line number.)
	size_t line();

	/// Returns the number of remaining bytes to be read from the file.
	size_t remaining();

protected:
	void growBuf();
};

} // namespace GClasses

#endif // __GFILE_H__
