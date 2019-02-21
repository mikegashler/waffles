/*
  The contents of this file are dedicated by all of its authors, including

    Michael S. Gashler,
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

#ifndef __GFILE_H__
#define __GFILE_H__

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <istream>
#include <vector>
#include <map>
#include <fstream>

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

	/// Appends szString to the specified file. (This is a convenient method
	/// for logging. It is not efficient, because it does not keep the file open.)
	static void appendToFile(const char* szFilename, const char* szString);

	/// Adds the names of all the files (excluding folders)
	/// in the specified directory to "list".
	/// The filenames do not include any path information.
	static void fileList(std::vector<std::string>& list, const char* dir = ".");

	/// Adds the names of all the folders in the specified directory to "list".
	/// If excludeDots is true, then folders named "." or ".." will be excluded.
	/// The folder names do not include any path information.
	static void folderList(std::vector<std::string>& list, const char* dir = ".", bool excludeDots = true);

	/// Produces a list of all the folders in "dir" recursively, including "dir".
	/// Relative paths to all the folders will be added to the list.
	/// "dir" is guaranteed to be the first item in the list.
	static void folderListRecursive(std::vector<std::string>& list, const char* dir = ".");

	/// Produces a list of all the files in "dir" recursively.
	/// Relative paths to all the files will be added to the list.
	static void fileListRecursive(std::vector<std::string>& list, const char* dir = ".");

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



// A simple memory-cache for files that can be loaded as strings.
class GFileCache
{
public:
	std::map<std::string, std::string> m_filename_to_page;

	// Loads a file into the cach (if it has not already been loaded) and returns a referent to it in string form.
	std::string& get(const char* szFilename);
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

	static void test();
};



} // namespace GClasses

#endif // __GFILE_H__
