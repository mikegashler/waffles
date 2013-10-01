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

#ifndef __GDIRLIST_H__
#define __GDIRLIST_H__

#include <fstream>
#include <stack>
#include <vector>

namespace GClasses {

class GBlobQueue;

/// This class contains a list of files and a list of folders.
/// The constructor populates these lists with the names of files and folders in
/// the current working directory
class GDirList
{
public:
	GDirList();
	~GDirList() {}

	std::vector<std::string> m_folders;
	std::vector<std::string> m_files;
};

/// This turns a file or a folder (and its contents recursively) into a stream of bytes
class GFolderSerializer
{
protected:
	const char* m_szPath;
	char* m_szOrigPath;
	char* m_pBuf;
	char* m_pPos;
	size_t m_size;
	size_t m_state;
	size_t m_remaining;
	std::ifstream* m_pInStream;
	std::stack<GDirList*> m_dirStack;
	unsigned char* m_pCompressedBuf;
	char* m_pUncompressedBuf;
	size_t m_uncompressedPos;
	unsigned int m_compressedSize;
	bool m_compressedBufReady;
	size_t m_bytesOut;

public:
	/// szPath can be a filename or a foldername
	GFolderSerializer(const char* szPath, bool compress);
	~GFolderSerializer();

	/// Returns a pointer to the next chunk of bytes. Returns NULL
	/// if it is done.
	char* next(size_t* pOutSize);

	/// Returns the number of bytes that have been sent out so far
	size_t bytesOut() { return m_bytesOut; }

protected:
	char* nextPiece(size_t* pOutSize);
	void addName(const char* szName);
	void startFile(const char* szFilename);
	void continueFile();
	void startDir(const char* szDirName);
	void continueDir();
};

/// This class complements GFolderSerializer
class GFolderDeserializer
{
protected:
	GBlobQueue* m_pBQ1;
	GBlobQueue* m_pBQ2;
	size_t m_compressedBlockSize;
	size_t m_state;
	unsigned int m_nameLen;
	unsigned long long m_fileLen;
	std::ofstream* m_pOutStream;
	size_t m_depth;
	std::string* m_pBaseName;

public:
	GFolderDeserializer(std::string* pBaseName = NULL);
	~GFolderDeserializer();

	void doNext(const char* pBuf, size_t bufLen);

protected:
	void pump1();
	void pump2();
};


} // namespace GClasses

#endif // __GDIRLIST_H__
