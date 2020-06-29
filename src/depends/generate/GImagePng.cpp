// -------------------------------------------------------------
// The contents of this file may be distributed under the CC0
// license (http://creativecommons.org/publicdomain/zero/1.0/).
// Of course you may also distribute it under a more restrictive
// license, such as any of the OSI-approved licenses
// (http://www.opensource.org/licenses).
// -------------------------------------------------------------

#ifdef WINDOWS
#	include "png.h"
#	include "zlib.h"
#else
#	include <png.h>
#	include <zlib.h>
#endif
#include "../../GClasses/GError.h"
#include "../../GClasses/GImage.h"
#include "../../GClasses/GHolders.h"
#include "../../GClasses/GFile.h"
#include "../../GClasses/GBits.h"
#include <memory>
#include <string.h>

namespace GClasses {

class GPNGReader
{
public:
	png_structp m_pReadStruct;
	png_infop m_pInfoStruct;
	png_infop m_pEndInfoStruct;
	const unsigned char* m_pData;
	int m_nPos;

	GPNGReader(const unsigned char* pData)
	{
		m_pData = pData;
		m_nPos = 0;
		m_pReadStruct = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
		if(!m_pReadStruct)
		{
			m_pReadStruct = NULL;
			throw Ex("Failed to create read struct");
			return;
		}
		m_pInfoStruct = png_create_info_struct(m_pReadStruct);
		m_pEndInfoStruct = png_create_info_struct(m_pReadStruct);
	}

	~GPNGReader()
	{
		if(m_pReadStruct)
			png_destroy_read_struct(&m_pReadStruct, &m_pInfoStruct, &m_pEndInfoStruct);
	}

	void ReadBytes(unsigned char* pBuf, int nBytes)
	{
		memcpy(pBuf, m_pData + m_nPos, nBytes);
		m_nPos += nBytes;
	}
};

void readFunc(png_struct* pReadStruct, png_bytep pBuf, png_size_t nSize)
{
	GPNGReader* pReader = (GPNGReader*)png_get_io_ptr(pReadStruct);
	pReader->ReadBytes((unsigned char*)pBuf, (int)nSize);
}

void loadPng(GImage* pImage, const unsigned char* pData, size_t nDataSize)
{
	// Check for the PNG signature
	if(nDataSize < 8 || png_sig_cmp((png_bytep)pData, 0, 8) != 0)
		throw Ex("not a png file");

	// Read all PNG data up until the image data chunk.
	GPNGReader reader(pData);
	png_set_read_fn(reader.m_pReadStruct, (png_voidp)&reader, (png_rw_ptr)readFunc);
	png_read_info(reader.m_pReadStruct, reader.m_pInfoStruct);

	// Get the image data
	int depth, color;
	png_uint_32 width, height;
	png_get_IHDR(reader.m_pReadStruct, reader.m_pInfoStruct, &width, &height, &depth, &color, NULL, NULL, NULL);
	GAssert(depth == 8); // unexpected depth
	pImage->setSize(width, height);

	// Set gamma correction
	double dGamma;
	if (png_get_gAMA(reader.m_pReadStruct, reader.m_pInfoStruct, &dGamma))
		png_set_gamma(reader.m_pReadStruct, 2.2, dGamma);
	else
		png_set_gamma(reader.m_pReadStruct, 2.2, 1.0 / 2.2); // 1.0 = viewing gamma, 2.2 = screen gamma

	// Update the 'info' struct with the gamma information
	png_read_update_info(reader.m_pReadStruct, reader.m_pInfoStruct);

	// Tell it to expand palettes to full channels
	png_set_expand(reader.m_pReadStruct);
	png_set_gray_to_rgb(reader.m_pReadStruct);

	// Allocate the row pointers
	unsigned long rowbytes = png_get_rowbytes(reader.m_pReadStruct, reader.m_pInfoStruct);
	unsigned long channels = rowbytes / width;
	std::unique_ptr<unsigned char[]> hData(new unsigned char[rowbytes * height]);
	png_bytep pRawData = (png_bytep)hData.get();
	unsigned int i;
	{
		std::unique_ptr<unsigned char[]> hRows(new unsigned char[sizeof(png_bytep) * height]);
		png_bytep* pRows = (png_bytep*)hRows.get();
		for(i = 0; i < height; i++)
			pRows[i] = pRawData + i * rowbytes;
		png_read_image(reader.m_pReadStruct, pRows);
	}

	// Copy to the GImage
	unsigned long nPixels = width * height;
	unsigned int* pRGBQuads = pImage->pixels();
	unsigned char *pBytes = pRawData;
	if(channels > 3)
	{
		GAssert(channels == 4); // unexpected number of channels
		for(i = 0; i < nPixels; i++)
		{
			*pRGBQuads = gARGB(pBytes[3], pBytes[0], pBytes[1], pBytes[2]);
			pBytes += channels;
			pRGBQuads++;
		}
	}
	else if(channels == 3)
	{
		for(i = 0; i < nPixels; i++)
		{
			*pRGBQuads = gARGB(0xff, pBytes[0], pBytes[1], pBytes[2]);
			pBytes += channels;
			pRGBQuads++;
		}
	}
	else
	{
		throw Ex("Sorry, loading ", to_str(channels), "-channel pngs not supported");
/*		GAssert(channels == 1); // unexpected number of channels
		for(i = 0; i < nPixels; i++)
		{
			*pRGBQuads = gARGB(0xff, pBytes[0], pBytes[0], pBytes[0]);
			pBytes += channels;
			pRGBQuads++;
		}*/
	}

	// Check for additional tags
	png_read_end(reader.m_pReadStruct, reader.m_pEndInfoStruct);
}

// -----------------------------------------------------------------------

class GPNGWriter
{
public:
	png_structp m_pWriteStruct;
	png_infop m_pInfoStruct;

	GPNGWriter()
	{
		m_pWriteStruct = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, error_handler, NULL);
		if(!m_pWriteStruct)
			throw Ex("Failed to create write struct. Out of mem?");
		m_pInfoStruct = png_create_info_struct(m_pWriteStruct);
		if(!m_pInfoStruct)
			throw Ex("Failed to create info struct. Out of mem?");
	}

	~GPNGWriter()
	{
		png_destroy_write_struct(&m_pWriteStruct, &m_pInfoStruct);
	}

	static void error_handler(png_structp png_ptr, png_const_charp msg)
	{
		throw Ex("Error writing PNG file: ", msg);
	}
};


void savePng(GImage* pImage, FILE* pFile, bool bIncludeAlphaChannel)
{
	// Set the jump value (This has something to do with enabling the error handler)
	GPNGWriter writer;
	if(setjmp(png_jmpbuf(writer.m_pWriteStruct)))
		throw Ex("Failed to set the jump value");

	// Init the IO
	png_init_io(writer.m_pWriteStruct, pFile);
	png_set_compression_level(writer.m_pWriteStruct, Z_BEST_COMPRESSION);

	// Write image stats and settings
	unsigned long width = pImage->width();
	unsigned long height = pImage->height();
	png_set_IHDR(writer.m_pWriteStruct, writer.m_pInfoStruct,
		width, height, 8,
		bIncludeAlphaChannel ? PNG_COLOR_TYPE_RGB_ALPHA : PNG_COLOR_TYPE_RGB,
		PNG_INTERLACE_NONE,	PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
	png_write_info(writer.m_pWriteStruct, writer.m_pInfoStruct);
	png_set_packing(writer.m_pWriteStruct);

	// Write the image data
	unsigned long channels = bIncludeAlphaChannel ? 4 : 3;
	unsigned long rowbytes = width * channels;
	unsigned char* pRow = new unsigned char[rowbytes];
	std::unique_ptr<unsigned char[]> hRow(pRow);
	unsigned int* pPix = pImage->pixels();
	if(channels == 4)
	{
		for(unsigned int i = 0; i < height; i++)
		{
			unsigned char* pBytes = pRow;
			for(unsigned int j = 0; j < width; j++)
			{
				*(pBytes++) = gRed(*pPix);
				*(pBytes++) = gGreen(*pPix);
				*(pBytes++) = gBlue(*pPix);
				*(pBytes++) = gAlpha(*pPix);
				pPix++;
			}
			png_write_row(writer.m_pWriteStruct, pRow);
		}
	}
	else if(channels == 3)
	{
		for(unsigned int i = 0; i < height; i++)
		{
			unsigned char* pBytes = pRow;
			for(unsigned int j = 0; j < width; j++)
			{
				*(pBytes++) = gRed(*pPix);
				*(pBytes++) = gGreen(*pPix);
				*(pBytes++) = gBlue(*pPix);
			}
			png_write_row(writer.m_pWriteStruct, pRow);
		}
	}
	else
		throw Ex("Unsupported number of channels");
	png_write_end(writer.m_pWriteStruct, writer.m_pInfoStruct);
}

void loadPng(GImage* pImage, const char* szFilename)
{
	size_t nSize;
	char* pRawData = GFile::loadFile(szFilename, &nSize);
	std::unique_ptr<char[]> hRawData(pRawData);
	loadPng(pImage, (const unsigned char*)pRawData, nSize);
}

void loadPngFromHex(GImage* pImage, const char* szHex)
{
	size_t len = strlen(szHex);
	unsigned char* pBuf = new unsigned char[len / 2];
	std::unique_ptr<unsigned char[]> hBuf(pBuf);
	GBits::hexToBuffer(szHex, len, pBuf);
	loadPng(pImage, pBuf, len / 2);
}

void savePng(GImage* pImage, FILE* pFile)
{
	savePng(pImage, pFile, true);
}

void savePng(GImage* pImage, const char* szFilename)
{
	FILE* pFile = fopen(szFilename, "wb");
	if(!pFile)
		throw Ex("Failed to create file: ", szFilename);
	FileHolder hFile(pFile);
	savePng(pImage, pFile);
}

} // namespace GClasses
