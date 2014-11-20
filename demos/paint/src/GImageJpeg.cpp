#include "GImageJpeg.h"
#include <exception>
#include <iostream>
#include <GClasses/GApp.h>
#include <GClasses/GError.h>
#include <GClasses/GImage.h>
#include <GClasses/GFile.h>
#include <GClasses/GBits.h>
#include <stdio.h>
#include <jpeglib.h>
#include <stdlib.h>

using std::cerr;
using std::cout;

namespace GClasses {


/**
 * read_jpeg_file Reads from a jpeg file on disk specified by filename and saves into the 
 * raw_image buffer in an uncompressed format.
 * 
 * \returns positive integer if successful, -1 otherwise
 * \param *filename char string specifying the file name to read from
 *
 */



void loadJpeg(GImage* pImage, const char *filename )
{
	
	/* these are standard libjpeg structures for reading(decompression) */
	struct jpeg_decompress_struct cinfo;
	struct jpeg_error_mgr jerr;
	/* libjpeg data structure for storing one row, that is, scanline of an image */
	JSAMPROW row_pointer[1];
	
	FILE *infile = fopen( filename, "rb" );
	
	if ( !infile )
		throw Ex("Error opening jpeg file ", filename );
	/* here we set up the standard libjpeg error handler */
	cinfo.err = jpeg_std_error( &jerr );
	/* setup decompression process and source, then read JPEG header */
	jpeg_create_decompress( &cinfo );
	/* this makes the library read from infile */
	jpeg_stdio_src( &cinfo, infile );
	/* reading the image header which contains image information */
	jpeg_read_header( &cinfo, TRUE );


	/* Uncomment the following to output image information, if needed. */
	/*--#include <stdlib.h>

	printf( "JPEG File Information: \n" );
	printf( "Image width and height: %d pixels and %d pixels.\n", cinfo.image_width, cinfo.image_height );
	printf( "Color components per pixel: %d.\n", cinfo.num_components );
	printf( "Color space: %d.\n", cinfo.jpeg_color_space );
	--*/
	/* Start decompression jpeg here */

	jpeg_start_decompress( &cinfo ); 

	//std::size_t buffer;

	//Set GImage and make raw image point to it.
	
	pImage->setSize(cinfo.output_width, cinfo.output_height);
	unsigned int* raw = pImage->pixelRef(0, 0);

	int y, cb, cr;

	/* allocate memory to hold the uncompressed image */
	
	//raw_image = (unsigned char*)malloc( cinfo.output_width*cinfo.output_height*cinfo.num_components );
	/* now actually read the jpeg into the raw buffer */
	row_pointer[0] = (unsigned char *)malloc( cinfo.output_width*cinfo.num_components );


	/* read one scan line at a time */
	while( cinfo.output_scanline < cinfo.image_height )
	{

		jpeg_read_scanlines( &cinfo, row_pointer, 1 );
		int input_location = 0;
		for(unsigned int i=0; i<cinfo.image_width;i++) 
		{
			y = (int)row_pointer[0][input_location++];
			cb = (int)row_pointer[0][input_location++];
			cr = (int)row_pointer[0][input_location++];
			
			*(raw++) = gRGB(y, cb, cr);
			
		}
	}

	/* wrap up decompression, destroy objects, free pointers and close open files */
	jpeg_finish_decompress( &cinfo );
	jpeg_destroy_decompress( &cinfo );
	free( row_pointer[0] );
	fclose( infile );
	


	/* yup, we succeeded! */


	


}




/**
 * write_jpeg_file Writes the raw image data stored in the raw_image buffer
 * to a jpeg image with default compression and smoothing options in the file
 * specified by *filename.
 *
 * \returns positive integer if successful, -1 otherwise
 * \param *filename char string specifying the file name to save to
 *
 */

 // LEE IMPLEMENT THIS LATER


void saveJpeg(GImage* pImage, const char *filename )
{
	struct jpeg_compress_struct cinfo;
	struct jpeg_error_mgr jerr;
	
	/* this is a pointer to one row of image data */
	JSAMPROW row_pointer[1];
	FILE *outfile = fopen( filename, "wb" );
	
	if ( !outfile )
		throw Ex("Error opening output jpeg file ", filename );
	cinfo.err = jpeg_std_error( &jerr );
	jpeg_create_compress(&cinfo);
	jpeg_stdio_dest(&cinfo, outfile);

	/* Setting the parameters of the output file here */
	cinfo.image_width = pImage->width();	
	cinfo.image_height = pImage->height();
	cinfo.input_components = 3;
	cinfo.in_color_space = (J_COLOR_SPACE)JCS_RGB;
    /* default compression parameters, we shouldn't be worried about these */
	jpeg_set_defaults( &cinfo );
	/* Now do the compression .. */
	jpeg_start_compress( &cinfo, TRUE );


	unsigned int* raw = pImage->pixelRef(0, 0);


	/* like reading a file, this time write one row at a time */
	while( cinfo.next_scanline < cinfo.image_height )
	{
		row_pointer[0] = (unsigned char*)&raw[cinfo.next_scanline * cinfo.image_width *  cinfo.input_components];
		jpeg_write_scanlines( &cinfo, row_pointer, 1 );
	}
	/* similar to read file, clean up after we're done compressing */
	jpeg_finish_compress( &cinfo );
	jpeg_destroy_compress( &cinfo );
	fclose( outfile );

}

} // namespace GClasses
