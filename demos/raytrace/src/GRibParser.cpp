//#pragma warning( disable : 4786 )
//#include <fstream.h>
#include "GRibParser.h"
#include <GClasses/GRayTrace.h>
#include <GClasses/GImage.h>
#include <stdlib.h>
#include <string.h>

// I use uint for readability.
#ifndef uint
typedef unsigned int uint;
#endif

using namespace GClasses;
using std::vector;
using std::map;
using std::string;
using std::list;

// Hopefully you'll never need this.
//#define PARSER_DEBUG

RibParser::RibParser(GRand* pRand)
: m_pScene(NULL), image_width(256), image_height(256), fov( 35 ), focal_distance(0), lens_diameter(0), baseDir( "" ), in_object_definition( false ), m_pRand(pRand)
{
	// Add the root state block.
	state.push_back( StateBlock() );
}

RibParser::~RibParser()
{
	// Delete the memory that the files were read into, since it's no longer needed.
	for ( uint i = 0; i < lineBuffs.size(); i++ )
		delete[] lineBuffs[ i ];
	delete(m_pScene);
}

/*static*/ GRayTraceScene* RibParser::LoadScene(const char* szFilename, GRand* pRand)
{
	RibParser parser(pRand);
	if(!parser.createScene(szFilename))
		return NULL;
	return parser.releaseScene();
}

// This is just a convenience function, taking a filename instead of a stream.
// See the other overloaded method for actual behavior
bool RibParser::createScene(string filename)
{
	FILE *F = fopen(filename.c_str(), "r");
	if (!F) return false;

	// Set the base name on the parser, so that it knows where to find relative files
	// TODO: test for bugs
	{
		string delims = "\\/";
		string blank = "";
#ifdef WINDOWS
		char mydelim = '\\';
		char otherdelim = '/';
#else
		char mydelim = '/';
		char otherdelim = '\\';
#endif
		// First, see if there's any directory delimiters
		string::size_type last_delim = filename.find_last_of( delims );
		if ( last_delim != string::npos )
		{
			baseDir = filename;
			baseDir.erase( last_delim + 1 );
			for ( string::size_type i = 0; i < baseDir.size(); i++ )
				if ( baseDir[i] == otherdelim )
					baseDir[i] = mydelim;
		}
	}
	return createScene(F);
}

// This reads the whole file at once, then handles each command
bool RibParser::createScene(FILE* F)
{
	// If we're recursing, then m_pScene has already been allocated
	if(!m_pScene)
		m_pScene = new GRayTraceScene(m_pRand);

	bool success = true; // Let's be optimistic

	vector< int > lines, line_nums;
	int total_length;
	RibData line;

	total_length = readLines( F, lines, line_nums );
	//printf("Read %d lines, parsing...\n", lines.size());
	char* line_buff = lineBuffs.back();

	// build RibData and handle each line.
	for ( uint i = 0; i < lines.size(); i++ )
	{
		buildData( line_buff, lines[ i ], line_nums[ i ], line );
		success = handleLine( line ) && success;
		line.clear();
	}
	//printf("Done parsing scene\n");
	return success;
}

int RibParser::readLines( FILE* F, vector< int >& lines, vector< int >& line_nums )
{
	char *line_buff;
	string temp;
	int length;
	int length_guess; // Oddly enough, length_guess != length.  It filters out '\r' because it's in ascii mode..
	int num_file_lines = 0;
	int seek_ahead;
	int brackets_match = 0;
	bool empty_line = true;

	fseek(F, 0, SEEK_END);
	length_guess = ftell(F);
	line_buff = new char[length_guess + 4];
	fseek(F, 0, SEEK_SET);
	length = fread(line_buff, 1, length_guess, F);
	line_buff[length] = '\0';
	fclose(F);

	if (length == 0) {
		printf("RIB Error! Could not load file.\n");
		delete[] line_buff;
		return 0;
	}

	// store the location of line_buff on the RibParser object, so that it will be deleted properly later
	lineBuffs.push_back( line_buff );

	int prev_pos = 0;
	for ( int i = 0; i < length; i++ )
	{
		switch( line_buff[ i ] )
		{
                case '\r':
                        line_buff[i] = '\n'; // Then fall through
		case '\n':
			{ // End of line
				num_file_lines++;

				if ( empty_line )
				{ // Whitespace only line.
					line_buff[ i ] = ' '; // Concat blank lines as trailing whitespace to previous line
					prev_pos = i+1;
					continue;
				}
				else
				{ // Line with something useful on it is ending.
					// To make sure it's really the end of the line,
					// we need to seek ahead to find the first non-whitespace character.
					// If it's " or [, then this isn't the end of a line (because the next line doesn't
					// begin with a Rib command - it's a continuation)
					// We also have to ignore comments, and blank lines.
					// We also need to make sure we're not in the middle of a list.
					// This can be kind of a pain.
					if ( brackets_match != 0 )
					{ // We're in the middle of a list - keep it all as one line.
						line_buff[ i ] = ' ';
						continue;
					}
					else
					{
						bool end_of_line = true;
						for ( seek_ahead = i; seek_ahead < length; seek_ahead++ )
						{
							switch( line_buff[ seek_ahead ] )
							{
							case ' ':
							case '\t':
							case '\n':
								// Keep going through whitespace - this isn't what we're looking for.
								break;
							case '#':
								// Eat the rest of the comment - this isn't what we're looking for either.
								while ( seek_ahead < length )
								{
									if ( line_buff[ seek_ahead ] == '\n' )
										break;
									seek_ahead++;
								}
								break;
							case '\"':
							case '[':
								// Not the end of the line
								end_of_line = false;
								seek_ahead = length; // This is a way of breaking out of the for loop
								break;
							default:
								end_of_line = true;
								seek_ahead = length; // This is a way of breaking out of the for loop
								break;
							}
						}

						if ( !end_of_line )
						{ // This isn't the end of the line
							line_buff[ i ] = ' ';
							continue;
						}
					}
					// At this point, we're sure that this was the end of a line.
					// Add this line to the vector
					line_buff[ i ] = '\0';
					lines.push_back( prev_pos );
					line_nums.push_back( num_file_lines );
					// Reset for next line
					prev_pos = i+1;
					empty_line = true;
				}
			}
			break;
		case ' ':
		case '\t':
			{ // Whitespace
			}
			break;
		case '#':
			{ // Just blank out the comments until the end of the line.  We don't need to parse comments.
				while ( i < length && line_buff[i+1] != '\n' )
				{
					line_buff[i] = ' ';
					i++;
				}
				line_buff[i] = ' ';
			}
			break;
		case '[':
			brackets_match++;
			empty_line = false;
			break;
		case ']':
			brackets_match--;
			empty_line = false;
			break;
		default:
			empty_line = false;
		}
	}
	// We'll check to see if we had a final line that hasn't been added yet
	if ( !empty_line )
	{
		lines.push_back( prev_pos );
		line_nums.push_back( num_file_lines );
	}

	// Return the total length of the stream read.
	return length;
}

void RibParser::buildData( char* big_str, int begin, int line_num, RibData& line )
{
	char* str = big_str + begin;
	int length = strlen( str );
	int prev_pos = 0;
	char temp;
	//int brackets_balance = 0;
	int i;

	// Find first non-whitespace
	for ( i = 0; i < length; i++ )
	{
		if ( str[ i ] != ' ' || str[ i ] != '\t' )
			break;
	}
#ifdef PARSER_DEBUG
	if ( str[ i ] == '\"' || str[ i ] == '[' )
	{
		// This shouldn't happen, unless there's a bug in read_lines, or a problem in the file
		fprintf(stderr, "First token on line is quote or an open bracket.\nThis is an error in the rib file (or perhaps the parser).\n");
	}
#endif
	// Set position to first non-whitespace
	prev_pos = i;

	// The main state machine loop.
	for ( i = prev_pos; i < length; i++ )
	{
		switch( str[ i ] )
		{
		case ' ':
		case '\t':
			// This is delimiting whitespace.  Make it a string terminator
			str[ i ] = '\0';
			break;
		case '[':
			{
				// It's possible to have an open bracket terminate a token.  So, let's make it a string terminator also.
				str[i] = '\0';
				// Make the parent RibData and add it to the line.
				line.append( RibData( RibData::LEAF_ARRAY ) );
				// Actually do the extraction.  This is somewhat involved, so I made it a separate function
				this->extractArray( str, length, i, line.back() );
			}
			break;
#ifdef PARSER_DEBUG
		case ']':
			fprintf(stderr, "Mismatched close bracket\n"); break;
		case '#':
			fprintf(stderr, "All the comments should be removed by now..\n"); break;
#endif
		case '\"':
			// Find the whole quoted string
			i++;
			prev_pos = i;
			while ( i < length && str[ i ] != '\"' )
				i++;
#ifdef PARSER_DEBUG
			if ( !( i < length ) )
				fprintf(stderr, "Error: Unterminated string.\n");
#endif
			str[ i ] = '\0'; // Zero out the final quote
			line.append( RibData( &str[ prev_pos ], true ) );
			break;
		default:
			// Find the whole token
			prev_pos = i;
			while ( i < length )
			{
				i++;
				char c = str[ i ];
				if ( c == ' ' || c == '\t' || c == '[' || c == '\"' )
					break;
			};
			// The next character might be significant for the state ( '[' and '"' are significant ).
			// So we'll have to save it, so it can be processed by the state machine.
			temp = str[ i ];
			str[ i ] = '\0';
			line.append( RibData( &str[ prev_pos ], false ) );
			str[ i ] = temp;
			i--; // Back up, so as not to miss a character after the loop increment
			break;
		}
	}
}

void RibParser::extractArray( char* str, int length, int& i, RibData& parent )
{
	list< char* > temp_vals; // Just store it as a list temporarily.  We'll move it to a vector when we know the size

	// Let's chew through the array values.
	for ( i++; i < length; i++ )
	{
		if ( str[i] == ' ' || str[i] == '\t' )
		{
			str[i] = '\0';
			continue;
		}
		else if ( str[i] == ']' )
		{
			str[i] = '\0';
			// The array is done - build the vector and return.
			parent.reserveCapacity( temp_vals.size() );
			vector< const char* >& vals = parent.accessVals();
			vals.clear();
			vals.resize( temp_vals.size(), (const char*)0xdeafbeef );
			int ival = 0;
			std::list< char* >::iterator lit;
			for ( lit = temp_vals.begin(); lit != temp_vals.end(); lit++, ival++ )
			{
				vals[ ival ] = *lit;
			}
			return;
		}
		else if ( str[i] == '\"' )
		{
			char* start_ptr = str + i; // We start at the quote, so we know it's a quoted string.
						               // Although the end quote is overwritten by a null, to make
									   // string manipulation a little bit easier.
			// Get off the current quote
			i++;
			// Find matching quote
			while ( i < length && str[i] != '\"' )
				i++;
			// At this point, we're either at the end of string ( which means malformed RIB )
			// or, str[i] == '\"'
#ifdef PARSER_DEBUG
			if ( ! (i < length) )
				fprintf(stderr, "Malformed RIB: Unterminated quote, unterminated array\n");
#endif
			// Add the entry to the parent's vals.
			str[i] = '\0'; // Overwrite the quote with nul, to make string manipulation easier
			temp_vals.push_back( start_ptr );
		}
		else
		{   // Some normal token: either a string, double, or int.  We'll find out when we are asked for it.
			// This is inner loop for large scenes, so I'm going to use pointer arithmetic instead of indices
			char* start_ptr = str + i;
			char* next_ptr = start_ptr + 1;
			char* end_ptr = start_ptr + length;
			// Find end of the token.  Can be terminated by whitespace, or close bracket.
			while ( next_ptr != end_ptr )
			{
				if ( *next_ptr == ' ' || *next_ptr == ']' )
					break; // Note that a null is written to these positions, to ease string manipulation
							// The null is actually written to str in the next round of the outer for loop
				next_ptr++;
			}
			i += next_ptr - start_ptr - 1;
			temp_vals.push_back( start_ptr );
		}
	}
	fprintf(stderr, "Error in Rib file: Unterminated array\n");
}

bool RibParser::handleLine( const RibData& data )
{
#ifdef PARSER_DEBUG
	if ( data.size() == 0 )
	{
		fprintf(stderr, "The data passed in should always be an array\n");
		return false;
	}
#endif
	bool success = true;

	string command;
	command = data[ 0 ].asString();

	// If defining an object, store commands to be instanced later.
	if ( in_object_definition && command != "ObjectEnd" )
	{
		RibData& object_parent = getObjectDefinition( curr_object );
		object_parent.append( data );
		return true;
	}

	// Block management
	if ( command == "TransformBegin" ) {
		pushState( StateBlock::TRANSFORMBLOCK );
	} else if ( command == "TransformEnd" ) {
		popState( StateBlock::TRANSFORMBLOCK );
	} else if ( command == "AttributeBegin" ) {
		pushState( StateBlock::ATTRIBUTEBLOCK );
	} else if ( command == "AttributeEnd" ) {
		popState( StateBlock::ATTRIBUTEBLOCK );
	} else if ( command == "WorldBegin" ) {
		pushState( StateBlock::WORLDBLOCK );
	} else if ( command == "WorldEnd" ) {
		popState( StateBlock::WORLDBLOCK );
	} else if ( command == "FrameBegin" ) {
		pushState( StateBlock::FRAMEBLOCK );
	} else if ( command == "FrameEnd" ) {
		popState( StateBlock::FRAMEBLOCK );
	}
	// These are the handlers for various commands
	else if ( command == "Transform" ) {
		success = handleTransform( data );
	} else if ( command == "ConcatTransform" ) {
		success = handleConcatTransform( data );
	} else if ( command == "Translate" ) {
		success = handleTranslate( data );
	} else if ( command == "ReadArchive" ) {
		success = handleReadArchive( data );
	} else if ( command == "ObjectBegin" ) {
		success = handleObjectBegin( data );
	} else if ( command == "ObjectEnd" ) {
		success = handleObjectEnd( data );
	} else if ( command == "ObjectInstance" ) {
		success = handleObjectInstance( data );
	} else if ( command == "Format" ) {
		success = handleFormat( data );
	} else if ( command == "PixelSamples" ) {
		success = handlePixelSamples( data );
	} else if ( command == "Imager" ) {
		success = handleImager( data );
	} else if ( command == "Projection" ) {
		success = handleProjection( data );
	} else if ( command == "DepthOfField" ) {
		success = handleDepthOfField( data );
	} else if ( command == "Attribute" ) {
		success = handleAttribute( data );
	} else if ( command == "Color" ) {
		success = handleColor( data );
	} else if ( command == "Surface" ) {
		success = handleSurface( data );
	} else if ( command == "LightSource" ) {
		success = handleLight( data );
	} else if ( command == "Sphere" ) {
		success = handleSphere( data );
	} else if ( command == "NuPatch" ) {
		success = handleNuPatch( data );
	} else if ( command == "PointsGeneralPolygons" ) {
		success = handlePointsGeneralPolygons( data );
	} else if ( command == "AreaLightSource" ) {
		success = handleAreaLightSource( data );
	} else if ( command == "Display" || command == "Clipping" || command == "ReverseOrientation" || command == "Orientation" ) {
		// Silently ignore these commands.
	} else {
		printf("Warning: Unhandled RIB data: %s\n", command.c_str());
		//data.debugPrint( cout, 3 );
	}

	return success;
}

bool RibParser::handleTransform( const RibData& data )
{
	if ( data[ 1 ].getType() != RibData::LEAF_ARRAY )
	{
		fprintf(stderr, "Bad Transform Syntax\n");
		return false;
	}
	const RibData& arr = data[ 1 ];
	if ( arr.size() != 16 )
	{
		fprintf(stderr, "Array is not of size 16\n");
		return false;
	}
	double tempmatrix[16];
	for ( uint i = 0; i < 16; i++ )
		tempmatrix[i] = arr.doubleVal( i );
	matrixTranspose( tempmatrix, tempmatrix );
	setObjectTransform( tempmatrix );
	return true;
}

bool RibParser::handleConcatTransform( const RibData& data )
{
	if ( data[ 1 ].getType() != RibData::LEAF_ARRAY )
	{
		fprintf(stderr, "Bad Transform Syntax");
		return false;
	}
	const RibData& arr = data[ 1 ];
	if ( arr.size() != 16 )
	{
		fprintf(stderr, "Array is not of size 16\n");
		return false;
	}
	double tempmatrix[16];
	for ( uint i = 0; i < 16; i++ )
		tempmatrix[i] = arr.doubleVal( i );
	matrixTranspose( tempmatrix, tempmatrix );
	multiplyTransform( tempmatrix );
	return true;
}

bool RibParser::handleTranslate( const RibData& data )
{
	if ( data.size() < 4 )
	{
		fprintf(stderr, "Bad translate syntax.  Expected dx dy dz, only have %d arguments", data.size() - 1);
		//data.debugPrint( cerr, 3 );
		return false;
	}
	else
	{
		double translate[16];
		matrixMakeIdentity( translate );
		translate[ 3 ] = data[ 1 ].asDouble();
		translate[ 7 ] = data[ 2 ].asDouble();
		translate[ 11 ] = data[ 3 ].asDouble();
		multiplyTransform( translate );
		return true;
	}
}

bool RibParser::handleReadArchive( const RibData& data )
{
	if ( data.size() < 2 )
	{
		fprintf(stderr, "ReadArchive needs 1 argument, only has %d\n", data.size() - 1);
		//data.debugPrint( cerr, 3 );
		return false;
	}
	string filename = baseDir + data[1].asString();
	// Store the old baseDir, it gets clobbered.
	string old_baseDir = baseDir;
	baseDir = "";

	// Let's hear it for recursion
	bool success = createScene(filename);

	baseDir = old_baseDir;
	return success;
}

bool RibParser::handleObjectBegin( const RibData& data )
{
	if ( data.size() < 2 )
	{
		fprintf(stderr, "Error in RIB: ObjectBegin requires 1 argument, has %d\n", data.size() - 1);
		return false;
	}
	// These variables are acted on in handleLine
	in_object_definition = true;
	curr_object = data[ 1 ].asInt();
	return true;
}

bool RibParser::handleObjectEnd( const RibData& data )
{
	in_object_definition = false;
	return true;
}

bool RibParser::handleObjectInstance( const RibData& data )
{
	if ( data.size() < 2 )
	{
		fprintf(stderr, "Error in RIB: ObjectBegin requires 1 argument, has %d\n", data.size() - 1);
		return false;
	}
	bool success = true;
	int instanced_object = data[ 1 ].asInt();
	RibData &object = getObjectDefinition( instanced_object );
	//object.recreatePointers();
	for ( int i = 0; i < object.size(); i++ )
	{
		//object[ i ].recreatePointers();
		success = handleLine( object[ i ] ) && success;
	}
	return success;
}

bool RibParser::handleFormat( const RibData& data )
{
	if ( data.size() < 4 )
	{
		fprintf(stderr, "Format requires 3 arguments, %d provided.\n", data.size() - 1);
		return false;
	}
	else
	{
		// Because these aren't attached to the camera in Rib like Dart wants, we'll
		// just store them, and then when we're done parsing, go back and set the
		// values on the camera
		image_width = data[ 1 ].asInt();
		image_height = data[ 2 ].asInt();
		// We could get the pixel aspect ratio here if we wanted it.
		// We'll just assume square pixels
		return true;
	}
}

bool RibParser::handlePixelSamples( const RibData& data )
{
	if ( data.size() < 3 )
	{
		fprintf(stderr, "Format requires 2 arguments, %d provided.\n", data.size() - 1);
		return false;
	}
	else
	{
		int pixel_samples_x = data[ 1 ].asInt();
		int pixel_samples_y = data[ 2 ].asInt();
		pixel_samples = pixel_samples_x * pixel_samples_y;
//		m_pScene->camera()->setRaysPerPixel(pixel_samples);
 		return true;
	}
}

bool RibParser::handleImager( const RibData& data )
{
	// I'll just assume that they're using the 'background' imager, if they're using anything.
	double background_color[ 4 ] = { 0, 0, 0, 1 };
	bool success = data.fillDoubleArrayFromOptionalArg( "background", background_color, 3 );
	m_pScene->setBackgroundColor(0, background_color[0], background_color[1], background_color[2]);
	return success;
}

bool RibParser::handleProjection( const RibData& data )
{
	// I'll just assume that they're using perspective.
	if ( data.hasOptionalArg( "fov" ) )
	{
		// This is much the same deal as the resolution - store for later
		double fov_as_degrees;
		if ( data.fillDoubleArrayFromOptionalArg( "fov", &fov_as_degrees, 1 ) )
		{
			// Only set the fov if we successfully get one.
			// Convert to radians
			fov = fov_as_degrees / 180.0 * 3.141592653589793238462643;
		}
	}
	return true;
}

bool RibParser::handleDepthOfField( const RibData& data )
{
	if ( data.size() < 4 )
	{
		fprintf(stderr, "Bad number of arguments to DepthOfField.  Requires at least 3, has %d\n", data.size() - 1);
		//data.debugPrint( cerr, 3 );
		return false;
	}
	else
	{
		double fstop;
		double focal_length;

		fstop = data[1].asDouble();
		focal_length = data[2].asDouble();
		focal_distance = data[3].asDouble();

		// Compute lens diameter
		lens_diameter = focal_length / fstop;
	}
	return true;
}

bool RibParser::handleAttribute( const RibData& data )
{
	if ( data.size() < 4 )
	{
		fprintf(stderr, "Bad number of arguments to Attribute.  Requires at least 3, has %d\n", data.size() - 1);
		//data.debugPrint( cerr, 3 );
		return false;
	}
	else
	{
		string attr_class = data[ 1 ].asString();
		string attr_name = data[ 2 ].asString();
		setAttribute( attr_class, attr_name, data[ 3 ] );
		return true;
	}
}

bool RibParser::handleColor( const RibData& data )
{
	if ( data.size() < 2 || data[1].size() != 3 )
	{
		fprintf(stderr, "Color command invalid argument sizes\n");
		//data.debugPrint( cerr, 3 );
		return false;
	}
	else
	{
		double temp_color[3];
		for ( uint icolor = 0; icolor < 3; icolor++ )
			temp_color[ icolor ] = data[ 1 ].doubleVal( icolor );

		setColor( temp_color );
		return true;
	}
}

bool RibParser::handleSurface( const RibData& data )
{
	// Colors, and their defaults
	double cd[ 4 ] = { .5, .5, .5, 1 };
	double cs[ 4 ] = { 1, 1, 1, 1 };
	double cr[ 4 ] = { 1, 1, 1, 1 };
	double ct[ 4 ] = { 0, 0, 0, 1 };
	double ca[ 4 ] = { .1, .1, .1, 1 };
	double ce[ 4 ] = { 0, 0, 0, 1 }; // This emissive color comes from the state block, via the command AreaLightSource

	// Scalar properties, and their defaults
	double ior = 1.0; // index of refraction
	double phong = 1.0;
	double glossiness = 1.0;
	double translucency = 0.0;

	if ( data.size() < 2 )
	{
		fprintf(stderr, "Bad number of arguments to Surface.  Requires at least 1, has %d\n", data.size() - 1);
		//data.debugPrint( cerr, 3 );
		return false;
	}
	else
	{
		string surface_type = data[ 1 ].asString();
		if ( surface_type == "cs655surface" ) {
			getColor( cd ); // Diffuse color is from a previous command.
			data.fillDoubleArrayFromOptionalArg( "specularcolor", cs, 3 );
			data.fillDoubleArrayFromOptionalArg( "reflectedcolor", cr, 3 );
			data.fillDoubleArrayFromOptionalArg( "transmittedcolor", ct, 3 );
			data.fillDoubleArrayFromOptionalArg( "ambientcolor", ca, 3 );
			data.fillDoubleArrayFromOptionalArg( "indexofrefraction", &ior, 1 );
			data.fillDoubleArrayFromOptionalArg( "phong", &phong, 1 );
			data.fillDoubleArrayFromOptionalArg( "glossiness", &glossiness, 1 );
			data.fillDoubleArrayFromOptionalArg( "translucency", &translucency, 1 );
			// Emissive color comes from the AreaLightSource shader
			for ( uint irgb = 0; irgb < 3; irgb++ )
				ce[ irgb ] = state.back().emissive_color[ irgb ];
		} else {
			fprintf(stderr, "Unhandled surface shader %s\n", surface_type.c_str());
			//data.debugPrint( cerr, 3 );
			return false;
		}

		// Make the material
		GRayTraceMaterial* pMat;

		// It should be noted that this way of doing textures isn't really RIB friendly.
		// But it's easier than parsing Pixar's .tex format.
		if ( data.hasOptionalArg( "texturefile" ) )
		{
			GRayTraceImageTexture* pMaterial = new GRayTraceImageTexture();
			GImage* pTextureImage = new GImage();
			const RibData& filename_data = data.findOptionalArg( "texturefile" );
			string filename = baseDir + filename_data.stringVal( 0 );
			pTextureImage->loadByExtension(const_cast< char* >(filename.c_str()));
			pMaterial->setTextureImage(pTextureImage, true);
			pMat = pMaterial;
		}
		else
		{
			GRayTracePhysicalMaterial* pMaterial = new GRayTracePhysicalMaterial();
			pMaterial->setColor(GRayTraceMaterial::Diffuse, cd[0], cd[1], cd[2]);
			pMaterial->setColor(GRayTraceMaterial::Specular, cs[0], cs[1], cs[2]);
			pMaterial->setColor(GRayTraceMaterial::Reflective, cr[0], cr[1], cr[2]);
			pMaterial->setColor(GRayTraceMaterial::Transmissive, ct[0], ct[1], ct[2]);
			pMaterial->setColor(GRayTraceMaterial::Ambient, ca[0], ca[1], ca[2]);
			pMaterial->setColor(GRayTraceMaterial::Emissive, ce[0], ce[1], ce[2]);
			pMaterial->setIndexOfRefraction(ior);
			pMaterial->setSpecularExponent(phong);
			pMaterial->setGlossiness(glossiness);
			pMaterial->setCloudiness(translucency);
			pMat = pMaterial;
		}

		// Set the current material in the parser state
		m_pScene->addMaterial(pMat);
		setMaterial(pMat);

		return true;
	}
}

bool RibParser::handleLight( const RibData& data )
{
	bool is_ambient = false, is_directional = false, is_point = false;
	double color[ 4 ] = { 1, 1, 1, 1 };
	double direction[ 4 ] = { 0, 0, 1, 1 };
	double location[ 4 ] = { 0, 0, 0, 1 };
	double jitter = 0;
	uint i;

	if ( data.size() < 3 )
	{
		fprintf(stderr, "Bad number of arguments to LightSource.  Requires at least 1, has %d\n", data.size() - 1);
		//data.debugPrint( cerr, 3 );
		return false;
	}
	else
	{
		string light_type = data[ 1 ].asString();
		if ( light_type == "cs655ambientLight" ) {
			is_ambient = true;
			data.fillDoubleArrayFromOptionalArg( "lightcolor", color, 3 );
		} else if ( light_type == "cs655directionalLight" ) {
			is_directional = true;
			data.fillDoubleArrayFromOptionalArg( "lightcolor", color, 3 );
			data.fillDoubleArrayFromOptionalArg( "to", direction, 3 );
			// Reverse the 'to' to become the direction
			for ( uint idir = 0; idir < 3; idir++ )
				direction[idir] = -direction[idir];
			data.fillDoubleArrayFromOptionalArg( "jitter", &jitter, 1 );
		} else if ( light_type == "cs655pointLight" ) {
			is_point = true;
			data.fillDoubleArrayFromOptionalArg( "lightcolor", color, 3 );
			data.fillDoubleArrayFromOptionalArg( "from", location, 3 );
			data.fillDoubleArrayFromOptionalArg( "jitter", &jitter, 1 );
		} else if ( light_type == "pointlight" ) {
			// This is an example of supporting other shader types with our model.
			// We just need to fill the global params of the function, and tell it
			// that we want to create a point light.
			is_point = true;
			double intensity; // This is a multiplier for color.
			data.fillDoubleArrayFromOptionalArg( "intensity", &intensity, 1 );
			data.fillDoubleArrayFromOptionalArg( "lightcolor", color, 3 );
			data.fillDoubleArrayFromOptionalArg( "from", location, 3 );
			// Multiply in the intensity.
			for ( i = 0; i < 3; i++ )
				color[i] *= intensity;
			// That's all we need to do to support Renderman's default pointlight shader.
		} else if ( light_type == "distantlight" ) {
			// This is Renderman's built-in directional light shader
			is_directional = true;
			double intensity;
			double from[3] = { 0, 0, 0 };
			double to[3] = { 0, 0, 1 };
			data.fillDoubleArrayFromOptionalArg( "intensity", &intensity, 1 );
			data.fillDoubleArrayFromOptionalArg( "lightcolor", color, 3 );
			data.fillDoubleArrayFromOptionalArg( "from", from, 3 );
			data.fillDoubleArrayFromOptionalArg( "to", to, 3 );
			for ( i = 0; i < 3; i++ )
				color[i] *= intensity;
			// Subtract to and from to get the direction
			for ( i = 0; i < 3; i++ )
				direction[ i ] = from[ i ] - to[ i ];
		} else if ( light_type == "mtorDirectionalLight" ) {
			// This is mtor's default directional light shader
			is_directional = true;
			double intensity;
			data.fillDoubleArrayFromOptionalArg( "intensity", &intensity, 1 );
			data.fillDoubleArrayFromOptionalArg( "lightcolor", color, 3 );
			data.fillDoubleArrayFromOptionalArg( "float intensity", &intensity, 1 );
			data.fillDoubleArrayFromOptionalArg( "color lightcolor", color, 3 );
			for ( i = 0; i < 3; i++ )
				color[i] *= intensity;
			// The direction of this light is based on the current transformation.
			double direction_point[4] = { 0, 0, -1, 1 };
			double transform[16];
			getObjectTransform( transform );
			matrixMultiplyPoint( direction, transform, direction_point );
		} else {
			fprintf(stderr, "Unhandled light shader: %s\n", light_type.c_str());
			//data.debugPrint( cerr, 3 );
			return false;
		}

		// At this point, we know the light params, and just need to build it.
		if ( is_ambient )
			m_pScene->setAmbientLight(color[0], color[1], color[2]);
		if ( is_directional )
		{
			m_pScene->addLight(new GRayTraceDirectionalLight(direction[0], direction[1], direction[2], color[0], color[1], color[2], jitter));
		}
		if ( is_point )
		{
			m_pScene->addLight(new GRayTracePointLight(location[0], location[1], location[2], color[0], color[1], color[2], jitter));
		}
		return true;
	}
}

bool RibParser::handleSphere( const RibData& data )
{
	// We only pay attention to the radius
	double radius = 1.0;

	if ( data.size() < 3 )
	{
		fprintf(stderr, "Bad number of arguments to Sphere.  Requires at least 1, has %d\n", data.size() - 1);
		//data.debugPrint( cerr, 3 );
		return false;
	}
	else
	{
		radius = data[ 1 ].asDouble();

		// Probably take a point at the origin of object space, and translate it with the object-to-world matrix
		// This will let us find the position of the center of the sphere in world space
		double point[4] = { 0, 0, 0, 1 };
		double curr_transform[16];
		getObjectTransform( curr_transform );
		matrixMultiplyPoint( point, curr_transform, point );

		// Create the sphere
		GRayTraceMaterial* pMaterial = getMaterial();
		GRayTraceSphere* pSphere = new GRayTraceSphere(pMaterial, point[0], point[1], point[2], radius);
		m_pScene->addObject(pSphere);

		if(pMaterial)
		{
			GRayTraceColor* pEmissive = pMaterial->color(GRayTraceMaterial::Emissive, NULL);
			if ( !pEmissive->isBlack() )
			{
				GRayTraceAreaLight* pAreaLight = new GRayTraceAreaLight(pSphere, pEmissive->r, pEmissive->g, pEmissive->b);
				m_pScene->addLight( pAreaLight );
			}
		}

		return true;
	}
}

bool RibParser::handleNuPatch( const RibData& data )
{
	// We don't actually expect you to handle nurbs patches..
	// This is just a hack to easily produce files with spheres.
	// Any nurbs patch will be replaced with a unit sphere at the origin of the current object space
	// So, as long as your nurbs patches are unit spheres at the origin, this should be correct.. :)
	RibData hack_sphere;
	hack_sphere.append( RibData( "Sphere" ) );
	hack_sphere.append( RibData( "1.0" ) );
	hack_sphere.append( RibData( "-1.0" ) );
	hack_sphere.append( RibData( "1.0" ) );
	hack_sphere.append( RibData( "360.0" ) );
	return handleSphere( hack_sphere );
}

bool RibParser::handlePointsGeneralPolygons( const RibData& data )
{
	uint npolys;
	uint npoints;
	uint nnormals = 0;
	uint ntexcoords = 0;
	vector< uint > nvertices;
	vector< vector< uint > > vertex_indices;
	double *points = NULL;
	double *normals = NULL;
	double *texcoords = NULL;
	bool homogeneous;

	uint ipt;
	uint ipoly;

	if ( data.size() < 6 )
	{
		fprintf(stderr, "Not enough arguments to PointsGeneralPolygons, needs at least 6, got %d\n", data.size() - 1);
		//data.debugPrint( cerr, 3 );
		return false;
	}

	if ( data.hasOptionalArg( "P" ) )
		homogeneous = false;
	else if ( data.hasOptionalArg( "Pw" ) )
		homogeneous = true;
	else
	{
		fprintf(stderr, "Doesn't have points or homogeneous points - skipping\n");
		//data.debugPrint( cerr, 3 );
		return false;
	}

	npolys = data[1].size();
	nvertices.reserve( npolys );
	vertex_indices.resize( npolys );
	uint sum_idx = 0;
	for ( ipoly = 0; ipoly < npolys; ipoly++ )
	{
		if ( data[1].intVal( ipoly ) != 1 )
		{
			fprintf(stderr, "I only handle polys with a single loop.\n");
			//data.debugPrint( cerr, 3 );
			return false;
		}
		int nvertices_temp = data[2].intVal( ipoly );
		nvertices.push_back( nvertices_temp );
		vertex_indices[ ipoly ].reserve( nvertices_temp );
		for ( uint ivtx = sum_idx; ivtx < sum_idx + nvertices_temp; ivtx++ )
		{
			vertex_indices[ ipoly ].push_back( data[3].intVal( ivtx ) );
		}
		sum_idx += nvertices_temp;
	}

	// Get geometry of points
	const RibData& points_arg = data.findOptionalArg( homogeneous ? "Pw" : "P" );
	npoints = points_arg.size();
	if ( homogeneous )
		npoints /= 4;
	else
		npoints /= 3;
	points = new double[ npoints * 4 ];
	if ( homogeneous )
	{
		for ( ipt = 0; ipt < npoints; ipt++ )
		{
			points[ ipt*4 + 0 ] = points_arg.doubleVal( ipt*4 + 0 );
			points[ ipt*4 + 1 ] = points_arg.doubleVal( ipt*4 + 1 );
			points[ ipt*4 + 2 ] = points_arg.doubleVal( ipt*4 + 2 );
			points[ ipt*4 + 3 ] = points_arg.doubleVal( ipt*4 + 3 );
		}
	}
	else
	{
		for ( ipt = 0; ipt < npoints; ipt++ )
		{
			points[ ipt*4 + 0 ] = points_arg.doubleVal( ipt*3 + 0 );
			points[ ipt*4 + 1 ] = points_arg.doubleVal( ipt*3 + 1 );
			points[ ipt*4 + 2 ] = points_arg.doubleVal( ipt*3 + 2 );
			points[ ipt*4 + 3 ] = 1.0;
		}
	}

	// Get normals
	if ( data.hasOptionalArg( "N" ) )
	{
		const RibData& normals_arg = data.findOptionalArg( "N" );
		uint total_size = normals_arg.size();
		nnormals = total_size / 3; // 3 components per normal
		normals = new double[ total_size ];
		for ( uint inormal = 0; inormal < total_size; inormal++ )
		{
			normals[ inormal ] = normals_arg.doubleVal( inormal );
		}
	}

	// Get texture coordinates
	if ( data.hasOptionalArg( "st" ) )
	{
		const RibData& texcoords_arg = data.findOptionalArg( "st" );
		uint total_size = texcoords_arg.size();
		ntexcoords = total_size / 2; // 2 components per texture coordinate
		texcoords = new double[ total_size ];
		for ( uint ist = 0; ist < total_size; ist++ )
		{
			texcoords[ ist ] = texcoords_arg.doubleVal( ist );
		}
	}

	// Transform points
	double curr_transform[16];
	getObjectTransform( curr_transform );
	for ( ipt = 0; ipt < npoints; ipt++ )
	{
		double *curr_pt = &points[ ipt*4 ];
		matrixMultiplyPoint( curr_pt, curr_transform, curr_pt );
	}

	// There are a couple of different ways of storing a polygonal mesh.
	// We've implemented a subset of Renderman's pointsGeneralPolygons command.
	// The basic idea is that you have an array of loops, an array of the number of vertices in each loop,
	//     an array of those vertices, and then the data for the vertices.
	// The data consists of points, and then optionally normals and texture coordinates.
	// The index for a given vertex is the same for vertices, points, and texture coordinates, so a single vertex only has one normal
	// However, they compensate for that by just duplicating data, if needed.
	//
	// On to what you'll actually need to do.
	//
	// The data you'll need is stored in the following variables:
	//
	// npolys: The number of polygons
	// npoints: The number of points
	// nnormals: The number of normals
	// ntexcoords: The number of texture coordinates
	// nvertices: An array of size npolys.  Tells how many vertices each polygon has.
	// vertex_indices: A 2d array of size [(npolys)][(nvertices[npolys])].
	//                 This keeps track of the indices of the vertices around a polygon
	//                 For example, vertex_indices[ 5 ][ 2 ] would be the vertex 2 on polygon 5.
	// points: An array of doubles, 4 for each point (whether the points were specified homogeneously or not)
	// normals: An array of doubles, 3 for each point, if normals are provided for this polyMesh.  If not, you should compute face normals.
	// texcoords: An array of doubles, 2 for each point.
	// homogeneous: A boolean, telling you whether you need to use homogeneous coordinates or not.
	//              You might be able to get some speed out of not dividing out w, especially for large meshes.
	//              In the sample files for the class, none of them are homogeneous, so this is only really for loading other files.
	//
        GRayTraceMaterial *material = getMaterial();
        GRayTraceTriMesh *mesh = new GRayTraceTriMesh(material, npoints, npolys, nnormals, ntexcoords);

	// Points
	G3DVector v;
	for (ipt = 0; ipt < npoints; ipt++)
	{
		v.m_vals[0] = points[ipt * 4];
		v.m_vals[1] = points[ipt * 4 + 1];
		v.m_vals[2] = points[ipt * 4 + 2];
	        mesh->setPoint(ipt, &v);
	}

	// Normals
	for (ipt = 0; ipt < nnormals; ipt++)
	{
		v.m_vals[0] = normals[ipt * 3];
		v.m_vals[1] = normals[ipt * 3 + 1];
		v.m_vals[2] = normals[ipt * 3 + 2];
	        mesh->setNormal(ipt, &v);
	}

	// Texture coords
	for (ipt = 0; ipt < ntexcoords; ipt++)
	    mesh->setTextureCoord(ipt, texcoords[ipt * 2], texcoords[ipt * 2 + 1]);

	// Triangles
	for(ipoly = 0; ipoly < npolys; ipoly++ )
	{
                vector< uint > & idx = vertex_indices[ ipoly ];
		mesh->setTriangle(ipoly, idx[0], idx[1], idx[2]);
	}

        // Add to the scene
        m_pScene->addMesh( mesh );
	if(material)
	{
		GRayTraceColor* pEmissive = material->color(GRayTraceMaterial::Emissive, NULL);
		if (!pEmissive->isBlack() )
		{
			for(ipoly = 0; ipoly < npolys; ipoly++)
			{
				GRayTraceTriangle* pTriangle = new GRayTraceTriangle(mesh, ipoly);
				m_pScene->addObject(pTriangle); // This is a duplicate object of the one the mesh added
				GRayTraceAreaLight* pAreaLight = new GRayTraceAreaLight(pTriangle, pEmissive->r, pEmissive->g, pEmissive->b);
				m_pScene->addLight( pAreaLight );
			}
		}
	}

	// Clean up the allocated memory.
	if ( points != NULL )
		delete[] points;
	if ( normals != NULL )
		delete[] normals;
	if ( texcoords != NULL )
		delete[] texcoords;

	return true;
}

bool RibParser::handleAreaLightSource( const RibData& data )
{
	// The default emissive color is white for any area light source
	double emissive_color[ 4 ] = { 1, 1, 1, 1 };

	if ( data.size() < 3 )
	{
		fprintf(stderr, "Bad number of arguments to AreaLightSource.  Requires at least 1, has %d\n", data.size() - 1);
		//data.debugPrint( cerr, 3 );
		return false;
	}
	else
	{
		if ( data.hasOptionalArg( "lightcolor" ) )
			data.fillDoubleArrayFromOptionalArg( "lightcolor", emissive_color, 3 );
		// Set the emissive color in the appropriate state block(s)
		int i = state.size() - 1;
		do {
			for ( uint j = 0; j < 3; j++ )
				state[ i ].emissive_color[ j ] = emissive_color[ j ];
			i--;
		} while ( state[ i ].type == StateBlock::TRANSFORMBLOCK && i > 0 );

		//       In project 4 we do geometry light sources.
		//       The basic idea is just setting the emissive on the material.
		//       We do it with the AreaLightSource command, for more compatibility with how
		//       real-world RIBs do it.
		//
		//       You'll need to replace this with your own material class.

		// Make a copy of the current material, replace the emissive color, and set
		GRayTracePhysicalMaterial* new_material = new GRayTracePhysicalMaterial();
		new_material->setColor( GRayTraceMaterial::Emissive, emissive_color[0], emissive_color[1], emissive_color[2] );
		// Add the material to the scene, so it'll be deleted properly
		m_pScene->addMaterial( new_material );
		// Set the current material on the parser state
		setMaterial( new_material );

		return true;
	}
}

void RibParser::pushState( int block_type )
{
	state.push_back( StateBlock( block_type, state.back() ) );
	if ( block_type == StateBlock::WORLDBLOCK )
	{
		// Move the object transform to the camera transform, and clear the object transform.
		double transform[16];
		getObjectTransform( transform );
		setCameraTransform( transform );
		matrixMakeIdentity( transform );
		setObjectTransform( transform );
	}
}

void RibParser::popState( int block_type )
{
	if ( state.back().getType() != block_type )
		fprintf(stderr, "Block error: Mismatched block types: %d terminated by %d\n", state.back().getType(), block_type);

	// If we're about to pop the world block, create a camera.
	if ( state.back().getType() == StateBlock::WORLDBLOCK )
	{
		// These variables record the type of render requested, for easy construction of the camera.
		bool ray_tracing_camera = false, path_tracing_camera = false;

		// Will be made the default camera in the scene.
		GRayTraceCamera *camera = NULL;

		// Camera parameters: Samples per pixel, ray recursion depth, and tone mapping constant
		int ray_depth;
//		double tone_mapping_constant;

		double from[4] = { 0, 0, 0, 1 };
		double to[4] = { 0, 0, 1, 1 }; // This 'to' point isn't what you'd expect,
		                               // but it allows us to change the handedness easily.
		double up[4] = { 0, 1, 0, 1 };

		double transform[16];  // Camera to world transform.
		getCameraTransform( transform );

		double inverted[16];
		if ( !matrixInvert( inverted, transform ) )
		{
			fprintf(stderr, "Transform matrix wasn't invertible.  Expect weird behavior\n");
		}

		matrixMultiplyPoint( from, inverted, from );
		matrixMultiplyPoint( to, inverted, to );
		matrixMultiplyPoint( up, inverted, up );
		// Change up point to up vector.
		for ( uint i = 0; i < 4; i++ )
			up[i] = up[i] - from[i];

		if ( hasAttribute( "cs655Setting", "renderType" ) )
		{
			string render_type = getAttribute( "cs655Setting", "renderType" ).stringVal( 0 );
			if ( render_type == "rayTracing" )
				ray_tracing_camera = true;
			else if ( render_type == "pathTracing" )
				path_tracing_camera = true;
		}
		if ( !ray_tracing_camera && !path_tracing_camera )
		{
			printf("No renderType attribute specified: Assuming ray tracing\n");
			ray_tracing_camera = true;
		}

		// At this point it's either a ray tracing camera, or path tracing.
		// Get other attributes common to both cameras
		ray_depth = getAttribute( "cs655Setting", "rayDepth" ).intVal( 0 );

		camera = m_pScene->camera();
		if ( ray_tracing_camera )
		{
		}
		else if ( path_tracing_camera )
		{
			// Get the tone mapping constant from the attributes.
			G3DReal tone_mapping_constant = getAttribute( "cs655Setting", "toneMappingConstant" ).doubleVal( 0 );
			m_pScene->setToneMappingConstant( (G3DReal)tone_mapping_constant );
		}

//		camera->setRaysPerPixel( pixel_samples );
		camera->setMaxDepth( ray_depth );
		camera->setFocalDistance( focal_distance );
		camera->setLensDiameter( lens_diameter );
		camera->lookFromPoint()->set(from[0], from[1], from[2]);
		G3DVector vecUp(up[0], up[1], up[2]);
		G3DVector vecForward(to[0] - from[0], to[1] - from[1], to[2] - from[2]);
		camera->setDirection(&vecForward, &vecUp);
		camera->setViewAngle(fov);
		camera->setImageSize(image_width, image_height);
		//camera->setScene(scene);
		//scene->setDefaultCamera(camera);
	}

	// We'll crash if we remove the root block, so um.. don't.
	if ( state.size() == 1 )
		fprintf(stderr, "Block error: Too many closing blocks, removing the file block from the stack is invalid.\n");
	else
		state.pop_back();
}


void RibParser::setMaterial(GRayTraceMaterial* material)
{
	int i = state.size() - 1;
	state[ i ].material = material;
	// Transform blocks only catch transforms, so propagate the material until you hit a different block type
	while ( state[ i ].type == StateBlock::TRANSFORMBLOCK && i > 0 )
	{
		--i;
		state[ i ].material = material;
	}
}

GRayTraceMaterial* RibParser::getMaterial()
{
	if ( state.back().material == NULL )
	{
		// If there is no current material, make and return the default material.
#ifdef PARSER_DEBUG
		fprintf(stderr, "Material requested, but no material currently defined.  Returning default material.\n");
#endif
		GRayTracePhysicalMaterial *material = new GRayTracePhysicalMaterial();
		m_pScene->addMaterial(material);
		setMaterial(material);
		return material;
	}
	else
	{
		return state.back().material;
	}
}

void RibParser::setColor( const double* color )
{
	StateBlock& block = state.back();
	for ( uint i = 0; i < 3; i++ )
		block.color[ i ] = color[ i ];
}

void RibParser::getColor( double *color )
{
	StateBlock& block = state.back();
	for ( uint i = 0; i < 3; i++ )
		color[ i ] = block.color[ i ];
}

void RibParser::setCameraTransform( const double transform[16] )
{
	for ( uint i = 0; i < 16; i++ )
		camera_to_world[ i ] = transform[ i ];
}

void RibParser::getCameraTransform( double transform[16] )
{
	for ( uint i = 0; i < 16; i++ )
		transform[ i ] = camera_to_world[ i ];
}

void RibParser::setObjectTransform( const double transform[16] )
{
	StateBlock& block = state.back();
	for ( uint i = 0; i < 16; i++ )
		block.transform[ i ] = transform[ i ];
}

void RibParser::getObjectTransform( double transform[16] )
{
	StateBlock& block = state.back();
	for ( uint i = 0; i < 16; i++ )
		transform[ i ] = block.transform[ i ];
}

void RibParser::multiplyTransform( const double transform[16] )
{
	matrixMultiply( state.back().transform, state.back().transform, transform );
}

bool RibParser::hasAttribute( const string& attr_class, const string& attr_name )
{
	int i = state.size() - 1;
	bool success = false;
	do {
		state[ i ].getAttribute( attr_class, attr_name, success );
		if ( success )
			return true;
		--i;
	} while ( i >= 0 );

	return false;
}

RibData& RibParser::getAttribute( const string& attr_class, const string& attr_name )
{
	bool success = false;
	int i = state.size() - 1;
	do {
		RibData& data = state[ i ].getAttribute( attr_class, attr_name, success );
		if ( success )
			return data;
		--i;
	} while ( i >= 0 );

	success = false;
	return *((RibData*)NULL);
}

void RibParser::setAttribute( const string& attr_class, const string& attr_name, const RibData& data )
{
	int i = state.size() - 1;
	do {
		if ( state[ i ].type != StateBlock::TRANSFORMBLOCK )
			break; // This is the one we want, don't decrement any more.
		i--;
	}
	while ( i > 0 ); // If we ever fail this test, we should be at the root block.

	// Either at the root, or the topmost non-transform, catch the attribute.
	// Since we look up attributes from the top of the stack,
	// this will properly allow shadowing of attribute values
	state[ i ].setAttribute( attr_class, attr_name, data );
}

RibData& RibParser::getObjectDefinition( unsigned int index )
{
	if ( object_definitions.size() <= index )
		object_definitions.resize( index + 1 );
	return object_definitions[ index ];
}

////////////////////
// Utility Matrix Functions
////////////////////

void RibParser::matrixZero( double matrix[16] )
{
	for ( uint i = 0; i < 16; i++ )
		matrix[ i ] = 0;
}

void RibParser::matrixMakeIdentity( double matrix[16] )
{
	matrixZero( matrix );
	for ( uint i = 0; i < 4; i++ )
		matrix[ i * 4 + i ] = 1;
}

void RibParser::matrixMultiply( double dest[16], const double left[16], const double right[16] )
{
	uint i;
	double temp[16];
	matrixZero( temp );
	for ( i = 0; i < 4; i++ )
		for ( uint j = 0; j < 4; j++ )
			for ( uint k = 0; k < 4; k++ )
				temp[ i*4 + j ] += left[ k*4 + j ] * right[ i*4 + k ];
	// Copy to the dest
	for ( i = 0; i < 16; i++ )
		dest[ i ] = temp[ i ];
}

void RibParser::matrixMultiplyPoint( double dest[4], const double matrix[16], const double point[4] )
{
	double temp[4] = { 0, 0, 0, 0 };
	uint i;

	for ( i = 0; i < 4; i++ )
		for ( uint j = 0; j < 4; j++ )
			temp[i] += matrix[ i*4 + j ] * point[ j ];
	// Copy to the dest
	for ( i = 0; i < 4; i++ )
		dest[i] = temp[i];
}

void RibParser::matrixTranspose( double dest[16], const double matrix[16] )
{
	double temp[16];
	uint i;
	for ( i = 0; i < 4; i++ )
		for ( uint j = 0; j < 4; j++ )
			temp[ i*4 + j ] = matrix[ j*4 + i ];
	for ( i = 0; i < 16; i++ )
		dest[i] = temp[i];
}

void RibParser::matrixScaleAugRow( double matrix[32], int irow, double factor )
{
	int offset = irow * 8;
	for ( uint i = 0; i < 8; i++ )
		matrix[ offset + i ] *= factor;
}

void RibParser::matrixAddScaledAugRow( double matrix[32], int source_row, int dest_row, double factor )
{
	int source_offset = source_row * 8;
	int dest_offset = dest_row * 8;
	for ( uint i = 0; i < 8; i++ )
		matrix[ dest_offset + i ] += factor * matrix[ source_offset + i ];
}

bool RibParser::matrixInvert( double dest[16], const double matrix[16] )
{
	double val;
	double aug_matrix[32] = { 0, 0, 0, 0, 1, 0, 0, 0,
							  0, 0, 0, 0, 0, 1, 0, 0,
							  0, 0, 0, 0, 0, 0, 1, 0,
							  0, 0, 0, 0, 0, 0, 0, 1 };
	int i, j;

	// Copy matrix to left block of augmented matrix
	for ( i = 0; i < 4; i++ )
		for ( j = 0; j < 4; j++ )
			aug_matrix[ i*8 + j ] = matrix[ i*4 + j ];

	// Zero out below the diagonal
	for ( i=0; i<4; i++ )
	{
		val = aug_matrix[i*8+i];
		if (val==0.0) return 0; // too hard or non-invertible.
		matrixScaleAugRow( aug_matrix, i, 1.0/val );
		aug_matrix[ i*8+i ] = 1.0; // Make sure it's exactly 1, like it should be.
		for ( uint j=i+1; j<4; j++ )
		{
			val = aug_matrix[ j*8+i ];
			if ( val == 0.0 ) continue; // Already zero'd out
			matrixAddScaledAugRow( aug_matrix, i, j, -val );
			aug_matrix[ j*8+i ] = 0.0; // Make sure it's 0.
		}
	}

	// Zero out above the diagonal
	for ( i=3; i>=1; i-- ) {
		for ( j=i-1; j>=0; j-- ) {
			val = aug_matrix[ j*8 + i ];
			if ( val == 0.0 ) continue;
			matrixAddScaledAugRow( aug_matrix, i, j, -val );
			aug_matrix[ j*8 + i ] = 0.0;
		}
	}

	// Copy right block of the aug matrix to the dest
	for ( i = 0; i < 4; i++ )
		for ( j = 0; j < 4; j++ )
			dest[ i*4 + j ] = aug_matrix[ i*8 + j+4 ];

	return true;
}

////////////////////
// Rib Data
////////////////////

// By default, this will be an array.
RibData::RibData() :
type( NODE_ARRAY ), file_line( -1 )
{}

RibData::RibData(int type) :
type( type ), file_line( -1 )
{}

RibData::RibData( const RibData& other )
: type( other.type ), file_line( other.file_line ), dataPtr( other.dataPtr ), childVector( other.childVector ), valArray( other.valArray )
{
	// Note that we don't create the pointers vector, because most of the time it's not used.
	// If you're going to use it, make sure the lengths match first.
	// Or, you could always call recreatePointers();
}

RibData& RibData::operator=( const RibData& other )
{
	type = other.type;
	file_line = other.file_line;
	dataPtr = other.dataPtr;
	childVector = other.childVector;
	valArray = other.valArray;
	//recreatePointers();
	return *this;
}

RibData::RibData( const char* data, bool quoted )
: dataPtr( data )
{
	if ( quoted )
		type = QUOTED_STRING;
	else
		type = UNDETECTED;
}

void RibData::append( const RibData& new_data )
{
#ifdef PARSER_DEBUG
	if ( type != NODE_ARRAY )
		fprintf(stderr, "Trying to append datum to a non-nodearray RibData.\n");
#endif
	childVector.push_back( new_data );
}

void RibData::appendVal( const char* new_val )
{
#ifdef PARSER_DEBUG
	if ( type != LEAF_ARRAY ) {
		fprintf(stderr, "Error: Trying to append a value to something that's not an Array\n");
	}
#endif
	valArray.push_back( new_val );
}

vector< const char* >& RibData::accessVals()
{
	return valArray;
}

std::vector< RibData >::iterator RibData::getBackIt()
{
#ifdef PARSER_DEBUG
	if ( childVector.empty() )
		fprintf(stderr, "Trying to get the end of an empty RibData array\n");
	if ( type != NODE_ARRAY )
		fprintf(stderr, "Trying to get the back iterator from a non-nodearray RibData\n");
#endif
	std::vector< RibData >::iterator retme = childVector.end();
	--retme;
	return retme;
}

RibData& RibData::back()
{
#ifdef PARSER_DEBUG
	if ( type != NODE_ARRAY )
		fprintf(stderr, "Trying to get the back of non-nodearray ribdata\n");
#endif
	return childVector.back();
}

void RibData::clear()
{
	childVector.clear();
	valArray.clear();
}
/*
void RibData::recreatePointers()
{
	if ( childPtrs.size() != childList.size() )
	{
		childPtrs.clear();
		childPtrs.resize( childList.size(), (RibData*)0xbeefdeed );
		int i = 0;
		for ( list< RibData >::iterator lit = childList.begin(); lit != childList.end(); ++lit, ++i )
			childPtrs[ i ] = &*lit;
	}
}
*/
/*
// This debugPrint isn't the cleanest..
void RibData::debugPrint( ostream& out, int offset ) const
{
#ifdef PARSER_DEBUG
	string lines( "=== " );
	string pad( offset, ' ' );
	out << pad;
	if ( type == NODE_ARRAY || type == LEAF_ARRAY )
		out << lines;
	out << "Type: ";
	switch( type )
	{
	case QUOTED_STRING:
		out << "Quoted String"; break;
	case NODE_ARRAY:
		out << "Node"; break;
	case LEAF_ARRAY:
		out << "Leaf"; break;
	case UNDETECTED:
		out << "Data"; break;
	default:
		out << "Invalid"; break;
	}
	out << " Value: ";

	if ( type == NODE_ARRAY )
	{
		out << endl;
		for ( vector< RibData >::const_iterator cit = childVector.begin(); cit != childVector.end(); ++cit )
		{
			cit->debugPrint( out, offset + 3 );
		}
	}
	else if ( type == LEAF_ARRAY )
	{
		for ( vector< const char* >::const_iterator vit = valArray.begin(); vit != valArray.end(); ++vit )
			out << *vit << " ";
		out << endl;
	}
	else
	{
		out << dataPtr << endl;
	}
#endif
}
*/
int RibData::size() const
{
	switch ( type )
	{
	case NODE_ARRAY:
		return childVector.size();
	case LEAF_ARRAY:
		return valArray.size();
	default:
		fprintf(stderr, "Trying to get the size of a non-array RibData\n");
		return 0;
	}
}

RibData& RibData::operator[]( int index )
{
#ifdef PARSER_DEBUG
	if ( type != NODE_ARRAY )
		fprintf(stderr, "Trying to get an array element on a RibData that's not an Array\n");
#endif
	return childVector[ index ];
}

const RibData& RibData::operator[]( int index ) const
{
#ifdef PARSER_DEBUG
	if ( type != NODE_ARRAY )
		fprintf(stderr, "Trying to get an array element on a RibData that's not an Array\n");
#endif
	return childVector[ index ];
}

double RibData::doubleVal( int index ) const
{
#ifdef PARSER_DEBUG
	if ( type != LEAF_ARRAY )
		fprintf(stderr, "Trying to get a double val from a non-leafarray\n");
#endif
	return atof( valArray[ index ] );
}

int RibData::intVal( int index ) const
{
#ifdef PARSER_DEBUG
	if ( type != LEAF_ARRAY )
		fprintf(stderr, "Trying to get an int val from a non-leafarray\n");
#endif
	return atoi( valArray[ index ] );
}

const char* RibData::stringVal( int index ) const
{
#ifdef PARSER_DEBUG
	if ( type != LEAF_ARRAY )
	{
		fprintf(stderr, "Trying to get a string val from a non-leafarray\n");
		return "";
	}
#endif
	if ( valArray[ index ][0] == '\"' )
		return &valArray[ index ][ 1 ];
	return valArray[ index ];
}

int RibData::getType() const
{
	return type;
}

void RibData::reserveCapacity( int new_capacity )
{
	if ( type == NODE_ARRAY )
		childVector.reserve( new_capacity );
	else if ( type == LEAF_ARRAY )
		valArray.reserve( new_capacity );
#ifdef PARSER_DEBUG
	else
		fprintf(stderr, "Trying to reserve capacity on non-array RibData\n");
#endif
}

string RibData::asString() const
{
#ifdef PARSER_DEBUG
	if ( type == NODE_ARRAY || type == LEAF_ARRAY )
		fprintf(stderr, "Trying to get a data value from an array.\n");
#endif
	return string( dataPtr );
};

double RibData::asDouble() const
{
#ifdef PARSER_DEBUG
	if ( type == NODE_ARRAY || type == LEAF_ARRAY )
		fprintf(stderr, "Trying to get a double from an array\n");
#endif
	return  atof( dataPtr );
}

int RibData::asInt() const
{
#ifdef PARSER_DEBUG
	if ( type == NODE_ARRAY || type == LEAF_ARRAY )
		fprintf(stderr, "Trying to get an int from an array\n");
#endif
	return atoi( dataPtr );
}

bool RibData::hasOptionalArg( const string& arg_name ) const
{
	if ( type != NODE_ARRAY )
		return false;
	// Two iterators: next is always one greater than next.
	std::vector< RibData >::const_iterator curr = childVector.begin();
	curr++;
	std::vector< RibData >::const_iterator next = curr;
	next++;

	if ( curr == childVector.end() )
		return false;

	while ( next != childVector.end() )
	{
		if ( curr->getType() == QUOTED_STRING
			&& next->getType() == LEAF_ARRAY
			&& curr->dataPtr == arg_name )
		{
			return true;
		}
		next++;
		curr++;
	}
	return false;
}

const RibData& RibData::findOptionalArg( const string& arg_name ) const
{
	if ( type != NODE_ARRAY )
	{
		fprintf(stderr, "findOptionalArg called on a non-nodearray RibData.\n");
		//debugPrint( cerr, 3 );
		return *this;
	}
	// Two iterators: next is always one greater than next.
	std::vector< RibData >::const_iterator curr = childVector.begin();
	curr++;
	std::vector< RibData >::const_iterator next = curr;
	next++;

	if ( curr != childVector.end() )
	{
		while ( next != childVector.end() )
		{
			if ( curr->getType() == QUOTED_STRING
				&& next->getType() == LEAF_ARRAY
				&& curr->dataPtr == arg_name )
			{
				return *next;
			}
			curr++;
			next++;
		}
	}
	fprintf(stderr, "findOptionalArg couldn't find requested arg %s - are you sure it's there?\n", arg_name.c_str());
	//debugPrint( cerr, 3 );
	return *this;
}

bool RibData::fillDoubleArrayFromOptionalArg( const string& arg_name, double* fill_me, int fill_size ) const
{
	if ( hasOptionalArg( arg_name ) )
	{
		const RibData& sub_array = findOptionalArg( arg_name );
		if ( sub_array.size() != fill_size )
		{
			fprintf(stderr, "I have vs6\n");
		//	fprintf(stderr, "Bad size of arg: %s expected: %s, got %d, %s, fill_size.c_str(), sub_array.size(), arg_name.c_str());
			//debugPrint( cerr, 3 );
			return false;
		}
		for ( int i = 0; i < sub_array.size(); i++ )
		{
			fill_me[ i ] = sub_array.doubleVal( i );
		}
		return true;
	}
	// Didn't have that optional arg.
	return false;
}

////////////////////
// State Block
////////////////////

StateBlock::StateBlock() :
type( ROOTBLOCK )
{
	RibParser::matrixMakeIdentity( transform );
	color[ 0 ] = color[ 1 ] = color[ 2 ] = 0.5; // Dull gray
	emissive_color[ 0 ] = emissive_color[ 1 ] = emissive_color[ 2 ] = 0.0; // Not a light
	material = NULL;
	// Set up the default attribute values.
	RibData default_renderType( RibData::LEAF_ARRAY );
	RibData default_rayDepth( RibData::LEAF_ARRAY );
	RibData default_toneMappingConstant( RibData::LEAF_ARRAY );

	default_renderType.appendVal( "rayTracing" );
	default_rayDepth.appendVal( "4" );
	default_toneMappingConstant.appendVal( "1" );

	setAttribute( "cs655Setting", "renderType", default_renderType );
	setAttribute( "cs655Setting", "rayDepth", default_rayDepth );
	setAttribute( "cs655Setting", "toneMappingConstant", default_toneMappingConstant );
}

StateBlock::StateBlock( int type, const StateBlock& parent )
: type( type )
{
	uint i;
	for ( i = 0; i < 16; i++ )
		transform[ i ] = parent.transform[ i ];
	for ( i = 0; i < 3; i++ )
		color[ i ] = parent.color[ i ];
	for ( i = 0; i < 3; i++ )
		emissive_color[ i ] = parent.emissive_color[ i ];
	material = parent.material;
}

RibData& StateBlock::getAttribute( const string& attr_class, const string& attr_name, bool& success )
{
	std::pair< string, string > key( attr_class, attr_name );
	std::map< std::pair< string, string>, RibData >::iterator it;
	it = attributes.find( key );
	if ( it == attributes.end() )
	{
		success = false;
		return (*(RibData*)(NULL));
	}
	success = true;
	return it->second;
}

void StateBlock::setAttribute( const string& attr_class, const string& attr_name, const RibData& data )
{
	std::pair< string, string > key( attr_class, attr_name );
	attributes[ key ] = data;
}


int StateBlock::getType() const
{
	return type;
}
