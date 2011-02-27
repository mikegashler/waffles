#ifndef __GRIBPARSER_H__
#define __GRIBPARSER_H__

#include <string>
#include <vector>
#include <map>
#include <list>
#include <cmath>
#include <limits>
#include <stdio.h>
#include <GClasses/GRand.h>
#include <GClasses/GRayTrace.h>

class RibParser;


class RibData
{
private:
	int type;
	int file_line;
	const char* dataPtr;
	std::vector< RibData > childVector;
	std::vector< const char* > valArray;

public:
	// Different types of data.
	enum
	{
		UNDETECTED = 0,
		QUOTED_STRING,
		NODE_ARRAY,
		LEAF_ARRAY
	};

	RibData(); // By default, this will be an node array.
	RibData( int type );
	RibData( const char* data, bool quoted = false );
	RibData( const RibData& other ); // Copy constructor

	// Assignment operator.
	RibData& operator=( const RibData& other );

	// Array access operators.  Only works for node arrays
	RibData& operator[]( int index );
	const RibData& operator[]( int index ) const;

	// Data creation and manipulation
	void append( const RibData& new_data );
	void appendVal( const char* new_val );
	std::vector< const char* >& accessVals(); // Allows direct access to the vals array, for speed.
	std::vector< RibData >::iterator getBackIt();
	RibData& back();
	void clear();
	void reserveCapacity( int new_capacity );

	// Convenience functions
	bool hasOptionalArg( const std::string& arg_name ) const;
	const RibData& findOptionalArg( const std::string& arg_name ) const;
	bool fillDoubleArrayFromOptionalArg( const std::string& arg_name, double* fill_me, int fill_size ) const;

	// Data retrieval from a node
	std::string asString() const;
	int asInt() const;
	double asDouble() const;

	// Data retrieval from a leaf
	double doubleVal( int index ) const;
	int intVal( int index ) const;
	const char* stringVal( int index ) const;

	// Returns the number of children nodes or values
	int size() const;
	// Returns the type
	int getType() const;
};



class StateBlock
{
friend class RibParser;
private:
	std::map< std::pair< std::string, std::string >, RibData > attributes;
	int type;
	double transform[ 16 ];
	double color[ 3 ];
	double emissive_color[ 3 ];
	GClasses::GRayTraceMaterial* material;

public:
	// These are types of blocks.  They're mostly the same.
	enum { ROOTBLOCK, WORLDBLOCK, FRAMEBLOCK, TRANSFORMBLOCK, ATTRIBUTEBLOCK };
	StateBlock();
	StateBlock( int type, const StateBlock& parent );

	// These getAttribute methods only work on the local values.
	// The methods on RibParser work on the whole stack
	RibData& getAttribute( const std::string& attr_class, const std::string& attr_name, bool& success );
	void setAttribute( const std::string& attr_class, const std::string& attr_name, const RibData& data );

	int getType() const;
};


// This parsers Pixar/Renderman RIB files for 3D scenes. It doesn't support all
// the features of the RIB format, so if you use advanced features like NURBS
// surfaces, they won't appear in the model.
class RibParser
{
private:
	std::vector< StateBlock > state; // The state is stack-based, and a vector works well.
	GClasses::GRayTraceScene* m_pScene;
	int image_width, image_height;
	int pixel_samples;
	double fov;
	double camera_to_world[ 16 ];
	double focal_distance; // Distance from the camera to things that are in focus
	double lens_diameter;

	std::vector< char* > lineBuffs; // This is where files are read, anything here will be delete[]'d in the destructor
	std::string baseDir; // This is the directory where the file resides, relative to the executable.  For loading archives/textures
	// These properties support flattening object instancing
	int curr_object;
	bool in_object_definition;
	std::vector< RibData > object_definitions;
	GClasses::GRand* m_pRand;

public:
	RibParser(GClasses::GRand* pRand);
	~RibParser();

	static GClasses::GRayTraceScene* LoadScene(const char* szFilename, GClasses::GRand* pRand);

	bool createScene(FILE* F);
	bool createScene(std::string filename);

	GClasses::GRayTraceScene* releaseScene()
	{
		GClasses::GRayTraceScene* pScene = m_pScene;
		m_pScene = NULL;
		return pScene;
	}

	// Utility right-handed matrix functions, included here to keep the file independent
	static void matrixZero( double matrix[16] );
	static void matrixMakeIdentity( double matrix[16] );
	static void matrixMultiply( double dest[16], const double left[16], const double right[16] );
	static void matrixMultiplyPoint( double dest[4], const double matrix[16], const double point[4] );
	static void matrixTranspose( double dest[16], const double matrix[16] );
	static void matrixScaleAugRow( double matrix[32], int irow, double factor );
	static void matrixAddScaledAugRow( double matrix[32], int source_row, int dest_row, double factor );
	static bool matrixInvert( double dest[16], const double matrix[16] );

private:
	// File input and parse tree construction
	int readLines( FILE* F, std::vector< int >& lines, std::vector< int >& line_nums );
	void buildData( char* big_str, int begin, int line_num, RibData& line );
	void extractArray( char* str, int length, int& i, RibData& parent );

	// A switch statement to execute each line's command
	bool handleLine( const RibData& data );

	// Handlers for each of the commands (well, the subset you support).
	// See https://renderman.pixar.com/products/rispec/ for specs.
	// The quick reference is a good index
	bool handleTransform( const RibData& data );
	bool handleConcatTransform( const RibData& data );
	bool handleTranslate( const RibData& data );
	bool handleReadArchive( const RibData& data );
	bool handleObjectBegin( const RibData& data );
	bool handleObjectEnd( const RibData& data );
	bool handleObjectInstance( const RibData& data );
	bool handleFormat( const RibData& data );
	bool handlePixelSamples( const RibData& data );
	bool handleImager( const RibData& data );
	bool handleProjection( const RibData& data );
	bool handleDepthOfField( const RibData& data );
	bool handleAttribute( const RibData& data );
	bool handleColor( const RibData& data );
	bool handleSurface( const RibData& data );
	bool handleLight( const RibData& data );
	bool handleSphere( const RibData& data );
	bool handleNuPatch( const RibData& data ); // Hack - outputs spheres
	bool handlePointsGeneralPolygons( const RibData& data );
	bool handleAreaLightSource( const RibData& data );

	// State management functions
	void pushState( int block_type );
	void popState( int block_type );

	// These functions allow you to query the current state.
	// They abstract some of the details of the stack away
	void setMaterial(GClasses::GRayTraceMaterial* material);
	GClasses::GRayTraceMaterial* getMaterial();
	void setColor( const double* color );
	void getColor( double *color );

	void setCameraTransform( const double transform[16] );
	void getCameraTransform( double transform[16] );
	void setObjectTransform( const double transform[16] );
	void getObjectTransform( double transform[16] );
	void multiplyTransform( const double transform[16] );

	bool hasAttribute( const std::string& attr_class, const std::string& attr_name );
	RibData& getAttribute( const std::string& attr_class, const std::string& attr_name );
	void setAttribute( const std::string& attr_class, const std::string& attr_name, const RibData& data );

	// These support object instancing, which is flattened.
	// Maya's built-in ribexport produces files that use this
	RibData& getObjectDefinition( unsigned int index );
};

#endif // __GRIBPARSER_H__
