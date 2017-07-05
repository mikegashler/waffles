#-------------------------------------------------------------------------------
#CMakeLists.txt for the GClasses library
#-------------------------------------------------------------------------------
PROJECT( GClasses )
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

IF(WIN32)
  IF (NOT ONCE_SET_CMAKE_INSTALL_PREFIX)
    SET(ONCE_SET_CMAKE_INSTALL_PREFIX true CACHE BOOL
        "Have we set the install prefix yet?" FORCE)
    SET(CMAKE_INSTALL_PREFIX /usr/local CACHE PATH
        "Install path prefix, prepended onto install directories" FORCE)
  ENDIF()

  #Remove console from the line if we do not want to have the console window)
  #SET(LINK_FLAGS ${LINK_FLAGS} "-mwindows")
  ADD_DEFINITIONS(-D_CRT_SECURE_NO_WARNINGS -DWINDOWS)
  ADD_DEFINITIONS( "/W3 /wd4005 /wd4996 /nologo /wd4291 /wd4267 /wd4244 /wd4305 /EHsc" )
ENDIF()

ADD_DEFINITIONS(-Wall -Werror -Wshadow -pedantic -std=c++11)

#-------------------------------------------------------------------------------
#Build the Waffles Library here
#-------------------------------------------------------------------------------
FILE(GLOB WAFFLES_SRCS *.cpp)
FILE(GLOB WAFFLES_HDRS *.h)

#Omit these files from the list
#LIST(REMOVE_ITEM WAFFLES_SRCS findAugMatrix.cpp)

#Add the include directores
INCLUDE_DIRECTORIES(.)

#And build a static library
SET (LIBRARY_OUTPUT_PATH ../../lib/ CACHE PATH "Output directory libraries.")
SET (EXECUTABLE_OUTPUT_PATH ../../bin/ CACHE PATH "Output directory for executables.")

IF(BUILD_SHARED)
	SET(LIB_TYPE "SHARED")
ELSE()
	SET(LIB_TYPE "STATIC")
ENDIF()
ADD_LIBRARY(GClasses ${LIB_TYPE} ${WAFFLES_SRCS} ${WAFFLES_HDRS})
#SET_TARGET_PROPERTIES( GClasses PROPERTIES COMPILE_FLAGS "-DWAFFLES_EXPORTS") 

#-------------------------------------------------------------------------------
INSTALL(
  TARGETS GClasses
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)



#-----------------------------------------------------------------------------
# Add compiler flags
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${Waffles_REQUIRED_C_FLAGS}")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Waffles_REQUIRED_CXX_FLAGS}")
SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${Waffles_REQUIRED_LINK_FLAGS}")
SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${Waffles_REQUIRED_LINK_FLAGS}")
SET(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} ${Waffles_REQUIRED_LINK_FLAGS}")
