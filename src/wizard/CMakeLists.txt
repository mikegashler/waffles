#-------------------------------------------------------------------------------
#CMakeLists.txt
#-------------------------------------------------------------------------------
PROJECT( wizard )
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
FILE(GLOB SOURCE_FILES *.cpp)
FILE(GLOB HEADER_FILES *.h)

#Add the include directores
INCLUDE_DIRECTORIES(../GClasses)
INCLUDE_DIRECTORIES(../wizard)

#And build a static library
SET (LIBRARY_OUTPUT_PATH ../../lib/ CACHE PATH "Output directory libraries.")
SET (EXECUTABLE_OUTPUT_PATH ../../bin/ CACHE PATH "Output directory for executables.")
ADD_EXECUTABLE(waffles_wizard ../GClasses/usage.cpp ${SOURCE_FILES} ../GClasses/usage.h ${HEADER_FILES})
IF(WIN32)
  TARGET_LINK_LIBRARIES(waffles_wizard GClasses Ws2_32lib)
ELSE()
  TARGET_LINK_LIBRARIES(waffles_wizard GClasses pthread)
ENDIF()
#-------------------------------------------------------------------------------
INSTALL(
  TARGETS
    waffles_wizard
  ARCHIVE DESTINATION 
    lib
  RUNTIME DESTINATION
    bin
)



#-----------------------------------------------------------------------------
# Add compiler flags
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${Waffles_REQUIRED_C_FLAGS}")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Waffles_REQUIRED_CXX_FLAGS}")
SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${Waffles_REQUIRED_LINK_FLAGS}")
SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${Waffles_REQUIRED_LINK_FLAGS}")
SET(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} ${Waffles_REQUIRED_LINK_FLAGS}")
