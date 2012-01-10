################
# Paths and Flags
################
SHELL = /bin/bash
TARGET_PATH = ../bin
TARGET_NAME_OPT = ubp
TARGET_NAME_DBG = $(TARGET_NAME_OPT)dbg
OBJ_PATH = ../obj
INSTALL_LOCATION_INCLUDE ?= /usr/local/include
UNAME = $(shell uname -s)
ifeq ($(UNAME),Darwin)
#CFLAGS = -I/usr/local/include/SDL -D_THREAD_SAFE -DDARWIN -I/sw/include -I../../../src -DBYTE_ORDER_BIG_ENDIAN -no-cpp-precomp
CFLAGS = -I/usr/local/include/SDL -D_THREAD_SAFE -DDARWIN -I/sw/include -I../../../src -no-cpp-precomp
DBG_CFLAGS = $(CFLAGS) -g -D_DEBUG
OPT_CFLAGS = $(CFLAGS) -O3
DBG_LFLAGS = $(DBG_LIBRARIES) /sw/lib/libSDLmain.a /sw/lib/libSDL.dylib /sw/lib/libpng.dylib -lz -framework AppKit ../../../lib/GClassesDbg.a
OPT_LFLAGS = $(OPT_LIBRARIES) /sw/lib/libSDLmain.a /sw/lib/libSDL.dylib /sw/lib/libpng.dylib -lz -framework AppKit ../../../lib/GClasses.a
else
CFLAGS = -Wall $(shell sdl-config --cflags) -I$(INSTALL_LOCATION_INCLUDE)
DBG_CFLAGS = $(CFLAGS) -g -D_DEBUG
OPT_CFLAGS = $(CFLAGS) -O3
DBG_LFLAGS = -lGClassesDbg $(shell sdl-config --libs) -lpng -lpthread
OPT_LFLAGS = -lGClasses $(shell sdl-config --libs) -lpng -lpthread
endif

################
# Source
################

CPP_FILES =\
	main.cpp\
	Gui.cpp\

################
# Lists
################

TEMP_LIST_OPT = $(CPP_FILES:%=$(OBJ_PATH)/opt/%)
TEMP_LIST_DBG = $(CPP_FILES:%=$(OBJ_PATH)/dbg/%)
OBJECTS_OPT = $(TEMP_LIST_OPT:%.cpp=%.o)
OBJECTS_DBG = $(TEMP_LIST_DBG:%.cpp=%.o)
DEPS_OPT = $(TEMP_LIST_OPT:%.cpp=%.d)
DEPS_DBG = $(TEMP_LIST_DBG:%.cpp=%.d)

################
# Rules
################

.DELETE_ON_ERROR:

dbg : $(TARGET_PATH)/$(TARGET_NAME_DBG)

opt : $(TARGET_PATH)/$(TARGET_NAME_OPT)

usage:
	@echo ""
	@echo "Usage:"
	@echo "  make usage   (to see this info)"
	@echo "  make clean   (to delete all the .o files)"
	@echo "  make dbg     (to build a debug version)"
	@echo "  make opt     (to build an optimized version)"
	@echo ""

# This rule makes the optimized binary by using g++ with the optimized ".o" files
$(TARGET_PATH)/$(TARGET_NAME_OPT) : partialcleanopt $(OBJECTS_OPT)
	g++ -O3 -o $(TARGET_PATH)/$(TARGET_NAME_OPT) $(OBJECTS_OPT) $(OPT_LFLAGS)

# This rule makes the debug binary by using g++ with the debug ".o" files
$(TARGET_PATH)/$(TARGET_NAME_DBG) : partialcleandbg $(OBJECTS_DBG)
	g++ -g -o $(TARGET_PATH)/$(TARGET_NAME_DBG) $(OBJECTS_DBG) $(DBG_LFLAGS)

# This includes all of the ".d" files. Each ".d" file contains a
# generated rule that tells it how to make .o files. (The reason these are generated is so that
# dependencies for these rules can be generated.)
-include $(DEPS_OPT)

-include $(DEPS_DBG)

# This rule makes the optimized ".d" files by using "g++ -MM" with the corresponding ".cpp" file
# The ".d" file will contain a rule that says how to make an optimized ".o" file.
# "$<" refers to the ".cpp" file, and "$@" refers to the ".d" file
$(DEPS_OPT) : $(OBJ_PATH)/opt/%.d : %.cpp
	@echo -e "Computing opt dependencies for $<"
	@-rm -f $$(dirname $@)/$$(basename $@ .d).o
	@if [ ! -d "$$(dirname $@)" ]; then umask 0;mkdir -p "$$(dirname $@)"; fi
	@umask 0;echo -en "$$(dirname $@)/" > $@
	@g++ $(OPT_CFLAGS) -MM $< >> $@
	@echo -e "	g++ $(OPT_CFLAGS) -c $< -o $$(dirname $@)/$$(basename $@ .d).o" >> $@

# This rule makes the debug ".d" files by using "g++ -MM" with the corresponding ".cpp" file
# The ".d" file will contain a rule that says how to make a debug ".o" file.
# "$<" refers to the ".cpp" file, and "$@" refers to the ".d" file
$(DEPS_DBG) : $(OBJ_PATH)/dbg/%.d : %.cpp
	@echo -e "Computing dbg dependencies for $<"
	@-rm -f $$(dirname $@)/$$(basename $@ .d).o
	@if [ ! -d "$$(dirname $@)" ]; then umask 0;mkdir -p "$$(dirname $@)"; fi
	@umask 0;echo -en "$$(dirname $@)/" > $@
	@g++ $(DBG_CFLAGS) -MM $< >> $@
	@echo -e "	g++ $(DBG_CFLAGS) -c $< -o $$(dirname $@)/$$(basename $@ .d).o" >> $@

partialcleandbg :
	rm -f $(TARGET_PATH)/$(TARGET_NAME_DBG)

partialcleanopt :
	rm -f $(TARGET_PATH)/$(TARGET_NAME_OPT)

clean : partialcleandbg partialcleanopt
	rm -f $(OBJECTS_OPT)
	rm -f $(OBJECTS_DBG)
	rm -f $(DEPS_OPT)
	rm -f $(DEPS_DBG)
