################
# Paths and Flags
################
SHELL = /bin/bash
TARGET_PATH = ../../lib
TARGET_NAME_OPT = GSup.a
TARGET_NAME_DBG = GSupDbg.a
OBJ_PATH = ../../obj/GClasses
INSTALL_LOCATION_LIB ?= /usr/local/lib
INSTALL_LOCATION_INCLUDE ?= /usr/local/include
UNAME = $(shell uname -s)

# If colorgcc is installed, use it, otherwise use g++
ifeq ($(wildcard /usr/bin/colorgcc),)
	COMPILER=g++
else
	COMPILER=colorgcc
endif

# Set platform-specific compiler flags
ifeq ($(UNAME),Darwin)
	CFLAGS = -D_THREAD_SAFE -DDARWIN -I/sw/include -no-cpp-precomp
else
	CFLAGS = -Wall
endif

DBG_CFLAGS = $(CFLAGS) -g -D_DEBUG
OPT_CFLAGS = $(CFLAGS) -O3

################
# Source
################

CPP_FILES = \
	GBigInt.cpp\
	GChess.cpp\
	GDate.cpp\
	GDiff.cpp\
	GKeyPair.cpp\
	GSDL.cpp\
	GVideo.cpp\
	GXML.cpp\

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

dbg : titledbg $(TARGET_PATH)/$(TARGET_NAME_DBG)

opt : titleopt $(TARGET_PATH)/$(TARGET_NAME_OPT)

ifeq ($(UNAME),Darwin)
copyheaderfiles :
	for f in *.h;do cp $${f} ../GClasses/;done
else
copyheaderfiles :
endif

titleopt : copyheaderfiles
	@for ((i=0;i<10;i+=1));do echo "";done
	@echo "Building $(TARGET_NAME_OPT)..."
	@echo ""

titledbg : copyheaderfiles
	@for ((i=0;i<10;i+=1));do echo "";done
	@echo "Building $(TARGET_NAME_DBG)..."
	@echo ""

usage:
	#
	# Usage:
	#  make usage   (to see this info)
	#  make clean   (to delete all the .o files)
	#  make dbg     (to build a debug version)
	#  make opt     (to build an optimized version)
	#

# This rule makes the optimized ".a" file by using the "ar" tool with all of the optimized ".o" files
$(TARGET_PATH)/$(TARGET_NAME_OPT) : partialcleanopt $(OBJECTS_OPT)
	ar -cr $(TARGET_PATH)/$(TARGET_NAME_OPT) $(OBJECTS_OPT)

# This rule makes the debug ".a" file by using the "ar" tool with all of the debug ".o" files
$(TARGET_PATH)/$(TARGET_NAME_DBG) : partialcleandbg $(OBJECTS_DBG)
	ar -cr $(TARGET_PATH)/$(TARGET_NAME_DBG) $(OBJECTS_DBG)

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
	@$(COMPILER) $(OPT_CFLAGS) -MM $< >> $@
	@echo -e "	umask 0;$(COMPILER) $(OPT_CFLAGS) -c $< -o $$(dirname $@)/$$(basename $@ .d).o" >> $@

# This rule makes the debug ".d" files by using "g++ -MM" with the corresponding ".cpp" file
# The ".d" file will contain a rule that says how to make a debug ".o" file.
# "$<" refers to the ".cpp" file, and "$@" refers to the ".d" file
$(DEPS_DBG) : $(OBJ_PATH)/dbg/%.d : %.cpp
	@echo -e "Computing dbg dependencies for $<"
	@-rm -f $$(dirname $@)/$$(basename $@ .d).o
	@if [ ! -d "$$(dirname $@)" ]; then umask 0;mkdir -p "$$(dirname $@)"; fi
	@umask 0;echo -en "$$(dirname $@)/" > $@
	@$(COMPILER) $(DBG_CFLAGS) -MM $< >> $@
	@echo -e "	umask 0;$(COMPILER) $(DBG_CFLAGS) -c $< -o $$(dirname $@)/$$(basename $@ .d).o" >> $@

partialcleandbg :
	rm -f $(TARGET_PATH)/$(TARGET_NAME_DBG)

partialcleanopt :
	rm -f $(TARGET_PATH)/$(TARGET_NAME_OPT)

clean : partialcleandbg partialcleanopt
	rm -f $(OBJECTS_OPT)
	rm -f $(OBJECTS_DBG)
	rm -f $(DEPS_OPT)
	rm -f $(DEPS_DBG)

install: $(TARGET_PATH)/$(TARGET_NAME_OPT) $(TARGET_PATH)/$(TARGET_NAME_DBG)
	mkdir -m 0755 -p $(INSTALL_LOCATION_LIB)
	mkdir -m 0755 -p $(INSTALL_LOCATION_INCLUDE)/GClasses
	install $(TARGET_PATH)/GSup.a $(INSTALL_LOCATION_LIB)/libGSup.a
	install $(TARGET_PATH)/GSupDbg.a $(INSTALL_LOCATION_LIB)/libGSupDbg.a
	install *.h $(INSTALL_LOCATION_INCLUDE)/GClasses

uninstall:
	rm -f $(INSTALL_LOCATION_LIB)/libGSup.a
	rm -f $(INSTALL_LOCATION_LIB)/libGSupDbg.a

.PHONY: clean partialcleandbg partialcleanopt install uninstall dbg opt copyheaderfiles
