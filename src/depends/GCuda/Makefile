################
# Paths and Flags
################
SHELL = /bin/bash
TARGET_PATH = ../../../lib
TARGET_NAME_OPT = libGCuda.a
TARGET_NAME_DBG = libGCudaDbg.a
OBJ_PATH = ../../../obj/GCuda
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
	DARWIN_BASE ?= /usr/X11
	CFLAGS = -D_THREAD_SAFE -DDARWIN -I$(DARWIN_BASE)/include -no-cpp-precomp
else
	CFLAGS = -Wall
endif

DBG_CFLAGS = $(CFLAGS) -g -D_DEBUG
OPT_CFLAGS = $(CFLAGS) -O3

################
# Source
################

CU_FILES = \
	GCudaLayers.cu\
	GCudaMatrix.cu\
	GCudaMatrixHelper.cu\

################
# Lists
################

TEMP_LIST_OPT = $(CU_FILES:%=$(OBJ_PATH)/opt/%)
TEMP_LIST_DBG = $(CU_FILES:%=$(OBJ_PATH)/dbg/%)
OBJECTS_OPT = $(TEMP_LIST_OPT:%.cu=%.o)
OBJECTS_DBG = $(TEMP_LIST_DBG:%.cu=%.o)

################
# Rules
################

.DELETE_ON_ERROR:

dbg : $(TARGET_PATH)/$(TARGET_NAME_DBG)

opt : $(TARGET_PATH)/$(TARGET_NAME_OPT)

usage:
	#
	# Usage:
	#  make usage   (to see this info)
	#  make clean   (to delete all the .o files)
	#  make dbg     (to build a debug version)
	#  make opt     (to build an optimized version)
	#  make fam     (build the fam executable for help in generating
	#                LAPVJRCT test cases)
	#

# This rule makes the optimized ".a" file by using the "ar" tool with all of the optimized ".o" files
$(TARGET_PATH)/$(TARGET_NAME_OPT) : partialcleanopt $(OBJECTS_OPT) $(OBJ_PATH)/opt/dlink.o
	@if [ ! -d "$(TARGET_PATH)" ]; then mkdir -p "$(TARGET_PATH)"; fi
	@ar -crs $(TARGET_PATH)/$(TARGET_NAME_OPT) $(OBJECTS_OPT) $(OBJ_PATH)/opt/dlink.o

# This rule makes the debug ".a" file by using the "ar" tool with all of the debug ".o" files
$(TARGET_PATH)/$(TARGET_NAME_DBG) : partialcleandbg $(OBJECTS_DBG) $(OBJ_PATH)/dbg/dlink.o
	@if [ ! -d "$(TARGET_PATH)" ]; then mkdir -p "$(TARGET_PATH)"; fi
	@ar -crs $(TARGET_PATH)/$(TARGET_NAME_DBG) $(OBJECTS_DBG) $(OBJ_PATH)/dbg/dlink.o

$(OBJ_PATH)/opt/dlink.o :
	@if [ ! -d "$(OBJ_PATH)/opt" ]; then mkdir -p "$(OBJ_PATH)/opt"; fi
	nvcc -arch=sm_20 -O3 -dlink $(OBJECTS_OPT) -o $(OBJ_PATH)/opt/dlink.o

$(OBJ_PATH)/dbg/dlink.o :
	@if [ ! -d "$(OBJ_PATH)/dbg" ]; then mkdir -p "$(OBJ_PATH)/dbg"; fi
	nvcc -arch=sm_20 -g -dlink $(OBJECTS_DBG) -o $(OBJ_PATH)/dbg/dlink.o

$(OBJECTS_OPT) : $(CU_FILES)
	@if [ ! -d "$(OBJ_PATH)/opt" ]; then mkdir -p "$(OBJ_PATH)/opt"; fi
	nvcc -O3 -c $(subst .o,.cu,$(subst $(OBJ_PATH)/opt/,,$@)) -o $@ -arch=sm_20

$(OBJECTS_DBG) : $(CU_FILES)
	@if [ ! -d "$(OBJ_PATH)/dbg" ]; then mkdir -p "$(OBJ_PATH)/dbg"; fi
	nvcc -g -c $(subst .o,.cu,$(subst $(OBJ_PATH)/dbg/,,$@)) -o $@ -arch=sm_20

partialcleanopt :
	rm -f $(TARGET_PATH)/$(TARGET_NAME_OPT)

partialcleandbg :
	rm -f $(TARGET_PATH)/$(TARGET_NAME_DBG)

clean : partialcleandbg partialcleanopt
	rm -f $(OBJECTS_OPT)
	rm -f $(OBJECTS_DBG)

install:
	@if [ "$${SUDO_USER}" == "" ]; then echo "You must use sudo to install"; false; fi
	@sudo -u $${SUDO_USER} $(MAKE) -C . dbg
	@sudo -u $${SUDO_USER} $(MAKE) -C . opt
	mkdir -m 0755 -p $(INSTALL_LOCATION_LIB)
	mkdir -m 0755 -p $(INSTALL_LOCATION_INCLUDE)/depends/GCuda
	install $(TARGET_PATH)/$(TARGET_NAME_OPT) $(INSTALL_LOCATION_LIB)/$(TARGET_NAME_OPT)
	install $(TARGET_PATH)/$(TARGET_NAME_DBG) $(INSTALL_LOCATION_LIB)/$(TARGET_NAME_DBG)
	install *.h $(INSTALL_LOCATION_INCLUDE)/depends/GCuda

uninstall:
	rm -f $(INSTALL_LOCATION_LIB)/$(TARGET_NAME_OPT)
	rm -f $(INSTALL_LOCATION_LIB)/$(TARGET_NAME_DBG)
	rm -rf $(INSTALL_LOCATION_INCLUDE)/depends/GCuda

.PHONY: clean partialcleandbg partialcleanopt install uninstall dbg opt
