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
COMPILER=nvcc

# Set platform-specific compiler flags
ifeq ($(UNAME),Darwin)
	DARWIN_BASE ?= /usr/X11
	CFLAGS = -std=c++11 -D_THREAD_SAFE -DDARWIN -I$(DARWIN_BASE)/include -no-cpp-precomp
else
	CFLAGS = -std=c++11
endif

DBG_CFLAGS = $(CFLAGS) -g -D_DEBUG
OPT_CFLAGS = $(CFLAGS) -O3 -D_FORCE_INLINES

################
# Source
################

CU_FILES = \
	GCudaLayers.cpp\
	GCudaMatrix.cpp\
	GCudaMatrixKernels.cu\

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
$(TARGET_PATH)/$(TARGET_NAME_OPT) : partialcleanopt $(OBJECTS_OPT)
	@if [ ! -d "$(TARGET_PATH)" ]; then mkdir -p "$(TARGET_PATH)"; fi
	@ar -crs $(TARGET_PATH)/$(TARGET_NAME_OPT) $(OBJECTS_OPT)

# This rule makes the debug ".a" file by using the "ar" tool with all of the debug ".o" files
$(TARGET_PATH)/$(TARGET_NAME_DBG) : partialcleandbg $(OBJECTS_DBG)
	@if [ ! -d "$(TARGET_PATH)" ]; then mkdir -p "$(TARGET_PATH)"; fi
	@ar -crs $(TARGET_PATH)/$(TARGET_NAME_DBG) $(OBJECTS_DBG)

$(OBJECTS_OPT) : $(CU_FILES)
	@if [ ! -d "$(OBJ_PATH)/opt" ]; then mkdir -p "$(OBJ_PATH)/opt"; fi
	$(COMPILER) $(OPT_CFLAGS) -c $(subst .o,.cu,$(subst $(OBJ_PATH)/opt/,,$@)) -o $@

$(OBJECTS_DBG) : $(CU_FILES)
	@if [ ! -d "$(OBJ_PATH)/dbg" ]; then mkdir -p "$(OBJ_PATH)/dbg"; fi
	$(COMPILER) $(DBG_CFLAGS) -c $(subst .o,.cu,$(subst $(OBJ_PATH)/dbg/,,$@)) -o $@

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
