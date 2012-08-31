################
# Paths and Flags
################
SHELL = /bin/bash
TARGET_PATH = ../../bin
TARGET_NAME_OPT = waffles_wizard
TARGET_NAME_DBG = $(TARGET_NAME_OPT)dbg
OBJ_PATH = ../../obj/$(TARGET_NAME_OPT)
INSTALL_LOCATION_BIN ?= /usr/local/bin
UNAME = $(shell uname -s)

# If colorgcc is installed, use it, otherwise use g++
ifeq ($(wildcard /usr/bin/colorgcc),)
	COMPILER=g++
else
	COMPILER=colorgcc
endif

# Set platform-specific compiler and linker flags
ifeq ($(UNAME),Darwin)
	DARWIN_BASE ?= /usr/X11
	CFLAGS = -I/usr/local/include/SDL -D_THREAD_SAFE -DDARWIN -I$(DARWIN_BASE)/include -no-cpp-precomp
	DBG_LFLAGS = $(DARWIN_BASE)/lib/libpng.dylib -lz -framework AppKit ../../lib/GClassesDbg.a
	OPT_LFLAGS = $(DARWIN_BASE)/lib/libpng.dylib -lz -framework AppKit ../../lib/GClasses.a
else
	CFLAGS = -Wall
	DBG_LFLAGS = ../../lib/GClassesDbg.a -lpng -lpthread
	OPT_LFLAGS = ../../lib/GClasses.a -lpng -lpthread
endif

DBG_CFLAGS = $(CFLAGS) -g -D_DEBUG
OPT_CFLAGS = $(CFLAGS) -O3

################
# Source
################

CPP_FILES =\
	main.cpp\
	usage.cpp\

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
	#
	# Usage:
	#  make usage   (to see this info)
	#  make clean   (to delete all the .o files)
	#  make dbg     (to build a debug version)
	#  make opt     (to build an optimized version)
	#

../../lib/GClassesDbg.a :
	$(MAKE) -C ../GClasses dbg

../../lib/GClasses.a :
	$(MAKE) -C ../GClasses opt

# This rule makes the optimized binary by using g++ with the optimized ".o" files
$(TARGET_PATH)/$(TARGET_NAME_OPT) : partialcleanopt $(OBJECTS_OPT) ../../lib/GClasses.a
	@if [ ! -d "$(TARGET_PATH)" ]; then mkdir -p "$(TARGET_PATH)"; fi
	g++ -O3 -o $(TARGET_PATH)/$(TARGET_NAME_OPT) $(OBJECTS_OPT) $(OPT_LFLAGS)

# This rule makes the debug binary by using g++ with the debug ".o" files
$(TARGET_PATH)/$(TARGET_NAME_DBG) : partialcleandbg $(OBJECTS_DBG) ../../lib/GClassesDbg.a
	@if [ ! -d "$(TARGET_PATH)" ]; then mkdir -p "$(TARGET_PATH)"; fi
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
	@if [ "$${USER}" == "root" ]; then false; fi # don't generate dependencies as root
	@echo -e "Computing opt dependencies for $<"
	@-rm -f $$(dirname $@)/$$(basename $@ .d).o
	@if [ ! -d "$$(dirname $@)" ]; then mkdir -p "$$(dirname $@)"; fi
	@echo -en "$$(dirname $@)/" > $@
	@$(COMPILER) $(OPT_CFLAGS) -MM $< >> $@
	@echo -e "	$(COMPILER) $(OPT_CFLAGS) -c $< -o $$(dirname $@)/$$(basename $@ .d).o" >> $@

# This rule makes the debug ".d" files by using "g++ -MM" with the corresponding ".cpp" file
# The ".d" file will contain a rule that says how to make a debug ".o" file.
# "$<" refers to the ".cpp" file, and "$@" refers to the ".d" file
$(DEPS_DBG) : $(OBJ_PATH)/dbg/%.d : %.cpp
	@if [ "$${USER}" == "root" ]; then false; fi # don't generate dependencies as root
	@echo -e "Computing dbg dependencies for $<"
	@-rm -f $$(dirname $@)/$$(basename $@ .d).o
	@if [ ! -d "$$(dirname $@)" ]; then mkdir -p "$$(dirname $@)"; fi
	@echo -en "$$(dirname $@)/" > $@
	@$(COMPILER) $(DBG_CFLAGS) -MM $< >> $@
	@echo -e "	$(COMPILER) $(DBG_CFLAGS) -c $< -o $$(dirname $@)/$$(basename $@ .d).o" >> $@

partialcleandbg :
	rm -f $(TARGET_PATH)/$(TARGET_NAME_DBG)

partialcleanopt :
	rm -f $(TARGET_PATH)/$(TARGET_NAME_OPT)

clean : partialcleandbg partialcleanopt
	rm -f $(OBJECTS_OPT)
	rm -f $(OBJECTS_DBG)
	rm -f $(DEPS_OPT)
	rm -f $(DEPS_DBG)

install :
	@if [ "$${SUDO_USER}" == "" ]; then echo "You must use sudo to install"; false; fi
	@sudo -u $${SUDO_USER} $(MAKE) -C . opt
	@rm -f $(INSTALL_LOCATION_BIN)/$(TARGET_NAME_OPT)
	install $(TARGET_PATH)/$(TARGET_NAME_OPT) $(INSTALL_LOCATION_BIN)
	if [ -d /etc/bash_completion.d ];then $(TARGET_PATH)/$(TARGET_NAME_OPT) make_bash_completion_file > /etc/bash_completion.d/waffles;fi

uninstall:
	rm -f $(INSTALL_LOCATION_BIN)/$(TARGET_NAME_OPT)
	rm -f /etc/bash_completion.d/waffles

.PHONY: clean partialcleandbg partialcleanopt install uninstall dbg opt
