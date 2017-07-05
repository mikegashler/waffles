################
# Paths and Flags
################
SHELL = /bin/bash
TARGET_PATH = ../../bin
TARGET_NAME_OPT = test
TARGET_NAME_DBG = $(TARGET_NAME_OPT)dbg
TARGET_NAME_GCOV = $(TARGET_NAME_OPT)gcov
OBJ_PATH = ../../obj/$(TARGET_NAME_OPT)
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
	CFLAGS = -std=c++11 -stdlib=libc++ -I/opt/local/include -I/usr/local/include -I/sw/include -I$(INSTALL_LOCATION_INCLUDE) -I$(DARWIN_BASE)/include -D_THREAD_SAFE -DDARWIN -no-cpp-precomp
	DBG_LFLAGS = -stdlib=libc++ -L/opt/local/lib -L/usr/local/lib -L/sw/lib -framework AppKit ../../lib/libGClassesDbg.a -lpthread
	OPT_LFLAGS = -stdlib=libc++ -L/opt/local/lib -L/usr/local/lib -L/sw/lib -framework AppKit ../../lib/libGClasses.a -lpthread
	GCOV_LFLAGS = -fprofile-arcs $(DARWIN_BASE)/lib/libpng.dylib -lz -framework AppKit ../../lib/libGClassesGcov.a
else
	CFLAGS = -Wall -Werror -pedantic -std=c++11
	DBG_LFLAGS = ../../lib/libGClassesDbg.a -lpthread
	OPT_LFLAGS = ../../lib/libGClasses.a -lpthread
	GCOV_LFLAGS = -fprofile-arcs ../../lib/libGClassesGcov.a -lpthread
endif

DBG_CFLAGS = $(CFLAGS) -g -D_DEBUG
OPT_CFLAGS = $(CFLAGS) -O3
GCOV_CFLAGS = $(CFLAGS) -fprofile-arcs -ftest-coverage -g

################
# Source
################

CPP_FILES =\
	main.cpp\

################
# Lists
################

TEMP_LIST_OPT = $(CPP_FILES:%=$(OBJ_PATH)/opt/%)
TEMP_LIST_DBG = $(CPP_FILES:%=$(OBJ_PATH)/dbg/%)
TEMP_LIST_GCOV = $(CPP_FILES:%=$(OBJ_PATH)/gcov/%)
OBJECTS_OPT = $(TEMP_LIST_OPT:%.cpp=%.o)
OBJECTS_DBG = $(TEMP_LIST_DBG:%.cpp=%.o)
OBJECTS_GCOV = $(TEMP_LIST_GCOV:%.cpp=%.o)
DEPS_OPT = $(TEMP_LIST_OPT:%.cpp=%.d)
DEPS_DBG = $(TEMP_LIST_DBG:%.cpp=%.d)
DEPS_GCOV = $(TEMP_LIST_GCOV:%.cpp=%.d)

GCNO_FILES = $(TEMP_LIST_GCOV:%.cpp=%.gcno)
GCDA_FILES = $(TEMP_LIST_GCOV:%.cpp=%.gcda)
################
# Rules
################

.DELETE_ON_ERROR:



dbg : $(TARGET_PATH)/$(TARGET_NAME_DBG)

opt : $(TARGET_PATH)/$(TARGET_NAME_OPT)

gcov : $(TARGET_PATH)/$(TARGET_NAME_GCOV)

#To do the lcov analysis: 
#  1) make clean 
#  2) make gcov 
#  3) run ../../bin/testgcov 
#  4) make lcov

lcov:
	rm -rf lcov_output
	mkdir -p lcov_output
	lcov  --capture --directory ../../obj/test/gcov --output-file lcov_output/lcov.info.test -b .  --ignore-errors source
	lcov  --capture --directory ../../obj/GClasses/gcov --output-file lcov_output/lcov.info.gclasses -b ../GClasses --ignore-errors source
	lcov  --add-tracefile lcov_output/lcov.info.test --output-file lcov_output/lcov.info
	lcov  --add-tracefile lcov_output/lcov.info.gclasses --output-file lcov_output/lcov.info
	genhtml -o lcov_output lcov_output/lcov.info

.PHONY: lcov


usage:
	#
	# Usage:
	#  make usage   (to see this info)
	#  make clean   (to delete all the .o files)
	#  make dbg     (to build a debug version)
	#  make opt     (to build an optimized version)
	#  make gcov    (to build a profiling/test-coverage version)
	#  make lcov    (generate the test coverage html files in lcov_output 
	#                after running ../../bin/testgcov)
	#

../../lib/libGClasses.a :
	$(MAKE) -C ../GClasses opt

../../lib/libGClassesDbg.a :
	$(MAKE) -C ../GClasses dbg

../../lib/libGClassesGcov.a :
	$(MAKE) -C ../GClasses gcov


# This rule makes the optimized binary by using g++ with the optimized ".o" files
$(TARGET_PATH)/$(TARGET_NAME_OPT) : partialcleanopt $(OBJECTS_OPT) ../../lib/libGClasses.a
	@if [ ! -d "$(TARGET_PATH)" ]; then mkdir -p "$(TARGET_PATH)"; fi
	$(COMPILER) -O3 -o $(TARGET_PATH)/$(TARGET_NAME_OPT) $(OBJECTS_OPT) $(OPT_LFLAGS)

# This rule makes the debug binary by using g++ with the debug ".o" files
$(TARGET_PATH)/$(TARGET_NAME_DBG) : partialcleandbg $(OBJECTS_DBG) ../../lib/libGClassesDbg.a
	@if [ ! -d "$(TARGET_PATH)" ]; then mkdir -p "$(TARGET_PATH)"; fi
	$(COMPILER) -g -o $(TARGET_PATH)/$(TARGET_NAME_DBG) $(OBJECTS_DBG) $(DBG_LFLAGS)

# This rule makes the gcov binary by using g++ with the gcov ".o" files
$(TARGET_PATH)/$(TARGET_NAME_GCOV) : partialcleangcov $(OBJECTS_GCOV) ../../lib/libGClassesGcov.a
	$(COMPILER) -g -o $(TARGET_PATH)/$(TARGET_NAME_GCOV) $(OBJECTS_GCOV) $(GCOV_LFLAGS)

# This includes all of the ".d" files. Each ".d" file contains a
# generated rule that tells it how to make .o files. (The reason these are generated is so that
# dependencies for these rules can be generated.)
-include $(DEPS_OPT)

-include $(DEPS_DBG)

-include $(DEPS_GCOV)

# This rule makes the optimized ".d" files by using "g++ -MM" with the corresponding ".cpp" file
# The ".d" file will contain a rule that says how to make an optimized ".o" file.
# "$<" refers to the ".cpp" file, and "$@" refers to the ".d" file
$(DEPS_OPT) : $(OBJ_PATH)/opt/%.d : %.cpp
	@if [ "$${USER}" == "root" ] && [ "$${SUDO_USER}" != "" ]; then false; fi
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
	@if [ "$${USER}" == "root" ] && [ "$${SUDO_USER}" != "" ]; then false; fi
	@echo -e "Computing dbg dependencies for $<"
	@-rm -f $$(dirname $@)/$$(basename $@ .d).o
	@if [ ! -d "$$(dirname $@)" ]; then mkdir -p "$$(dirname $@)"; fi
	@echo -en "$$(dirname $@)/" > $@
	@$(COMPILER) $(DBG_CFLAGS) -MM $< >> $@
	@echo -e "	$(COMPILER) $(DBG_CFLAGS) -c $< -o $$(dirname $@)/$$(basename $@ .d).o" >> $@

# This rule makes the debug ".d" files by using "g++ -MM" with the corresponding ".cpp" file
# The ".d" file will contain a rule that says how to make a debug ".o" file.
# "$<" refers to the ".cpp" file, and "$@" refers to the ".d" file
$(DEPS_GCOV) : $(OBJ_PATH)/gcov/%.d : %.cpp
	@if [ "$${USER}" == "root" ] && [ "$${SUDO_USER}" != "" ]; then false; fi
	@echo -e "Computing gcov dependencies for $<"
	@-rm -f $$(dirname $@)/$$(basename $@ .d).o
	@if [ ! -d "$$(dirname $@)" ]; then mkdir -p "$$(dirname $@)"; fi
	@echo -en "$$(dirname $@)/" > $@
	@$(COMPILER) $(GCOV_CFLAGS) -MM $< >> $@
	@echo -e "	rm -f $(GCNO_FILES)" >> $@
	@echo -e "	$(COMPILER) $(GCOV_CFLAGS) -c $< -o $$(dirname $@)/$$(basename $@ .d).o" >> $@

partialcleanopt :
	rm -f $(TARGET_PATH)/$(TARGET_NAME_OPT)

partialcleandbg :
	rm -f $(TARGET_PATH)/$(TARGET_NAME_DBG)

partialcleangcov :
	rm -f $(TARGET_PATH)/$(TARGET_NAME_GCOV)
	rm -f $(GCDA_FILES)

clean : partialcleandbg partialcleanopt
	rm -f $(OBJECTS_OPT)
	rm -f $(OBJECTS_DBG)
	rm -f $(OBJECTS_GCOV)
	rm -f $(DEPS_OPT)
	rm -f $(DEPS_DBG)
	rm -f $(DEPS_GCOV)

install:
	@if [ "$${SUDO_USER}" == "" ]; then echo "You must use sudo to install"; false; fi
	@sudo -u $${SUDO_USER} $(MAKE) -C . opt

uninstall:

.PHONY: clean partialcleandbg partialcleangcov partialcleanopt install uninstall dbg opt gcov
