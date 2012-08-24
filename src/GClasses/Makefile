################
# Paths and Flags
################
SHELL = /bin/bash
TARGET_PATH = ../../lib
TARGET_NAME_OPT = GClasses.a
TARGET_NAME_DBG = GClassesDbg.a
TARGET_NAME_GCOV= GClassesGcov.a
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
	DARWIN_BASE ?= /usr/X11
	CFLAGS = -D_THREAD_SAFE -DDARWIN -I$(DARWIN_BASE)/include -no-cpp-precomp
else
	CFLAGS = -Wall
endif

GCOV_CFLAGS = $(CFLAGS) -g -fprofile-arcs -ftest-coverage 
DBG_CFLAGS = $(CFLAGS) -g -D_DEBUG
OPT_CFLAGS = $(CFLAGS) -O3

################
# Source
################

CPP_FILES = \
	G3D.cpp\
	GActivation.cpp\
	GApp.cpp\
	GAssignment.cpp\
	GBayesianNetwork.cpp\
	GBezier.cpp\
	GBigInt.cpp\
	GBits.cpp\
	GBitTable.cpp\
	GBlob.cpp\
	GChess.cpp\
	GCluster.cpp\
	GCrypto.cpp\
	GDecisionTree.cpp\
	GDiff.cpp\
	GDirList.cpp\
	GDistance.cpp\
	GDistribution.cpp\
	GDom.cpp\
	GDynamicPage.cpp\
	GEnsemble.cpp\
	GError.cpp\
	GEvolutionary.cpp\
	GFile.cpp\
	GFourier.cpp\
	GFunction.cpp\
	GGaussianProcess.cpp\
	GGraph.cpp\
	GHashTable.cpp\
	GHeap.cpp\
	GHiddenMarkovModel.cpp\
	GHillClimber.cpp\
	GHistogram.cpp\
	GHolders.cpp\
	GHtml.cpp\
	GHttp.cpp\
	GImage.cpp\
	GKalman.cpp\
	GKernelTrick.cpp\
	GKeyPair.cpp\
	GKNN.cpp\
	GLearner.cpp\
	GLinear.cpp\
	GManifold.cpp\
	GMath.cpp\
	GMatrix.cpp\
	GMixtureOfGaussians.cpp\
	GNaiveBayes.cpp\
	GNaiveInstance.cpp\
	GNeighborFinder.cpp\
	GNeuralNet.cpp\
	GOptimizer.cpp\
	GParticleSwarm.cpp\
	GPlot.cpp\
	GPolicyLearner.cpp\
	GPolynomial.cpp\
	GPriorityQueue.cpp\
	GRayTrace.cpp\
	GReverseBits.cpp\
	GRect.cpp\
	GRand.cpp\
	GRecommender.cpp\
	GRegion.cpp\
	GReinforcement.cpp\
	GSDL.cpp\
	GSelfOrganizingMap.cpp\
	GSmtp.cpp\
	GSocket.cpp\
	GSparseMatrix.cpp\
	GSpinLock.cpp\
	GStabSearch.cpp\
	GStemmer.cpp\
	GString.cpp\
	GSystemLearner.cpp\
	GText.cpp\
	GThread.cpp\
	GTime.cpp\
	GTokenizer.cpp\
	GTransform.cpp\
	GVec.cpp\
	GWave.cpp\
	GWidgets.cpp\
	GWindows.cpp\
	sha1.cpp\
	sha2.cpp\

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

gcov: $(TARGET_PATH)/$(TARGET_NAME_GCOV)

fam: findAugMatrix.cpp ../../lib/GClassesDbg.a
	g++ -o fam findAugMatrix.cpp ../../lib/GClassesDbg.a -lpng -lz

usage:
	#
	# Usage:
	#  make usage   (to see this info)
	#  make clean   (to delete all the .o files)
	#  make dbg     (to build a debug version)
	#  make opt     (to build an optimized version)
	#  make gcov    (to build a profiling/test-coverage version)
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

# This rule makes the debug ".a" file by using the "ar" tool with all of the debug ".o" files
$(TARGET_PATH)/$(TARGET_NAME_GCOV) : partialcleangcov $(OBJECTS_GCOV)
	@if [ ! -d "$(TARGET_PATH)" ]; then mkdir -p "$(TARGET_PATH)"; fi
	@ar -crs $(TARGET_PATH)/$(TARGET_NAME_GCOV) $(OBJECTS_GCOV)

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
	@if [ "$${USER}" == "root" ]; then false; fi # don't generate dependencies as root
	@if [ ! -d "$$(dirname $@)" ]; then mkdir -p "$$(dirname $@)"; fi
	@echo -e "Computing opt dependencies for $<"
	@-rm -f $$(dirname $@)/$$(basename $@ .d).o
	@#lockfile-create -r 0 $@
	@echo -en "$$(dirname $@)/" > $@
	@$(COMPILER) $(OPT_CFLAGS) -MM $< >> $@
	@echo -e "	$(COMPILER) $(OPT_CFLAGS) -c $< -o $$(dirname $@)/$$(basename $@ .d).o" >> $@
	@#lockfile-remove $@

# This rule makes the debug ".d" files by using "g++ -MM" with the corresponding ".cpp" file
# The ".d" file will contain a rule that says how to make a debug ".o" file.
# "$<" refers to the ".cpp" file, and "$@" refers to the ".d" file
$(DEPS_DBG) : $(OBJ_PATH)/dbg/%.d : %.cpp
	@if [ "$${USER}" == "root" ]; then false; fi # don't generate dependencies as root
	@if [ ! -d "$$(dirname $@)" ]; then mkdir -p "$$(dirname $@)"; fi
	@echo -e "Computing dbg dependencies for $<"
	@-rm -f $$(dirname $@)/$$(basename $@ .d).o
	@#lockfile-create -r 0 $@
	@echo -en "$$(dirname $@)/" > $@
	@$(COMPILER) $(DBG_CFLAGS) -MM $< >> $@
	@echo -e "	$(COMPILER) $(DBG_CFLAGS) -c $< -o $$(dirname $@)/$$(basename $@ .d).o" >> $@
	@#lockfile-remove $@

# This rule makes the test/profile ".d" files by using "g++ -MM" with the corresponding ".cpp" file
# The ".d" file will contain a rule that says how to make a test/profile ".o" file.
# "$<" refers to the ".cpp" file, and "$@" refers to the ".d" file
$(DEPS_GCOV) : $(OBJ_PATH)/gcov/%.d : %.cpp
	@if [ "$${USER}" == "root" ]; then false; fi # don't generate dependencies as root
	@if [ ! -d "$$(dirname $@)" ]; then mkdir -p "$$(dirname $@)"; fi
	@echo -e "Computing gcov dependencies for $<"
	@-rm -f $$(dirname $@)/$$(basename $@ .d).o
	@#lockfile-create -r 0 $@
	@echo -en "$$(dirname $@)/" > $@
	@$(COMPILER) $(GCOV_CFLAGS) -MM $< >> $@
	@#Delete old gcno file
	@echo -en "	rm -f " >> $@
	@echo $$(dirname $@)/$$(basename $@ .d).gcno >> $@
	@echo -e "	$(COMPILER) $(GCOV_CFLAGS) -c $< -o $$(dirname $@)/$$(basename $@ .d).o" >> $@
	@#lockfile-remove $@

partialcleanopt :
	rm -f $(TARGET_PATH)/$(TARGET_NAME_OPT)

partialcleandbg :
	rm -f $(TARGET_PATH)/$(TARGET_NAME_DBG)

partialcleangcov :
	rm -f $(TARGET_PATH)/$(TARGET_NAME_GCOV)
	@echo "Removing all gcda test-coverage/profiling files from ../../obj/GClasses directory"
	@rm -f $(GCDA_FILES)

clean : partialcleandbg partialcleanopt partialcleangcov
	rm -f $(OBJECTS_OPT)
	rm -f $(OBJECTS_DBG)
	rm -f $(OBJECTS_GCOV)
	rm -f $(DEPS_OPT)
	rm -f $(DEPS_DBG)
	rm -f $(DEPS_GCOV)
	rm -f $(OBJ_PATH)/opt/*.lock
	rm -f $(OBJ_PATH)/dbg/*.lock
	rm -f $(OBJ_PATH)/gcov/*.lock

install:
	@if [ "$${SUDO_USER}" == "" ]; then echo "You must use sudo to install"; false; fi
	@sudo -u $${SUDO_USER} $(MAKE) -C . dbg
	@sudo -u $${SUDO_USER} $(MAKE) -C . opt
	mkdir -m 0755 -p $(INSTALL_LOCATION_LIB)
	mkdir -m 0755 -p $(INSTALL_LOCATION_INCLUDE)/GClasses
	install $(TARGET_PATH)/GClasses.a $(INSTALL_LOCATION_LIB)/libGClasses.a
	install $(TARGET_PATH)/GClassesDbg.a $(INSTALL_LOCATION_LIB)/libGClassesDbg.a
	install *.h $(INSTALL_LOCATION_INCLUDE)/GClasses

uninstall:
	rm -f $(INSTALL_LOCATION_LIB)/libGClasses.a
	rm -f $(INSTALL_LOCATION_LIB)/libGClassesDbg.a
	rm -rf $(INSTALL_LOCATION_INCLUDE)/GClasses

.PHONY: clean partialcleandbg partialcleanopt install uninstall dbg opt
