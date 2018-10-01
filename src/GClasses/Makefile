################
# Paths and Flags
################
SHELL = /bin/bash
TARGET_PATH = ../../lib
TARGET_NAME_OPT = libGClasses.a
TARGET_NAME_DBG = libGClassesDbg.a
TARGET_NAME_CUDA = libGClassesCuda.a
TARGET_NAME_CUDA_DBG = libGClassesCudaDbg.a
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

# Detect compiler type
ifeq ($(shell $(COMPILER) 2>/dev/null --version | grep clang),)
	CLANG=false
else
	CLANG=true
endif

# Set shared compiler flags
CFLAGS = -Wall -pedantic -std=c++11

# Add platform-specific compiler flags
ifeq ($(UNAME),Darwin)
	DARWIN_BASE ?= /usr/X11
	CFLAGS += -DDARWIN -I$(DARWIN_BASE)/include
else
	CFLAGS += -Werror -Wno-error=unused-variable
endif

# Add compiler-specific flags
ifeq ($(CLANG),true)
	CFLAGS += -no-cpp-precomp -stdlib=libc++ -D_THREAD_SAFE
endif

DBG_CFLAGS = $(CFLAGS) -g -D_DEBUG
OPT_CFLAGS = $(CFLAGS) -O3
#OPT_CFLAGS = $(CFLAGS)  -Ofast -ffp-contract=fast -floop-parallelize-all -funroll-loops -falign-loops -ffast-math -msse -msse2 -msse3 -mfpmath=sse

################
# Source
################

CPP_FILES = \
	G3D.cpp\
	GApp.cpp\
	GAssignment.cpp\
	GBayesianNetwork.cpp\
	GBezier.cpp\
	GBigInt.cpp\
	GBits.cpp\
	GBitTable.cpp\
	GBlob.cpp\
	GCluster.cpp\
	GCrypto.cpp\
	GCudaMatrix.cpp\
	GDecisionTree.cpp\
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
	GGridSearch.cpp\
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
	GLearnerLib.cpp\
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
	GRand.cpp\
	GRecommender.cpp\
	GRecommenderLib.cpp\
	GRect.cpp\
	GRegion.cpp\
	GReinforcement.cpp\
	GSDL.cpp\
	GSelfOrganizingMap.cpp\
	GSmtp.cpp\
	GSocket.cpp\
	GSparseMatrix.cpp\
	GStemmer.cpp\
	GString.cpp\
	GText.cpp\
	GThread.cpp\
	GTime.cpp\
	GTokenizer.cpp\
	GTransform.cpp\
	GTree.cpp\
	GVec.cpp\
	GWave.cpp\
	GWidgets.cpp\
	GWindows.cpp\
	sha2.cpp\
	usage.cpp\

CU_FILES = \
	GCudaMatrixKernels.cu\

################
# Lists
################

TEMP_LIST_OPT = $(CPP_FILES:%=$(OBJ_PATH)/opt/%)
TEMP_LIST_DBG = $(CPP_FILES:%=$(OBJ_PATH)/dbg/%)
TEMP_LIST_CUDA = $(CPP_FILES:%=$(OBJ_PATH)/cuda/%)
TEMP_LIST_CUDA_DBG = $(CPP_FILES:%=$(OBJ_PATH)/cudadbg/%)
OBJECTS_OPT = $(TEMP_LIST_OPT:%.cpp=%.o)
OBJECTS_DBG = $(TEMP_LIST_DBG:%.cpp=%.o)
OBJECTS_CUDA = $(TEMP_LIST_CUDA:%.cpp=%.o)
OBJECTS_CUDA_DBG = $(TEMP_LIST_CUDA_DBG:%.cpp=%.o)
DEPS_OPT = $(TEMP_LIST_OPT:%.cpp=%.d)
DEPS_DBG = $(TEMP_LIST_DBG:%.cpp=%.d)

################
# Rules
################

.DELETE_ON_ERROR:

dbg : $(TARGET_PATH)/$(TARGET_NAME_DBG)

opt : $(TARGET_PATH)/$(TARGET_NAME_OPT)

cudadbg : $(TARGET_PATH)/$(TARGET_NAME_CUDA_DBG)

cuda : $(TARGET_PATH)/$(TARGET_NAME_CUDA)

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

# This rule makes the CUDA ".a" file by using the "ar" tool with all of the CUDA ".o" files
$(TARGET_PATH)/$(TARGET_NAME_CUDA_DBG) : partialcleancudadbg $(OBJECTS_CUDA_DBG) $(OBJ_PATH)/cudadbg/GCudaMatrixKernels.o
	@if [ ! -d "$(TARGET_PATH)" ]; then mkdir -p "$(TARGET_PATH)"; fi
	@ar -crs $(TARGET_PATH)/$(TARGET_NAME_CUDA_DBG) $(OBJECTS_CUDA_DBG) $(OBJ_PATH)/cudadbg/GCudaMatrixKernels.o

# This rule makes the CUDA ".a" file by using the "ar" tool with all of the CUDA ".o" files
$(TARGET_PATH)/$(TARGET_NAME_CUDA) : partialcleancuda $(OBJECTS_CUDA) $(OBJ_PATH)/cuda/GCudaMatrixKernels.o
	@if [ ! -d "$(TARGET_PATH)" ]; then mkdir -p "$(TARGET_PATH)"; fi
	@ar -crs $(TARGET_PATH)/$(TARGET_NAME_CUDA) $(OBJECTS_CUDA) $(OBJ_PATH)/cuda/GCudaMatrixKernels.o

# This includes all of the ".d" files. Each ".d" file contains a
# generated rule that tells it how to make .o files. (The reason these are generated is so that
# dependencies for these rules can be generated.)
-include $(DEPS_OPT)

-include $(DEPS_DBG)

# This rule makes the optimized ".d" files by using "g++ -MM" with the corresponding ".cpp" file
# The ".d" file will contain a rule that says how to make an optimized ".o" file.
# "$<" refers to the ".cpp" file, and "$@" refers to the ".d" file
$(DEPS_OPT) : $(OBJ_PATH)/opt/%.d : %.cpp
	@if [ "$${USER}" == "root" ] && [ "$${SUDO_USER}" != "" ]; then false; fi
	@if [ ! -d "$$(dirname $@)" ]; then mkdir -p "$$(dirname $@)"; fi
	@echo -e "Computing opt dependencies for $<"
	@echo -en "$$(dirname $@)/" > $@
	@$(COMPILER) $(OPT_CFLAGS) -MM $< >> $@
	@echo -e "	$(COMPILER) $(OPT_CFLAGS) -c $< -o $$(dirname $@)/$$(basename $@ .d).o" >> $@

# This rule makes the debug ".d" files by using "g++ -MM" with the corresponding ".cpp" file
# The ".d" file will contain a rule that says how to make a debug ".o" file.
# "$<" refers to the ".cpp" file, and "$@" refers to the ".d" file
$(DEPS_DBG) : $(OBJ_PATH)/dbg/%.d : %.cpp
	@if [ "$${USER}" == "root" ] && [ "$${SUDO_USER}" != "" ]; then false; fi
	@if [ ! -d "$$(dirname $@)" ]; then mkdir -p "$$(dirname $@)"; fi
	@echo -e "Computing dbg dependencies for $<"
	@echo -en "$$(dirname $@)/" > $@
	@$(COMPILER) $(DBG_CFLAGS) -MM $< >> $@
	@echo -e "	$(COMPILER) $(DBG_CFLAGS) -c $< -o $$(dirname $@)/$$(basename $@ .d).o" >> $@

$(OBJ_PATH)/cuda/GCudaMatrixKernels.o : GCudaMatrixKernels.cu
	@if [ ! -d "$(OBJ_PATH)/cuda" ]; then mkdir -p "$(OBJ_PATH)/cuda"; fi
	nvcc -std=c++11 -DGCUDA -c GCudaMatrixKernels.cu -o $(OBJ_PATH)/cuda/GCudaMatrixKernels.o

$(OBJ_PATH)/cudadbg/GCudaMatrixKernels.o : GCudaMatrixKernels.cu
	@if [ ! -d "$(OBJ_PATH)/cudadbg" ]; then mkdir -p "$(OBJ_PATH)/cudadbg"; fi
	nvcc -std=c++11 -g -D_DEBUG -DGCUDA -c GCudaMatrixKernels.cu -o $(OBJ_PATH)/cudadbg/GCudaMatrixKernels.o

$(OBJECTS_CUDA) : $(CPP_FILES)
	@if [ ! -d "$(OBJ_PATH)/cuda" ]; then mkdir -p "$(OBJ_PATH)/cuda"; fi
	nvcc -std=c++11 -O3 -DGCUDA -c $(subst .o,.cpp,$(subst $(OBJ_PATH)/cuda/,,$@)) -o $@

$(OBJECTS_CUDA_DBG) : $(CPP_FILES)
	@if [ ! -d "$(OBJ_PATH)/cudadbg" ]; then mkdir -p "$(OBJ_PATH)/cudadbg"; fi
	nvcc -std=c++11 -g -D_DEBUG -DGCUDA -c $(subst .o,.cpp,$(subst $(OBJ_PATH)/cudadbg/,,$@)) -o $@

partialcleanopt :
	rm -f $(TARGET_PATH)/$(TARGET_NAME_OPT)

partialcleandbg :
	rm -f $(TARGET_PATH)/$(TARGET_NAME_DBG)

partialcleancuda :
	rm -f $(TARGET_PATH)/$(TARGET_NAME_CUDA)

partialcleancudadbg :
	rm -f $(TARGET_PATH)/$(TARGET_NAME_CUDA_DBG)

clean : partialcleandbg partialcleanopt partialcleancuda partialcleancudadbg
	rm -f $(OBJECTS_OPT)
	rm -f $(OBJECTS_DBG)
	rm -f $(OBJECTS_CUDA)
	rm -f $(OBJECTS_CUDA_DBG)
	rm -f $(DEPS_OPT)
	rm -f $(DEPS_DBG)

install:
	@if [ "$${SUDO_USER}" == "" ]; then echo "You must use sudo to install"; false; fi
	@sudo -u $${SUDO_USER} $(MAKE) -j 5 -C . dbg
	@sudo -u $${SUDO_USER} $(MAKE) -j 5 -C . opt
	mkdir -m 0755 -p $(INSTALL_LOCATION_LIB)
	mkdir -m 0755 -p $(INSTALL_LOCATION_INCLUDE)/GClasses
	install $(TARGET_PATH)/$(TARGET_NAME_OPT) $(INSTALL_LOCATION_LIB)/$(TARGET_NAME_OPT)
	install $(TARGET_PATH)/$(TARGET_NAME_DBG) $(INSTALL_LOCATION_LIB)/$(TARGET_NAME_DBG)
	install *.h $(INSTALL_LOCATION_INCLUDE)/GClasses
	if [ -f $(TARGET_PATH)/$(TARGET_NAME_CUDA) ]; then install $(TARGET_PATH)/$(TARGET_NAME_CUDA) $(INSTALL_LOCATION_LIB)/$(TARGET_NAME_CUDA); fi
	if [ -f $(TARGET_PATH)/$(TARGET_NAME_CUDA_DBG) ]; then install $(TARGET_PATH)/$(TARGET_NAME_CUDA_DBG) $(INSTALL_LOCATION_LIB)/$(TARGET_NAME_CUDA_DBG); fi

uninstall:
	rm -f $(INSTALL_LOCATION_LIB)/$(TARGET_NAME_OPT)
	rm -f $(INSTALL_LOCATION_LIB)/$(TARGET_NAME_DBG)
	rm -f $(INSTALL_LOCATION_LIB)/$(TARGET_NAME_CUDA)
	rm -f $(INSTALL_LOCATION_LIB)/$(TARGET_NAME_CUDADBG)
	rm -rf $(INSTALL_LOCATION_INCLUDE)/GClasses

.PHONY: clean partialcleandbg partialcleanopt install uninstall dbg opt cuda cudadbg
