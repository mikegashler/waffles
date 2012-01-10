################
# Paths and Flags
################
SHELL = /bin/bash
TARGET_PATH = ../lib
TARGET_NAME_OPT = bullet.a
TARGET_NAME_DBG = bulletdbg.a
OBJ_PATH = ../obj
UNAME = $(shell uname -s)
ifeq ($(UNAME),Darwin)
#CFLAGS = -D_THREAD_SAFE -DDARWIN -I/sw/include -DBYTE_ORDER_BIG_ENDIAN -no-cpp-precomp
CFLAGS = -D_THREAD_SAFE -DDARWIN -I/sw/include -no-cpp-precomp -I.
else
CFLAGS = -Wall -I.
endif
DBG_CFLAGS = $(CFLAGS) -g -D_DEBUG -I.
OPT_CFLAGS = $(CFLAGS) -O3 -I.
UNIX_LIB_DIR = /usr/local/lib
UNIX_INCLUDE_DIR = /usr/local/include

################
# Source
################

CPP_FILES = \
	BulletDynamics/Character/btKinematicCharacterController.cpp\
	BulletDynamics/ConstraintSolver/btConeTwistConstraint.cpp\
	BulletDynamics/ConstraintSolver/btContactConstraint.cpp\
	BulletDynamics/ConstraintSolver/btGeneric6DofConstraint.cpp\
	BulletDynamics/ConstraintSolver/btGeneric6DofSpringConstraint.cpp\
	BulletDynamics/ConstraintSolver/btHinge2Constraint.cpp\
	BulletDynamics/ConstraintSolver/btHingeConstraint.cpp\
	BulletDynamics/ConstraintSolver/btPoint2PointConstraint.cpp\
	BulletDynamics/ConstraintSolver/btSequentialImpulseConstraintSolver.cpp\
	BulletDynamics/ConstraintSolver/btSliderConstraint.cpp\
	BulletDynamics/ConstraintSolver/btSolve2LinearConstraint.cpp\
	BulletDynamics/ConstraintSolver/btTypedConstraint.cpp\
	BulletDynamics/ConstraintSolver/btUniversalConstraint.cpp\
	BulletDynamics/Dynamics/btContinuousDynamicsWorld.cpp\
	BulletDynamics/Dynamics/btDiscreteDynamicsWorld.cpp\
	BulletDynamics/Dynamics/btRigidBody.cpp\
	BulletDynamics/Dynamics/btSimpleDynamicsWorld.cpp\
	BulletDynamics/Dynamics/Bullet-C-API.cpp\
	BulletDynamics/Vehicle/btRaycastVehicle.cpp\
	BulletDynamics/Vehicle/btWheelInfo.cpp\
	BulletCollision/BroadphaseCollision/btAxisSweep3.cpp\
	BulletCollision/BroadphaseCollision/btBroadphaseProxy.cpp\
	BulletCollision/BroadphaseCollision/btCollisionAlgorithm.cpp\
	BulletCollision/BroadphaseCollision/btDbvt.cpp\
	BulletCollision/BroadphaseCollision/btDbvtBroadphase.cpp\
	BulletCollision/BroadphaseCollision/btDispatcher.cpp\
	BulletCollision/BroadphaseCollision/btMultiSapBroadphase.cpp\
	BulletCollision/BroadphaseCollision/btOverlappingPairCache.cpp\
	BulletCollision/BroadphaseCollision/btQuantizedBvh.cpp\
	BulletCollision/BroadphaseCollision/btSimpleBroadphase.cpp\
	BulletCollision/CollisionDispatch/btActivatingCollisionAlgorithm.cpp\
	BulletCollision/CollisionDispatch/btBoxBoxCollisionAlgorithm.cpp\
	BulletCollision/CollisionDispatch/btBox2dBox2dCollisionAlgorithm.cpp\
	BulletCollision/CollisionDispatch/btBoxBoxDetector.cpp\
	BulletCollision/CollisionDispatch/btCollisionDispatcher.cpp\
	BulletCollision/CollisionDispatch/btCollisionObject.cpp\
	BulletCollision/CollisionDispatch/btCollisionWorld.cpp\
	BulletCollision/CollisionDispatch/btCompoundCollisionAlgorithm.cpp\
	BulletCollision/CollisionDispatch/btConvexConcaveCollisionAlgorithm.cpp\
	BulletCollision/CollisionDispatch/btConvexConvexAlgorithm.cpp\
	BulletCollision/CollisionDispatch/btConvexPlaneCollisionAlgorithm.cpp\
	BulletCollision/CollisionDispatch/btConvex2dConvex2dAlgorithm.cpp\
	BulletCollision/CollisionDispatch/btDefaultCollisionConfiguration.cpp\
	BulletCollision/CollisionDispatch/btEmptyCollisionAlgorithm.cpp\
	BulletCollision/CollisionDispatch/btGhostObject.cpp\
	BulletCollision/CollisionDispatch/btInternalEdgeUtility.cpp\
	BulletCollision/CollisionDispatch/btManifoldResult.cpp\
	BulletCollision/CollisionDispatch/btSimulationIslandManager.cpp\
	BulletCollision/CollisionDispatch/btSphereBoxCollisionAlgorithm.cpp\
	BulletCollision/CollisionDispatch/btSphereSphereCollisionAlgorithm.cpp\
	BulletCollision/CollisionDispatch/btSphereTriangleCollisionAlgorithm.cpp\
	BulletCollision/CollisionDispatch/btUnionFind.cpp\
	BulletCollision/CollisionDispatch/SphereTriangleDetector.cpp\
	BulletCollision/CollisionShapes/btBoxShape.cpp\
	BulletCollision/CollisionShapes/btBox2dShape.cpp\
	BulletCollision/CollisionShapes/btBvhTriangleMeshShape.cpp\
	BulletCollision/CollisionShapes/btCapsuleShape.cpp\
	BulletCollision/CollisionShapes/btCollisionShape.cpp\
	BulletCollision/CollisionShapes/btCompoundShape.cpp\
	BulletCollision/CollisionShapes/btConcaveShape.cpp\
	BulletCollision/CollisionShapes/btConeShape.cpp\
	BulletCollision/CollisionShapes/btConvexHullShape.cpp\
	BulletCollision/CollisionShapes/btConvexInternalShape.cpp\
	BulletCollision/CollisionShapes/btConvexPointCloudShape.cpp\
	BulletCollision/CollisionShapes/btConvexShape.cpp\
	BulletCollision/CollisionShapes/btConvex2dShape.cpp\
	BulletCollision/CollisionShapes/btConvexTriangleMeshShape.cpp\
	BulletCollision/CollisionShapes/btCylinderShape.cpp\
	BulletCollision/CollisionShapes/btEmptyShape.cpp\
	BulletCollision/CollisionShapes/btHeightfieldTerrainShape.cpp\
	BulletCollision/CollisionShapes/btMinkowskiSumShape.cpp\
	BulletCollision/CollisionShapes/btMultimaterialTriangleMeshShape.cpp\
	BulletCollision/CollisionShapes/btMultiSphereShape.cpp\
	BulletCollision/CollisionShapes/btOptimizedBvh.cpp\
	BulletCollision/CollisionShapes/btPolyhedralConvexShape.cpp\
	BulletCollision/CollisionShapes/btScaledBvhTriangleMeshShape.cpp\
	BulletCollision/CollisionShapes/btShapeHull.cpp\
	BulletCollision/CollisionShapes/btSphereShape.cpp\
	BulletCollision/CollisionShapes/btStaticPlaneShape.cpp\
	BulletCollision/CollisionShapes/btStridingMeshInterface.cpp\
	BulletCollision/CollisionShapes/btTetrahedronShape.cpp\
	BulletCollision/CollisionShapes/btTriangleBuffer.cpp\
	BulletCollision/CollisionShapes/btTriangleCallback.cpp\
	BulletCollision/CollisionShapes/btTriangleIndexVertexArray.cpp\
	BulletCollision/CollisionShapes/btTriangleIndexVertexMaterialArray.cpp\
	BulletCollision/CollisionShapes/btTriangleMesh.cpp\
	BulletCollision/CollisionShapes/btTriangleMeshShape.cpp\
	BulletCollision/CollisionShapes/btUniformScalingShape.cpp\
	BulletCollision/Gimpact/btContactProcessing.cpp\
	BulletCollision/Gimpact/btGenericPoolAllocator.cpp\
	BulletCollision/Gimpact/btGImpactBvh.cpp\
	BulletCollision/Gimpact/btGImpactCollisionAlgorithm.cpp\
	BulletCollision/Gimpact/btGImpactQuantizedBvh.cpp\
	BulletCollision/Gimpact/btGImpactShape.cpp\
	BulletCollision/Gimpact/btTriangleShapeEx.cpp\
	BulletCollision/Gimpact/gim_box_set.cpp\
	BulletCollision/Gimpact/gim_contact.cpp\
	BulletCollision/Gimpact/gim_memory.cpp\
	BulletCollision/Gimpact/gim_tri_collision.cpp\
	BulletCollision/NarrowPhaseCollision/btContinuousConvexCollision.cpp\
	BulletCollision/NarrowPhaseCollision/btConvexCast.cpp\
	BulletCollision/NarrowPhaseCollision/btGjkConvexCast.cpp\
	BulletCollision/NarrowPhaseCollision/btGjkEpa2.cpp\
	BulletCollision/NarrowPhaseCollision/btGjkEpaPenetrationDepthSolver.cpp\
	BulletCollision/NarrowPhaseCollision/btGjkPairDetector.cpp\
	BulletCollision/NarrowPhaseCollision/btMinkowskiPenetrationDepthSolver.cpp\
	BulletCollision/NarrowPhaseCollision/btPersistentManifold.cpp\
	BulletCollision/NarrowPhaseCollision/btRaycastCallback.cpp\
	BulletCollision/NarrowPhaseCollision/btSubSimplexConvexCast.cpp\
	BulletCollision/NarrowPhaseCollision/btVoronoiSimplexSolver.cpp\
	LinearMath/btAlignedAllocator.cpp\
	LinearMath/btConvexHull.cpp\
	LinearMath/btGeometryUtil.cpp\
	LinearMath/btQuickprof.cpp\
	LinearMath/btSerializer.cpp\

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

titleopt :
	@for ((i=0;i<10;i+=1));do echo "";done
	@echo "Building $(TARGET_NAME_OPT)..."
	@echo ""
	@-umask 0;mkdir -p ../obj/opt/BulletDynamics/Character
	@-umask 0;mkdir -p ../obj/opt/BulletDynamics/ConstraintSolver
	@-umask 0;mkdir -p ../obj/opt/BulletDynamics/Dynamics
	@-umask 0;mkdir -p ../obj/opt/BulletDynamics/Vehicle
	@-umask 0;mkdir -p ../obj/opt/BulletCollision/BroadphaseCollision
	@-umask 0;mkdir -p ../obj/opt/BulletCollision/CollisionDispatch
	@-umask 0;mkdir -p ../obj/opt/BulletCollision/CollisionShapes
	@-umask 0;mkdir -p ../obj/opt/BulletCollision/Gimpact
	@-umask 0;mkdir -p ../obj/opt/BulletCollision/NarrowPhaseCollision
	@-umask 0;mkdir -p ../obj/opt/LinearMath

titledbg :
	@for ((i=0;i<10;i+=1));do echo "";done
	@echo "Building $(TARGET_NAME_DBG)..."
	@echo ""
	@-umask 0;mkdir -p ../obj/dbg/BulletDynamics/Character
	@-umask 0;mkdir -p ../obj/dbg/BulletDynamics/ConstraintSolver
	@-umask 0;mkdir -p ../obj/dbg/BulletDynamics/Dynamics
	@-umask 0;mkdir -p ../obj/dbg/BulletDynamics/Vehicle
	@-umask 0;mkdir -p ../obj/dbg/BulletCollision/BroadphaseCollision
	@-umask 0;mkdir -p ../obj/dbg/BulletCollision/CollisionDispatch
	@-umask 0;mkdir -p ../obj/dbg/BulletCollision/CollisionShapes
	@-umask 0;mkdir -p ../obj/dbg/BulletCollision/Gimpact
	@-umask 0;mkdir -p ../obj/dbg/BulletCollision/NarrowPhaseCollision
	@-umask 0;mkdir -p ../obj/dbg/LinearMath

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
	@g++ $(OPT_CFLAGS) -MM $< >> $@
	@echo -e "	umask 0;g++ $(OPT_CFLAGS) -c $< -o $$(dirname $@)/$$(basename $@ .d).o" >> $@

# This rule makes the debug ".d" files by using "g++ -MM" with the corresponding ".cpp" file
# The ".d" file will contain a rule that says how to make a debug ".o" file.
# "$<" refers to the ".cpp" file, and "$@" refers to the ".d" file
$(DEPS_DBG) : $(OBJ_PATH)/dbg/%.d : %.cpp
	@echo -e "Computing dbg dependencies for $<"
	@-rm -f $$(dirname $@)/$$(basename $@ .d).o
	@if [ ! -d "$$(dirname $@)" ]; then umask 0;mkdir -p "$$(dirname $@)"; fi
	@umask 0;echo -en "$$(dirname $@)/" > $@
	@g++ $(DBG_CFLAGS) -MM $< >> $@
	@echo -e "	umask 0;g++ $(DBG_CFLAGS) -c $< -o $$(dirname $@)/$$(basename $@ .d).o" >> $@

partialcleandbg :
	rm -f $(TARGET_PATH)/$(TARGET_NAME_DBG)

partialcleanopt :
	rm -f $(TARGET_PATH)/$(TARGET_NAME_OPT)

clean : partialcleandbg partialcleanopt
	rm -f $(OBJECTS_OPT)
	rm -f $(OBJECTS_DBG)
	rm -f $(DEPS_OPT)
	rm -f $(DEPS_DBG)

.PHONY: clean partialcleandbg partialcleanopt dbg opt
