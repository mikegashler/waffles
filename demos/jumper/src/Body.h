/*
Bullet Continuous Collision Detection and Physics Library
RagdollDemo
Copyright (c) 2007 Starbreeze Studios

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it freely, 
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.

Written by: Marten Svanfeldt
*/

#ifndef RAGDOLLDEMO_H
#define RAGDOLLDEMO_H

#include "GlutDemoApplication.h"
#include "LinearMath/btAlignedObjectArray.h"


class btBroadphaseInterface;
class btCollisionShape;
class btOverlappingPairCache;
class btCollisionDispatcher;
class btConstraintSolver;
struct btCollisionAlgorithmCreateFunc;
class btDefaultCollisionConfiguration;

namespace GClasses
{
	class GSupervisedLearner;
}


class RagDoll
{
	enum
	{
		BODYPART_PELVIS = 0,
		BODYPART_SPINE,
		BODYPART_HEAD,

		BODYPART_LEFT_UPPER_LEG,
		BODYPART_LEFT_LOWER_LEG,

		BODYPART_RIGHT_UPPER_LEG,
		BODYPART_RIGHT_LOWER_LEG,

		BODYPART_LEFT_UPPER_ARM,
		BODYPART_LEFT_LOWER_ARM,

		BODYPART_RIGHT_UPPER_ARM,
		BODYPART_RIGHT_LOWER_ARM,

		BODYPART_LEFT_FOOT,
		BODYPART_RIGHT_FOOT,

		BODYPART_COUNT
	};

	enum
	{
		JOINT_SPINE_HEAD = 0,
		JOINT_PELVIS_SPINE,

		JOINT_LEFT_HIP,
		JOINT_LEFT_KNEE,

		JOINT_RIGHT_HIP,
		JOINT_RIGHT_KNEE,

		JOINT_LEFT_SHOULDER,
		JOINT_LEFT_ELBOW,

		JOINT_RIGHT_SHOULDER,
		JOINT_RIGHT_ELBOW,

		JOINT_LEFT_ANKLE,
		JOINT_RIGHT_ANKLE,

		JOINT_COUNT
	};

	double m_positiveStrength[JOINT_COUNT];
	double m_negativeStrength[JOINT_COUNT];

	btDynamicsWorld* m_ownerWorld;
	btCollisionShape* m_shapes[BODYPART_COUNT];
	btRigidBody* m_bodies[BODYPART_COUNT];
	btTypedConstraint* m_joints[JOINT_COUNT];

	btRigidBody* localCreateRigidBody (btScalar mass, const btTransform& startTransform, btCollisionShape* shape);

public:
	RagDoll(btDynamicsWorld* ownerWorld, const btVector3& positionOffset);
	virtual	~RagDoll();

	// Prints the angles of all the joints
	void PrintJoints();

	// Returns the height above the ground
	btScalar getLowestPoint();

	// Returns the sum of angular and linear velocities of the torso
	btScalar getMovingness();

	// Returns 1 if standing vertical, -1 if standing on head, 0 if laying flat, some value in-between for other positions
	btScalar getUprightness();

	// Returns 1 if laying on tummy, -1 if laying on back, 0 if laying on side, some value in-between for other positions
	btScalar getFaceDownness();

	void DoAction(GClasses::GSupervisedLearner* pPolicy, double time);

protected:
	void makeFeatureVector(double* pFeatures, double time);
};




class RagdollDemo : public GlutDemoApplication
{
	btAlignedObjectArray<class RagDoll*> m_ragdolls;

	//keep the collision shapes, for deletion/cleanup
	btAlignedObjectArray<btCollisionShape*>	m_collisionShapes;

	btBroadphaseInterface*	m_broadphase;

	btCollisionDispatcher*	m_dispatcher;

	btConstraintSolver*	m_solver;

	btDefaultCollisionConfiguration* m_collisionConfiguration;

	GClasses::GSupervisedLearner* m_pPolicy;

	double m_time;

public:
	RagdollDemo()
	: GlutDemoApplication(), m_pPolicy(NULL)
	{
		//myinit();
		initPhysics();
		m_time = 0;
	}

	virtual ~RagdollDemo()
	{
		exitPhysics();
	}

	virtual void onReset()
	{
		m_time = 0;
	}

	void SetPolicy(GClasses::GSupervisedLearner* pPolicy)
	{
		m_pPolicy = pPolicy;
	}

	void spawnRagdoll(const btVector3& startOffset);

	void advanceTime(float fTimeDelta);

	virtual void clientMoveAndDisplay();

	virtual void displayCallback();

	virtual void keyboardCallback(unsigned char key, int x, int y);

	RagDoll* getRagDoll(int i);

protected:
	void initPhysics();

	void exitPhysics();
};


#endif
