/*
Bullet Continuous Collision Detection and Physics Library
Ragdoll Demo
Copyright (c) 2007 Starbreeze Studios

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it freely, 
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.

Originally Written by: Marten Svanfeldt
In March 2008, Mike Gashler made some alterations:
	added feet
	dropped second dummy
	changed some cone constraints to hinge constraints to simplify the model
*/


#include "btBulletDynamicsCommon.h"
#include "GlutStuff.h"
#include "GL_ShapeDrawer.h"

#include "LinearMath/btIDebugDraw.h"

#include "GLDebugDrawer.h"
#include "Body.h"
#include <GClasses/GError.h>
#include <GClasses/GLearner.h>
#include <GClasses/GBits.h>
#include <GClasses/GThread.h>
#include <GClasses/GVec.h>
#include "PolicyLearner.h"
#include <iostream>

using namespace GClasses;
using std::cout;

#ifndef M_PI
#define M_PI       3.14159265358979323846
#endif

#ifndef M_PI_2
#define M_PI_2     1.57079632679489661923
#endif

#ifndef M_PI_4
#define M_PI_4     0.785398163397448309616
#endif

btRigidBody* RagDoll::localCreateRigidBody (btScalar mass, const btTransform& startTransform, btCollisionShape* shape)
{
	bool isDynamic = (mass != 0.f);

	btVector3 localInertia(0,0,0);
	if (isDynamic)
		shape->calculateLocalInertia(mass,localInertia);

	btDefaultMotionState* myMotionState = new btDefaultMotionState(startTransform);
	
	btRigidBody::btRigidBodyConstructionInfo rbInfo(mass,myMotionState,shape,localInertia);
	btRigidBody* body = new btRigidBody(rbInfo);

	m_ownerWorld->addRigidBody(body);

	return body;
}

RagDoll::RagDoll(btDynamicsWorld* ownerWorld, const btVector3& positionOffset)
 : m_ownerWorld (ownerWorld)
{
	// Setup the geometry
	m_shapes[BODYPART_PELVIS] = new btCapsuleShape(btScalar(0.15), btScalar(0.20));
	m_shapes[BODYPART_SPINE] = new btCapsuleShape(btScalar(0.15), btScalar(0.28));
	m_shapes[BODYPART_HEAD] = new btCapsuleShape(btScalar(0.10), btScalar(0.05));
	m_shapes[BODYPART_LEFT_UPPER_LEG] = new btCapsuleShape(btScalar(0.07), btScalar(0.45));
	m_shapes[BODYPART_LEFT_LOWER_LEG] = new btCapsuleShape(btScalar(0.05), btScalar(0.37));
	m_shapes[BODYPART_RIGHT_UPPER_LEG] = new btCapsuleShape(btScalar(0.07), btScalar(0.45));
	m_shapes[BODYPART_RIGHT_LOWER_LEG] = new btCapsuleShape(btScalar(0.05), btScalar(0.37));
	m_shapes[BODYPART_LEFT_UPPER_ARM] = new btCapsuleShape(btScalar(0.05), btScalar(0.33));
	m_shapes[BODYPART_LEFT_LOWER_ARM] = new btCapsuleShape(btScalar(0.04), btScalar(0.25));
	m_shapes[BODYPART_RIGHT_UPPER_ARM] = new btCapsuleShape(btScalar(0.05), btScalar(0.33));
	m_shapes[BODYPART_RIGHT_LOWER_ARM] = new btCapsuleShape(btScalar(0.04), btScalar(0.25));
	m_shapes[BODYPART_LEFT_FOOT] = new btBoxShape(btVector3(0.06, 0.02, 0.14));
	m_shapes[BODYPART_RIGHT_FOOT] = new btBoxShape(btVector3(0.06, 0.02, 0.14));

	// Setup all the rigid bodies
	btTransform offset; offset.setIdentity();
	offset.setOrigin(positionOffset);

	btTransform transform;
	transform.setIdentity();
	transform.setOrigin(btVector3(btScalar(0.), btScalar(1.), btScalar(0.)));
	m_bodies[BODYPART_PELVIS] = localCreateRigidBody(btScalar(1.2)/*mass*/, offset*transform, m_shapes[BODYPART_PELVIS]);

	transform.setIdentity();
	transform.setOrigin(btVector3(btScalar(0.), btScalar(1.2), btScalar(0.)));
	m_bodies[BODYPART_SPINE] = localCreateRigidBody(btScalar(1.4)/*mass*/, offset*transform, m_shapes[BODYPART_SPINE]);

	transform.setIdentity();
	transform.setOrigin(btVector3(btScalar(0.), btScalar(1.6), btScalar(0.)));
	m_bodies[BODYPART_HEAD] = localCreateRigidBody(btScalar(0.8)/*mass*/, offset*transform, m_shapes[BODYPART_HEAD]);

	transform.setIdentity();
	transform.setOrigin(btVector3(btScalar(-0.18), btScalar(0.65), btScalar(0.)));
	m_bodies[BODYPART_LEFT_UPPER_LEG] = localCreateRigidBody(btScalar(1.1)/*mass*/, offset*transform, m_shapes[BODYPART_LEFT_UPPER_LEG]);

	transform.setIdentity();
	transform.setOrigin(btVector3(btScalar(-0.18), btScalar(0.2), btScalar(0.)));
	m_bodies[BODYPART_LEFT_LOWER_LEG] = localCreateRigidBody(btScalar(0.8)/*mass*/, offset*transform, m_shapes[BODYPART_LEFT_LOWER_LEG]);

	transform.setIdentity();
	transform.setOrigin(btVector3(btScalar(0.18), btScalar(0.65), btScalar(0.)));
	m_bodies[BODYPART_RIGHT_UPPER_LEG] = localCreateRigidBody(btScalar(1.1)/*mass*/, offset*transform, m_shapes[BODYPART_RIGHT_UPPER_LEG]);

	transform.setIdentity();
	transform.setOrigin(btVector3(btScalar(0.18), btScalar(0.2), btScalar(0.)));
	m_bodies[BODYPART_RIGHT_LOWER_LEG] = localCreateRigidBody(btScalar(0.8)/*mass*/, offset*transform, m_shapes[BODYPART_RIGHT_LOWER_LEG]);

	transform.setIdentity();
	transform.setOrigin(btVector3(btScalar(-0.18), btScalar(1.2), btScalar(0.)));
	//transform.getBasis().setEulerZYX(0,0,M_PI_2);
	m_bodies[BODYPART_LEFT_UPPER_ARM] = localCreateRigidBody(btScalar(0.7)/*mass*/, offset*transform, m_shapes[BODYPART_LEFT_UPPER_ARM]);

	transform.setIdentity();
	transform.setOrigin(btVector3(btScalar(-0.18), btScalar(0.7), btScalar(0.)));
	//transform.getBasis().setEulerZYX(0,0,M_PI_2);
	m_bodies[BODYPART_LEFT_LOWER_ARM] = localCreateRigidBody(btScalar(0.6)/*mass*/, offset*transform, m_shapes[BODYPART_LEFT_LOWER_ARM]);

	transform.setIdentity();
	transform.setOrigin(btVector3(btScalar(0.18), btScalar(1.2), btScalar(0.)));
	//transform.getBasis().setEulerZYX(0,0,-M_PI_2);
	m_bodies[BODYPART_RIGHT_UPPER_ARM] = localCreateRigidBody(btScalar(0.7)/*mass*/, offset*transform, m_shapes[BODYPART_RIGHT_UPPER_ARM]);

	transform.setIdentity();
	transform.setOrigin(btVector3(btScalar(0.18), btScalar(0.7), btScalar(0.)));
	//transform.getBasis().setEulerZYX(0,0,-M_PI_2);
	m_bodies[BODYPART_RIGHT_LOWER_ARM] = localCreateRigidBody(btScalar(0.6)/*mass*/, offset*transform, m_shapes[BODYPART_RIGHT_LOWER_ARM]);

	transform.setIdentity();
	transform.setOrigin(btVector3(btScalar(-0.18), btScalar(0.15), btScalar(0.)));
	m_bodies[BODYPART_LEFT_FOOT] = localCreateRigidBody(btScalar(0.3)/*mass*/, offset*transform, m_shapes[BODYPART_LEFT_FOOT]);

	transform.setIdentity();
	transform.setOrigin(btVector3(btScalar(0.18), btScalar(0.15), btScalar(0.)));
	m_bodies[BODYPART_RIGHT_FOOT] = localCreateRigidBody(btScalar(0.3)/*mass*/, offset*transform, m_shapes[BODYPART_RIGHT_FOOT]);

	// Setup some damping on the m_bodies
	for (int i = 0; i < BODYPART_COUNT; ++i)
	{
		m_bodies[i]->setDamping(0.05, 0.85);
		m_bodies[i]->setDeactivationTime(0.8);
		m_bodies[i]->setSleepingThresholds(0.8, 1.0); // when to stop simulating the object
	}

	// Now setup the constraints
	btHingeConstraint* hingeC;
	btConeTwistConstraint* coneC;

	btTransform localA, localB;

	// Head
	localA.setIdentity(); localB.setIdentity();
	localA.getBasis().setEulerZYX(0,0,M_PI_2); localA.setOrigin(btVector3(btScalar(0.), btScalar(0.30), btScalar(0.)));
	//localB.getBasis().setEulerZYX(0,0,M_PI_2); localB.setOrigin(btVector3(btScalar(0.), btScalar(-0.14), btScalar(0.)));
	localB.getBasis().setEulerZYX(0,0,-M_PI_2); localB.setOrigin(btVector3(btScalar(0.), btScalar(-0.14), btScalar(0.)));
	coneC = new btConeTwistConstraint(*m_bodies[BODYPART_SPINE], *m_bodies[BODYPART_HEAD], localA, localB);
	coneC->setLimit(M_PI_4, M_PI_4, M_PI_2);
	m_joints[JOINT_SPINE_HEAD] = coneC;
	m_ownerWorld->addConstraint(m_joints[JOINT_SPINE_HEAD], true);

	// Spine (0=straight, positive=natural, negative=awkward)
	localA.setIdentity(); localB.setIdentity();
	localA.getBasis().setEulerZYX(0,M_PI_2,0); localA.setOrigin(btVector3(btScalar(0.), btScalar(0.15), btScalar(0.)));
	localB.getBasis().setEulerZYX(0,M_PI_2,0); localB.setOrigin(btVector3(btScalar(0.), btScalar(-0.15), btScalar(0.)));
	hingeC = new btHingeConstraint(*m_bodies[BODYPART_PELVIS], *m_bodies[BODYPART_SPINE], localA, localB);
	//hingeC->setLimit(btScalar(-M_PI_4), btScalar(M_PI_2));
	hingeC->setLimit(btScalar(-M_PI) / 8/*leaning back a little*/, btScalar(M_PI_4)/*sitting or bending forward*/);
	m_joints[JOINT_PELVIS_SPINE] = hingeC;
	m_ownerWorld->addConstraint(m_joints[JOINT_PELVIS_SPINE], true);
	m_positiveStrength[0] = 1.0;
	m_negativeStrength[0] = 0.8;

	// Left Hip (0=straignt, negative=natural, positive=awkward)
	localA.setIdentity(); localB.setIdentity();
	localA.getBasis().setEulerZYX(0,M_PI_2,0); localA.setOrigin(btVector3(btScalar(-0.16), btScalar(-0.10), btScalar(0.)));
	localB.getBasis().setEulerZYX(0,M_PI_2,0); localB.setOrigin(btVector3(btScalar(0.), btScalar(0.225), btScalar(0.)));
	hingeC = new btHingeConstraint(*m_bodies[BODYPART_PELVIS], *m_bodies[BODYPART_LEFT_UPPER_LEG], localA, localB);
	hingeC->setLimit(btScalar(-M_PI_2)/*upper leg points forward*/, btScalar(M_PI) / 8/*upper leg points down and back a little*/);
	m_joints[JOINT_LEFT_HIP] = hingeC;
	m_ownerWorld->addConstraint(m_joints[JOINT_LEFT_HIP], true);
	m_positiveStrength[1] = 0.5;
	m_negativeStrength[1] = 1.0;

	// Left Knee (0=straight, positive=natural, negative=awkward)
	localA.setIdentity(); localB.setIdentity();
	localA.getBasis().setEulerZYX(0,M_PI_2,0); localA.setOrigin(btVector3(btScalar(0.), btScalar(-0.225), btScalar(0.)));
	localB.getBasis().setEulerZYX(0,M_PI_2,0); localB.setOrigin(btVector3(btScalar(0.), btScalar(0.185), btScalar(0.)));
	hingeC =  new btHingeConstraint(*m_bodies[BODYPART_LEFT_UPPER_LEG], *m_bodies[BODYPART_LEFT_LOWER_LEG], localA, localB);
	hingeC->setLimit(btScalar(0)/*straight down*/, btScalar(M_PI_2)/*straight back*/);
//hingeC->enableAngularMotor(true, -1.0, 0.01);
	m_joints[JOINT_LEFT_KNEE] = hingeC;
	m_ownerWorld->addConstraint(m_joints[JOINT_LEFT_KNEE], true);
	m_positiveStrength[2] = 0.7;
	m_negativeStrength[2] = 1.0;

	// Right Hip (0=straignt, negative=natural, positive=awkward)
	localA.setIdentity(); localB.setIdentity();
	localA.getBasis().setEulerZYX(0,M_PI_2,0); localA.setOrigin(btVector3(btScalar(0.16), btScalar(-0.10), btScalar(0.)));
	localB.getBasis().setEulerZYX(0,M_PI_2,0); localB.setOrigin(btVector3(btScalar(0.), btScalar(0.225), btScalar(0.)));
	hingeC = new btHingeConstraint(*m_bodies[BODYPART_PELVIS], *m_bodies[BODYPART_RIGHT_UPPER_LEG], localA, localB);
	hingeC->setLimit(btScalar(-M_PI_2)/*upper leg points forward*/, btScalar(M_PI) / 8/*upper leg points down and back a little*/);
	m_joints[JOINT_RIGHT_HIP] = hingeC;
	m_ownerWorld->addConstraint(m_joints[JOINT_RIGHT_HIP], true);
	m_positiveStrength[3] = 0.5;
	m_negativeStrength[3] = 1.0;

	// Right Knee (0=straight, positive=natural, negative=awkward)
	localA.setIdentity(); localB.setIdentity();
	localA.getBasis().setEulerZYX(0,M_PI_2,0); localA.setOrigin(btVector3(btScalar(0.), btScalar(-0.225), btScalar(0.)));
	localB.getBasis().setEulerZYX(0,M_PI_2,0); localB.setOrigin(btVector3(btScalar(0.), btScalar(0.185), btScalar(0.)));
	hingeC =  new btHingeConstraint(*m_bodies[BODYPART_RIGHT_UPPER_LEG], *m_bodies[BODYPART_RIGHT_LOWER_LEG], localA, localB);
	hingeC->setLimit(btScalar(0)/*straight down*/, btScalar(M_PI_2)/*straight back*/);
	m_joints[JOINT_RIGHT_KNEE] = hingeC;
	m_ownerWorld->addConstraint(m_joints[JOINT_RIGHT_KNEE], true);
	m_positiveStrength[4] = 0.7;
	m_negativeStrength[4] = 1.0;

	// Left shoulder (0=straight down, negative=natural, positive=awkward)
	localA.setIdentity(); localB.setIdentity();
	localA.getBasis().setEulerZYX(M_PI / 16,M_PI_2,0); localA.setOrigin(btVector3(btScalar(-0.2), btScalar(0.15), btScalar(0.)));
	localB.getBasis().setEulerZYX(0,M_PI_2,0); localB.setOrigin(btVector3(btScalar(0.), btScalar(0.18), btScalar(0.)));
	hingeC = new btHingeConstraint(*m_bodies[BODYPART_SPINE], *m_bodies[BODYPART_LEFT_UPPER_ARM], localA, localB);
	hingeC->setLimit(btScalar(-M_PI_2)/*straight forward*/, btScalar(M_PI) / 8/*back a little*/);
	m_joints[JOINT_LEFT_SHOULDER] = hingeC;
	m_ownerWorld->addConstraint(m_joints[JOINT_LEFT_SHOULDER], true);
	m_positiveStrength[5] = 1.0;
	m_negativeStrength[5] = 1.0;

	// Left elbow (0=straight, negative=natural, positive=awkward)
	localA.setIdentity(); localB.setIdentity();
	localA.getBasis().setEulerZYX(0,M_PI_2,0); localA.setOrigin(btVector3(btScalar(0.), btScalar(-0.18), btScalar(0.)));
	localB.getBasis().setEulerZYX(0,M_PI_2,0); localB.setOrigin(btVector3(btScalar(0.), btScalar(0.14), btScalar(0.)));
	hingeC =  new btHingeConstraint(*m_bodies[BODYPART_LEFT_UPPER_ARM], *m_bodies[BODYPART_LEFT_LOWER_ARM], localA, localB);
	hingeC->setLimit(btScalar(-M_PI_4 * 3)/*up a little*/, btScalar(0)/*straight down*/);
	m_joints[JOINT_LEFT_ELBOW] = hingeC;
	m_ownerWorld->addConstraint(m_joints[JOINT_LEFT_ELBOW], true);
	m_positiveStrength[6] = 1.0;
	m_negativeStrength[6] = 1.0;

	// Right shoulder (0=straight down, negative=natural, positive=awkward)
	localA.setIdentity(); localB.setIdentity();
	localA.getBasis().setEulerZYX(-M_PI / 16,M_PI_2,0); localA.setOrigin(btVector3(btScalar(0.2), btScalar(0.15), btScalar(0.)));
	localB.getBasis().setEulerZYX(0,M_PI_2,0); localB.setOrigin(btVector3(btScalar(0.), btScalar(0.18), btScalar(0.)));
	hingeC =  new btHingeConstraint(*m_bodies[BODYPART_SPINE], *m_bodies[BODYPART_RIGHT_UPPER_ARM], localA, localB);
	hingeC->setLimit(btScalar(-M_PI_2)/*straight forward*/, btScalar(M_PI) / 8/*back a little*/);
	m_joints[JOINT_RIGHT_SHOULDER] = hingeC;
	m_ownerWorld->addConstraint(m_joints[JOINT_RIGHT_SHOULDER], true);
	m_positiveStrength[7] = 1.0;
	m_negativeStrength[7] = 1.0;

	// Right elbow (0=straight, negative=natural, positive=awkward)
	localA.setIdentity(); localB.setIdentity();
	localA.getBasis().setEulerZYX(0,M_PI_2,0); localA.setOrigin(btVector3(btScalar(0.), btScalar(-0.18), btScalar(0.)));
	localB.getBasis().setEulerZYX(0,M_PI_2,0); localB.setOrigin(btVector3(btScalar(0.), btScalar(0.14), btScalar(0.)));
	hingeC =  new btHingeConstraint(*m_bodies[BODYPART_RIGHT_UPPER_ARM], *m_bodies[BODYPART_RIGHT_LOWER_ARM], localA, localB);
	hingeC->setLimit(btScalar(-M_PI_4 * 3)/*up a little*/, btScalar(0)/*straight down*/);
	m_joints[JOINT_RIGHT_ELBOW] = hingeC;
	m_ownerWorld->addConstraint(m_joints[JOINT_RIGHT_ELBOW], true);
	m_positiveStrength[8] = 1.0;
	m_negativeStrength[8] = 1.0;

	// Left foot (0=straight, negative=up, positive=down)
	localA.setIdentity(); localB.setIdentity();
	localA.getBasis().setEulerZYX(0,M_PI_2,0); localA.setOrigin(btVector3(btScalar(0.), btScalar(-0.215), btScalar(0.)));
	localB.getBasis().setEulerZYX(0,M_PI_2,0); localB.setOrigin(btVector3(btScalar(0.), btScalar(0.02), btScalar(0.08)));
	hingeC =  new btHingeConstraint(*m_bodies[BODYPART_LEFT_LOWER_LEG], *m_bodies[BODYPART_LEFT_FOOT], localA, localB);
	hingeC->setLimit(btScalar(-1.0)/*toes up*/, btScalar(0.6)/*toes pointed*/);
	m_joints[JOINT_LEFT_ANKLE] = hingeC;
	m_ownerWorld->addConstraint(m_joints[JOINT_LEFT_ANKLE], true);
	m_positiveStrength[9] = 1.0;
	m_negativeStrength[9] = 0.2;

	// Right foot (0=straight, negative=up, positive=down)
	localA.setIdentity(); localB.setIdentity();
	localA.getBasis().setEulerZYX(0,M_PI_2,0); localA.setOrigin(btVector3(btScalar(0.), btScalar(-0.215), btScalar(0.)));
	localB.getBasis().setEulerZYX(0,M_PI_2,0); localB.setOrigin(btVector3(btScalar(0.), btScalar(0.02), btScalar(0.08)));
	hingeC =  new btHingeConstraint(*m_bodies[BODYPART_RIGHT_LOWER_LEG], *m_bodies[BODYPART_RIGHT_FOOT], localA, localB);
	hingeC->setLimit(btScalar(-1.0)/*toes up*/, btScalar(0.6));
	m_joints[JOINT_RIGHT_ANKLE] = hingeC;
	m_ownerWorld->addConstraint(m_joints[JOINT_RIGHT_ANKLE], true);
	m_positiveStrength[10] = 1.0;
	m_negativeStrength[10] = 0.2;
}

// virtual
RagDoll::~RagDoll ()
{
	int i;

	// Remove all constraints
	for ( i = 0; i < JOINT_COUNT; ++i)
	{
		m_ownerWorld->removeConstraint(m_joints[i]);
		delete m_joints[i]; m_joints[i] = 0;
	}

	// Remove all bodies and shapes
	for ( i = 0; i < BODYPART_COUNT; ++i)
	{
		m_ownerWorld->removeRigidBody(m_bodies[i]);
	
		delete m_bodies[i]->getMotionState();

		delete m_bodies[i]; m_bodies[i] = 0;
		delete m_shapes[i]; m_shapes[i] = 0;
	}
}

void RagDoll::PrintJoints()
{
	printf("Joints:\n");
	printf("JOINT_PELVIS_SPINE=%g\n", ((btHingeConstraint*)m_joints[JOINT_PELVIS_SPINE])->getHingeAngle());
	printf("JOINT_LEFT_HIP=%g\n", ((btHingeConstraint*)m_joints[JOINT_LEFT_HIP])->getHingeAngle());
	printf("JOINT_LEFT_KNEE=%g\n", ((btHingeConstraint*)m_joints[JOINT_LEFT_KNEE])->getHingeAngle());
	printf("JOINT_RIGHT_HIP=%g\n", ((btHingeConstraint*)m_joints[JOINT_RIGHT_HIP])->getHingeAngle());
	printf("JOINT_RIGHT_KNEE=%g\n", ((btHingeConstraint*)m_joints[JOINT_RIGHT_KNEE])->getHingeAngle());
	printf("JOINT_LEFT_SHOULDER=%g\n", ((btHingeConstraint*)m_joints[JOINT_LEFT_SHOULDER])->getHingeAngle());
	printf("JOINT_LEFT_ELBOW=%g\n", ((btHingeConstraint*)m_joints[JOINT_LEFT_ELBOW])->getHingeAngle());
	printf("JOINT_RIGHT_SHOULDER=%g\n", ((btHingeConstraint*)m_joints[JOINT_RIGHT_SHOULDER])->getHingeAngle());
	printf("JOINT_RIGHT_ELBOW=%g\n", ((btHingeConstraint*)m_joints[JOINT_RIGHT_ELBOW])->getHingeAngle());
	printf("JOINT_LEFT_ANKLE=%g\n", ((btHingeConstraint*)m_joints[JOINT_LEFT_ANKLE])->getHingeAngle());
	printf("JOINT_RIGHT_ANKLE=%g\n", ((btHingeConstraint*)m_joints[JOINT_RIGHT_ANKLE])->getHingeAngle());
}

btScalar RagDoll::getLowestPoint()
{
	btVector3 vMin, vMax;
	m_shapes[BODYPART_HEAD]->getAabb(m_bodies[BODYPART_HEAD]->getWorldTransform(), vMin, vMax); // (Aabb = axis aligned bounding box)
	btScalar y = vMin.getY();
	m_shapes[BODYPART_LEFT_FOOT]->getAabb(m_bodies[BODYPART_LEFT_FOOT]->getWorldTransform(), vMin, vMax);
	y = std::min(y, vMin.getY());
	m_shapes[BODYPART_RIGHT_FOOT]->getAabb(m_bodies[BODYPART_RIGHT_FOOT]->getWorldTransform(), vMin, vMax);
	y = std::min(y, vMin.getY());
	m_shapes[BODYPART_LEFT_UPPER_ARM]->getAabb(m_bodies[BODYPART_LEFT_UPPER_ARM]->getWorldTransform(), vMin, vMax);
	y = std::min(y, vMin.getY());
	m_shapes[BODYPART_RIGHT_UPPER_ARM]->getAabb(m_bodies[BODYPART_RIGHT_UPPER_ARM]->getWorldTransform(), vMin, vMax);
	y = std::min(y, vMin.getY());
	return y;
}

btScalar RagDoll::getMovingness()
{
	return m_bodies[BODYPART_SPINE]->getLinearVelocity().length2() + m_bodies[BODYPART_SPINE]->getAngularVelocity().length2();
}

btScalar RagDoll::getUprightness()
{
	/*
	btScalar yaw, pitch, roll;
	m_bodies[BODYPART_SPINE]->getWorldTransform().getBasis().getEuler(yaw, pitch, roll);
	return pitch / (-M_PI_2);
	*/
	btVector3 v(0, 1, 0);
	return m_bodies[BODYPART_SPINE]->getWorldTransform().getBasis().tdoty(v);
}

btScalar RagDoll::getFaceDownness()
{
	btVector3 v(0, 1, 0);
	return m_bodies[BODYPART_SPINE]->getWorldTransform().getBasis().tdotz(v);
}

void RagDoll::makeFeatureVector(double* pFeatures, double time)
{
	GAssert(LABEL_DIMS == JOINT_COUNT - 1); // We control all of the joints except for the neck
	GAssert(FEATURE_DIMS == 3); // Unexpected value
	pFeatures[0] = time;
	pFeatures[1] = getFaceDownness();
	pFeatures[2] = getUprightness();
	//cout << "time=" << pPat[0] << ", facedownness=" << pPat[1] << ", uprightness=" << pPat[2] << "\n";
}

void RagDoll::DoAction(GSupervisedLearner* pPolicy, double time)
{
	// Make a feature vector
	GTEMPBUF(double, pPat, FEATURE_DIMS + LABEL_DIMS);
	makeFeatureVector(pPat, time);

	// Decide how to move the joints
	int i;
	if(pPolicy)
	{
		// Use the policy
		GAssert(pPolicy->featureDims() == FEATURE_DIMS);
		GAssert(pPolicy->labelDims() == LABEL_DIMS);
		pPolicy->predict(pPat, pPat + FEATURE_DIMS);
	}
	else
	{
		// Use the hard-coded manual policy
		manualPolicy(pPat, pPat + FEATURE_DIMS);
	}

	// Apply impulse to the joints
	btScalar velocity, impulse;
	for(i = 0; i < JOINT_COUNT - 1; i++)
	{
		velocity = 8.0 * GBits::sign(pPat[FEATURE_DIMS + i]);
		if(pPat[FEATURE_DIMS + i] < 0)
			impulse = std::min(m_negativeStrength[i], -pPat[FEATURE_DIMS + i]);
		else
			impulse = std::min(m_positiveStrength[i], pPat[FEATURE_DIMS + i]);
		((btHingeConstraint*)m_joints[i + 1])->enableAngularMotor(true, velocity, impulse);
	}
}



void RagdollDemo::initPhysics()
{
	//setTexturing(true);
	//setShadows(true);


	// Setup the basic world
#ifndef NOGUI
	setCameraDistance(btScalar(5.));
#endif
	m_collisionConfiguration = new btDefaultCollisionConfiguration();

	m_dispatcher = new btCollisionDispatcher(m_collisionConfiguration);

	btVector3 worldAabbMin(-10000,-10000,-10000); // (Aabb = axis aligned bounding box)
	btVector3 worldAabbMax(10000,10000,10000);
	m_broadphase = new btAxisSweep3 (worldAabbMin, worldAabbMax);

	m_solver = new btSequentialImpulseConstraintSolver;

	m_dynamicsWorld = new btDiscreteDynamicsWorld(m_dispatcher,m_broadphase,m_solver,m_collisionConfiguration);


	// Setup a big ground box
	{
		btCollisionShape* groundShape = new btBoxShape(btVector3(btScalar(200.),btScalar(10.),btScalar(200.)));
		m_collisionShapes.push_back(groundShape);
		btTransform groundTransform;
		groundTransform.setIdentity();
		groundTransform.setOrigin(btVector3(0,-10,0));

#define CREATE_GROUND_COLLISION_OBJECT 1
#ifdef CREATE_GROUND_COLLISION_OBJECT
		btCollisionObject* fixedGround = new btCollisionObject();
		fixedGround->setCollisionShape(groundShape);
		fixedGround->setWorldTransform(groundTransform);
		m_dynamicsWorld->addCollisionObject(fixedGround);
#else
		localCreateRigidBody(btScalar(0.),groundTransform,groundShape);
#endif //CREATE_GROUND_COLLISION_OBJECT

	}

	// Spawn one ragdoll
	btVector3 startOffset(0,0,0);
	spawnRagdoll(startOffset);

#ifndef NOGUI
	clientResetScene();
#endif
}

void RagdollDemo::spawnRagdoll(const btVector3& startOffset)
{
	RagDoll* ragDoll = new RagDoll (m_dynamicsWorld, startOffset);
	m_ragdolls.push_back(ragDoll);
}	

// This method is called when not in display mode
void RagdollDemo::advanceTime(float fTimeDelta)
{
	//if(m_time > 1)
	//	GThread::sleep(200); // slow-motion
	//printf("Time: %g, Lowest point: %g, Uprightness: %g, FaceDownness: %g\n", m_time, getRagDoll(0)->getLowestPoint(), getRagDoll(0)->getUprightness(), getRagDoll(0)->getFaceDownness());
	//getRagDoll(0)->PrintJoints();
	if (m_dynamicsWorld)
	{
		m_ragdolls[0]->DoAction(m_pPolicy, m_time);
		getDynamicsWorld()->stepSimulation(fTimeDelta);
		m_time += fTimeDelta;
	}
}

extern void APIENTRY glutSwapBuffers(void);


// This method is called when in display mode
void RagdollDemo::clientMoveAndDisplay()
{
#ifndef NOGUI
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); 
#endif

	//simple dynamics world doesn't handle fixed-time-stepping
	float ms = getDeltaTimeMicroseconds();

	float minFPS = 1000000.f/60.f;
	if (ms > minFPS)
		ms = minFPS;

	if (m_dynamicsWorld)
	{
		float fTimeDelta = ms / 1000000.f;
		m_ragdolls[0]->DoAction(m_pPolicy, m_time);
		m_dynamicsWorld->stepSimulation(fTimeDelta);
		m_time += fTimeDelta;

		//optional but useful: debug drawing
		//m_dynamicsWorld->debugDrawWorld();
	}

#ifndef NOGUI
	renderme();

	glFlush();

	glutSwapBuffers();
#endif
}

void RagdollDemo::displayCallback()
{
#ifndef NOGUI
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); 

	renderme();

	glFlush();
	glutSwapBuffers();
#endif
}

void RagdollDemo::keyboardCallback(unsigned char key, int x, int y)
{
#ifndef NOGUI
	switch (key)
	{
		case 'e':
			{
				btVector3 startOffset(0,2,0);
				spawnRagdoll(startOffset);
				break;
			}
		default:
			DemoApplication::keyboardCallback(key, x, y);
	}
#endif
}

RagDoll* RagdollDemo::getRagDoll(int i)
{
	return m_ragdolls[i];
}

void	RagdollDemo::exitPhysics()
{

	int i;

	for (i=0;i<m_ragdolls.size();i++)
	{
		RagDoll* doll = m_ragdolls[i];
		delete doll;
	}

	//cleanup in the reverse order of creation/initialization

	//remove the rigidbodies from the dynamics world and delete them
	
	for (i=m_dynamicsWorld->getNumCollisionObjects()-1; i>=0 ;i--)
	{
		btCollisionObject* obj = m_dynamicsWorld->getCollisionObjectArray()[i];
		btRigidBody* body = btRigidBody::upcast(obj);
		if (body && body->getMotionState())
		{
			delete body->getMotionState();
		}
		m_dynamicsWorld->removeCollisionObject( obj );
		delete obj;
	}

	//delete collision shapes
	for (int j=0;j<m_collisionShapes.size();j++)
	{
		btCollisionShape* shape = m_collisionShapes[j];
		delete shape;
	}

	//delete dynamics world
	delete m_dynamicsWorld;

	//delete solver
	delete m_solver;

	//delete broadphase
	delete m_broadphase;

	//delete dispatcher
	delete m_dispatcher;

	delete m_collisionConfiguration;
}




