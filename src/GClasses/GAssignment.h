/*
	Copyright (C) 2012, Mike Gashler

	This library is free software; you can redistribute it and/or
	modify it under the terms of the GNU Lesser General Public
	License as published by the Free Software Foundation; either
	version 2.1 of the License, or (at your option) any later version.

	see http://www.gnu.org/copyleft/lesser.html
*/
///\file Defines an assignment between two sets and holds algorithms
///      for generating them.


#ifndef __GASSIGNMENT_H__
#define __GASSIGNMENT_H__

#include "GError.h"
#include "GHolders.h"
#include <cassert>

namespace GClasses {

class GMatrix;
class GDom;
class GDomNode;

///\brief An abstract base class defining an assignment between two sets
///
/// An assignment is a mapping M between two sets A and B.  If a and b
/// are elements of A and B respectively, then the following
/// properties hold:
///
/// - M(a) is a single element of B or is undefined
/// - M(b) is a single element of A or is undefined
/// - M(a) == b if and only if M(b) == a
/// 
/// More specifically, an assignment is a graph-theoretic matching on
/// a bipartite graph.
///
/// Practically, this is implemented by denoting the members of the
/// two sets with non-negative integers.  Undefined mappings are just
/// given as mapping to -1.  sizeA() and sizeB() give the number of
/// elements in each set.  The members of A are then denoted as
/// 0..sizeA()-1 and the members of B as the integers 0..sizeB()-1
///
/// operator()(int) gives the mapping from A->B (as in 
/// \code if(assignment(1)==-1){do_something();} \endcode )
///
/// inverse(int) gives the mapping from B->A (as in 
/// \code if(assignment.inverse(1)==-1){do_something();} \endcode )
class GAssignment{
public:
	///\brief Return the number of elements in the A set of this GAssignment
	///
	///See the class comment for more information.
	///
	///\return the number of elements in the A set of this GAssignment
	virtual std::size_t sizeA() const = 0;

	///\brief Return the number of elements in the B set of this GAssignment
	///
	///See the class comment for more information.
	///
	///\return the number of elements in the B set of this GAssignment
	virtual std::size_t sizeB() const = 0;

	///\brief Return the number of the member of set B corresponding to
	///       the \a memberOfA element in set A or -1 if there is no
	///       corresponding member
	///
	///\param memberOfA the number of the member of set A whose
	///                 assignment is desired
	///
	///\return the number of the member of set B corresponding to the \a
	///        memberOfA element in set A or -1 if there is no
	///        corresponding member
	virtual int operator()(unsigned memberOfA) const = 0;

	///\brief Return the number of the member of set A corresponding to
	///       the \a memberOfB element in set B or -1 if there is no
	///       corresponding member.
	///
	///\param memberOfB the number of the member of set B whose
	///                 assignment is desired
	///
	///\return the number of the member of set A corresponding to the \a
	///        memberOfB element in set B or -1 if there is no
	///        corresponding member.
	virtual int inverse(unsigned memberOfB) const = 0;

	///\brief A do-nothing destructor needed since there will be subclasses
	virtual ~GAssignment(){};
};

///\brief A simple concrete implementation of the GAssignment protocol
///\brief using std::vector<int>
class GSimpleAssignment:public GAssignment{
protected:
	///\brief aForB[i[ has the member of A assigned to a given member of
	///\brief B (or -1 when no member)
	std::vector<int> aForB;

	///\brief bForA[i] has the member of B assigned to a given member of
	///\brief A (or -1 when no member)
	std::vector<int> bForA;
public:
	///\brief Create an assignment between two sets of the given sizes
	///\brief that assigns everything to -1
	///
	///\param aSize the number of elements in set A (the first set in
	///             the assignment)
	///
	///\param bSize the number of elements in set B (the second set in
	///             the assignment)
	GSimpleAssignment(std::size_t aSize, std::size_t bSize):aForB(bSize, -1), 
																													bForA(aSize, -1){}

	///\brief Create an assignment between the two members, setting any
	///\brief previously corresponding members to unassigned.
	///
	///First breaks any assignment between memberOfA and any other
	///member and memberOfB and any other member, then makes a new
	///assignment between memberOfA and memberOfB
	///
	///\param memberOfA the number for the member of set A that will get
	///                 assigned to \a memberOfB
	///
	///\param memberOfB the number for the member of set B that will get
	///                 assigned to \a memberOfA
	virtual void assign(unsigned memberOfA, unsigned memberOfB){
		//Erase any old assignments
		unassignA(memberOfA);
		unassignB(memberOfB);

		//Create the new assignment
		bForA[memberOfA] = memberOfB;
		aForB[memberOfB] = memberOfA;
	}

	///\brief Remove any assignment for the given member of A
	///
	///Does nothing if A is unassigned.
	///
	///\param memberOfA The number for the member of set A which should
	///       be unassigned
	virtual void unassignA(unsigned memberOfA){
		int curBForA = (*this)(memberOfA);
		if(curBForA != -1){
			aForB[curBForA] = -1;
			bForA[memberOfA] = -1;
		}
		assert(bForA[memberOfA] == -1);
	}

	///\brief Remove any assignment for the given member of B
	///
	///Does nothing if B is unassigned.
	///
	///\param memberOfB The number for the member of set B which should
	///       be unassigned
	virtual void unassignB(unsigned memberOfB){
		int curAForB = inverse(memberOfB);
		if(curAForB != -1){
			bForA[curAForB] = -1;
			aForB[memberOfB] = -1;
		}
		assert(aForB[memberOfB] == -1);
	}

	///\brief Return the number of elements in the A set of this GAssignment
	///
	///See the class comment for more information.
	///
	///\return the number of elements in the A set of this GAssignment
	virtual std::size_t sizeA() const{ return bForA.size(); }

	///\brief Return the number of elements in the B set of this GAssignment
	///
	///See the class comment for more information.
	///
	///\return the number of elements in the B set of this GAssignment
	virtual std::size_t sizeB() const{ return aForB.size(); }

	///\brief Return the number of the member of set B corresponding to
	///       the \a memberOfA element in set A or -1 if there is no
	///       corresponding member
	///
	///\param memberOfA the number of the member of set A whose
	///                 assignment is desired
	///
	///\return the number of the member of set B corresponding to the \a
	///        memberOfA element in set A or -1 if there is no
	///        corresponding member
	virtual int operator()(unsigned memberOfA) const{ 
		return bForA.at(memberOfA); }

	///\brief Return the number of the member of set A corresponding to
	///       the \a memberOfB element in set B or -1 if there is no
	///       corresponding member.
	///
	///\param memberOfB the number of the member of set B whose
	///                 assignment is desired
	///
	///\return the number of the member of set A corresponding to the \a
	///        memberOfB element in set B or -1 if there is no
	///        corresponding member.
	virtual int inverse(unsigned memberOfB) const{ return aForB.at(memberOfB); }

	///\brief A do-nothing destructor needed since there may be subclasses
	virtual ~GSimpleAssignment(){};
};

///\brief Rectangular linear assignment problem solver by Volgenant and Jonker
///
///Solves the rectangular linar assignment problem:
///
///Given a non-negative cost matrix cost, create a binary assignment
///matrix x to minimize the cost of the assignments, that is, minimize
///sum cost[i][j]*x[i][j].  numRows <= numColumns
///
///The assignments are 1 if assigned, and zero if unassigned.  At most
///one row can be assigned to each column and vice versa.  That fixing
///k, sum x[i][k] <= 1 and sum x[k][j] == 1.
///
///The x matrix is encoded in rowAssign (and for convenience)
///colAssign.  rowAssign[row] is the column assigned to that row.
///colAssign[column] is the row assigned to that column or -1 if
///no row was assigned.
///
///rowPotential[row] and colPotential[column] give the dual potentials
///after the assignment.  The dual problem is:
///
///maximize ( sum rowPotential[row] ) + ( sum colPotential[column] )
///
///subject to: 
///
///cost[i][j]-rowPotential[i]-colPotential[j] >= 0 for all i,j
///
///rowPotential[row] >= 0
///
///colPotential[column] >= 0
///
///totalCost is the cost of the generated assignment
///
///For more information, see:
///
///"Linear and semi-assignment problems: a core-oriented approach",
///Computers in Operations Research, Vol. 23, No. 10, pp. 917-932,
///1996, Elsevier Science Ltd
///
///\param cost The cost matrix - must be non-negative and have at
///            least as many columns as rows
///
///\param rowAssign Output vector.  rowAssign[row] is the index of the
///                 column that was assigned to that row
///
///\param colAssign Output vector.  colAssign[column] is the index of
///                 the row that was assigned to that column, or -1 if
///                 no row was assigned.
///
///\param rowPotential Output vector.  The potential assigned to a
///                    given row in the solution to the dual problem
///                    (see main description).
///
///\param colPotential Output vector.  The potential assigned to a
///                    given column in the solution to the dual problem
///                    (see main description).
///
///\param totalCost Output double.  The total cost of the optimal
///                 assignment
///
///\param epsilon Threshold to use in floating point comparisons.  If
///               two values differ by epsilon, they are considered to
///               be equal
void LAPVJRCT(GMatrix cost, std::vector<int>& rowAssign, 
							std::vector<int>& colAssign, 
							std::vector<double>& rowPotential, 
							std::vector<double>& colPotential, double& totalCost,
							const double epsilon=1e-8);
	
}
#endif //  __GASSIGNMENT_H__
