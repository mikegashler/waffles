/*
  The contents of this file are dedicated by all of its authors, including

    Michael S. Gashler,
    Eric Moyer,
    anonymous contributors,

  to the public domain (http://creativecommons.org/publicdomain/zero/1.0/).

  Note that some moral obligations still exist in the absence of legal ones.
  For example, it would still be dishonest to deliberately misrepresent the
  origin of a work. Although we impose no legal requirements to obtain a
  license, it is beseeming for those who build on the works of others to
  give back useful improvements, or find a way to pay it forward. If
  you would like to cite us, a published paper about Waffles can be found
  at http://jmlr.org/papers/volume12/gashler11a/gashler11a.pdf. If you find
  our code to be useful, the Waffles team would love to hear how you use it.
*/

///\file
///\brief Defines an assignment between two sets and holds algorithms
///       for generating them.

#ifndef __GASSIGNMENT_H__
#define __GASSIGNMENT_H__

#include "GError.h"
#include "GHolders.h"
#include <cassert>
#include <algorithm>

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
///using std::vector<int>
///
///\see linearAssignment(GMatrix, ShouldMinimize, const double)
///\see linearAssignment(GMatrix, ShouldMaximize, const double)
///\see linearAssignmentBruteForce(GMatrix, ShouldMinimize)
///\see linearAssignmentBruteForce(GMatrix, ShouldMaximize)
class GSimpleAssignment:public GAssignment{
protected:
	///\brief aForB[i] has the member of A assigned to a given member of
	///B (or -1 when no member)
	std::vector<int> aForB;

	///\brief bForA[i] has the member of B assigned to a given member of
	///A (or -1 when no member)
	std::vector<int> bForA;
public:
	///\brief Create an assignment between two sets of the given sizes
	///that assigns everything to -1
	///
	///\param aSize the number of elements in set A (the first set in
	///             the assignment)
	///
	///\param bSize the number of elements in set B (the second set in
	///             the assignment)
	GSimpleAssignment(std::size_t aSize, std::size_t bSize):aForB(bSize, -1), 
																													bForA(aSize, -1){}
	
	///\brief Create an assignment between two sets of the given sizes
	///that assigns everything according to the array \a bForA
	///
	///Just like writing:
	///\code
	///GSimpleAssignment foo(bForA.size(), \a bSize);
	///foo.setBForA(bForA);
	///\endcode
	///
	///\param bSize the number of elements in set B (the second set in
	///             the assignment)
	///
	///\param bForA bForA[i] gives the assignment for element i in set
	///             A.  It is the index of the corresponding element of
	///             set B or -1 if there is no corresponding element.
	///
	///\see GSimpleAssignment(std::size_t, std::size_t)
	///\see bForA(const std::vector<int>&)
	GSimpleAssignment(std::size_t bSize, const std::vector<int>& bfA)
		:aForB(bSize, -1), bForA(bfA){
		for(std::size_t i = 0; i < bForA.size(); ++i){
			int match = bForA.at(i);
			if(match >= 0) { aForB.at(match) = (int)i; }
		}
	}

	///\brief Create an assignment between the two members, setting any
	///previously corresponding members to unassigned.
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

	///\brief Set the assignment to the one expressed in the vector \a bForA
	///
	///Given a vector \a bForA for which bForA[i] is the index of the
	///member of set B that corresponds to the i-th member of set A or
	///is -1 if there is no corresponding element, make the assignments
	///in this object identical.
	///
	///It is assumed that this operation will not change the sizes of
	///the sets referred to by this assignment.  That is, -1 <= bForA[i]
	///and bForA[i] < sizeB() and bForA.size() == sizeA()
	///
	///\param bForA bForA[i] gives the assignment for element i in set
	///             A.  It is the index of the corresponding element of
	///             set B or -1 if there is no corresponding element.
	virtual void setBForA(const std::vector<int>& bfA){
		assert(bfA.size() == this->bForA.size());
		this->bForA = bfA; 
		std::fill(aForB.begin(), aForB.end(), -1);
		for(std::size_t i = 0; i < bfA.size(); ++i){
			int match = bfA.at(i);
			if(match >= 0) { aForB.at(match) = (int)i; }
		}
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

	///\brief Return true if *this and other represent the same
	///       assignments among the same sized sets
	///
	///If A is the same size as set A for other and B is the same size
	///as B for other and the members with identical indices are
	///assigned to one another and the unassigned members also have
	///identical indices, then the two assignment objects are equal.
	///
	///In other words, returns true if and only if sizeA(), sizeB(),
	///operator()(unsigned) and inverse()(unsigned) give identical
	///results on both *this and other for all valid input values.
	///
	///\param other the GSimpleAssignment object to compare to this
	///
	///\return true if the assignments are the same, false otherwise.
	///        See main description for details.
	virtual bool operator==(const GSimpleAssignment& other) const{
		return 
			sizeA() == other.sizeA() && sizeB() == other.sizeB()
			&&
			std::equal(aForB.begin(),aForB.end(),other.aForB.begin()) 
			&&
			std::equal(bForA.begin(),bForA.end(),other.bForA.begin());
	}

	///\brief Return true iff *this, expressed as the input to setBForA
	/// with sizeB prepended is lexiographically less than other
	///
	///\param other the GSimpleAssignment being compared to this one
	///
	///\return true iff *this, expressed as the input to setBForA with
	///sizeB prepended is lexiographically less than other
	virtual bool operator<(const GSimpleAssignment& other) const{
		if(sizeB() < other.sizeB()){ 
			return true;
		}else if(other.sizeB() < sizeB()){
			return false;
		}else{ //sizeB() == other.sizeB()
			return std::lexicographical_compare
				(bForA.begin(), bForA.end(),
				 other.bForA.begin(), other.bForA.end());
		}
	}

	///\brief Swaps the A and B set.  The assignments stay the same.
	///
	///If A element i was assigned to B element j before the swap, A
	///element j will be asigned to B element i after the swap.
	virtual void swapAAndB(){
		std::vector<int> tmp = aForB;
		aForB = bForA;
		bForA = tmp;
	}

	///\brief A do-nothing destructor needed since there may be subclasses
	virtual ~GSimpleAssignment(){};


	///\brief Run unit tests for GSimpleAssignment - throws an exception
	///if an error is found.
	static void test();
};

///\brief Return the cost of the assignment \a assign for the matrix
///\a costs
///
///costs[i][j] is the cost of assigning element i from set A to
///element j from set B.  This retuns the sum of the costs of making
///the assignments given in assign.  Non-assigned members have zero
///costs.
///
///\param assign The assignment whose cost will be computed
///
///\param costs costs[i][j] is the cost of assigning element i from
///             set A to element j from set B.
double cost(const GSimpleAssignment& assign, const GMatrix& costs);

///\brief Print assignment \a gsa to \a out in a human-readable form
///
///The GSimpleAssignment \a gsa is written to out as a sequence of pairs
///{a,b} where a is an index in set A and b is an index into set B.
///Any indices that are not listed are unassigned.
///
///Thus the assignment from the bForA array [3,2,-1,9] would be output
///as [{0,3},{1,2},{3,9}]
///
///\param out the stream on which the assignment will be printed
///
///\param gsa the GSimpleAssignment object to print
///
///\return out after gsa has been written to it
std::ostream& operator<<(std::ostream& out, const GSimpleAssignment& gsa);

///\brief Tag class to indicate that the linearAssignment routine
///should maximize the cost of the assignment
class ShouldMaximize{};

///\brief Tag class to indicate that the linearAssignment routine
///should minimize the cost of the assignment
class ShouldMinimize{};

///\brief Return a GSimpleAssignment that minimizes the cost of the assignment
///
///costs[i][j] is the cost of assigning element i in set A to element
///j in set B.  The GSimpleAssignment object returned minimizes the sum
///of the costs for all assignments made.  All elements in the smaller
///set will be assigned a corresponding element in the larger set.
///
///\param costs costs[i][j] is the cost of assigning element i in set
///             A to element j in set B
///
///\param sm a tag parameter to differentiate this linearAssignment
///          function from the version that maximizes the benefit of the
///          assignment.  Just pass ShouldMinimize()
///
///\param epsilon fudge factor for floating-point equality comparisons
///               after multiple additions and subtractions.  If two
///               values lie within this distance of one another, they
///               are considered equal.  Make sure that the costs in
///               your cost matrix that are truely non-equal have
///               distances much greater than epsilon.
///
///\return a Simple assignment object that minimizes the cost of the
///        assignment.
///
///\see linearAssignment(GMatrix, ShouldMaximize, const double)
///
///\see LAPVJRCT
GSimpleAssignment linearAssignment(GMatrix costs,	
																	 ShouldMinimize sm=ShouldMinimize(), 
																	 const double epsilon = 1e-8);


///\brief Return a GSimpleAssignment that maximizes the benefit of the
///assignment
///
///benefits[i][j] is the benefit of assigning element i in set A to element
///j in set B.  The GSimpleAssignment object returned maximizes the sum
///of the benefits for all assignments made.  All elements in the smaller
///set will be assigned a corresponding element in the larger set.
///
///\param benefits benefits[i][j] is the benefit of assigning element i in set
///             A to element j in set B
///
///\param sm a tag parameter to differentiate this linearAssignment
///          function from the version that minimizes the cost of the
///          assignment.  Just pass ShouldMaximize()
///
///\param epsilon fudge factor for floating-point equality comparisons
///               after multiple additions and subtractions.  If two
///               values lie within this distance of one another, they
///               are considered equal.  Make sure that the costs in
///               your cost matrix that are truely non-equal have
///               distances much greater than epsilon.
///
///\return a Simple assignment object that maximizes the cost of the
///        assignment.
///
///\see linearAssignment(GMatrix, ShouldMinimize, const double)
///
///\see LAPVJRCT
GSimpleAssignment linearAssignment(GMatrix benefits,	
																	 ShouldMaximize sm, 
																	 const double epsilon = 1e-8);

///\brief Solve a linear assignment minimization problem by the VERY SLOW brute
///force method
///
///This method is mainly here for testing.
///
///Finds all simple assignments that minimize the cost matrix.  (For
///certain degenerate cases (like the matrix where all costs are the
///same), this can be a large number of assignments, be careful).
///Takes time n choose m where n is the larger matrix dimension and m
///is the smaller.  Always assigns one element to each of the smaller
///dimensions of the matrix.
///
///The rows are the A set and the columns are the B set.
///
///\param costs costs[i][j] is the cost of assigning element i from
///             set A to element j from set B
///
///\param sm A tag class to distinguish this from the code which
///          maximizes the benefits.  Just pass ShouldMinimize
std::vector<GSimpleAssignment> 
linearAssignmentBruteForce(GMatrix costs, ShouldMinimize sm=ShouldMinimize());


///\brief Solve a linear assignment maximization problem by the VERY SLOW brute
///force method
///
///This method is mainly here for testing.
///
///Finds all simple assignments that maximize the benefit matrix.  (For
///certain degenerate cases (like the matrix where all benefits are the
///same), this can be a large number of assignments, be careful).
///Takes time n choose m where n is the larger matrix dimension and m
///is the smaller.  Always assigns one element to each of the smaller
///dimensions of the matrix.
///
///The rows are the A set and the columns are the B set.
///
///\param benefits benefits[i][j] is the benefit of assigning element i from
///                set A to element j from set B
///
///\param sm A tag class to distinguish this from the code which
///          maximizes the benefits.  Just pass ShouldMaximize
std::vector<GSimpleAssignment> 
linearAssignmentBruteForce(GMatrix benefits, ShouldMaximize sm);


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


///\brief Runs unit tests on the linear assignment code and supporting
///       code.  Throws an exception if it detects an error.
void testLinearAssignment();

	
}
#endif //  __GASSIGNMENT_H__
