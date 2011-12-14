/*
	Copyright (C) 2012, Mike Gashler

	This library is free software; you can redistribute it and/or
	modify it under the terms of the GNU Lesser General Public
	License as published by the Free Software Foundation; either
	version 2.1 of the License, or (at your option) any later version.

	see http://www.gnu.org/copyleft/lesser.html
*/

///\file
///\brief Defines those things declared in GAssignment.h

#include "GMatrix.h"
#include "GAssignment.h"
#include <limits>
#include <cmath> //For abs, isfinite, isinf

namespace GClasses{


std::ostream& operator<<(std::ostream& out, const GSimpleAssignment& gsa){
	out << '[';
	for(unsigned i = 0; i < gsa.sizeA(); ++i){
		int j = gsa(i);
		if(j >= 0){
			if( i > 0 ){ out << ','; }
			out << '{' << i << ',' << j << '}';
		}		
	}
	return out << ']';
}


void LAPVJRCT(GMatrix c, std::vector<int>& x, std::vector<int>& y, 
							std::vector<double>& u, std::vector<double>& v, 
							double& totalCost, const double epsilon){
	//Infinity
	const double inf = std::numeric_limits<double>::infinity();

	//Number of rows in the cost matrix
	const std::size_t n = c.rows();
	//Number of columns in the cost matrix
	const std::size_t m = c.cols();

	//Check input that it is a square matrix or a rectangular matrix
	//with fewer rows than columns, and that no costs are negative
	if(n > m){ 
		ThrowError("There must be at least as many columns as rows in the "
							 "cost matrix passed to LAPVJRCT"); return;
	}

	for(std::size_t i = 0; i < n; ++i){
		for(std::size_t j = 0; j < m; ++j){
			if(c[i][j] < 0){
				ThrowError("The cost matrix passed to LAPVJRCT must have only "
									 "nonnegative entries."); return;
			}
		}
	}

	//Resize the various vectors to the number of columns (the original
	//code has everything sized to some maximum value
	x.resize(m);
	y.resize(m);
	u.resize(m);
	v.resize(m);

	//Declare other variables used for communicating between the augrowred
	//and augmentation steps
 
	
	//Array of columns.  col[k-1] is:
	//
	//scanned                if 1   <= k <= low-1  (READY in figure 5)
	//labeled and unscanned  if low <= k <= up-1   (SCAN  in figure 5)
	//unlabeled              if up  <= k <= m      (TODO  in figure 5)
	std::vector<int> col(m);
	
	//Unassigned rows.  Number is f0.  Index is f.
	std::vector<int> free(m);

	//Index into the free vector.
	int f;

	//In the following, i and i1 are row indices and j, j1, and j2 are
	//column indices

	//########################################
	//#   "augrowred" routine from original
	//########################################

	{
		//Init arrays to unassigned (0 is unassigned since pascal arrays
		//are 1-based).  Mark all rows as unassigned
		for(unsigned i = 1; i <= n; ++i){
			x.at(i-1) = 0;
			free.at(i-1) = i;
		}
		
		//Mark all columns as unlabeled and unassigned and set their
		//column potentials to 0
		for(unsigned j= 1; j <= m; ++j){
			col[j-1] = j;
			y[j-1] = 0;
			v[j-1] = 0;
		}
		
		//Do two iterations of augmenting row-reduction
		//
		//Find augmenting paths starting in unassigned rows and transfer
		//the reduction to them.  In this process unassigned columns
		//remain unassigned, but rows may become assigned, unassigned, or
		//reassigned.
		//
		//
		//This is an implementation of the following pseudocode from the
		//paper (with some of my own comments added):
		//
		//LIST:={all unassigned rows};
		//for all i in LIST 
		//   repeat

		//      (* Find two indices q which have the two lowest values of
		//         c[i, q]-v[q] *)
		//      u1 := min { c[i,j]-v[j] | j=1..n }
		//      select j1 for which c[i,j1] - v[j1] == u1
		//      u2 := min { c[i,j]-v[j] | j=1..n j != j1}
		//      select j2 for which c[i,j1] - v[j2] == u2
		//
		//      (* Make the second lowest the potential of the current row *)
		//      u[i] := u2;
		//
		//      if u1 < u2
		//         v[j1] := v[j1] - (c[j2]-v[j2]) + (c[j1] - v[j1]) 
	  //                = c[j1] - (c[j2]-v[j2])
	  //                = c[j1] - u2
		//         v[j1] := v[j1] - (u2-u1)
		//      else (* they are equal *) if j1 is assigned
		//         j1 := j2
		//      endif
		//
		//      k := y[j1] (* k is the index of the row to which column 
		//                    j1 is assigned *)
		//      if k > 0   (* if j1 was assigned *)
		//         x[k] := 0  (* unassign the row assigned to j1 *)
		//         x[i] := j1 (* assign column j1 to row i *) 
		//         y[j1]:= i
		//         i:=k       (* do the row assigned to j1 next *)
		//      endif
		//   until u1=u2 (*that is, until no reduction transfer*) or k=0 (* augmentation *)
		//end for

		//Start at the last unassigned row
		f = n;
		for(int cnt = 0; cnt < 2; ++cnt){ 
			int k = 1;
			//f0 is number of unassigned rows in "free"
			int f0 = f;
			
			f = 0;
			while (k <= f0) {
				//i is the index of the current unassigned row in the list of
				//unassigned rows
				int i = free[k-1]; ++k;
				//u1 and u2 are the lowest and second lowest values of
				//c[i-1,j-1] - v[j-1] for row i
				double u1 = c[i-1][0] - v[0];
				double u2 = inf;
				//j1 and j2 are the indices at which u1 and u2 occur
				int j1 = 1;
				int j2 = 0; //Unused value, just gets rid of warning
				//j is the index of the current candidate for lowest or second
				//lowest u value
				for(int j = 2; j <= (int)m; ++j){
					//h is the u value of the candidate
					double h = c[i-1][j-1] - v[j-1];
					//I don't use epsilon here because we're seeing where h lies
					//in respect to the interval [u1,u2) for which extending
					//equality is not necessary
					if(h < u2){ 
						if(h >= u1) { 
							u2 = h; j2 = j;
						}else{ 
							u2 = u1; u1 = h; j2 = j1; j1 = j; 
						}
					}
				}
				int i1 = y[j1 -1];

				if(u1 < u2){
					v[j1 - 1] = c[i-1][j1-1] - c[i-1][j2-1] + v[j2-1];
					//was: v[j1 - 1] = v[j1 -1] - u2 + u1; 
					//I put this in to reduce accumulated round-off error
				}else if(i1 > 0){ 
					j1 = j2; i1 = y[j1 - 1]; 
				}

				if(i1 > 0){ 
					//This comparison doesn't need epsilon because we only care
					//about whether u1 is the minimum or if u2 is equal
					if(u1 < u2){
						--k; free[k-1] = i1; x[i1-1] = 0;
					}else{
						++f; free[f-1] = i1; x[i1-1] = 0;
					}
				}
				x[i-1] = j1; y[j1-1] = i;
			}
		}//Took out an until cnt=2 here
		
	}


	//########################################
	//#   "augmentation" routine from original
	//########################################

	{

		//Find augmenting path for each unassigned row

		//These variables are Used for communication between the augment
		//subsection and the rest of this section

		//An upper bound on part of the shortest path length tree
		double min=-inf; 
		//The number of unassigned rows
		int f0 = f;
		int j;

		//The last column in col-array (with pascal indexing) with d[j-1] < min
		int last=-1; //The initial value is to remove a compiler warning
		for(f = 1; f <= f0; ++f){
			int i1 = free[f-1];
			//The index of the first labeled but unscanned column in col
			//(using pascal indexing)
			int low = 1;
			//The index of the first unlabeled column in col (using pascal
			//labeling)
			int up = 1;
			//d is shortest path lengths 
			std::vector<double> d(m);
			//predecessor array for shortest path tree
			std::vector<int> pred(m);
			for(j = 1; j <= (int)m; ++j){
				d[j-1] = c[i1-1][j-1] - v[j-1];
				pred[j-1] = i1;
			}
			while(true){
				//Find columns with a new value for the minimum d
				if(up == low){
					last = low - 1; min = d[col[up-1]-1]; ++up;
					for(int k = up; k <= (int)m; ++k){
						j = col[k-1]; 
						double h=d[j-1];
						//Note: I don't use epsilon in these comparisons because
						//they are computing a minimum, so using the
						//interval-bounded equality would not be appropriate (a
						//greater number coluld be selected)
						if(h <= min){ 
							if(h < min){ 
								up=low; min=h;
							}
							col[k-1]=col[up-1]; col[up-1]=j; ++up;
						}
					}
					for(int q=low; q <= up-1; ++q){
						j=col[q-1];
						if(y[j-1] == 0){ goto augment; }
					}
				} //up == low
				//Scan a row

				int j1=col[low-1]; ++low; 
				int i=y[j1-1];
				double u1 = c[i-1][j1-1] - v[j1-1] - min;
				for(int k=up; k <= (int)m; ++k){
					j=col[k-1];
					double h = c[i-1][j-1] - v[j-1] - u1;
					if(h < d[j-1]){
						d[j-1] = h; pred[j-1] = i;
						if(fabs(h - min) < epsilon){ //h==min
							if(y[j-1]==0){ 
								goto augment; 
							}else{ 
								col[k-1]=col[up-1]; col[up-1]=j; ++up;
							}
						}
					}
				}//for k
			}//Infinite loop

		augment:

			//Update column prices
			for(int k=1; k <= last; ++k){
				int j1 = col[k-1];
				v[j1-1] += d[j1-1] - min; 
			}

			//Perform actual augmentation
			int i;
			do{
				i = pred[j-1]; y[j-1] = i; int k=j; j=x[i-1]; x[i-1]=k;
			}while(i != i1);
			
		}//for f
	}


	//########################################
	//#   "determine" routine from original
	//########################################

	{
		totalCost = 0; //This is called optimum in the original
		for(std::size_t i=1; i <= n; ++i){
			int j=x[i-1];
			u[i-1]=c[i-1][j-1] - v[j-1];
			totalCost += u[i-1] + v[j-1];
		}
	}

	//#######################################
	//# Change x and y to zero-based indices for return and resize the
	//# vectors giving row characteristics to the number of rows
	//#######################################

	for(std::size_t i=0; i < m; ++i){
		--x.at(i); --y.at(i);
	}

	x.resize(n);
	u.resize(n);
}

namespace{

	///Return true iff d represents a non-NAN finite value
	///
	///This is a stand-in until the isfinite(double) function is added
	///in the next c++ standard.  It will work on well-behaved
	///processors, but if you want to compile for an older chip that
	///does strange things with NaNs and infinities, cross your fingers.
	///
	///\param d the double value whose finitude is assessed
	///
	///\return true if d is not NaN, infinity or -infinity, return false
	///        if d is NaN, infinity or -infinity.
	bool isfinite(double d){
		const double inf = std::numeric_limits<double>::infinity();
		return d != inf && d != -inf && d==d;
	}

	///\brief Applies a linear transformation a*x + b to all entries in
	///the matrix m.
	///
	///Does: m[i][j] = a*m[i][j] + b for all entries in the matrix m
	///
	///\param m the matrix to transform
	///
	///\param a the multiplicative factor in the a*x+b transformation
	///         applied to the matrix entries
	///
	///\param b the constant term in the a*x+b transformation applied to
	///         the matrix entries.
	void linearTransformMatrixEntries(GMatrix&m, 
																		const double a, const double b){
		for(unsigned i = 0; i < m.rows(); ++i){
			for(unsigned j = 0; j < m.cols(); ++j){
				if(a==1){
					m[i][j] += b;
				}else if(a == -1){
					m[i][j] = b - m[i][j];
				}else{
					m[i][j] = b + a*m[i][j];
				}
			}
		}
	}

	
	///\brief Set \a min to the minimum finite value in the matrix \a m
	///
	///Of all the values x in the matrix for which isfinite(x) is true,
	///sets \a min to the minimum such value (and sets \a
	///matrixHadFiniteMinimum to true).  If there is no such value, \a
	///min is undefined and \a matrixHadFiniteMinimum is set to false.
	///
	///\param m the matrix whose minimum is found
	///
	///\param min the minimum finite value in the matrix \a m.
	///           undefined if there is no such value
	///
	///\param matrixHadFiniteMinimum If \a m had a finite minimum,
	///                              this will be true.  If the matrix
	///                              was empty or had no finite value as
	///                              the minimum, this will be false
	void minFiniteValue(GMatrix& m, double& min, bool& matrixHadFiniteMinimum){
		const double inf = std::numeric_limits<double>::infinity();
		
		//Find the smallest finite value in the matrix 
		min = inf;
		for(std::size_t row = 0; row < m.rows(); ++row){
			double const* r = m[row];
			double const* rend = r + m.cols();
			for(double const* cur = r; cur != rend; ++cur){
				if(min > *cur && isfinite(*cur)){
					min = *cur;
				}
			}
		}
		
		//If min is not finite, there are no finite values in the matrix
		matrixHadFiniteMinimum = isfinite(min);
	}

	///\brief Set \a max to the maximum finite value in the matrix \a m
	///
	///Of all the values x in the matrix for which isfinite(x) is true,
	///sets \a max to the maximum such value (and sets \a
	///matrixHadFiniteMaximum to true).  If there is no such value, \a
	///max is undefined and \a matrixHadFiniteMaximum is set to false.
	///
	///\param m the matrix whose maximum is found
	///
	///\param max the maximum finite value in the matrix \a m.
	///           undefined if there is no such value
	///
	///\param matrixHadFiniteMaximum If \a m had a finite maximum,
	///                              this will be true.  If the matrix
	///                              was empty or had no finite value as
	///                              the maximum, this will be false
	void maxFiniteValue(GMatrix& m, double& max, bool& matrixHadFiniteMaximum){
		const double inf = std::numeric_limits<double>::infinity();
		
		//Find the largest finite value in the matrix 
		max = -inf;
		for(std::size_t row = 0; row < m.rows(); ++row){
			double const* r = m[row];
			double const* rend = r + m.cols();
			for(double const* cur = r; cur != rend; ++cur){
				if(max < *cur && isfinite(*cur)){
					max = *cur;
				}
			}
		}
		
		//If max is not finite, there are no finite values in the matrix
		matrixHadFiniteMaximum = isfinite(max);
	}


	///\brief transforms the benefits matrix for a linear assignment
	///maximization problem into a non-negative cost matrix for a
	///minimization problem with the same solution.
	///
	///\param benefits the matrix of benefits for the linear-assignment
	///                maximization problem.  Will become a cost matrix
	///                with non-negative entries and the same solution.
	///
	///\param success true if the matrix could be standardized, false if
	///               it could not.  Success is not a return value to
	///               ensure that the caller notices it.
	void standardizeMaxLinAssignProb(GMatrix& benefits, bool& success){
		
		success = false;

		//Find the minimum and maximum for the matrix
		
		double min; bool matrixHasFiniteMinimum;
		minFiniteValue(benefits, min, matrixHasFiniteMinimum);
		
		double max; bool matrixHasFiniteMaximum;
		maxFiniteValue(benefits, max, matrixHasFiniteMaximum);

		if(! matrixHasFiniteMinimum){
			//The matrix cannot be standardized when the matrix has no
			//finite entries
			return;
		}
		//Since there is a finite minimum, there must also be a finite maximum
		assert(matrixHasFiniteMaximum);
		
		// Add the minimum to all entries if the minimum is negative
		//
		// then
		//
		// Subtract all entries from the maximum so the maximum becomes the
		// minimum and vice-versa
		//
		// This is equivalent to doing the transformation 
		// y = x - min then z = max - y  (if min was negative)
		// or 
		// z = max - x (if min was non-negative)
		//
		// This is equivalent to 
		//
		// z = max - (x-min) = max + min - x (if min was negative)
		// or
		// z = max - x (if min was non-negative
		//
		// I do all these gyrations to allow one pass over the matrix rather
		// than two for the two transformations.
		double toAdd;
		if(min < 0){
			toAdd = max + min;
		}else{
			toAdd = max;
		}
		
		linearTransformMatrixEntries(benefits, -1, toAdd);
		success = true;
	}


	///\brief Produce the next k-permutation of n integers
	///
	///Set \a perm to the next permutation in lexiographic order of k
	///integers selected from the range 0..n-1 if there is such a
	///k-permutation and return true.  Otherwise, produce the first such
	///permutation (0,1,2,3...k-1) and return false.
	///
	///\note perm.size() == k
	///
	///\param perm the current permutation in the sequence.  After
	///            return, holds the next permutation or 0,1,...k-1 if
	///            there is no next greater permutation.
	///
	///\param n the number of integers that can be selected, must be at
	///         least k (n>=k)
	///
	///\return true if the next k-permutation was produced, otherwise,
	///             produce the first k-permutation and return false
	///
	///\see std::next_permutation
	bool nextKPermutation(std::vector<int>& perm, int n){
		assert(n>=0); 
		assert((unsigned)n >= perm.size());
		std::vector<int> origPerm = perm; //DEBUG CODE
		//Mark which elements are used
		std::vector<bool> used(n,false);
		for(std::vector<int>::const_iterator it = perm.begin(); it != perm.end();
				++it){
			used.at(*it) = true;
		}

		//Find the int with the greatest index that can be advanced
		std::vector<int>::reverse_iterator it = perm.rbegin();
		for(; it != perm.rend(); ++it){
			int replacement = *it+1;
			while(replacement < n && used[replacement]){
				++replacement;
			}
			used.at(*it) = false;
			if(replacement < n){
				//We've found a valid replacement, do the replacement, fill
				//the rest of the array and return;
				used.at(replacement) = true;
				*it = replacement;
				//Fill the rest of the perm with sequential elements from the
				//used array.  From here on, it points to the last element
				//filled with a replacement
				for(std::vector<bool>::iterator isUsed = used.begin(); 
						isUsed != used.end(); ++isUsed){
					if(it == perm.rbegin()){ 
						return true;
					}else{
						//For each unused element fill the next element of perm
						if(! *isUsed ){
							--it;
							*it = isUsed - used.begin();
							*isUsed = true; //isUsed doesn't need to be updated, but
															//I want to keep the data structure
															//consistent with reality
							if(it == perm.rbegin()){	return true;	}
						}
					}
				}
				assert("We should never get here" == NULL);
			}
			//No valid replacement exists for the element at *it, go to the
			//next
		}

		//No replacement existed for any element of perm, thus, we were
		//already at the maximum permutation, start over at the beginning
		for(std::size_t i = 0; i < perm.size(); ++i){
			perm.at(i) = i;
		}
		return false;
	}
}//Anonymous Namespace

GSimpleAssignment linearAssignment(GMatrix benefits,	
																	 ShouldMaximize /*Not used*/, 
																	 const double epsilon){

	bool success;
	standardizeMaxLinAssignProb(benefits, success);

	GSimpleAssignment result(benefits.rows(), benefits.cols());
	if(!success){
		//No assignment is possible if the matrix could not be standardized
		return result;
	}

	bool hadToTranspose = false;
	GMatrix* costs = &benefits;
	if(benefits.rows() > benefits.cols()){
		hadToTranspose = true;
	  costs = benefits.transpose();
	}

	std::vector<int> rowAssign;
	std::vector<int> colAssign;
	std::vector<double> rowPotential;
	std::vector<double> colPotential;
	double totalCost;
	LAPVJRCT(*costs, rowAssign, colAssign, rowPotential, colPotential,
					 totalCost, epsilon);

	if(hadToTranspose){
		result.setBForA(colAssign);
		delete costs;
	}else{
		result.setBForA(rowAssign);
	}

	return result;
}

GSimpleAssignment linearAssignment(GMatrix costs,	
																	 ShouldMinimize /*Not used*/, 
																	 const double epsilon){

	//Convert the cost matrix to one with non-negative entries but an
	//identical solution
	double min; bool matrixHasFiniteMinimum;
	minFiniteValue(costs, min, matrixHasFiniteMinimum);

	GSimpleAssignment result(costs.rows(), costs.cols());
	if(! matrixHasFiniteMinimum){
		//The matrix cannot be standardized when the matrix has no
		//finite entries
		return result;
	}

	if(min < 0){
		linearTransformMatrixEntries(costs, 1, min);
	}
		
	//Transpose if necessary so fewer rows than columns
	bool hadToTranspose = false;
	GMatrix* cstd = &costs; //standardized costs
	if(costs.rows() > costs.cols()){
		hadToTranspose = true;
	  cstd = costs.transpose();
	}

	//Run LAPVJRCT
	std::vector<int> rowAssign;
	std::vector<int> colAssign;
	std::vector<double> rowPotential;
	std::vector<double> colPotential;
	double totalCost;
	LAPVJRCT(*cstd, rowAssign, colAssign, rowPotential, colPotential,
					 totalCost, epsilon);

	//Set the result and free the standardized copy
	if(hadToTranspose){
		result.setBForA(colAssign);
		delete cstd;
	}else{
		result.setBForA(rowAssign);
	}

	return result;
}

std::vector<GSimpleAssignment>	
linearAssignmentBruteForce(GMatrix costs, ShouldMinimize /*not used*/){
	const double inf = std::numeric_limits<double>::infinity();

	//Make fewer rows than columns
	bool hadToTranspose = false;
	if(costs.rows() > costs.cols()){
		hadToTranspose = true;
	  GMatrix* tmp = costs.transpose();
		costs = *tmp;
		delete tmp;
	}

	//The minimum cost encountered (will be reinitialized later)
	double minCost = inf;

	//All assignments found that have cost == minCost
	std::vector<GSimpleAssignment> result;
	

	//Only empty assignment possible for an empty matrix
	if(costs.rows() == 0){ 
		if(hadToTranspose){
			result.push_back(GSimpleAssignment(costs.cols(), costs.rows()));
		}else{
			result.push_back(GSimpleAssignment(costs.rows(), costs.cols()));
		}
		return result; 
	}

	//The current assignment
	std::vector<int> cur(costs.rows());

	//Initialize the current assignment to 0...n-1 where n is the number
	//of rows
	for(std::size_t i = 0; i < cur.size(); ++i){
		cur.at(i) = i;
	}

	//Set minCost to the cost of the current assignment
	minCost = 0;
	for(std::size_t i = 0; i < cur.size(); ++i){
		minCost += costs[i][cur.at(i)];
	}

	//If the current solution is feasible, add it to the list of
	//solutions
	if(minCost < inf){ 
		result.push_back(GSimpleAssignment(costs.cols(), cur));
	}

	//Go through all permutations and add to the list any that have a
	//cost equal to the current minimum.  If the current assignment has
	//a lower cost, make it the current element in the list and set the
	//minimum cost to its cost
	while(nextKPermutation(cur, costs.cols())){
		double curCost = 0;
		for(std::size_t i = 0; i < cur.size(); ++i){
			curCost += costs[i][cur.at(i)];
		}
		if(curCost < minCost){
			minCost = curCost;
			result.clear();
		}

		if(curCost == minCost && minCost < inf){
			result.push_back(GSimpleAssignment(costs.cols(), cur));
		}
	}

	if(hadToTranspose){
		for(std::size_t i = 0; i < result.size(); ++i){
			result[i].swapAAndB();
		}
	}

	return result;
}


std::vector<GSimpleAssignment>	
linearAssignmentBruteForce(GMatrix benefits, ShouldMaximize /*not used*/){
	const double inf = std::numeric_limits<double>::infinity();

	//Make fewer rows than columns
	bool hadToTranspose = false;
	if(benefits.rows() > benefits.cols()){
		hadToTranspose = true;
	  GMatrix* tmp = benefits.transpose();
		benefits = *tmp;
		delete tmp;
	}

	//The maximum benefit encountered (will be reinitialized later)
	double maxBenefit = -inf;

	//All assignments found that have benefit == maxBenefit
	std::vector<GSimpleAssignment> result;
	

	//Only empty assignment possible for an empty matrix
	if(benefits.rows() == 0){ 
		if(hadToTranspose){
			result.push_back(GSimpleAssignment(benefits.cols(), benefits.rows()));
		}else{
			result.push_back(GSimpleAssignment(benefits.rows(), benefits.cols()));
		}
		return result; 
	}

	//The current assignment
	std::vector<int> cur(benefits.rows());

	//Initialize the current assignment to 0...n-1 where n is the number
	//of rows
	for(std::size_t i = 0; i < cur.size(); ++i){
		cur.at(i) = i;
	}

	//Set maxBenefit to the benefit of the current assignment
	maxBenefit = 0;
	for(std::size_t i = 0; i < cur.size(); ++i){
		maxBenefit += benefits[i][cur.at(i)];
	}

	//If the current solution is feasible, add it to the list of
	//solutions
	if(maxBenefit > -inf){ 
		result.push_back(GSimpleAssignment(benefits.cols(), cur));
	}

	//Go through all permutations and add to the list any that have a
	//benefit equal to the current maximum.  If the current assignment has
	//a greater benefit, make it the current element in the list and set the
	//maximum benefit to its benefit
	while(nextKPermutation(cur, benefits.cols())){
		double curBenefit = 0;
		for(std::size_t i = 0; i < cur.size(); ++i){
			curBenefit += benefits[i][cur.at(i)];
		}
		if(curBenefit > maxBenefit){
			maxBenefit = curBenefit;
			result.clear();
		}

		if(curBenefit == maxBenefit && maxBenefit > -inf){
			result.push_back(GSimpleAssignment(benefits.cols(), cur));
		}
	}

	if(hadToTranspose){
		for(std::size_t i = 0; i < result.size(); ++i){
			result[i].swapAAndB();
		}
	}

	return result;
}


#ifndef  NO_TEST_CODE

///\brief Runs unit tests on nextKPermutation
void testNextKPermutation(){
	using std::string;
	{
		int perms[][3]={{0,1,2},
									 {0,2,1},
									 {1,0,2},
									 {1,2,0},
									 {2,0,1},
									 {2,1,0}};
		int n = 3;
		int k = 3;
		unsigned numPerms = sizeof(perms)/(k*sizeof(int));
		for(unsigned prev = 0; prev < numPerms; ++prev){
			std::vector<int> prevPerm(&perms[prev][0], k+&perms[prev][0]); 
			std::vector<int> pp      (&perms[prev][0], k+&perms[prev][0]); 

			unsigned cur = (prev+1)%numPerms;
			std::vector<int> curPerm(&perms[cur][0], k+&perms[cur][0]); 

			bool retVal = nextKPermutation(prevPerm, n);
			TestEqual(cur > prev, retVal, 
								string("Wrong return value for nextKPermutation(")+
								to_str(pp)+","+to_str(n)+")");
			TestEqual(curPerm, prevPerm,
								string("Wrong permutation generated for nextKPermutation(")+
								to_str(pp)+","+to_str(n)+")");
		}
	}

	{
		int perms[][3]={{0,1,2},{0,1,3},{0,1,4},
										{0,2,1},{0,2,3},{0,2,4},
										{0,3,1},{0,3,2},{0,3,4},
										{0,4,1},{0,4,2},{0,4,3},
										{1,0,2},{1,0,3},{1,0,4},
										{1,2,0},{1,2,3},{1,2,4},
										{1,3,0},{1,3,2},{1,3,4},
										{1,4,0},{1,4,2},{1,4,3},
										{2,0,1},{2,0,3},{2,0,4},
										{2,1,0},{2,1,3},{2,1,4},
										{2,3,0},{2,3,1},{2,3,4},
										{2,4,0},{2,4,1},{2,4,3},
										{3,0,1},{3,0,2},{3,0,4},
										{3,1,0},{3,1,2},{3,1,4},
										{3,2,0},{3,2,1},{3,2,4},
										{3,4,0},{3,4,1},{3,4,2},
										{4,0,1},{4,0,2},{4,0,3},
										{4,1,0},{4,1,2},{4,1,3},
										{4,2,0},{4,2,1},{4,2,3},
										{4,3,0},{4,3,1},{4,3,2}
		};
		int n = 5;
		int k = 3;
		unsigned numPerms = sizeof(perms)/(k*sizeof(int));
		for(unsigned prev = 0; prev < numPerms; ++prev){
			std::vector<int> prevPerm(&perms[prev][0], k+&perms[prev][0]); 
			std::vector<int> pp      (&perms[prev][0], k+&perms[prev][0]); 

			unsigned cur = (prev+1)%numPerms;
			std::vector<int> curPerm(&perms[cur][0], k+&perms[cur][0]); 

			bool retVal = nextKPermutation(prevPerm, n);
			TestEqual(cur > prev, retVal, 
								string("Wrong return value for nextKPermutation(")+
								to_str(pp)+","+to_str(n)+")");
			TestEqual(curPerm, prevPerm,
								string("Wrong permutation generated for nextKPermutation(")+
								to_str(pp)+","+to_str(n)+")");
		}
	}
}

namespace{
	class LinearAssignmentTestCase{
		///The input matrix for the test case
		GMatrix input;

		///True if the problem is a minimization problem, false if it is a
		///maximization problem
		bool isMinimization;

		///The set of all correct assignments for the problem
		std::set<GSimpleAssignment> solutionSet;

		///\brief Finish initializing a partially initialized test case
		///
		///Assumes that this object has been partially initialized by the
		///constructor.  In particular, this->isMinimization has been set
		///and this->input was constructed with the correct number of rows
		///and columns.
		///
		///For the precise specifications of the parameters, look at 
		///LinearAssignmentTestCase(const std::size_t, const std::size_t, 
		///												 double const*, const std::size_t, int const*, 
		///                        shouldMinimize);
		///
		///\param input the input matrix in row-major form.  If m is the
		///             input matrix then m[i][j]=input[cols*i+j]
		///
		///\param numSolutions the number of correct solutions to
		///                           the linear assignment problem
		///
		///\param correct the solution set of assignments in row-major
		///               form.  The correct assignment to row i in the
		///               k'th solution is given by correct[k*rows + i].
		///               If row i should be unassigned, then the correct
		///               assignment is -1
		void init(double const* input, 
							const std::size_t numSolutions,
							int const*correct){
			const std::size_t rows = this->input.rows();
			const std::size_t cols = this->input.cols();

			this->input.fromVector(input, rows);
			for(std::size_t curSol = 0; curSol < numSolutions; ++curSol){
				int const* start = correct + curSol * rows;
				std::vector<int> tmp(start, start+rows);
				solutionSet.insert(GSimpleAssignment(cols, tmp));
			}
		}
	public:
		///\brief Make a test case for a linear assignment minimization problem
		///
		///\param rows the number of rows in the input matrix (also the
		///            size of set A for the final solution)
		///
		///\param cols the number of columns in the input matrix (also the
		///            size of set B for the final solution)
		///
		///\param input the input matrix in row-major form.  If m is the
		///             input matrix then m[i][j]=input[cols*i+j]
		///
		///\param numSolutions the number of correct solutions to
		///                           the linear assignment problem
		///
		///\param correct the solution set of assignments in row-major
		///               form.  The correct assignment to row i in the
		///               k'th solution is given by correct[k*rows + i].
		///               If row i should be unassigned, then the correct
		///               assignment is -1
		LinearAssignmentTestCase(const std::size_t rows, const std::size_t cols, 
														 double const* input, 
														 const std::size_t numSolutions,
														 int const*correct, 
														 ShouldMinimize):
			input(rows, cols), isMinimization(true){
			init(input, numSolutions, correct);
		}
		///\brief Make a test case for a linear assignment maximization problem
		///
		///For full description, see the minimization constructor which is
		///identical, except for passing a ShouldMinimize parameter
		///
		///\see LinearAssignmentTestCase(const std::size_t rows, const std::size_t cols, double const* input, const std::size_t numSolutions, int const*correct, ShouldMinimize);
		LinearAssignmentTestCase(const std::size_t rows, const std::size_t cols, 
														 double const* input, 
														 const std::size_t numSolutions,
														 int const*correct, ShouldMaximize):
			input(rows, cols), isMinimization(false){
			init(input, numSolutions, correct);
		}

		///\brief Runs this test case for the brute-force linear
		///assignment problem solver, throws an exception if there is an error
		void testBruteForce(){
			std::vector<GSimpleAssignment> actual;
			if(isMinimization){
				actual = linearAssignmentBruteForce(input);
			}else{
				actual = linearAssignmentBruteForce(input, ShouldMaximize());
			}
			std::set<GSimpleAssignment> actualSet(actual.begin(), actual.end());
			if(actualSet.size() == solutionSet.size() && 
				 std::equal(actualSet.begin(), actualSet.end(), solutionSet.begin())){
				//No error
			}else{
				ThrowError
					("The brute-force linear assignment ", 
					 isMinimization?"minimization ":"maximization ",
					 "problem solver returned a different solution set.\n",
					 "Actual:",to_str(actualSet),"\nExpected:",to_str(solutionSet),
					 "\non the problem input:\n",to_str(input));
			}
		}

		///\brief Runs this test case for the standard linear assignment
		///problem solver, throws an exception if there is an error
		///
		///This is a separate function from the brute-force testing
		///function because you may want to try the standard algorithm on
		///a matrix much too big for the brute-force algorithm
		void testStandard(){
			GSimpleAssignment actual(input.rows(), input.cols());
			if(isMinimization){
				actual = linearAssignment(input);
			}else{
				actual = linearAssignment(input, ShouldMaximize());
			}

			std::set<GSimpleAssignment>::const_iterator location = 
				solutionSet.find(actual);
			if(location == solutionSet.end()){
				//Could not find the generated solution among the correct answers
				ThrowError
					("The standard linear assignment ", 
					 isMinimization?"minimization ":"maximization ",
					 "problem solver returned an incorrect solution\n",
					 "Returned answer:",to_str(actual),
					 "\nCorrect answers:",to_str(solutionSet),
					 "\non the problem input:\n",to_str(input));
			}
		}
	};
}



void testLinearAssignment(){
	testNextKPermutation();
	//Simple matrix test
	{ 
		const unsigned r=5, c=5;
		double input[r*c] = {
			0,1,2,3,4,
			1,0,1,2,3,
			2,1,0,1,2,
			3,2,1,0,1,
			4,3,2,1,0};
		const unsigned ns = 1;
		int solutions[r*ns] = {
			0,1,2,3,4};
		LinearAssignmentTestCase tc(r,c,input, ns, solutions, ShouldMinimize());		
		tc.testStandard();
		tc.testBruteForce();
	}
	//Test for distance matrix where original points were:
	//
	//A :0,B :2,C :5,D :8 and after moving, the points were
	//
	//A':1,B':2,C :3,D':6
	//
	//The rows are A,C,D,B and the columns are the primes as A',B',C',D'
	//
	//So, the expected permutation is 0,2,3,1
	{
		const unsigned r=4, c=4;
		double input[r*c] = {
		//A'B'C'D'
			1,2,3,6, //A
			4,3,2,1, //C
			7,6,5,2, //D
			1,0,1,4, //B
		};

		const unsigned ns = 1;
		int solutions[r*ns] = {
			0,2,3,1};
		LinearAssignmentTestCase tc(r,c,input, ns, solutions, ShouldMinimize());		
		tc.testBruteForce();
		tc.testStandard();
	}

	//10x10 matrix of uniform[0,1) after applying the exponent function
	//to each entry and then subtracting 1.  The expected value was
	//calculated by the brute-force routine
	{
		const unsigned r=10, c=10;
		double input[r*c] = {
			0.568312, 0.186491, 0.251822, 1.69097, 0.116055, 1.02668, 0.833451, 
			0.913244, 0.379057, 0.395985, 0.233801, 0.42134, 0.565648, 0.0175523, 
			0.740374, 0.892502, 0.0773453, 0.637876, 0.913435, 1.21444, 0.126032, 
			1.68264, 0.493317, 1.4361, 1.01254, 0.692319, 0.589151, 1.49253, 
			0.446722, 0.356082, 0.535721, 0.336294, 0.370396, 0.347027, 
			0.00762895, 0.655826, 0.769328, 0.638039, 0.355134, 1.2056, 0.981591, 
			0.409451, 1.55589, 0.861532, 0.772693, 0.492869, 1.01395, 0.467705, 
			0.710182, 1.38119, 1.02061, 0.144308, 0.408324, 0.115162, 0.267188, 
			1.20891, 1.36694, 0.962855, 0.497654, 0.0574919, 0.344201, 0.677325, 
			0.129867, 0.282101, 1.55384, 0.847254, 0.0174505, 1.27618, 0.037278, 
			1.01577, 0.554105, 0.592333, 0.00260338, 1.08652, 0.715492, 0.533419, 
			1.50578, 0.42504, 0.332691, 0.551155, 1.14577, 0.602076, 0.100099, 
			0.0906785, 0.71927, 0.205266, 0.141679, 0.180455, 0.0312792, 
			0.379333, 0.849287, 0.170698, 0.965801, 0.0165352, 0.311275, 
			0.0970228, 1.55921, 0.491228, 0.705571, 0.00843538
		};

		{
			//Minimization
			const unsigned ns = 1;
			int solutions[r*ns] = {
				1,3,0,4,7,9,6,2,8,5};

			LinearAssignmentTestCase tc(r,c,input, ns, solutions, ShouldMinimize());
			tc.testStandard();
		}
		{
			//Maximization
			const unsigned ns = 1;
			int solutions[r*ns] = {
				3,8,1,9,2,5,7,4,0,6};

			LinearAssignmentTestCase tc(r,c,input, ns, solutions, ShouldMaximize());
			tc.testStandard();
		}
	}

	//10x10 matrix of uniform[0,1) after applying 1/x to each entry.
	//The expected value was calculated by the brute-force routine
	{
		const unsigned r=10, c=10;
		double input[r*c] = {
			2.78149, 2.1004, 2.19674, 2.27427, 1.56916, 45.9015, 1.86884,		
			2.38739, 5.87695, 46.6571, 2.27358, 1.35429, 1.46073, 1.53818,	
			3.96977, 1.12193, 4.18369, 1.83308, 11.4366, 4.10773, 5.49231,	
			2.82751, 4.80075, 19.7531, 10.9935, 1.53742, 2.49098, 1.16571,	
			9425.96, 2.41417, 59.6294, 28.0838, 3.51606, 40.9338, 3.00573,	
			1.6941, 5.57562, 25.0501, 8.87884, 17.2603, 2.42773, 13.0047,		
			75.6144, 1.49303, 3.07248, 1625.91, 64.5151, 5.21972, 38.3878,	
			15.6599, 2.17477, 1.7034, 1.78778, 3.33483, 98.8107, 1.97094,		
			1.89674, 2.78892, 13.8196, 17.4921, 12.1236, 1.43426, 176.364,	
			1.78348, 1.85562, 1.11178, 6.95813, 206612., 2.64393, 11.1856,	
			8573.39, 9.31777, 39.5057, 11.491, 171.322, 1189.06, 1.02777,		
			3673.09, 493.827, 9.01803, 8573.39, 1.42843, 9.11632, 2.81141,	
			2.00117, 16.6055, 4.26531, 28.0838, 4.41539, 1.15595, 1.66322,	
			1.06633, 46.9774, 9.31209, 14.1013, 50.6579, 2.92106, 46.8489,	
			86.2122, 116.621
		};
		{
			//Minimization
			const unsigned ns = 1;
			int solutions[r*ns] = {
				4,2,7,5,3,0,8,6,9,1};

			LinearAssignmentTestCase tc(r,c,input, ns, solutions, ShouldMinimize());
			tc.testStandard();
		}
		{
			//Maximization
			const unsigned ns = 1;
			int solutions[r*ns] = {
				2,6,8,3,5,1,7,4,0,9};

			LinearAssignmentTestCase tc(r,c,input, ns, solutions, ShouldMaximize());
			tc.testStandard();
		}
	}
	{
		const unsigned r=3, c=4;
		double input[r*c] = {
			1, 2,4,8,
			8,-2,8,4,
			8, 4,8,8
		};
		{
			//Minimization
			const unsigned ns = 2;
			int solutions[r*ns] = {
				0,1,2,
				0,1,3};
		
			LinearAssignmentTestCase tc(r,c,input, ns, solutions, ShouldMinimize());
			tc.testBruteForce();
			tc.testStandard();
		}
		{
			//Maximization
			const unsigned ns = 2;
			int solutions[r*ns] = {
				3,0,2,
				3,2,0};
		
			LinearAssignmentTestCase tc(r,c,input, ns, solutions, ShouldMaximize());
			tc.testBruteForce();
			tc.testStandard();
		}
	}
	ThrowError("LinearAssignment is still experimental, not all tests have been implimented.  This message means it has passed all implemented tests.");
	
}
#endif //NO_TEST_CODE



}//Namespace GClasses

