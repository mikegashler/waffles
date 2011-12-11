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

}

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


}//Namespace GClasses

