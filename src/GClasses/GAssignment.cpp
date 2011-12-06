#include "GAssignment.h"
#include <limits>
#include "GMatrix.h"


void GClasses::LAPVJRCT(GMatrix c, std::vector<int>& x, std::vector<int>& y, 
												std::vector<double>& u, std::vector<double>& v, 
												double& totalCost, const double epsilon){
	//Infinity
	const double inf = std::numeric_limits<double>::infinity();

	//Number of rows in the cost matrix
	const std::size_t n = c.rows();
	//Number of columns in the cost matrix
	const std::size_t m = c.cols();

	if(n > m){ 
		ThrowError("There must be at least as many columns as rows in the "
							 "cost matrix passed to LAPVJRCT"); return;
	}

	//Resize the various vectors to the number of columns (the original
	//code has everything sized to some maximum value
	x.resize(m);
	y.resize(m);
	u.resize(m);
	v.resize(m);

	//Declare other vectors used for communicating between the augrowred
	//and augmentation steps
	std::vector<int> col(m), free(m);

	//Other global variables in the original that are actually used
	//globally
	int f;
	

	//########################################
	//#   "augrowred" routine from original
	//########################################

	{
		//Init arrays to unassigned (0 is unassigned since pascal arrays
		//are 1-based)
		for(unsigned i = 1; i <= n; ++i){
			x.at(i-1) = 0;
			free.at(i-1) = i;
		}
		
		for(unsigned j= 1; j <= m; ++j){
			col[j-1] = j;
			y[j-1] = 0;
			v[j-1] = 0;
		}
		
		//Do two iterations of augmenting row-reduction
		f = n;
		for(int cnt = 0; cnt < 2; ++cnt){ 
			int k = 1;
			int f0 = f;
			f = 0;
			while (k <= f0) {
				int i = free[k-1]; ++k;
				double u1 = c[i-1][0] - v[0];
				int j1 = 1;
				int j2 = 0; //Unused value, just gets rid of warning
				double u2 = inf;
				for(int j = 2; j <= (int)m; ++j){
					double h = c[i-1][j-1] - v[j-1];
					if(h < u2){
						if(h >= u1) { u2 = h; j2 = j;
						}else{ u2 = u1; u1 = h; j2 = j1; j1 = j; }
					}
				}
				int i1 = y[j1 -1];

				if(u1 < u2){ v[j1 - 1] = v[j1 -1] - u2 + u1;
				}else if(i1 > 0){ j1 = j2; i1 = y[j1 - 1]; }

				if(i1 > 0){ 
					if(u1 < u2){
						--k; free[k-1] = i1; x[i1-1] = 0;
					}else{
						++f; free[f-1] = i1; x[i1-1] = 0;
					}
				}
				x[i-1] = j1; y[j1] = i;
			}
		}//Took out an until cnt=2 here
		
	}


	//########################################
	//#   "augmentation" routine from original
	//########################################

	{

		//Find augmenting path for each unassigned row
		int f0 = f;
		int j,last=-1; //Used for communication between the augment
					         //subsection and the rest of this section
		double min=-inf; //Also used for communication
		for(f = 1; f <= f0; ++f){
			int i1 = free[f-1];
			int low = 1;
			int up = 1;
			std::vector<double> d(m);
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
						if(h <= min){
							if(h < min){ up=low; min=h;}
							col[k-1]=col[up-1]; col[up-1]=j; ++up;
						}
					}
					for(int q=low; q <= up-1; ++q){
						j=col[q-1];
						if(y[j-1] == 0){ goto augment; }
					}
				} //up == low
				//Scan a row
				int j1=col[low-1]; ++low; int i=y[j1-1];
				double u1 = c[i-1][j1-1] - v[j1-1] - min;
				for(int k=up; k <= (int)m; ++k){
					j=col[k-1];
					double h = c[i-1][j-1] - v[j-1] - u1;
					if(h < d[j-1]){
						d[j-1] = h; pred[j-1] = i;
						if(h == min){
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
