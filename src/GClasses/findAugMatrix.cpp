/*
	Copyright (C) 2012, Mike Gashler

	This library is free software; you can redistribute it and/or
	modify it under the terms of the GNU Lesser General Public
	License as published by the Free Software Foundation; either
	version 2.1 of the License, or (at your option) any later version.

	see http://www.gnu.org/copyleft/lesser.html
*/

///\file

///\brief An executable that searches (randomly) for a matrix that
///will require use of the augmenting path section of LAPVJRCT.
///
///Randomly generates matrices and then solves each matrix as a linear
///assignment problem.  If the matrix uses the augmentation portion of
///LAPVJRCT, prints it to stdout.

#include "GMatrix.h"
#include "GAssignment.h"
#include "GRand.h"
#include <stdint.h>
#include <iostream>
namespace GClasses{
  extern bool LAPVJRCT_augmentation_section_was_used;
}
using namespace GClasses;

int main(){
  GRand& rnd = GRand::global();
  const std::size_t r=5, c=10;
  GMatrix m(r,c);
  
  uint64_t num_searched = 0;
  while(true){
    for(unsigned i = 0; i < m.rows(); ++i){
      for(unsigned j = 0; j < m.cols(); ++j){
	m[i][j]=rnd.next(1000);
      }
    }
    
    LAPVJRCT_augmentation_section_was_used = false;
    std::vector<int> rowAssign;
    std::vector<int> colAssign;
    std::vector<double> rowPotential;
    std::vector<double> colPotential;
    double totalCost;
    LAPVJRCT(m, rowAssign, colAssign, rowPotential, colPotential,
	     totalCost);

    if(LAPVJRCT_augmentation_section_was_used){
      std::cout << "Found matrix for which the augmentation section was used:\n"
		<< to_str(m) << "\n";
      std::cerr << "Found one!\n";
    }

    ++num_searched;
    if(num_searched % 1000 == 0){
      std::cerr << num_searched << ' ';
    }
  }
}
