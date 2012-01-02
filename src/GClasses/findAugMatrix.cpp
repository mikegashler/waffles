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
///will require use of certain sections of LAPVJRCT.
///
///Randomly generates matrices and then solves each matrix as a linear
///assignment problem.  If the matrix uses the desired portion of
///LAPVJRCT, prints it to stdout.

#include "GMatrix.h"
#include "GAssignment.h"
#include "GRand.h"
#include "GApp.h"
#ifdef WINDOWS
#else
#	include <stdint.h>
#endif
#include <iostream>
#include <string>

#ifdef WINDOWS
#	ifndef uint64_t
typedef unsigned long long uint64_t;
#	endif
#endif

namespace GClasses{
  extern bool LAPVJRCT_augmentation_section_was_used;
	extern bool LAPVJRCT_h_eq_min_true_called;
	extern bool LAPVJRCT_h_eq_min_false_called;
}


int main(int argc, char** argv){
	using namespace GClasses;
	GArgReader args(argc, argv);
	args.pop_string(); //Skip the program name
	bool* success_ptr = NULL;
	std::string section_name;
	if(      args.if_pop("true")){
		success_ptr = &LAPVJRCT_h_eq_min_true_called;
		section_name = "the true branch of the h==min labeled if statement";
	}else if(args.if_pop("false")){
		success_ptr = &LAPVJRCT_h_eq_min_false_called;
		section_name = "the false branch of the h==min labeled if statement";
	}else if(args.if_pop("augment")){
		success_ptr = &LAPVJRCT_augmentation_section_was_used;
		section_name = "the augmentation section";
	}else{
		std::cerr 
			<< "Usage: " << argv[0] << " <true|false|augment>\n"
			<< "Finds matrices where either the true section or false section of\n"
			<< "the LAPVJRCT h==min if statement is executed, or where the LAPVJRCT\n"
			<< "augmentation section is exectuted.  Note that a returned matrix \n"
			<< "may satisfy criteria not given.  Only the first criterion given is \n"
			<< "searched for.  All matrices found are printed to stdout and \n"
			<< "the program stops after finding a certain number of them.\n"
			;
		return -1;
	}
	
	bool& success = *success_ptr;
	
  GRand& rnd = GRand::global();
  const std::size_t r=3, c=3;
	const bool mustTranspose = c < r;
  GMatrix m(r,c);

  uint64_t num_searched = 0;
	uint64_t num_found = 0;
  while(num_found < 10){
    for(unsigned i = 0; i < m.rows(); ++i){
      for(unsigned j = 0; j < m.cols(); ++j){
				m[i][j]=(double)rnd.next(10);
      }
    }

    GMatrix* tr = (mustTranspose)? m.transpose() : &m;
    
    success = false;
    std::vector<int> rowAssign;
    std::vector<int> colAssign;
    std::vector<double> rowPotential;
    std::vector<double> colPotential;
    double totalCost;
    LAPVJRCT(*tr, rowAssign, colAssign, rowPotential, colPotential,
	     totalCost);

		if(mustTranspose){
			delete tr; tr = 0;
		}

    if(success){
      std::cout << "Found matrix for which " << section_name << " was used:\n"
		<< to_str(m) << "\n";
      std::cerr << "Found one!\n";
			++num_found;
    }

    ++num_searched;
    if(num_searched % 10000 == 0){
      std::cerr << num_searched << ' ';
    }
  }
}
