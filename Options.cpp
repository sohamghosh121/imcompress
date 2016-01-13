/*
 * Options.cpp
 *
 *  Created on: 19 Oct 2015
 *      Author: sohamghosh
 */

#include "Options.h"
//#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>


float Options::Mk = 0.8;
float Options::Mw = 0.6;
int Options::blockSize = 8;
int Options::M = 2;
int Options::B = 32;
int Options::A = 509;
float Options::eta = 2.0;
float Options::tau = 0.02;
float Options::sigma = 0.001;
float Options::tolP = 0.0001;
int Options::M_safeguard = 5;
int Options::wavelet_level = 2;

Options::Options() {


}


void Options::parseAndSetKeyValue(std::string key, float value){
	if (key.compare("blockSize")){
		blockSize = int(value);
	} else if (key.compare("Mk")){
		Mk = value;
	} else if (key.compare("Mw")){
		Mw = value;
	} else if (key.compare("M")){
		M = int(value);
	} else if (key.compare("B")){ // SBHE
		B = int(value);
	} else if (key.compare("A")){
		A = int(value);
	} else if (key.compare("tau")){
		tau = value;
	} else if (key.compare("sigma")){
		sigma = value;
	} else if (key.compare("tolP")){
		tolP = value;
	} else if (key.compare("eta")){
		eta = int(value);
	} else if (key.compare("M_safeguard")){
		M_safeguard = int(value);
	} else if (key.compare("wavelet_level")){
		wavelet_level = int(value);
	} else {
		std::cout << "Error parsing option file. Unrecognised parameter " << key;
	}
}

void Options::parseOptionsFile(char * filename){
	std::ifstream fp(filename);
	assert(fp.is_open());
	std::string key;
	float value;
	std::cout << "Options\n------------------------------\n";
	while(fp >> key >> value){
		std::cout << key << ": " << value << "\n";
//		parseAndSetKeyValue(key, value);
	}
	fp.close();
	std::cout << "------------------------------\n";
}

Options::~Options() {
}

