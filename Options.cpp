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


float Options::Mk = 0.6;
float Options::Mw = 0.55;
int Options::blockSize = 16;
int Options::M = 2;
int Options::A = 255;
float Options::eta = 2.0;
float Options::tau = 0.00002; //increasing helps, but makes it slower (more sparse solution)
float Options::lambda = 0.05;
float Options::sigma = 0.001; // decreasing doesn't help much
float Options::tolP = 0.00001;
float Options::tolD = 0.001; // decreasing helps
int Options::M_safeguard = 5;
int Options::wavelet_level = 20; // increasing this helps a lot
size_t Options::maxIter = 500;
size_t Options::minIter = 10;
size_t Options::maxDebiasIter = 200;
size_t Options::maxItersPerCycle = 20;

Options::Options() {


}


void Options::parseAndSetKeyValue(std::string key, float value){
	if (key.compare(std::string("blockSize"))){
		blockSize = int(value);
	} else if (key.compare(std::string("Mk"))){
		Mk = value;
	} else if (key.compare(std::string("Mw"))){
		Mw = value;
	} else if (key.compare(std::string("M"))){
		M = int(value);
	} else if (key.compare(std::string("A"))){
		A = int(value);
	} else if (key.compare(std::string("tau"))){
		tau = value;
	} else if (key.compare(std::string("sigma"))){
		sigma = value;
	} else if (key.compare(std::string("tolP"))){
		tolP = value;
	} else if (key.compare(std::string("eta"))){
		eta = int(value);
	} else if (key.compare(std::string("M_safeguard"))){
		M_safeguard = int(value);
	} else if (key.compare(std::string("wavelet_level"))){
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
//	std::cout << "Options\n------------------------------\n";
	while(fp >> key >> value){
//		std::cout << key << ": " << value << "\n";
		parseAndSetKeyValue(key, value);
	}
	fp.close();
//	std::cout << "------------------------------\n";
}

void Options::dumpOptions()
{
	std::cout << "Mk: " <<  Options::Mk << "\n";
	std::cout << "Mw: " <<   Options::Mw << "\n";
	std::cout << "blockSize: " <<   Options::blockSize << "\n";
	std::cout << "M: " <<   Options::M << "\n";
	std::cout << "A: " <<   Options::A << "\n";
	std::cout << "eta: " <<   Options::eta << "\n";
	std::cout << "tau: " <<   Options::tau << "\n"; //increasing helps, but makes it slower (more sparse solution)
	std::cout << "lambda: " <<   Options::lambda << "\n";
	std::cout << "sigma: " <<   Options::sigma << "\n"; // decreasing doesn't help much
	std::cout << "tolP: " <<   Options::tolP << "\n";
	std::cout << "tolD: " <<   Options::tolD << "\n"; // decreasing helps
	std::cout << "M_safeguard: " <<   Options::M_safeguard << "\n";
	std::cout << "wavelet_level: " <<   Options::wavelet_level << "\n"; // increasing this helps a lot
	std::cout << "maxIter: " <<   Options::maxIter << "\n";
	std::cout << "minIter: " <<   Options::minIter << "\n";
	std::cout << "maxDebiasIter: " <<   Options::maxDebiasIter << "\n";
	std::cout << "maxItersPerCycle: " <<   Options::maxItersPerCycle << "\n";
}
Options::~Options() {
}

