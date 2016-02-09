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
int Options::A = 997;
float Options::eta = 2.0;
float Options::tau = 0.7; //increasing helps, but makes it slower (more sparse solution)
float Options::lambda = 0.05;
float Options::sigma = 0.001; // decreasing doesn't help much
float Options::tolP = 0.00001;
float Options::tolD = 0.0001; // decreasing helps
int Options::M_safeguard = 10;
int Options::wavelet_level = 10; // increasing this helps a lot
size_t Options::maxIter = 500;
size_t Options::minIter = 10;
size_t Options::maxDebiasIter = 200;
size_t Options::minDebiasIter = 5;
size_t Options::maxItersPerCycle = 100;

Options::Options() {


}


void Options::parseAndSetKeyValue(std::string key, float value){
	if (key.compare(std::string("blockSize")) == 0){
		blockSize = int(value);
	} else if (key.compare(std::string("Mk")) == 0){
		Mk = value;
	} else if (key.compare(std::string("Mw")) == 0){
		Mw = value;
	} else if (key.compare(std::string("M")) == 0){
		M = int(value);
	} else if (key.compare(std::string("A")) == 0){
		A = int(value);
	} else if (key.compare(std::string("tau")) == 0){
		tau = value;
	} else if (key.compare(std::string("sigma")) == 0){
		sigma = value;
	} else if (key.compare(std::string("tolP")) == 0){
		tolP = value;
	} else if (key.compare(std::string("tolD")) == 0){
		tolD = value;
	} else if (key.compare(std::string("eta")) == 0){
		eta = int(value);
	} else if (key.compare(std::string("M_safeguard")) == 0){
		M_safeguard = int(value);
	} else if (key.compare(std::string("wavelet_level")) == 0){
		wavelet_level = int(value);
	} else if (key.compare(std::string("maxIter")) == 0){
		maxIter = int(value);
	} else if (key.compare(std::string("minIter")) == 0){
		minIter = int(value);
	} else if (key.compare(std::string("maxDebiasIter")) == 0){
		maxDebiasIter = int(value);
	} else if (key.compare(std::string("minDebiasIter")) == 0){
		minDebiasIter = int(value);
	} else if (key.compare(std::string("maxItersPerCycle")) == 0){
		maxItersPerCycle = int(value);
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

