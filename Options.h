/*
 * Options.h
 *
 *  Created on: 19 Oct 2015
 *      Author: sohamghosh
 */

#ifndef OPTIONS_H_
#define OPTIONS_H_

#include <opencv2/opencv.hpp>

class Options {
private:
	Options();
	~Options();
public:
	static float Mk;
	static float Mw;
	static int blockSize;
	static int M;
	static int A;
	static float eta;
	static float tau;
	static float lambda;
	static float sigma;
	static float tolP;
	static float tolD;
	static int M_safeguard;
	static int wavelet_level;
	static size_t maxIter;
	static size_t minIter;
	static size_t maxDebiasIter;
	static size_t maxItersPerCycle;



	static void parseOptionsFile(char* filename);
	static void parseAndSetKeyValue(std::string key, float value);
	static void dumpOptions();
};

#endif /* OPTIONS_H_ */
