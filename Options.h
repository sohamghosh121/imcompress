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
	static int B;
	static int A;
	static float eta;
	static float tau;
	static float sigma;
	static float tolP;
	static int M_safeguard;
	static int wavelet_level;



	static void parseOptionsFile(char* filename);
	static void parseAndSetKeyValue(std::string key, float value);
};

#endif /* OPTIONS_H_ */
