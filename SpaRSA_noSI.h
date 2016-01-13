/*
 * SpaRSAL2L1.h
 *
 *  Created on: 12 Jan 2016
 *      Author: sohamghosh
 */

#ifndef SPARSA_NOSI_H_
#define SPARSA_NOSI_H_

#include "SpaRSA.h"

class SpaRSA_noSI : public SpaRSA{
protected:
	cv::Mat y, phi;

	void chooseAlpha();
	void solveSubproblem(); // solve x_{t+1} sub problem
	float objectiveFunctionValue(cv::Mat);
	cv::Mat del_f(cv::Mat x);
	float soft(float u, float a);
public:
	SpaRSA_noSI(cv::Mat, cv::Mat);
	virtual ~SpaRSA_noSI();
};

#endif /* SPARSA_NOSI_H_ */
