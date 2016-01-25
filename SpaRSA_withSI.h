/*
 * SpaRSAL1L1.h
 *
 *  Created on: 12 Jan 2016
 *      Author: sohamghosh
 */

#include "SpaRSA.h"

#ifndef SPARSAL1L1_H_
#define SPARSAL1L1_H_

class SpaRSA_withSI : public SpaRSA {
protected:
	cv::Mat si;
	float lambda;  // another regularisation term -_-

	void solveSubproblem(); // solve x_{t+1} sub problem
	float solve(float z, float u, float w);
	float objectiveFunctionValue(cv::Mat);
	cv::Mat del_f(cv::Mat x);
public:
	SpaRSA_withSI(cv::Mat, cv::Mat, cv::Mat);
	virtual ~SpaRSA_withSI();
};

#endif /* SPARSAL1L1_H_ */
