/*
 * SpaRSAjoint.h
 *
 *  Created on: 26 Jan 2016
 *      Author: sohamghosh
 */

#ifndef SPARSA_JOINT_H_
#define SPARSA_JOINT_H_


#include "SpaRSA.h"

class SpaRSA_joint : public SpaRSA {
protected:
//	void chooseAlpha();
	cv::Mat keyPhi, nonkeyPhi, keyPhi_T, nonkeyPhi_T;
	void chooseAlpha();
	void solveSubproblem(); // solve x_{t+1} sub problem
	float objectiveFunctionValue(cv::Mat&);
	float soft(float, float);
	cv::Mat A_x(cv::Mat);
	cv::Mat A_resid(cv::Mat);
	cv::Mat del_f(cv::Mat& x);
public:
	SpaRSA_joint(cv::Mat y, cv::Mat keyPhi, cv::Mat nonkeyPhi);
	virtual ~SpaRSA_joint();
};

#endif /* SPARSA_JOINT_H_ */
