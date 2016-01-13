/*
 * SpaRSA.cpp
 *
 *  Created on: 28 Oct 2015
 *      Author: sohamghosh
 */

#include "SpaRSA.h"

using namespace cv;
SpaRSA::SpaRSA(Size size) {
	x_t = Mat::zeros(size, CV_32FC1); // initialises to zero (lines 417-432 SpaRSA.m)
//	std::cout << "Initial objective: " << objectiveFunctionValue(x_t) << "\n";
	x_t_plus_1 = Mat::zeros(size, CV_32FC1);
	x_t_minus_1 = Mat::zeros(size, CV_32FC1);
}

void SpaRSA::runAlgorithm(){
	t = 0;
	runOuterIteration();
}

void SpaRSA::runOuterIteration(){
	do {
		runInnerIteration();
		x_t.copyTo(x_t_minus_1);
		x_t_plus_1.copyTo(x_t);
//		std::cout << "t: " << t << "\tobj: " << objectiveFunctionValue(x_t) << "\talpha: " << alpha_t ;
		t++;
		updateObjectiveValues(objectiveFunctionValue(x_t));
		chooseAlpha();
	}while(!checkStoppingCriterion());
}

void SpaRSA::runInnerIteration(){
	itersThisCycle = 0;
	do {
		itersThisCycle++;
		solveSubproblem();
		if (checkAcceptanceCriterion())
			break;
		updateAlpha();
	} while(true);
}

void SpaRSA::updateAlpha(){
//	std::cout << "(t=" << t << ") obj: " << objectiveFunctionValue(x_t_plus_1) << " not accepted, raising alpha to " << alpha_t * eta << "\n";
	alpha_t = eta * alpha_t;
}

void SpaRSA::updateObjectiveValues(float val){
	objectiveFunctionValues.push_back(val);
	if (t > M)
		objectiveFunctionValues.pop_front();
}

bool SpaRSA::checkAcceptanceCriterion(){
	if (t==0 || itersThisCycle > maxItersPerCycle)
		return true;
	double currObj;
	currObj = objectiveFunctionValue(x_t_plus_1);
	std::deque<double>::iterator res = std::max_element(objectiveFunctionValues.begin(), objectiveFunctionValues.end());
	if (isnan(*res))
		return false;
	double maxValue = *res;
	if (currObj <= maxValue - (sigma/2 * alpha_t * norm(x_t_plus_1 - x_t, NORM_L2))){
		return true;
	} else {
		return false;
	}
}

bool SpaRSA::checkStoppingCriterion(){ // first using simple termination criteria in equation 26
	if (t > maxIter){
		return true;
	}
//	std::cout << "\t rel_change: " << norm(x_t - x_t_minus_1, NORM_L2)/norm(x_t, NORM_L2) << "\n";
	if (norm(x_t - x_t_minus_1, NORM_L2)/norm(x_t, NORM_L2) <= tolP){
		return true;
	} else
		return false;
}

Mat SpaRSA::reconstructed(){
	return x_t;
}

SpaRSA::~SpaRSA() {
}

