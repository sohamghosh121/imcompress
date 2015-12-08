/*
 * SpaRSA.cpp
 *
 *  Created on: 28 Oct 2015
 *      Author: sohamghosh
 */

#include "SpaRSA.h"

using namespace cv;
SpaRSA::SpaRSA(Mat y, Mat phi) {
	this->y = y;
	this->phi = phi;
	runAlgorithm();
}

void SpaRSA::runAlgorithm(){
	t = 0;
	x_t = Mat(Size(phi.cols,1), CV_32FC1); // initialises to zero (lines 417-432 SpaRSA.m)
	runOuterIteration();
}

void SpaRSA::runOuterIteration(){

	do {
		alpha_t = ((double)rand() / RAND_MAX) * (alpha_max - alpha_min) + alpha_min;
		runInnerIteration();
		if (t >= 1)
			x_t_minus_1 = x_t; // probably might have problems with copying data
		x_t = x_t_plus_1;
		t++;
		updateObjectiveValues(objectiveFunctionValue(x_t));
	}while(!checkStoppingCriterion());
}

void SpaRSA::runInnerIteration(){
	do {
		solveSubproblem();
		updateAlpha();
	} while(!checkAcceptanceCriterion());
}

void SpaRSA::updateAlpha(){
	alpha_t = eta * alpha_t;
}

void SpaRSA::updateObjectiveValues(double val){
	objectiveFunctionValues.push_back(val);
	if (t > M)
		objectiveFunctionValues.pop_front();
}

bool SpaRSA::checkAcceptanceCriterion(){
	double currObj = objectiveFunctionValue(x_t_plus_1);
	std::deque<double>::iterator res = std::max_element(objectiveFunctionValues.begin(), objectiveFunctionValues.end());
	double maxValue = *res;
	if (currObj < maxValue - (sigma/2 * alpha_t * norm(x_t_plus_1 - x_t, 2))){
		return true;
	} else {
		return false;
	}
}

bool SpaRSA::checkStoppingCriterion(){ // first using simple termination criteria in equation 26
	if (t > 1) {
		return (norm(x_t - x_t_minus_1, 2)/norm(x_t, 2) <= tolP);
	} else {
		return false;
	}
}

double SpaRSA::objectiveFunctionValue(Mat x){
	Mat f;
	gemm(phi, x, -1.0, y, 1.0, f);
	return norm(f, 2) + tau * norm(x, 1);
}

double SpaRSA::soft(double u, double a){
	if (u < 0){
		return -1 * fmax(abs(u) - a, 0);
	} else {
		return 1 * fmax(abs(u) - a, 0);
	}
}

Mat SpaRSA::del_f(Mat x){
	Mat At_A;
	Mat firstTerm;
	Mat del_f;
	gemm(phi.t(), phi, 1.0, noArray(), 0.0, At_A);
	gemm(At_A, x, 2.0, noArray(), 0.0, firstTerm);
	gemm(phi.t(), y, -2.0, firstTerm, 1.0, del_f);
	return del_f;
}

void SpaRSA::solveSubproblem(){
	Mat u = x_t - 1/alpha_t * (del_f(x_t));  // equation 7
	for (int i = 0; i < u.rows; i++){
		x_t_plus_1.at<double>(i,0) = soft(u.at<double>(i,0), tau/alpha_t); // equation 12
	}
}

Mat SpaRSA::reconstructed(){
	return x_t;
}

SpaRSA::~SpaRSA() {
}

