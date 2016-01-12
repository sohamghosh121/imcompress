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
	x_t = Mat::zeros(Size(1,phi.cols), CV_32FC1); // initialises to zero (lines 417-432 SpaRSA.m)
//	std::cout << "Initial objective: " << objectiveFunctionValue(x_t) << "\n";
	x_t_plus_1 = Mat::zeros(Size(1,phi.cols), CV_32FC1);
	x_t_minus_1 = Mat::zeros(Size(1,phi.cols), CV_32FC1);
	runOuterIteration();
}

void SpaRSA::runOuterIteration(){
	do {
		runInnerIteration();
		x_t.copyTo(x_t_minus_1);
		x_t_plus_1.copyTo(x_t);
//		std::cout << "t: " << t << "\tobj: " << objectiveFunctionValue(x_t) << "\talpha: " << alpha_t << "\t";
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

void SpaRSA::chooseAlpha(){
//	alpha_t *= this->alphaFactor;

		Mat diff;
		subtract(x_t, x_t_minus_1, diff);
		double dd = pow(norm(diff, NORM_L2), 2);
		Mat phi_diff;
		gemm(phi, diff, 1.0, noArray(), 0.0, phi_diff);
		double dGd = pow(norm(phi_diff, NORM_L2), 2);
//		std::cout << "dd: " << dd << "\tdGd: " << dGd;
		alpha_t = fmin(this->alpha_max, fmax(this->alpha_min, dGd/dd));
}

void SpaRSA::updateAlpha(){
//	std::cout << "\n(t=" << t << ") obj: " << objectiveFunctionValue(x_t_plus_1) << " not accepted, raising alpha to " << alpha_t * eta << "\n";
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

float SpaRSA::objectiveFunctionValue(Mat x){
	Mat f;
	gemm(phi, x, -1.0, y, 1.0, f);
	double o = 0.5 * norm(f, NORM_L2) + tau * norm(x, NORM_L1); // l2 - l1
	return o;
}

float SpaRSA::soft(float u, float a){
	float y = fmax(abs(u) - a, 0);
	return u * y/(y + a);
}

Mat SpaRSA::del_f(Mat x){
	Mat resid;
	gemm(phi, x, 1.0, y, -1.0, resid);
	Mat grad_q;
	gemm(phi.t(), resid, 1.0, noArray(), 0.0, grad_q);
	return grad_q;
}

void SpaRSA::solveSubproblem(){
	Mat u = x_t - 1/alpha_t * (del_f(x_t));  // equation 7
	for (int i = 0; i < u.rows; i++){
		x_t_plus_1.at<float>(i,0) = soft(u.at<float>(i,0), tau/alpha_t); // equation 12
	}
//	std::cout << "\n" << x_t_plus_1.t() << "\n";
}

Mat SpaRSA::reconstructed(){
	return x_t;
}

SpaRSA::~SpaRSA() {
}

