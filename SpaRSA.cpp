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
	x_t = Mat(Size(1,phi.cols), CV_32FC1); // initialises to zero (lines 417-432 SpaRSA.m)
	x_t_plus_1 = Mat(Size(1,phi.cols), CV_32FC1);
	x_t_minus_1 = Mat(Size(1,phi.cols), CV_32FC1);
	runOuterIteration();
}

void SpaRSA::runOuterIteration(){
	do {
		runInnerIteration();
		chooseAlpha();
		x_t.copyTo(x_t_minus_1);
		x_t_plus_1.copyTo(x_t);
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

void SpaRSA::chooseAlpha(){
	alpha_t *= this->alphaFactor;
//	if (t == 0){
////		alpha_t = ((double)rand() / RAND_MAX) * (alpha_max - alpha_min) + alpha_min;
//	} else {
//
//		Mat s_t;
//		subtract(x_t, x_t_minus_1, s_t);
//		Mat r_t;
//		subtract(del_f(x_t), del_f(x_t_minus_1), r_t);
//		Mat num(1,1,CV_32FC1), den(1,1,CV_32FC1);
////		std::cout << s_t;
////		std::cout << r_t;
//		cv::gemm(s_t.t(), r_t, 1.0, noArray(), 0.0, num);
//		cv::gemm(s_t.t(), s_t, 1.0, noArray(), 0.0, den);
////		std :: cout<< t << "\t";
//		std::cout << num.at<double>(0,0)/den.at<double>(0,0) << "\n";
////		alpha_t = 1.0;
//	}
}

void SpaRSA::updateAlpha(){
	alpha_t = eta * alpha_t;
}

void SpaRSA::updateObjectiveValues(float val){
	objectiveFunctionValues.push_back(val);
	if (t > M)
		objectiveFunctionValues.pop_front();
}

bool SpaRSA::checkAcceptanceCriterion(){
	if (t==0)
		return true;
	double currObj;
	currObj = objectiveFunctionValue(x_t_plus_1);
	std::deque<double>::iterator res = std::max_element(objectiveFunctionValues.begin(), objectiveFunctionValues.end());
	if (isnan(*res))
		return false;
	double maxValue = *res;
	if (currObj <= maxValue - (sigma/2 * alpha_t * norm(x_t_plus_1 - x_t, 2))){
		return true;
	} else {
		return false;
	}
}

bool SpaRSA::checkStoppingCriterion(){ // first using simple termination criteria in equation 26
	if (t > minIter) {
		if (t > maxIter){
			return true;
		}
		if (norm(x_t - x_t_minus_1, 2)/norm(x_t, 2) <= tolP){
			return true;
		} else
			return false;
	} else {
		return false;
	}
}

float SpaRSA::objectiveFunctionValue(Mat x){
	Mat f;
	gemm(phi, x, -1.0, y, 1.0, f);
	double o = norm(f, 2) + tau * norm(x, 1); // l2 - l1
	return o;
}

float SpaRSA::soft(float u, float a){
	if (u < 0){
		return -1 * fmax(abs(u) - a, 0);
	} else {
		return 1 * fmax(abs(u) - a, 0);
	}
}

Mat SpaRSA::del_f(Mat x){
	// del_f computation is correct
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
		x_t_plus_1.at<float>(i,0) = soft(u.at<float>(i,0), tau/alpha_t); // equation 12
	}
}

Mat SpaRSA::reconstructed(){
	return x_t;
}

SpaRSA::~SpaRSA() {
}

