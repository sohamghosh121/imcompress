/*
 * SpaRSA.cpp
 *
 *  Created on: 28 Oct 2015
 *      Author: sohamghosh
 */

#include "SpaRSA.h"

using namespace cv;
SpaRSA::SpaRSA(Mat y, Mat phi) {
	std::cout << "SpaRSA reconstruction\n";
	this->y = y;
	this->phi = phi;
	runAlgorithm();
}

void SpaRSA::runAlgorithm(){
	t = 0;
	x_t = Mat(Size(1,phi.cols), CV_32FC1); // initialises to zero (lines 417-432 SpaRSA.m)
	x_t_plus_1 = Mat(Size(1,phi.cols), CV_32FC1);
	runOuterIteration();
}

void SpaRSA::runOuterIteration(){

	do {
		std::cout << "t: " << t << std::endl;
		alpha_t = ((double)rand() / RAND_MAX) * (alpha_max - alpha_min) + alpha_min;
		runInnerIteration();
		if (t >= 1)
			x_t_minus_1 = x_t; // probably might have problems with copying data
		x_t = x_t_plus_1;
		t++;
//		assert(false);
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
	std::cout << "alpha " << alpha_t << std::endl;
}

void SpaRSA::updateObjectiveValues(double val){
	objectiveFunctionValues.push_back(val);
	if (t > M)
		objectiveFunctionValues.pop_front();
}

bool SpaRSA::checkAcceptanceCriterion(){
	if (t==0 || objectiveFunctionValues.size() < this->M)
		return true;
	double currObj;
	currObj = objectiveFunctionValue(x_t_plus_1);
	std::deque<double>::iterator res = std::max_element(objectiveFunctionValues.begin(), objectiveFunctionValues.end());
	std::cout << "haila " << objectiveFunctionValues.size() << std::endl;
	if (isnan(*res))
		return false;
	double maxValue = *res;
//	return true;
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
	std::cout << f;
	assert(false);
	double o = norm(f, 2) + tau * norm(x, 1);
//	std::cout << "\n" << x;
	std::cout << "obj: " << o << std::endl;
	return o;
}

double SpaRSA::soft(double u, double a){
	if (u < 0){
		return -1 * fmax(abs(u) - a, 0);
	} else {
		return 1 * fmax(abs(u) - a, 0);
	}
}

Mat SpaRSA::del_f(Mat x){
//	std::cout << "del_f\n";
	Mat At_A;
	Mat firstTerm;
	Mat del_f;
//	printf("phi: (%d, %d)\tx: (%d, %d)\ty: (%d, %d)\n", phi.rows, phi.cols, x.rows, x.cols, y.rows, y.cols);
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

