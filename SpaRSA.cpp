/*
 * SpaRSA.cpp
 *
 *  Created on: 28 Oct 2015
 *      Author: sohamghosh
 */

#include "SpaRSA.h"

using namespace cv;

SpaRSA::SpaRSA(Mat y, Mat phi) {
	this->phi = phi;
	this->y = y;

	Size size = Size(1, phi.cols);
	this->tau = Options::tau;
	this->eta = Options::eta;
	this->M = Options::M_safeguard;
	this->maxIter = Options::maxIter;
	this->minIter = Options::minIter;
	this->tolP = Options::tolP;
	this->tolD = Options::tolD;
	this->maxItersPerCycle = Options::maxItersPerCycle;
	x_t = Mat::zeros(size, CV_32FC1); // initialises to zero (lines 417-432 SpaRSA.m)
	x_t_plus_1 = Mat::zeros(size, CV_32FC1);
	x_t_minus_1 = Mat::zeros(size, CV_32FC1);
}

void SpaRSA::warmStart(Mat x){
	x.copyTo(x_t);
	x_t_plus_1 = Mat::zeros(x.size(), CV_32FC1);
	x_t_minus_1 = Mat::zeros(x.size(), CV_32FC1);
}

void SpaRSA::runAlgorithm(){
	t = 0;
	this->objectiveFunctionValues.clear();
	updateObjectiveValues(objectiveFunctionValue(x_t)); // update with initial value
	runOuterIteration();
}

void SpaRSA::chooseAlpha(){
	Mat diff;
	subtract(x_t, x_t_minus_1, diff);
	double dd = pow(norm(diff, NORM_L2), 2);
	Mat phi_diff;
	gemm(phi, diff, 1.0, noArray(), 0.0, phi_diff);
	double dGd = pow(norm(phi_diff, NORM_L2), 2);
	alpha_t = fmin(this->alpha_max, fmax(this->alpha_min, dGd/dd));
}


void SpaRSA::runOuterIteration(){
	while(true) {
		runInnerIteration();
		x_t.copyTo(x_t_minus_1);
		x_t_plus_1.copyTo(x_t);
		t++;
		updateObjectiveValues(objectiveFunctionValue(x_t));
		if (checkStoppingCriterion())
			break;
		chooseAlpha();
	}
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

void SpaRSA::runDebiasingPhase(){
	Mat resid, x_debias;
	x_t.copyTo(x_debias);
	int debias_t = 0;
	std::vector<int> zeroind;
	for (int i = 0; i < x_t.cols; i++){
		if (x_debias.at<float>(0,i) == 0.0)
			zeroind.push_back(i);
	}
	Mat pvec, rvec;
	gemm(phi, x_t, 1.0, y, -1.0, resid);
	gemm(phi.t(), resid, 1.0, noArray(), 0.0, rvec);
	int idx;
	for (int const& idx: zeroind){  // mask vector
		rvec.at<float>(0, idx) = 0;
	}
	float rTr_cg = pow(norm(rvec, NORM_L2), 2), rTr_cg_plus;
	float tol_debias = tolD * rTr_cg;
	rvec.copyTo(pvec);
	pvec = -pvec;

	while(true){
		Mat RWpvec, Apvec;
		gemm(phi, pvec, 1.0, noArray(), 0.0, RWpvec);
		gemm(phi.t(), RWpvec, 1.0, noArray(), 0.0, Apvec);
		for (int const& idx: zeroind){  // mask vector
			Apvec.at<float>(0, idx) = 0;
		}

		Mat den;
		gemm(pvec.t(), Apvec, 1.0, noArray(), 0.0, den);
		float alpha_cg = rTr_cg/den.at<float>(0,0);
		x_debias = x_debias + alpha_cg * pvec;
		resid = resid + alpha_cg * RWpvec;
		rvec  = rvec  + alpha_cg * Apvec;
		rTr_cg_plus = pow(norm(rvec, NORM_L2), 2);
		float beta_cg = rTr_cg_plus/rTr_cg;
		pvec = -rvec + beta_cg * pvec;
		rTr_cg = rTr_cg_plus;
		debias_t++;
		if (rTr_cg < tol_debias || debias_t > maxDebiasIter){
			break;
		}
	}
	x_debias.copyTo(x_t);

}

void SpaRSA::updateAlpha(){
	alpha_t = eta * alpha_t;
}

void SpaRSA::updateObjectiveValues(float val){
	objectiveFunctionValues.push_back(val);
	if (t > M)
		objectiveFunctionValues.pop_front();
}
/*
 * For a solution to be accepted, it must be at least slightly lower than past M objective function values
 */
bool SpaRSA::checkAcceptanceCriterion(){
	if (itersThisCycle > maxItersPerCycle)
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

/*
 * Use relative change as stopping criterion
 */
bool SpaRSA::checkStoppingCriterion(){
	if (t > maxIter){
		return true;
	}
	if (t < minIter){
		return false;
	}
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

