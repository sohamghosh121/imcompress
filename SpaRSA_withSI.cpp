/*
 * SpaRSAL1L1.cpp
 *
 *  Created on: 12 Jan 2016
 *      Author: sohamghosh
 */

#include "SpaRSA_withSI.h"
#include "Options.h"

using namespace cv;

SpaRSA_withSI::SpaRSA_withSI(Mat y, Mat phi, Mat si) : SpaRSA(y, phi){
	this->si = si;
	this->lambda = Options::lambda;
}

SpaRSA_withSI::~SpaRSA_withSI() {
	// TODO Auto-generated destructor stub
}

float SpaRSA_withSI::objectiveFunctionValue(Mat x){
	Mat f;
	gemm(phi, x, -1.0, y, 1.0, f);
	double o = 0.5 * norm(f, NORM_L2) + tau * norm(x, NORM_L1) + lambda * norm(x - si, NORM_L2);
	return o;
}

void SpaRSA_withSI::solveSubproblem(){
	Mat u = x_t - 1/alpha_t * (del_f(x_t));  // equation 7
	for (int i = 0; i < u.rows; i++){
		x_t_plus_1.at<float>(i,0) = solve(x_t.at<float>(i, 0), u.at<float>(i,0), si.at<float>(i, 0)); // equation 12
	}
}

float SpaRSA_withSI::solve(float z, float u, float w){
	if (w < 0){
		if (u - tau/alpha_t - lambda < w){
			return u - tau/alpha_t - lambda;
		} else if (u - tau/alpha_t + lambda < 0){
			return u - tau/alpha_t + lambda;
		} else {
			return u + tau/alpha_t + lambda;
		}
	} else {
		if (u - tau/alpha_t - lambda < 0){
			return u - tau/alpha_t - lambda;
		} else if (u + tau/alpha_t - lambda < w){
			return u + tau/alpha_t - lambda;
		} else {
			return u + tau/alpha_t + lambda;
		}
	}
}

Mat SpaRSA_withSI::del_f(Mat x){
	Mat resid;
//	printf("x: (%d, %d)\tphi: (%d, %d)\n", x.rows, x.cols, phi.rows, phi.cols);
	gemm(phi, x, 1.0, y, -1.0, resid);
	Mat grad_q;
	gemm(phi.t(), resid, 1.0, noArray(), 0.0, grad_q);
	return grad_q;
}

