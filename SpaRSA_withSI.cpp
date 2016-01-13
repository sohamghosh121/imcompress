/*
 * SpaRSAL1L1.cpp
 *
 *  Created on: 12 Jan 2016
 *      Author: sohamghosh
 */

#include "SpaRSA_withSI.h"

using namespace cv;

SpaRSA_withSI::SpaRSA_withSI(Mat y, Mat phi, Mat si) : SpaRSA(Size(1,phi.cols)){
	this->y = y;
	this->phi = phi;
	this->si = si;
}

SpaRSA_withSI::~SpaRSA_withSI() {
	// TODO Auto-generated destructor stub
}

void SpaRSA_withSI::chooseAlpha(){
//	alpha_t *= this->alphaFactor;
}

float SpaRSA_withSI::objectiveFunctionValue(Mat x){
	Mat f;
	gemm(phi, x, -1.0, y, 1.0, f);
	double o = 0.5 * norm(f, NORM_L2) + tau * norm(x, NORM_L1) + lambda * norm(x - si, NORM_L2);
	return o;
}

void SpaRSA_withSI::solveSubproblem(){
}

Mat SpaRSA_withSI::del_f(Mat x){
	Mat resid;
//	printf("x: (%d, %d)\tphi: (%d, %d)\n", x.rows, x.cols, phi.rows, phi.cols);
	gemm(phi, x, 1.0, y, -1.0, resid);
	Mat grad_q;
	gemm(phi.t(), resid, 1.0, noArray(), 0.0, grad_q);
	return grad_q;
}

