/*
 * SpaRSAL2L1.cpp
 *
 *  Created on: 12 Jan 2016
 *      Author: sohamghosh
 */

#include "SpaRSA_noSI.h"

using namespace cv;

SpaRSA_noSI::SpaRSA_noSI(Mat y, Mat phi) : SpaRSA(y, phi){
	// TODO Auto-generated constructor stub
}

SpaRSA_noSI::~SpaRSA_noSI() {
	// TODO Auto-generated destructor stub
}

void SpaRSA_noSI::chooseAlpha(){
		Mat diff;
		subtract(x_t, x_t_minus_1, diff);
		double dd = pow(norm(diff, NORM_L2), 2);
		Mat phi_diff;
		gemm(phi, diff, 1.0, noArray(), 0.0, phi_diff);
		double dGd = pow(norm(phi_diff, NORM_L2), 2);
//		std::cout << "\tdd: " << dd << "\tdGd: " << dGd;
		alpha_t = fmin(this->alpha_max, fmax(this->alpha_min, dGd/dd));
}

float SpaRSA_noSI::objectiveFunctionValue(Mat x){
	Mat f;
	gemm(phi, x, -1.0, y, 1.0, f);
	double o = 0.5 * pow(norm(f, NORM_L2), 2) + tau * norm(x, NORM_L1); // l2 - l1
	return o;
}

void SpaRSA_noSI::solveSubproblem(){
	Mat u = x_t - 1/alpha_t * (del_f(x_t));  // equation 7
	for (int i = 0; i < u.rows; i++){
		x_t_plus_1.at<float>(i,0) = soft(u.at<float>(i,0), tau/alpha_t); // equation 12
	}
}

Mat SpaRSA_noSI::del_f(Mat x){
	Mat resid;
//	printf("x: (%d, %d)\tphi: (%d, %d)\n", x.rows, x.cols, phi.rows, phi.cols);
	gemm(phi, x, 1.0, y, -1.0, resid);
	Mat grad_q;
	gemm(phi.t(), resid, 1.0, noArray(), 0.0, grad_q);
	return grad_q;
}

float SpaRSA_noSI::soft(float u, float a){
	float y = fmax(abs(u) - a, 0);
	return u * y/(y + a);
}
