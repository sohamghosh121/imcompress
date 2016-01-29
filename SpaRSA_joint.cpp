/*
 * SpaRSAjoint.cpp
 *
 *  Created on: 26 Jan 2016
 *      Author: sohamghosh
 */

#include "SpaRSA_joint.h"

using namespace cv;
SpaRSA_joint::SpaRSA_joint(Mat y, Mat keyPhi, Mat nonkeyPhi) : SpaRSA(y, Mat()) {
	this->keyPhi = keyPhi;
	this->nonkeyPhi = nonkeyPhi;
	this->keyPhi_T = keyPhi.t();
	this->nonkeyPhi_T = nonkeyPhi.t();
}

void SpaRSA_joint::chooseAlpha(){
//		std::cout << "chooseAlpha\n";
		Mat diff;
		subtract(x_t, x_t_minus_1, diff);
		double dd = pow(norm(diff, NORM_L2), 2);
		Mat phi_diff;
		phi_diff = A_x(diff);
		double dGd = pow(norm(phi_diff, NORM_L2), 2);
//		std::cout << "\tdd: " << dd << "\tdGd: " << dGd;
		alpha_t = fmin(this->alpha_max, fmax(this->alpha_min, dGd/dd));
}

float SpaRSA_joint::objectiveFunctionValue(Mat x){
//	std::cout << "obj\n";
	Mat f;
	f = y - A_x(x);
	double o = 0.5 * pow(norm(f, NORM_L2), 2) + tau * norm(x, NORM_L1); // l2 - l1
	return o;
}

void SpaRSA_joint::solveSubproblem(){
//	std::cout << "solveSubproblem\n";
	Mat u = x_t - 1/alpha_t * (del_f(x_t));  // equation 7
	for (int i = 0; i < u.rows; i++){
		x_t_plus_1.at<float>(i,0) = soft(u.at<float>(i,0), tau/alpha_t); // equation 12
	}
}

Mat SpaRSA_joint::del_f(Mat x){
//	std::cout << "del_f\n";
	Mat resid;
	resid = y - A_x(x);
	Mat grad_q;
	A_resid(resid);
	return grad_q;
}

float SpaRSA_joint::soft(float u, float a){
	float y = fmax(abs(u) - a, 0);
	return u * y/(y + a);
}


// these methods are to calculate gemm of the joint reconstruction matrix without using much memory
Mat SpaRSA_joint::A_x(Mat x){
//	std::cout << "A_x\n";
	Mat A_x = Mat::zeros(y.size(), CV_32FC1);
	Mat temp;
	int last_idx = 0;
	for (int i = 0; i < x.rows/pow(Options::blockSize, 2); i++){
		if (i % (Options::M * Options::M) == 0){
			gemm(keyPhi, x.rowRange(i * pow(Options::blockSize, 2), (i + 1) * pow(Options::blockSize, 2)), 1.0, noArray(), 0.0, temp);
		} else {
			gemm(nonkeyPhi, x.rowRange(i * pow(Options::blockSize, 2), (i + 1) * pow(Options::blockSize, 2)), 1.0, noArray(), 0.0, temp);
		}
		temp.copyTo(A_x.rowRange(last_idx, last_idx + temp.rows));
		last_idx += temp.rows;
	}
	return A_x;
}

Mat SpaRSA_joint::A_resid(Mat resid){
//	std::cout << "A_resid\n";
	Mat A_resid = Mat::zeros(x_t.size(), CV_32FC1);
	Mat temp;
	int last_idx = 0;
	for (int i = 0; i < x_t.rows/pow(Options::blockSize, 2); i++){
		if (i % int(pow(Options::M, 2)) == 0){
			gemm(keyPhi_T, resid.rowRange(last_idx, last_idx + keyPhi_T.cols), 1.0, noArray(), 0.0, temp);
			last_idx += keyPhi_T.cols;
		} else {
			gemm(nonkeyPhi_T, resid.rowRange(last_idx, last_idx + nonkeyPhi_T.cols), 1.0, noArray(), 0.0, temp);
			last_idx += nonkeyPhi_T.cols;
		}

		temp.copyTo(A_resid.rowRange(i * pow(Options::blockSize, 2), (i + 1) * pow(Options::blockSize, 2)));
	}
	return A_resid;
}

SpaRSA_joint::~SpaRSA_joint() {
}

