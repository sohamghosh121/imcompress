/*
 * GradientDescent.cpp
 *
 *  Created on: 20 Oct 2015
 *      Author: sohamghosh
 */

#include "GradientDescent.h"


using namespace cv;

GradientDescent::GradientDescent(Mat y, Mat theta) {
	this->x = Mat::zeros(this->theta.cols, 1, CV_32FC1);
	cv::RNG rng(666);
	rng.fill(x, RNG::UNIFORM, 0.0, 255.0);
	this->y = y;
	this->theta = theta;
	this->thetaTtheta = norm(theta, NORM_L2, noArray());
}

void GradientDescent::doGradientDescent(){
	int numIterations = this->maxIterations;
	Mat grad;
	while (numIterations > 0){
		grad = getGradient();
		x = x - this->alpha * grad;
		numIterations --;
	}
}

Mat GradientDescent::getGradient(){
	Mat gradient = Mat::zeros(this->x.rows, 1, CV_32FC1);
	Mat lassoTerm = Mat::zeros(this->x.rows, 1, CV_32FC1);
	gemm(this->theta.t(), this->y.t(), -1.0, this->x, this->thetaTtheta, gradient);
	for (int i = 0; i < this->x.rows; i++){
		if (this->x.at<double>(i, 1) >= 0){
			lassoTerm.at<double>(i,1) = 1;
		} else {
			lassoTerm.at<double>(i,1) = -1;
		}
	}
	gradient = 2 * gradient + this->lambda * lassoTerm;
}

GradientDescent::~GradientDescent() {
	// TODO Auto-generated destructor stub
}

