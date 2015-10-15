/*
 * Encoder.cpp
 *
 *  Created on: 12 Oct 2015
 *      Author: sohamghosh
 */

#include "Encoder.h"

using namespace cv;

Encoder::Encoder(Mat img) {
	// TODO Auto-generated constructor stub
	cv::cvtColor(img, this->img, CV_RGB2GRAY, 0);
}

void Encoder::getPhi(int m, int n, cv::Mat dst){
	static cv::RNG rng(666);
	rng.fill(dst, RNG::NORMAL, 0.0, 1.0/(double(n)), false);
}

Mat Encoder::getNthBlock(int n){ // this is 0 indexed
	assert(n > 0 && n < (this->img.cols * this->img.rows/(this->blockSize * this->blockSize)));
	int rowStart, colStart;
	int nc = this->img.cols / this -> blockSize;
	int nr = this->img.rows / this -> blockSize;
	colStart = (n % nc) * this->blockSize;
	rowStart = (n / nr) * this->blockSize;
	return this->img.colRange(colStart, colStart+this->blockSize-1).rowRange(rowStart, rowStart+7);
}

Mat Encoder::encodeBlock(Mat x, Mat phi){
	Mat res = Mat();
	Mat zero = Mat::zeros(phi.rows, x.cols, CV_64FC1);
	gemm(x, phi, 1.0, zero, 0.0, res);
	return res;
}

Mat Encoder::encodeKeyBlock(Mat x){
	return encodeBlock(x, this->keyPhi);
}

Mat Encoder::encodeNonKeyBlock(Mat x){
	return encodeBlock(x, this->nonkeyPhi);
}

void Encoder::encodeImage(){
	int i, nr, nc;
	nr = this->img.rows / this->blockSize;
	nc = this->img.cols / this->blockSize;
//	if (hasPhi()){
	Mat y_w;
		for (i = 1; i < nr * nc; i++){
			y_w = encodeKeyBlock(getNthBlock(i));
			double tau = getTau();
			double lambda = getLambda(y_w);
			if (tau > lambda){
				// key block
			} else {
				// WZ block
			}
		}
//	}

}

double Encoder::getTau(){
	return 0.0;
}

double Encoder::getLambda(cv::Mat y_w){
	return 0.0;
}

double calculateMSE(Mat a, Mat b){
	Mat res = Mat::zeros(1, 1, CV_32FC1);
	Mat zero = Mat::zeros(1, 1, CV_32FC1);
	gemm((a-b).t(), (a-b), 1.0, zero, 0.0, res);
	return res.at<double>(0,0);
}

Encoder::~Encoder() {
}

