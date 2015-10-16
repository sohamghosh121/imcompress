/*
 * Encoder.cpp
 *
 *  Created on: 12 Oct 2015
 *      Author: sohamghosh
 */

#include "Encoder.h"

using namespace cv;

Encoder::Encoder(Mat img) {
	cv::cvtColor(img, this->img, CV_RGB2GRAY, 0);
	this->img.convertTo(this->img, CV_32FC1);
	this->keyPhi = getPhi(this->Mk);
	this->nonkeyPhi = getPhi(this->Mw);
}


cv::Mat Encoder::getPhi(double measurementRate){  // measurement rate = Mr/B^2
	double B_sq = this->blockSize * this->blockSize;
	return getPhi(measurementRate * B_sq, B_sq);
}

cv::Mat Encoder::getPhi(int m, int n){
	Mat dst = Mat(m, n, CV_32FC1);
	static cv::RNG rng(666);
	rng.fill(dst, RNG::NORMAL, 0.0, 1.0/(double(n)), false);
	return dst;
}

Mat Encoder::getNthBlock(int n){ // this is 0 indexed
	assert(n > 0 && n < (this->img.cols * this->img.rows/(this->blockSize * this->blockSize)));
	int rowStart, colStart;
	int nc = this->img.cols / this -> blockSize;
	int nr = this->img.rows / this -> blockSize;
	colStart = (n % nc) * this->blockSize;
	rowStart = (n / nc) * this->blockSize;
	Mat x = Mat(this->img.colRange(colStart, colStart+this->blockSize).rowRange(rowStart, rowStart+this->blockSize));
	return x.clone().reshape(1, this->blockSize*this->blockSize);
}

Mat Encoder::encodeBlock(Mat x, Mat phi){  // (MxN) x (Nx1)
	Mat res = Mat(phi.rows, 1, CV_32FC1);
	Mat zero = Mat::zeros(phi.rows, 1, CV_32FC1);
//	printf("res.size: (%d, %d)\tzero.size: (%d, %d)\tx.size: (%d, %d)\tphi.size: (%d, %d)\n", res.rows, res.cols, zero.rows, zero.cols, x.rows, x.cols, phi.rows, phi.cols);
	gemm(phi, x, 1.0, zero, 0.0, res);
	return res;
}

Mat Encoder::encodeKeyBlock(Mat x){
	return encodeBlock(x, this->keyPhi);
}

Mat Encoder::encodeNonKeyBlock(Mat x){
	return encodeBlock(x, this->nonkeyPhi);
}

void Encoder::encodeImage(){
	int i, nr, nc, M_count;
	nr = this->img.rows / this->blockSize;
	nc = this->img.cols / this->blockSize;
//	if (hasPhi()){
	Mat y_key, y_wz;
	for (i = 1; i < nr * nc; i++, M_count--){
		if (M == 0){ // finished encoding all GOB blocks, next GOB
			y_key = encodeKeyBlock(getNthBlock(i));
			this->encoded[i] = &y_key;
			M_count = this->M - 1;
		} else {
			y_wz = encodeNonKeyBlock(getNthBlock(i));
			this->encoded[i] = &y_wz;
		}
	}

}

Encoder::~Encoder() {
}

