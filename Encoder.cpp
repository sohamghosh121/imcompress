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
	this->keyPhi = getPhi(this->opts.getMk());
	this->nonkeyPhi = getPhi(this->opts.getMw());
}


cv::Mat Encoder::getPhi(double measurementRate){  // measurement rate = Mr/B^2
	double B_sq = this->opts.getBlockSize() * this->opts.getBlockSize();
	return getPhi(measurementRate * B_sq, B_sq);
}

cv::Mat Encoder::getPhi(int m, int n){
	Mat dst = Mat(m, n, CV_32FC1);
	static cv::RNG rng(666);
	rng.fill(dst, RNG::NORMAL, 0.0, 1.0/(double(n)), false);
	return dst;
}

Mat Encoder::getNthBlock(int n){ // this is 0 indexed
	assert(n > 0 && n < (this->img.cols * this->img.rows/(this->opts.getBlockSize() * this->opts.getBlockSize())));
	int rowStart, colStart;
	int nc = this->img.cols / this->opts.getBlockSize();
	int nr = this->img.rows / this->opts.getBlockSize();
	colStart = (n % nc) * this->opts.getBlockSize();
	rowStart = (n / nc) * this->opts.getBlockSize();
	Mat x = Mat(this->img.colRange(colStart, colStart+this->opts.getBlockSize()).rowRange(rowStart, rowStart+this->opts.getBlockSize()));
	return x.clone().reshape(1, this->opts.getBlockSize()*this->opts.getBlockSize());
}

Mat Encoder::encodeBlock(Mat x, Mat phi){  // (MxN) x (Nx1)
	Mat res = Mat(phi.rows, 1, CV_32FC1);
	Mat zero = Mat::zeros(phi.rows, 1, CV_32FC1);
//	printf("res.size: (%d, %d)\tzero.size: (%d, %d)\tx.size: (%d, %d)\tphi.size: (%d, %d)\n", res.rows, res.cols, zero.rows, zero.cols, x.rows, x.cols, phi.rows, phi.cols);
	gemm(phi, x, 1.0, noArray(), 0.0, res);
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
	nr = this->img.rows / this->opts.getBlockSize();
	nc = this->img.cols / this->opts.getBlockSize();
//	if (hasPhi()){
	M_count = 0;
	Mat y_key, y_wz;
	for (i = 1; i < nr * nc; i++, M_count--){
		if (M_count == 0){ // finished encoding all GOB blocks, next GOB
			y_key = encodeKeyBlock(getNthBlock(i));
			this->encoded[i] = &y_key;
			M_count = this->opts.getM() - 1;
		} else {
			y_wz = encodeNonKeyBlock(getNthBlock(i));
			this->encoded[i] = &y_wz;
		}
	}

}

cv::Mat Encoder::getKeyPhi(){
	return this->keyPhi;
}
cv::Mat Encoder::getnonkeyPhi(){
	return this->nonkeyPhi;
}

std::map<int, cv::Mat *> Encoder::getEncodedValues(){
	return this->encoded;
}

Encoder::~Encoder() {
}

