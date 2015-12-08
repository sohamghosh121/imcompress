/*
 * Encoder.cpp
 *
 *  Created on: 12 Oct 2015
 *      Author: sohamghosh
 */

#include "Encoder.h"
#include "SBHE.h"
#include "Wavelet.h"
#include <cmath>

using namespace cv;

Encoder::Encoder(Mat img) {
	cv::cvtColor(img, this->img, CV_RGB2GRAY, 0);
	this->img.convertTo(this->img, CV_32FC1);
	this->keyPhi = getPhi(this->opts.getMk());
	this->nonkeyPhi = getPhi(this->opts.getMw());
}


cv::Mat Encoder::getPhi(double measurementRate){  // measurement rate = Mr/B^2
	double B_sq = this->opts.getBlockSize() * this->opts.getBlockSize();
	return getPhi(int(measurementRate * B_sq), B_sq);
}

cv::Mat Encoder::getPhi(int m, int n){
	SBHE h = SBHE(m, n, n / 16, 11);
	return h.getSBHEmat();
}

Mat Encoder::getNthBlock(int n, Mat gob){ // this is 0 indexed
//	assert(n >= 0 && n < (this->img.cols * this->img.rows/(this->opts.getBlockSize() * this->opts.getBlockSize())));
	assert(n >= 0 && n < this->opts.getM() * this->opts.getM());
	return gob.rowRange(n*this->opts.getBlockSize(), (n+1)*this->opts.getBlockSize() - 1).clone();
}

Mat Encoder::getGOB(int n){
	assert(n >= 0 && n < (this->img.cols * this->img.rows/(pow(this->opts.getBlockSize(), 2) * pow(this->opts.getBlockSize(), 2))));
	int rowStart, colStart;
	int nc = this->img.cols / (this->opts.getBlockSize() * this->opts.getM());
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
	assert(img.cols == img.rows);
	int numGOBs = this->img.cols * this->img.rows/(pow(this->opts.getBlockSize(), 2) * pow(this->opts.getBlockSize(), 2));
	Mat y;
	this->f = Wavelet(this->img, Wavelet::DWT).getResult();
	for (int i = 0; i < numGOBs; i++){
		Mat gob = getGOB(i);
		for (int j = 0; j < pow(this->opts.getM(), 2); j++){
			Mat block = getNthBlock(j, gob);
			if (j == 0){  // key block
				y = encodeKeyBlock(block);
			} else {
				y = encodeNonKeyBlock(block);
			}
			encoded[i] = y;
		}
	}
}

cv::Mat Encoder::getKeyPhi(){
	return this->keyPhi;
}
cv::Mat Encoder::getnonkeyPhi(){
	return this->nonkeyPhi;
}

std::map<int, cv::Mat> Encoder::getEncodedValues(){
	return this->encoded;
}

Encoder::~Encoder() {
}

