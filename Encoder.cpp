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
	this->img = img;
	this->img.convertTo(this->img, CV_32FC1);
	this->keyPhi = getPhi(opts.getMk());
	this->nonkeyPhi = getPhi(opts.getMw());
}


cv::Mat Encoder::getPhi(double measurementRate){  // measurement rate = Mr/B^2
	double B_sq = opts.getBlockSize() * opts.getBlockSize();
	return getPhi(int(measurementRate * B_sq), B_sq);
}

cv::Mat Encoder::getPhi(int m, int n){
	SBHE h = SBHE(m, n, n / 8, 11);
	return h.getSBHEmat();
}

Mat Encoder::getNthBlock(int n, Mat gob){ // this is 0 indexed
	assert(n >= 0 && n < opts.getM() * opts.getM());
	return gob.rowRange(n*opts.getBlockSize()*opts.getBlockSize(), (n+1)*opts.getBlockSize()*opts.getBlockSize()).clone();
}

Mat Encoder::getGOB(int n){
	assert(n >= 0 && n < (this->img.cols * this->img.rows/(pow(opts.getBlockSize(), 2) * pow(opts.getM(), 2))));
	int rowStart, colStart;
	int nc = this->f.cols / (opts.getBlockSize() * opts.getM());
	colStart = (n % nc) * opts.getBlockSize() * opts.getM();
	rowStart = (n / nc) *opts.getBlockSize() * opts.getM();
	Mat x = Mat(this->f.colRange(colStart, colStart+opts.getBlockSize() * opts.getM()).rowRange(rowStart, rowStart+opts.getBlockSize() * opts.getM()));
	return x.clone().reshape(1, opts.getBlockSize()*opts.getBlockSize()*opts.getM()*opts.getM());
}

Mat Encoder::encodeBlock(Mat x, Mat phi){  // (MxN) x (Nx1)
	Mat res = Mat(phi.rows, 1, CV_32FC1);
	Mat zero = Mat::zeros(phi.rows, 1, CV_32FC1);
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
	int numGOBs = this->img.cols * this->img.rows/(pow(opts.getBlockSize(), 2) * pow(opts.getM(), 2));
	Mat y;
	this->f = Wavelet(this->img, Wavelet::DWT).getResult();  // wavelet transform (CDF 9/7)
	for (int i = 0; i < numGOBs; i++){
		Mat gob = getGOB(i);
		for (int j = 0; j < pow(opts.getM(), 2); j++){
			Mat block = getNthBlock(j, gob);
			if (j == 0){  // key block
				y = encodeKeyBlock(block);
			} else {
				y = encodeNonKeyBlock(block);
			}
			encoded[i*pow(opts.getM(),2) +j] = y;
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

