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
	this->keyPhi = getPhi(Options::Mk);
	this->nonkeyPhi = getPhi(Options::Mw);
	assert(img.rows % (Options::blockSize * Options::M) == 0); // divisibility assertions
	assert(img.cols % (Options::blockSize * Options::M) == 0);
}


cv::Mat Encoder::getPhi(double measurementRate){  // measurement rate = Mr/B^2
	double B_sq = Options::blockSize * Options::blockSize;
	return getPhi(int(measurementRate * B_sq), B_sq);
}

cv::Mat Encoder::getPhi(int m, int n){
	SBHE h = SBHE(m, n, 61);
	return h.getSBHEmat();
}

Mat Encoder::getNthBlock(int n, Mat gob){ // this is 0 indexed
	assert(n >= 0 && n < Options::M * Options::M);
	int rowStart = int(n / Options::M) * Options::blockSize;
	int colStart = int(n % Options::M) * Options::blockSize;
//	 std::cout << rowStart << ", " << colStart << "\n";
	return gob.colRange(colStart, colStart+Options::blockSize).rowRange(rowStart, rowStart + Options::blockSize).clone().reshape(1, pow(Options::blockSize, 2));
}


/*
 *  GOB: Group Of Blocks
 *  - Each GOB has M x M blocks
 */
Mat Encoder::getGOB(int n){
	assert(n >= 0 && n < (this->img.cols * this->img.rows/(pow(Options::blockSize, 2) * pow(Options::M, 2))));
	int rowStart, colStart;
	int nc = this->f.cols / (Options::blockSize * Options::M);
	colStart = (n % nc) * Options::blockSize * Options::M;
	rowStart = (n / nc) * Options::blockSize * Options::M;

	Mat x = Mat(this->f.colRange(colStart, colStart+Options::blockSize * Options::M).rowRange(rowStart, rowStart+Options::blockSize * Options::M));
	Mat GOB = x.clone();
	return GOB;
}

/*
 * Get CS measurements
 * y = phi * x
 */
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
	int numGOBs = this->img.cols * this->img.rows/(pow(Options::blockSize * Options::M, 2));
	Mat y;
	this->f = Wavelet(this->img, Wavelet::DWT).getResult();  // wavelet transform (CDF 9/7)
	f = f.reshape(1, this->img.cols * this->img.rows); // resize to N^2 x 1 vector
	f = SBHE::scrambleInputSignal(f, Options::A).reshape(1, img.rows);
	for (int i = 0; i < numGOBs; i++){
		Mat gob = getGOB(i);
		for (int j = 0; j < pow(Options::M, 2); j++){
			Mat block = getNthBlock(j, gob); // given offset within GOB, get block (Nx1)
			if (j == 0){ // first block of every GOB is a key block
				y = encodeKeyBlock(block);
			} else {
				y = encodeNonKeyBlock(block);
			}
			encoded[i*pow(Options::M,2) +j] = y;
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

