/*
 * Decoder.cpp
 *
 *  Created on: 12 Oct 2015
 *      Author: sohamghosh
 */

#include "Decoder.h"

#include "SpaRSA_noSI.h"
#include "SpaRSA_withSI.h"
#include "Wavelet.h"

using namespace cv;

// first try individual recovery

Decoder::Decoder(int nr, int nc, Mat keyPhi, Mat nonkeyPhi, std::map<int, Mat> encoded) {
	this->encoded = encoded;
	this->keyPhi = keyPhi;
	this->nonkeyPhi = nonkeyPhi;
	this->imsize = cv::Size(nr, nc);
	this->img = Mat::zeros(imsize, CV_32FC1);
	this->f = Mat::zeros(imsize, CV_32FC1);
}

void Decoder::decodeImage(){
	int i = 0;
	for (i = 0; i < this->encoded.size(); i++){
		if (i % (Options::M * Options::M) == 0) { // key block
			Mat block = decodeBlock(this->encoded[i], this->keyPhi);
			block = block.reshape(1, Options::blockSize);
			fillNthBlock(i, block);
		} else {
			Mat block = decodeBlock(this->encoded[i], this->nonkeyPhi);

			block = block.reshape(1, Options::blockSize);
			fillNthBlock(i, block);
		}
	}
	this->img = Wavelet(f, Wavelet::IDWT).getResult();
	this->img.convertTo(this->img, CV_8UC1);
}

Mat Decoder::decodeBlock(cv::Mat block, cv::Mat phi){
	SpaRSA_noSI l2l1_solver = SpaRSA_noSI(block, phi);
	SpaRSA *solver = &l2l1_solver;
	solver->runAlgorithm();
	Mat f_ = solver->reconstructed();
	return f_;
}

void Decoder::fillNthBlock(int n, cv::Mat block){ // this is 0 indexed
	assert(n >= 0 && n < (this->f.cols * this->f.rows)/(Options::blockSize * Options::blockSize));
	int whichGOB = int(double(n)/pow(Options::M,2));
	int blockOffset = n % int(pow(Options::M, 2));

	int n_gob_y = this->f.cols / (Options::blockSize * Options::M);
	int gob_colStart = (whichGOB % n_gob_y) * (Options::blockSize * Options::M);
	int gob_rowStart = int(whichGOB / n_gob_y) * (Options::blockSize * Options::M);
	int rowStart = gob_rowStart + (blockOffset % Options::M) * Options::blockSize;
	int colStart = gob_colStart + int(blockOffset / Options::M) * Options::blockSize;
//	printf("block %d: (%d,%d)\t\t\t gob_rowStart=%d  gob_colStart=%d  \n", n, rowStart, colStart, gob_rowStart, gob_colStart);
	Mat tmpblock = this->f.colRange(colStart, colStart+Options::blockSize).rowRange(rowStart, rowStart + Options::blockSize);
	assert(tmpblock.rows == block.rows && tmpblock.cols == block.cols);
	block.copyTo(tmpblock);
}

Mat Decoder::getDecodedImage(){
	return this->img;
}
Decoder::~Decoder() {
}

