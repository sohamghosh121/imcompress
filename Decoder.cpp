/*
 * Decoder.cpp
 *
 *  Created on: 12 Oct 2015
 *      Author: sohamghosh
 */

#include "Decoder.h"
#include "Wavelet.h"
#include "SpaRSA.h"

using namespace cv;

// first try individual recovery

Decoder::Decoder(int nr, int nc, Mat keyPhi, Mat nonkeyPhi, std::map<int, Mat> encoded) {
	this->encoded = encoded;
	this->keyPhi = keyPhi;
	this->nonkeyPhi = nonkeyPhi;
	this->imsize = cv::Size(nr, nc);
	this->img = Mat(imsize, CV_32FC1);
}

void Decoder::decodeImage(){
	int i = 0;
	for (i = 0; i < this->encoded.size(); i++){
		if (i % this->opts.getM() == 0) { // key block
			fillNthBlock(i, decodeBlock(this->encoded[i], this->keyPhi));
		} else {
			fillNthBlock(i, decodeBlock(this->encoded[i], this->nonkeyPhi));
		}
	}
}

Mat Decoder::decodeBlock(cv::Mat block, cv::Mat phi){
	SpaRSA solver = SpaRSA(block, phi);
	Mat f = solver.reconstructed();
	Mat decoded = Wavelet(f, Wavelet::IDWT).getResult();
	return decoded;
}

void Decoder::fillNthBlock(int n, cv::Mat block){ // this is 0 indexed
	assert(n >= 0 && n < (this->img.cols * this->img.rows)/(this->opts.getBlockSize() * this->opts.getBlockSize()));
	int rowStart, colStart;
	int nc = this->img.cols / this->opts.getBlockSize();
	int nr = this->img.rows / this->opts.getBlockSize();
	colStart = (n % nc) * this->opts.getBlockSize();
	rowStart = (n / nc) * this->opts.getBlockSize();
//	std::cout << block << std::endl;
	Mat tmpblock = this->img.colRange(colStart, colStart+this->opts.getBlockSize()).rowRange(rowStart, rowStart + this->opts.getBlockSize());
	block.copyTo(tmpblock);
}

Mat Decoder::getDecodedImage(){
	return this->img;
}
Decoder::~Decoder() {
}

