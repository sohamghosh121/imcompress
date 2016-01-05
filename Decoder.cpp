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
	this->img = Mat::zeros(imsize, CV_32FC1);
	this->f = Mat::zeros(imsize, CV_32FC1);
}

void Decoder::decodeImage(){
	int i = 0;
	for (i = 0; i < this->encoded.size(); i++){
		if (i % (opts.getM() * opts.getM()) == 0) { // key block
			Mat block = decodeBlock(this->encoded[i], this->keyPhi);
			block = block.reshape(1, opts.getBlockSize());
			fillNthBlock(i, block);
		} else {
			Mat block = decodeBlock(this->encoded[i], this->nonkeyPhi);
			block = block.reshape(1, opts.getBlockSize());
			fillNthBlock(i, block);
		}
	}
	this->img = Wavelet(f, Wavelet::IDWT).getResult();
	this->img.convertTo(this->img, CV_8UC1);

}

Mat Decoder::decodeBlock(cv::Mat block, cv::Mat phi){
	SpaRSA solver = SpaRSA(block, phi);
	Mat f_ = solver.reconstructed();
	return f_;
}

void Decoder::fillNthBlock(int n, cv::Mat block){ // this is 0 indexed
	assert(n >= 0 && n < (this->f.cols * this->f.rows)/(this->opts.getBlockSize() * this->opts.getBlockSize()));
	int rowStart, colStart;
	int nc = this->f.cols / this->opts.getBlockSize();
	int nr = this->f.rows / this->opts.getBlockSize();
	colStart = (n % nc) * this->opts.getBlockSize();
	rowStart = (n / nc) * this->opts.getBlockSize();
	Mat tmpblock = this->f.colRange(colStart, colStart+this->opts.getBlockSize()).rowRange(rowStart, rowStart + this->opts.getBlockSize());
	assert(tmpblock.rows == block.rows && tmpblock.cols == block.cols);
	block.copyTo(tmpblock);
}

Mat Decoder::getDecodedImage(){
	return this->img;
}
Decoder::~Decoder() {
}

