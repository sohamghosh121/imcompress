/*
 * Decoder.cpp
 *
 *  Created on: 12 Oct 2015
 *      Author: sohamghosh
 */

#include "Decoder.h"
#include "SBHE.h"

#include "SpaRSA_noSI.h"
#include "SpaRSA_joint.h"
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
	int i = 0; //, diag_idx_x = 0, diag_idx_y = 0, joint_y_idx = 0, joint_x_idx = 0;
	int num_rows = (keyPhi.rows + (pow(Options::M, 2) - 1) * nonkeyPhi.rows) * (pow(img.rows / double(Options::M * Options::blockSize), 2));
	Mat joint_y = Mat::zeros(num_rows, 1, CV_32FC1);
	Mat joint_x = Mat::zeros(pow(Options::blockSize, 2) * encoded.size(), 1,CV_32FC1);
	int joint_y_idx = 0, joint_x_idx = 0;
	for (i = 0; i < this->encoded.size(); i++){
		encoded[i].copyTo(joint_y.rowRange(joint_y_idx, joint_y_idx + encoded[i].rows));
		joint_y_idx += encoded[i].rows;
	}
	for (i = 0; i < this->encoded.size(); i++){
		Mat block, first_reconstruction;
		if (i % (Options::M * Options::M) == 0) { // key block
//			std::cout << "DECODE key block " << i << "\n";
			block = decodeBlock(this->encoded[i], this->keyPhi);
//			DecodedBlock db;
//			db.measurements = encoded[i];
//			db.decoded = block;
//			decodedKeyBlocks.push_back(db);
		} else {
//			std::cout << "DECODE nonkey block " << i << "\n";
			block = decodeBlock(this->encoded[i], this->nonkeyPhi);
//			first_reconstruction = decodeBlock(this->encoded[i], this->nonkeyPhi);
//			Mat si = findSI(first_reconstruction);
//			Mat block = decodeBlockWithSI(encoded[i], nonkeyPhi, si, first_reconstruction);
		}
		block.copyTo(joint_x.rowRange(joint_x_idx, joint_x_idx + block.rows));
		joint_x_idx += block.rows;
//		block = block.reshape(1, Options::blockSize);
//		fillNthBlock(i, block);
	}
	// do joint reconstruction
	Mat final = joint_x;
//	SpaRSA_joint joint_solver = SpaRSA_joint(joint_y, keyPhi, nonkeyPhi);
//	SpaRSA *solver = &joint_solver;
//	solver->warmStart(joint_x);
//	solver->runAlgorithm();
////	solver->runDebiasingPhase();
//	final = solver->reconstructed();
	std::cout << final.rows << ", " << final.cols << "\n";
	for (i = 0; i < this->encoded.size(); i++){
		Mat block;
		block = final.rowRange(i * pow(Options::blockSize, 2), (i + 1) * pow(Options::blockSize, 2)).clone();
		block = block.reshape(1, Options::blockSize);
		fillNthBlock(i, block);
	}
	f = f.reshape(1, this->img.cols * this->img.rows);
	f = SBHE::unscrambleInputSignal(f, Options::A).reshape(1, img.rows);
	this->img = Wavelet(f, Wavelet::IDWT).getResult();
	this->img.convertTo(img, CV_8UC1);
}

Mat Decoder::findSI(Mat decoded){
	double bestMSE = DBL_MAX;
	Mat si = Mat();
	Mat diff;
	for(std::vector<DecodedBlock>::iterator it = decodedKeyBlocks.begin(); it != decodedKeyBlocks.end(); ++it) {
		DecodedBlock db = *it;
//		printf("db (%d, %d)  me (%d, %d)", db.measurements.rows, db.measurements.cols, measurements.rows, measurements.cols);
		if (norm(db.decoded - decoded, NORM_L2) < bestMSE){
			si = db.decoded;
		}
	}
	return si;
}

Mat Decoder::decodeBlock(cv::Mat block, cv::Mat phi){
	SpaRSA_noSI nosi_solver = SpaRSA_noSI(block, phi);
	SpaRSA *solver = &nosi_solver;
	solver->runAlgorithm();
	solver->runDebiasingPhase();
	Mat f_ = solver->reconstructed();
//	return SBHE::unscrambleInputSignal(f_, Options::A);
	return f_;
}

Mat Decoder::decodeBlockWithSI(Mat block, Mat phi, Mat si, Mat rec){
	SpaRSA_withSI withsi_solver = SpaRSA_withSI(block, phi, si);
	SpaRSA *solver = &withsi_solver;
	solver->warmStart(rec);
	solver->runAlgorithm();
	solver->runDebiasingPhase();
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
	int rowStart = gob_rowStart + int(blockOffset / Options::M) * Options::blockSize;
	int colStart = gob_colStart + int(blockOffset % Options::M) * Options::blockSize;
//	printf("n: %d, GOB: %d, offset: %d, (%d,%d)\n", n, whichGOB, blockOffset, rowStart, colStart);
	Mat tmpblock = this->f.colRange(colStart, colStart+Options::blockSize).rowRange(rowStart, rowStart + Options::blockSize);
	assert(tmpblock.rows == block.rows && tmpblock.cols == block.cols);
	block.copyTo(tmpblock);
}

Mat Decoder::getDecodedImage(){
	return this->img;
}
Decoder::~Decoder() {
}

