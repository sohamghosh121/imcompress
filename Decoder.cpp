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

Decoder::Decoder(int nr, int nc, Mat keyPhi, Mat nonkeyPhi, std::vector<Mat> encoded) {
	this->encoded = encoded;
	this->keyPhi = keyPhi;
	this->nonkeyPhi = nonkeyPhi;
	this->imsize = cv::Size(nc, nr);
	this->img = Mat::zeros(imsize, CV_32FC1);
	this->f = Mat::zeros(imsize, CV_32FC1);
}

void Decoder::decodeImage(){
	int i = 0;
	int num_rows = (keyPhi.rows + (pow(Options::M, 2) - 1) * nonkeyPhi.rows) * (double(img.rows * img.cols)/ double(pow(Options::M * Options::blockSize, 2)));
	Mat joint_y = Mat::zeros(num_rows, 1, CV_32FC1);
	Mat joint_x = Mat::zeros(pow(Options::blockSize, 2) * encoded.size(), 1,CV_32FC1);
	int joint_y_idx = 0, joint_x_idx = 0;
	for (i = 0; i < this->encoded.size(); i++){
		// joint_y, joint_x  can be used for joint reconstruction
		encoded[i].copyTo(joint_y.rowRange(joint_y_idx, joint_y_idx + encoded[i].rows));
		joint_y_idx += encoded[i].rows;
	}
	for (i = 0; i < this->encoded.size(); i++){
		Mat block, first_reconstruction;
		if (i % (Options::M * Options::M) == 0) { // key block
			block = decodeBlock(this->encoded[i], this->keyPhi);
		} else {
			block = decodeBlock(this->encoded[i], this->nonkeyPhi);
		}
		block.copyTo(joint_x.rowRange(joint_x_idx, joint_x_idx + block.rows));
		joint_x_idx += block.rows;
	}
	Mat final = joint_x;
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

/*
 * Find key block with lowest MSE and use as
 */
Mat Decoder::findSI(Mat decoded){
	double bestMSE = DBL_MAX;
	Mat si = Mat();
	Mat diff;
	for(std::vector<DecodedBlock>::iterator it = decodedKeyBlocks.begin(); it != decodedKeyBlocks.end(); ++it) {
		DecodedBlock db = *it;
		if (norm(db.decoded - decoded, NORM_L2) < bestMSE){
			si = db.decoded;
		}
	}
	return si;
}

/*
 * Run SpaRSA solver for each block
 */
Mat Decoder::decodeBlock(cv::Mat block, cv::Mat phi){
	SpaRSA_noSI nosi_solver = SpaRSA_noSI(block, phi);
	SpaRSA *solver = &nosi_solver;
	solver->runAlgorithm();
	solver->runDebiasingPhase();
	Mat f_ = solver->reconstructed();
	return f_;
}

/*
 * If there is SI, use SI
 */
Mat Decoder::decodeBlockWithSI(Mat block, Mat phi, Mat si, Mat rec){
	SpaRSA_withSI withsi_solver = SpaRSA_withSI(block, phi, si);
	SpaRSA *solver = &withsi_solver;
	solver->warmStart(rec);
	solver->runAlgorithm();
	solver->runDebiasingPhase();
	Mat f_ = solver->reconstructed();
	return f_;
}

/*
 * Fill n-th block in image with decoded values
 */
void Decoder::fillNthBlock(int n, cv::Mat block){ // this is 0 indexed
	assert(n >= 0 && n < (this->f.cols * this->f.rows)/(Options::blockSize * Options::blockSize));
	int whichGOB = int(double(n)/pow(Options::M,2));
	int blockOffset = n % int(pow(Options::M, 2));

	int n_gob_y = this->f.cols / (Options::blockSize * Options::M);
	int gob_colStart = (whichGOB % n_gob_y) * (Options::blockSize * Options::M);
	int gob_rowStart = int(whichGOB / n_gob_y) * (Options::blockSize * Options::M);
	int rowStart = gob_rowStart + int(blockOffset / Options::M) * Options::blockSize;
	int colStart = gob_colStart + int(blockOffset % Options::M) * Options::blockSize;
	Mat tmpblock = this->f.colRange(colStart, colStart+Options::blockSize).rowRange(rowStart, rowStart + Options::blockSize);
	assert(tmpblock.rows == block.rows && tmpblock.cols == block.cols);
	block.copyTo(tmpblock);
}

Mat Decoder::getDecodedImage(){
	return this->img;
}
Decoder::~Decoder() {
}

