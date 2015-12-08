/*
 * Encoder.h
 *
 *  Created on: 12 Oct 2015
 *      Author: sohamghosh
 */

#ifndef ENCODER_H_
#define ENCODER_H_

#include "Options.h"
#include <opencv2/opencv.hpp>

// Assume image is multiple of block size for now

class Encoder {
private:
	const Options opts;

	// generate phi matrix
	cv::Mat getPhi(double);
	cv::Mat getPhi(int, int);

	std::map<int, cv::Mat> encoded;  // will find more efficient representations later.
	cv::Mat img;
	cv::Mat f; // 2D wavelet transform of image

	cv::Mat keyPhi;
	cv::Mat nonkeyPhi;

	cv::Mat getNthBlock(int, cv::Mat);
	cv::Mat getGOB(int);


	cv::Mat encodeBlock(cv::Mat, cv::Mat);
	cv::Mat encodeKeyBlock(cv::Mat);
	cv::Mat encodeNonKeyBlock(cv::Mat);

public:
	Encoder(cv::Mat);
	void encodeImage();
	std::map<int, cv::Mat> getEncodedValues();
	cv::Mat getKeyPhi();
	cv::Mat getnonkeyPhi();
	virtual ~Encoder();
};

#endif /* ENCODER_H_ */
