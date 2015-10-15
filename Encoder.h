/*
 * Encoder.h
 *
 *  Created on: 12 Oct 2015
 *      Author: sohamghosh
 */

#ifndef ENCODER_H_
#define ENCODER_H_

#include <opencv2/opencv.hpp>
#include <random>

// Assume image is multiple of block size for now

class Encoder {
private:
	const float Mk = 1.0;
	const float Mw = 2.0;
	const float C = 1.0;
	const int blockSize = 8;

	// generate random matrix
	void getPhi(int, int, cv::Mat);

	std::map<int, cv::Mat *> D;
	cv::Mat img;
	bool isKeyBlock();

	cv::Mat keyPhi;
	cv::Mat nonkeyPhi;

	cv::Mat getNthBlock(int);

	cv::Mat encodeBlock(cv::Mat, cv::Mat);
	cv::Mat encodeKeyBlock(cv::Mat);
	cv::Mat encodeNonKeyBlock(cv::Mat);

	// helper functions
	double calculateMSE(cv::Mat, cv::Mat);
	double getTau();  // get min MSE between y_w and columns of D
	double getLambda(cv::Mat);

public:
	Encoder(cv::Mat);
	void encodeImage();
	virtual ~Encoder();
};

#endif /* ENCODER_H_ */
