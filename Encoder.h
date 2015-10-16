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
	const double Mk = 1.0;
	const double Mw = 2.0;
	const double C = 1.0;
	const int blockSize = 8;
	const int M = 4;

	// generate phi matrix
	cv::Mat getPhi(double);
	cv::Mat getPhi(int, int);

	std::map<int, cv::Mat *> encoded;
	cv::Mat img;

	cv::Mat keyPhi;
	cv::Mat nonkeyPhi;

	cv::Mat getNthBlock(int);

	cv::Mat encodeBlock(cv::Mat, cv::Mat);
	cv::Mat encodeKeyBlock(cv::Mat);
	cv::Mat encodeNonKeyBlock(cv::Mat);

public:
	Encoder(cv::Mat);
	void encodeImage();
	virtual ~Encoder();
};

#endif /* ENCODER_H_ */
