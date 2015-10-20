/*
 * Decoder.h
 *
 *  Created on: 12 Oct 2015
 *      Author: sohamghosh
 */

#ifndef DECODER_H_
#define DECODER_H_

#include <opencv2/opencv.hpp>
#include "Options.h"

class Decoder {
private:
	const Options opts;

	cv::Mat keyPhi;
	cv::Mat nonkeyPhi;
	std::map<int, cv::Mat> encoded;
	cv::Size imsize;
	cv::Mat img;

	void fillNthBlock(int, cv::Mat);
	cv::Mat decodeBlock(cv::Mat, cv::Mat);
public:
	Decoder(int, int, cv::Mat, cv::Mat, std::map<int, cv::Mat>);
	Decoder(cv::Size);
	void decodeImage();
	cv::Mat getDecodedImage();
	virtual ~Decoder();
};

#endif /* DECODER_H_ */
