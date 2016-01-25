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
	typedef struct {
		cv::Mat measurements;
		cv::Mat decoded;
	} DecodedBlock;

	cv::Mat keyPhi;
	cv::Mat nonkeyPhi;
	std::map<int, cv::Mat> encoded;
	cv::Size imsize;
	cv::Mat img;
	cv::Mat f;
	std::vector<DecodedBlock> decodedKeyBlocks;

	void fillNthBlock(int, cv::Mat);
	cv::Mat findSI(cv::Mat);
	cv::Mat decodeBlock(cv::Mat, cv::Mat);
	cv::Mat decodeBlockWithSI(cv::Mat, cv::Mat, cv::Mat, cv::Mat);
public:
	Decoder(int, int, cv::Mat, cv::Mat, std::map<int, cv::Mat>);
	Decoder(cv::Size);
	void decodeImage();
	cv::Mat getDecodedImage();
	virtual ~Decoder();
};

#endif /* DECODER_H_ */
