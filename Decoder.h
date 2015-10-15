/*
 * Decoder.h
 *
 *  Created on: 12 Oct 2015
 *      Author: sohamghosh
 */

#ifndef DECODER_H_
#define DECODER_H_

#include <opencv2/opencv.hpp>

class Decoder {
public:
	Decoder();
	cv::Mat decodeImage();
	virtual ~Decoder();
};

#endif /* DECODER_H_ */
