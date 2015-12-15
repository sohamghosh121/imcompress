/*


 * Wavelet.h
 *
 *  Created on: 6 Dec 2015
 *      Author: sohamghosh
 */

#ifndef WAVELET_H_
#define WAVELET_H_

#include <opencv2/opencv.hpp>

class Wavelet {
	cv::Mat in;
	cv::Mat out;
public:
	static const int DWT = 1;
	static const int IDWT = -1;
	Wavelet(cv::Mat, int);
	void encode();
	void decode();
	cv::Mat getResult();
	virtual ~Wavelet();
};

#endif /* WAVELET_H_ */
