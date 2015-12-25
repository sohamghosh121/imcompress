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
	// wavelets
	static const int ROW = 1;
	static const int COL = 2;

	cv::Mat lp = (cv::Mat_<float>(1, 9) << 0.0267487574110000, -0.0168641184430000, -0.0782232665290000, 0.266864118443000, 0.602949018236000, 0.266864118443000, -0.0782232665290000, -0.0168641184430000, 0.0267487574110000);
	cv::Mat hp = (cv::Mat_<float>(1, 9) << 0.0456358815570000, -0.0287717631140000, -0.295635881557000, 0.557543526229000, -0.295635881557000, -0.0287717631140000, 0.0456358815570000);
	cv::Mat lpr = (cv::Mat_<float>(1, 9) << -0.0912717631140000, -0.0575435262280000, 0.591271763114000, 1.11508705245800, 0.591271763114000, -0.0575435262280000, -0.0912717631140000);
	cv::Mat hpr = (cv::Mat_<float>(1, 9) << 0.0534975148220000, 0.0337282368860000, -0.156446533058000, -0.533728236886000, 1.20589803647200, -0.533728236886000, -0.156446533058000, 0.0337282368860000, 0.0534975148220000);
	int level = 1;
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
	// helper functions
	cv::Mat symconv2(cv::Mat, cv::Mat, int rowCol);
	void downsample(cv::Mat&, cv::Mat&, int rowCol, int start); // start is either 0 or 1
};

#endif /* WAVELET_H_ */
