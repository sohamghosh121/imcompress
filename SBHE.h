/*
 * SBHE.h
 *
 *  Created on: 29 Oct 2015
 *      Author: sohamghosh
 */

// To generate the SBHE matrix

#ifndef SBHE_H_
#define SBHE_H_

#include <opencv2/opencv.hpp>

class SBHE {
private:
	cv::Mat SBHEmat;
	cv::Mat generateHadamardMatrix(int); // get hadamard matrix of size B
	cv::Mat scrambleMatrix(cv::Mat, int);
	cv::Mat chooseMrows(cv::Mat, int);
public:
	SBHE(int, int, int, int);
	cv::Mat getSBHEmat();
	virtual ~SBHE();
};

#endif /* SBHE_H_ */
