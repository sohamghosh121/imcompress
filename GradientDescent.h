/*
 * GradientDescent.h
 *
 *  Created on: 20 Oct 2015
 *      Author: sohamghosh
 */

#ifndef GRADIENTDESCENT_H_
#define GRADIENTDESCENT_H_

#include <opencv2/opencv.hpp>

class GradientDescent {
private:
	const double lambda = 1;
	const double alpha = 0.1;
	const double maxIterations = 1000;
	cv::Mat x;
	cv::Mat y;
	cv::Mat theta;
	double thetaTtheta; // store for later computations

	cv::Mat getGradient();

public:
	GradientDescent(cv::Mat, cv::Mat);
	void doGradientDescent();
	virtual ~GradientDescent();
};

#endif /* GRADIENTDESCENT_H_ */
