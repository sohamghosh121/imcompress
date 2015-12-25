/*
 * SpaRSA.h
 *
 *  Created on: 28 Oct 2015
 *      Author: sohamghosh
 */

#ifndef SPARSA_H_
#define SPARSA_H_

#include <opencv2/opencv.hpp>
#include <cmath>

class SpaRSA {
private:
	const float eta = 1.5;
	const float tau = 1.0;
	const float sigma = 0.01;
	const float tolP = 0.0001;
	const int M = 100;
	float alpha_t = 1.0;
	const float alpha_min = 1.0, alpha_max = 1000.0; // TODO: what values to initialize to?
	int t = 0;
	std::deque<double> objectiveFunctionValues;
	cv::Mat x_t, x_t_plus_1, x_t_minus_1;

	cv::Mat y, phi;

	void updateAlpha();
	void updateObjectiveValues(float);
	void runAlgorithm();
	void runOuterIteration();
	void runInnerIteration();
	bool checkAcceptanceCriterion(); // return true if acceptance criterion is met
	bool checkStoppingCriterion(); // return true if stopping criterion is met
	void solveSubproblem(); // solve x_{t+1} sub problem
	float objectiveFunctionValue(cv::Mat);

	float soft(float u, float a);
	cv::Mat del_f(cv::Mat x);
public:
	SpaRSA(cv::Mat y, cv::Mat phi);
	cv::Mat reconstructed();
	virtual ~SpaRSA();
};

#endif /* SPARSA_H_ */
