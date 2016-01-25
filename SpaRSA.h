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
protected:
	float eta = 2.0;
	float tau = 0.2;
	float sigma = 0.001;
	float tolP = 0.0001;
	float tolD = 0.0001;
	size_t maxIter = 500;
	size_t minIter = 10;
	size_t maxDebiasIter = 200;
	size_t maxItersPerCycle = 10;
	int M = 5;
	float alpha_t = 1.0;
	const float alpha_min = 0.00000001, alpha_max = 10000000.0; // TODO: what values to initialize to?
	int t = 0, itersThisCycle = 0;
	std::deque<double> objectiveFunctionValues;
	cv::Mat x_t, x_t_plus_1, x_t_minus_1;
	cv::Mat y, phi;


	void chooseAlpha();
	void updateAlpha();
	void updateObjectiveValues(float);
	void runOuterIteration();
	void runInnerIteration();
	bool checkAcceptanceCriterion(); // return true if acceptance criterion is met
	bool checkStoppingCriterion(); // return true if stopping criterion is met

	virtual void solveSubproblem() = 0; // solve x_{t+1} sub problem
	virtual float objectiveFunctionValue(cv::Mat) = 0;
	virtual cv::Mat del_f(cv::Mat x) = 0;
public:
	SpaRSA(cv::Mat y, cv::Mat phi);
	void runAlgorithm();
	void runDebiasingPhase();
	void warmStart(cv::Mat);
	cv::Mat reconstructed();
	virtual ~SpaRSA() = 0;
};

#endif /* SPARSA_H_ */
