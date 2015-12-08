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
	const double eta = 2.0;
	const double tau = 1.0;
	const double sigma = 0.01;
	const double tolP = 0.001;
	const int M = 10;
	float alpha_t = 1.0;
	const double alpha_min = 1.0, alpha_max = 10.0; // TODO: what values to initialize to?
	int t = 0;
	std::deque<double> objectiveFunctionValues;
	cv::Mat x_t, x_t_plus_1, x_t_minus_1;

	cv::Mat y, phi;

	void updateAlpha();
	void updateObjectiveValues(double);
	void runAlgorithm();
	void runOuterIteration();
	void runInnerIteration();
	bool checkAcceptanceCriterion(); // return true if acceptance criterion is met
	bool checkStoppingCriterion(); // return true if stopping criterion is met
	void solveSubproblem(); // solve x_{t+1} sub problem
	double objectiveFunctionValue(cv::Mat);

	double soft(double u, double a);
	cv::Mat del_f(cv::Mat x);
public:
	SpaRSA(cv::Mat y, cv::Mat phi);
	cv::Mat reconstructed();
	virtual ~SpaRSA();
};

#endif /* SPARSA_H_ */
