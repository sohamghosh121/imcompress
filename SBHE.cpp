/*
 * SBHE.cpp
 *
 *  Created on: 29 Oct 2015
 *      Author: sohamghosh
 */

#include "SBHE.h"
#include <stdlib.h>
#include <cmath>

using namespace cv;

int A = 13;

// helper methods
Point pt(int i, Mat H){
	return Point(int(i / H.cols), int(i % H.cols));
}

int getPi(int i, int n, int A){
	return (A * (i - 1) % n );
}

bool isPowerOfTwo (unsigned int x)
{
	while (((x % 2) == 0) && x > 1) /* While x is even and > 1 */
		x /= 2;
	return (x == 1);
}

SBHE::SBHE(int M, int N, int A) {
	assert(isPowerOfTwo(N));
	this->SBHEmat = Mat::zeros(M, N, CV_32FC1);
	Mat m = Mat::zeros(N, N, CV_32FC1);
	Mat hadamard = generateHadamardMatrix(int(log2(N)));
	this->SBHEmat = chooseMrows(hadamard, M);
}

Mat SBHE::chooseMrows(Mat m, int M){
	Mat m_ = Mat::zeros(M,  m.cols, CV_32FC1);
	std::vector<int> idx;
	std::srand (unsigned(std::time(0)));
	// set some values:
	for (int i=1; i<m.cols; ++i) idx.push_back(i);
	// using built-in random generator:
	std::random_shuffle ( idx.begin(), idx.end());
	std::sort(idx.begin(), idx.begin()+M);
	int i = 0;
	for (std::vector<int>::iterator it=idx.begin(); it!=idx.begin()+M; ++it, i++){
		m.row(*it).copyTo(m_.row(i));
	}
	return m_;
}


Mat SBHE::generateHadamardMatrix(int n){
	int half = pow(2,n-1), full = pow(2,n);
	Mat H = Mat::zeros(full, full, CV_32FC1);
	if (n == 1){
		return (Mat_<double>(2, 2) << 1, 1, 1, -1);
	} else {
		Mat H_ = generateHadamardMatrix(n-1);
		H_.copyTo(H.rowRange(0, half).colRange(0, half));
		H_.copyTo(H.rowRange(0, half).colRange(half, full));
		H_.copyTo(H.rowRange(half, full).colRange(0, half));
		H_ = - H_;
		H_.copyTo(H.rowRange(half, full).colRange(half, full));
	}
	return H;
}

Mat SBHE::scrambleInputSignal(Mat & signal, int A){
	assert(signal.cols == 1);
	Mat scrambled_signal = Mat::zeros(signal.size(), CV_32FC1);
	int pi;
	for (int i = 0; i < signal.rows; i++){
		pi = (A * i) % signal.rows;
		scrambled_signal.at<float>(i, 0) =  signal.at<float>(pi, 0);
	}
//	std::cout << signal;
//	std::cout << "\n" << scrambled_signal;
	return scrambled_signal;
}

Mat SBHE::unscrambleInputSignal(Mat & scrambled_signal, int A){
	assert(scrambled_signal.cols == 1);
	Mat signal = Mat::zeros(scrambled_signal.size(), CV_32FC1);
	int pi;
	for (int i = 0; i < signal.rows; i++){
		pi = (A * i) % signal.rows;
		signal.at<float>(pi, 0) = scrambled_signal.at<float>(i, 0);
	}
//	std::cout << "\n" << scrambled_signal;
//	std::cout << signal;

	return signal;
}

Mat SBHE::getSBHEmat(){
	return this->SBHEmat;
}

SBHE::~SBHE() {
}

