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

int A = 7;

// helper methods
Point pt(int i, Mat H){
	return Point(int(i / H.cols), int(i % H.cols));
}

int getPi(int i, int n, int A){
	return (A * (i + 1) % n );
}

bool isPowerOfTwo (unsigned int x)
{
	while (((x % 2) == 0) && x > 1) /* While x is even and > 1 */
		x /= 2;
	return (x == 1);
}

//////////

SBHE::SBHE(int M, int N, int B, int A) {
	assert(N % B == 0 && isPowerOfTwo(B));
	this->SBHEmat = Mat::zeros(M, N, CV_32FC1);
	Mat m = Mat::zeros(N, N, CV_32FC1);
	Mat hadamard = generateHadamardMatrix(int(log2(B)));
	int hadamardBlockSize = hadamard.cols;
	for (int i = 0; i < N/B; i++){
		hadamard.copyTo(m.rowRange(hadamardBlockSize*i,hadamardBlockSize*(i+1)).colRange(hadamardBlockSize*i,hadamardBlockSize*(i+1)));
	}
	std::cout << m;
	std::cout << "got hadamard\n";
	m = scrambleMatrix(m, A);
	std::cout << m;
	std::cout << "scrambled\n";
	this->SBHEmat = chooseMrows(m, M);
	std::cout << "chose m rows\n";
}

Mat SBHE::scrambleMatrix(Mat H, int A){
	Mat scrambled = H.clone();
	int n = H.rows * H.cols, i;
	for (i = 0; i < n; i++){
		scrambled.at<double>(pt(i, H)) = H.at<double>(pt(getPi(i,n,A), H));
	}
	return scrambled;
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

Mat SBHE::getSBHEmat(){
	return this->SBHEmat;
}

SBHE::~SBHE() {
	// TODO Auto-generated destructor stub
}

