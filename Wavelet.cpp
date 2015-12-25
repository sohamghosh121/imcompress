/*
 * Wavelet.cpp
 *
 *  Created on: 6 Dec 2015
 *      Author: sohamghosh
 */

#include "Wavelet.h"
# include <stddef.h>
using namespace cv;

Wavelet::Wavelet(Mat img, int type) {
	this->in = img;
	this->out = Mat(img.size(), CV_32FC1);
	switch(type){
	case DWT:
		encode();
		printf("dwt\n");
		break;
	case IDWT:
		decode();
		printf("idwt\n");
		break;
	}
}


void Wavelet::encode(){
	for (int i = 0; i < level; i++){
		// declarations
		printf("level: %d\n", i);
		Mat temp = Mat(in.size(), CV_32FC1);
		Mat temp_downsample = Mat(temp.rows/2, temp.cols, CV_32FC1);
		Mat ll = Mat(temp_downsample.size(), CV_32FC1);
		Mat ll_downsample = Mat(ll.rows, ll.cols/2, CV_32FC1);
		Mat lh = Mat(temp_downsample.size(), CV_32FC1);
		Mat lh_downsample = Mat(lh.rows, lh.cols/2, CV_32FC1);
		Mat hl = Mat(temp_downsample.size(), CV_32FC1);
		Mat hl_downsample = Mat(hl.rows, hl.cols/2, CV_32FC1);
		Mat hh = Mat(temp_downsample.size(), CV_32FC1);
		Mat hh_downsample = Mat(hh.rows, hh.cols/2, CV_32FC1);
		//
		temp = symconv2(in, hp, COL);
		downsample(temp, temp_downsample, ROW, 1);
		hh = symconv2(temp_downsample, hp, ROW);
		downsample(hh, hh_downsample, COL, 1);
		lh = symconv2(temp_downsample, lp, ROW);
		downsample(lh, lh_downsample, COL, 0);

		temp = symconv2(in, lp, COL);
		downsample(temp, temp_downsample, ROW, 1);
		hl = symconv2(temp_downsample, hp, ROW);
		downsample(hl, hl_downsample, COL, 1);
		ll = symconv2(temp_downsample, lp, ROW);
		downsample(ll, ll_downsample, COL, 0);

		printf("ll (%d,%d), lh (%d,%d), hl (%d,%d), hh (%d,%d)\n", ll_downsample.rows, ll_downsample.cols, lh_downsample.rows, lh_downsample.cols, hl_downsample.rows, hl_downsample.cols, hh_downsample.rows, hh_downsample.cols);
		ll_downsample.copyTo(out.rowRange(0, in.rows/2).colRange(0, in.rows/2));
		lh_downsample.copyTo(out.rowRange(in.rows/2, in.rows).colRange(0, in.rows/2));
		hl_downsample.copyTo(out.rowRange(0, in.rows/2).colRange(in.rows/2, in.rows));
		hh_downsample.copyTo(out.rowRange(in.rows/2, in.rows).colRange(in.rows/2, in.rows));
		in = ll_downsample;
	}
}

void Wavelet::decode(){
	Mat c = Mat(in.size(), CV_32FC1);
	for (int i = 0; i < level; i++){

	}

}

Mat Wavelet::symconv2(Mat x, Mat vec, int rowCol){
	int length = vec.cols;
//	printf("length: %d\n", length);
	int half = (length - 1)/2;
	Mat y = Mat::zeros(x.size(), CV_32FC1);
	switch(rowCol){
		case COL: {
			Mat new_x = Mat(x.rows + 2 * length, x.cols, CV_32FC1);
			Mat filtered_x = Mat(x.rows + 2 * length, x.cols, CV_32FC1);
//			printf("x: (%d,%d)\tnew_x: (%d, %d)\n", x.rows, x.cols, new_x.rows, new_x.cols);
			x.copyTo(new_x.rowRange(length, x.cols + length));
			for (int i = 0; i < length; i++){
				x.row(2*i + 1).copyTo(new_x.row(i));
				x.row(x.rows-2).copyTo(new_x.row(i+x.rows));
			}
			cv::filter2D(new_x, filtered_x, -1, vec.t());
			filtered_x.rowRange(length, x.cols + length).copyTo(y);
		};
		break;
		case ROW:{
			Mat new_x = Mat(x.rows, x.cols + 2 * length, CV_32FC1);
			Mat filtered_x = Mat(x.rows + 2 * length, x.cols, CV_32FC1);
			x.copyTo(new_x.colRange(length, x.cols + length));
			for (int i = 0; i < length; i++){
				x.col(2 * i + 1).copyTo(new_x.col(i));
				x.col(x.cols-2).copyTo(new_x.col(i + x.cols)); // might be wrong
			}
			cv::filter2D(new_x, filtered_x, -1, vec);
			filtered_x.colRange(length, x.cols + length).copyTo(y); // check dimensionality
		};
		break;
	}
	return y;
}

void Wavelet::downsample(Mat& src, Mat& dest, int rowCol, int start){
	switch(rowCol){
	case ROW: {
		assert(dest.rows == src.rows/2);
		for (int i = 0; i < dest.rows; i++){
			src.row(i * 2 + start).copyTo(dest.row(i));
		}
	};
	break;
	case COL: {
		assert(dest.cols == src.cols/2);
		for (int i = 0; i < dest.cols; i++){
			src.col(i * 2 + start).copyTo(dest.col(i));
		}
	};
	break;
	}
}

Mat Wavelet::getResult(){
	return out;
}

Wavelet::~Wavelet() {

}

