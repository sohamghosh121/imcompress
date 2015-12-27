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
//		printf("dwt\n");
		break;
	case IDWT:
		decode();
//		printf("idwt\n");
		break;
	}
}


void Wavelet::encode(){
	for (int i = 0; i < level; i++){
		// declarations
//		printf("level: %d\n", i);
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
		downsample(temp, temp_downsample, ROW, 0);
		hl = symconv2(temp_downsample, hp, ROW);
		downsample(hl, hl_downsample, COL, 1);
		ll = symconv2(temp_downsample, lp, ROW);
		downsample(ll, ll_downsample, COL, 0);

//		printf("ll (%d,%d), lh (%d,%d), hl (%d,%d), hh (%d,%d)\n", ll_downsample.rows, ll_downsample.cols, lh_downsample.rows, lh_downsample.cols, hl_downsample.rows, hl_downsample.cols, hh_downsample.rows, hh_downsample.cols);
		ll_downsample.copyTo(out.rowRange(0, in.rows/2).colRange(0, in.rows/2));
		lh_downsample.copyTo(out.rowRange(in.rows/2, in.rows).colRange(0, in.rows/2));
		hl_downsample.copyTo(out.rowRange(0, in.rows/2).colRange(in.rows/2, in.rows));
		hh_downsample.copyTo(out.rowRange(in.rows/2, in.rows).colRange(in.rows/2, in.rows));
		in = ll_downsample;
	}
}

void Wavelet::decode(){
	in.copyTo(out);
	for (int i = 0; i < level; i++){
		int sLL_dim1 = ceil(double(in.rows)/pow(2, level - i));
		int sLL_dim2 = ceil(double(in.cols)/pow(2, level - i));
		int sConstructed_dim1 = ceil(double(in.rows)/pow(2, level - i - 1));
		int sConstructed_dim2 = ceil(double(in.cols)/pow(2, level - i - 1));
		int sHH_dim1 = sConstructed_dim1 - sLL_dim1;
		int sHH_dim2 = sConstructed_dim2 - sLL_dim2;

//		printf("in (%d, %d) \tsLL (%d, %d)\tsConstructed (%d, %d)\tsHH (%d, %d)\n", in.rows, in.cols, sLL_dim1, sLL_dim2, sConstructed_dim1, sConstructed_dim2, sHH_dim1, sHH_dim2);

		Mat ll = Mat(sLL_dim1, sLL_dim2, CV_32FC1);
		Mat lh = Mat(sHH_dim1, sLL_dim2, CV_32FC1);
		Mat hl = Mat(sLL_dim1, sHH_dim2, CV_32FC1);
		Mat hh = Mat(sHH_dim1, sHH_dim2, CV_32FC1);


		out.rowRange(0, sLL_dim1).colRange(0, sLL_dim2).copyTo(ll);
		out.rowRange(0, sLL_dim1).colRange(sLL_dim2, sLL_dim2 + sHH_dim2).copyTo(hl);
		out.rowRange(sLL_dim1, sLL_dim1 + sHH_dim1).colRange(0, sLL_dim2).copyTo(lh);
		out.rowRange(sLL_dim1, sLL_dim1 + sHH_dim1).colRange(sLL_dim2, sLL_dim2 + sHH_dim2).copyTo(hh);

		Mat temp = Mat::zeros(sLL_dim1, sConstructed_dim2, CV_32FC1);
		upsample(ll, temp, COL, 0);
		ll = symconv2(temp, lpr, ROW);

		temp = Mat::zeros(sLL_dim1, sConstructed_dim2, CV_32FC1);
		upsample(hl, temp, COL, 1);
		hl = symconv2(temp, hpr, ROW);

		temp = Mat::zeros(sConstructed_dim1, sConstructed_dim2, CV_32FC1);
		Mat ll_plus_hl;
		add(ll, hl, ll_plus_hl);
		upsample(ll_plus_hl, temp, ROW, 0);
		Mat l = symconv2(temp, lpr, COL);


		temp = Mat::zeros(sHH_dim1, sConstructed_dim2, CV_32FC1);
		upsample(lh, temp, COL, 0);
		lh = symconv2(temp, lpr, ROW);

		temp = Mat::zeros(sHH_dim1, sConstructed_dim2, CV_32FC1);
		upsample(hh, temp, COL, 1);
		hh = symconv2(temp, hpr, ROW);

		temp = Mat::zeros(sConstructed_dim1, sConstructed_dim2, CV_32FC1);
		Mat lh_plus_hh;
		add(lh, hh, lh_plus_hh);
		upsample(lh_plus_hh, temp, ROW, 1);
		Mat h = symconv2(temp, hpr, COL);

		Mat l_plus_h;
		add(l, h, l_plus_h);
		l_plus_h.copyTo(out.rowRange(0, sConstructed_dim1).colRange(0, sConstructed_dim2));
	}

}

Mat Wavelet::symconv2(Mat x, Mat vec, int rowCol){
	int length = vec.cols;
//	printf("length: %d\n", length);
	int half = (length - 1)/2;
	Mat y = Mat::zeros(x.size(), CV_32FC1);
	switch(rowCol){
		case COL: {
			Mat new_x = Mat(x.rows + 2 * half, x.cols, CV_32FC1);
			Mat filtered_x = Mat(x.rows + 2 * half, x.cols, CV_32FC1);
//			printf("x: (%d,%d)\tnew_x: (%d, %d)\n", x.rows, x.cols, new_x.rows, new_x.cols);
			x.copyTo(new_x.rowRange(half, x.cols + half));
			for (int i = 0; i < half; i++){
				printf("copy %d to %d\n", i + 1, half - i - 1);
				x.row(i + 1).copyTo(new_x.row(half - i - 1));
				printf("copy %d to %d\n", x.rows - i - 2, i + x.rows);
				x.row(x.rows - i - 2).copyTo(new_x.row(i + x.rows));
			}
			cv::filter2D(new_x, filtered_x, -1, vec.t());
			filtered_x.rowRange(half, x.rows + half).copyTo(y);
		};
		break;
		case ROW:{
			Mat new_x = Mat(x.rows, x.cols + 2 * half, CV_32FC1);
			Mat filtered_x = Mat(x.rows + 2 * half, x.cols, CV_32FC1);
			x.copyTo(new_x.colRange(half, x.cols + half));
			for (int i = 0; i < half; i++){
				x.col(i + 1).copyTo(new_x.col(half - i - 1));
				x.col(x.cols - i - 2).copyTo(new_x.col(i + x.cols));
			}
			cv::filter2D(new_x, filtered_x, -1, vec);
			filtered_x.colRange(half, x.cols + half).copyTo(y);
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

void Wavelet::upsample(Mat& src, Mat& dest, int rowCol, int start){
//	printf("src (%d, %d) \t des (%d, %d)", src.rows, src.cols, dest.rows, dest.cols);
	switch(rowCol){
	case ROW: {
		assert(src.rows == dest.rows/2);
		for (int i = 0; i < src.rows; i++){
			src.row(i).copyTo(dest.row(i * 2 + start));
		}
	};
	break;
	case COL: {
		assert(src.cols == dest.cols/2);
		for (int i = 0; i < src.cols; i++){
			src.col(i).copyTo(dest.col(i * 2 + start));
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

