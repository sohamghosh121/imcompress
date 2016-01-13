/*
 * Wavelet.cpp
 *
 *  Created on: 6 Dec 2015
 *      Author: sohamghosh
 */

#include "Wavelet.h"
#include "Options.h"
#include "blitzwave/WaveletDecomp.h"
#include <blitz/array.h>
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


Mat naiveBlitzToCvMat(blitz::Array<float, 2> a){
	int i, j;
	Mat b = Mat(a.rows(), a.cols(), CV_32FC1);
	for (i=0; i< a.rows(); i++){
		for(j=0; j<a.cols(); j++){
			b.at<float>(i,j) = a(i, j);
		}
	}
	return b;
}

void Wavelet::encode(){
	blitz::Array<float,2> tmp(   (float*)(this->in.data),
	                    blitz::shape(this->in.rows,
	                    this->in.cols),
	                    blitz::neverDeleteData);
	bwave::WaveletDecomp<2> decomp(bwave::WL_CDF_97, bwave::NONSTD_DECOMP, Options::wavelet_level);
	decomp.apply(tmp);
	out = naiveBlitzToCvMat(tmp);
}

void Wavelet::decode(){
	blitz::Array<float,2> tmp(   (float*)(this->in.data),
		                    blitz::shape(this->in.rows,
		                    this->in.cols),
		                    blitz::neverDeleteData);
	bwave::WaveletDecomp<2> decomp(bwave::WL_CDF_97, bwave::NONSTD_DECOMP, Options::wavelet_level);
	decomp.applyInv(tmp);
	out = naiveBlitzToCvMat(tmp);
}

Mat Wavelet::getResult(){
	return out;
}

Wavelet::~Wavelet() {

}

