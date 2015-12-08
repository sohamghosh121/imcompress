/*
 * Wavelet.cpp
 *
 *  Created on: 6 Dec 2015
 *      Author: sohamghosh
 */

#include "Wavelet.h"

using namespace cv;

Wavelet::Wavelet(Mat img, int type) {
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
	// TODO: needs implementation
}

void Wavelet::decode(){
	// TODO: needs implementation
}

Mat Wavelet::getResult(){
	return out;
}

Wavelet::~Wavelet() {

}

