#include <opencv2/opencv.hpp>
#include "Encoder.h"
#include "Decoder.h"
#include "Wavelet.h"
#include "SBHE.h"
#include <cmath>

double getPSNR(cv::Mat original, cv::Mat reconstructed){
	double MSE = cv::norm(original, reconstructed, cv::NORM_L2, cv::noArray());
	cv::Point min_loc, max_loc;
	double min_I, max_I;
	cv::minMaxLoc(original, &min_I, &max_I, &min_loc, &max_loc);
	double PSNR = 10 * log10(pow(max_I, 2)/MSE);
	return PSNR;
}

int main(int argc, char** argv) {
	cv::Mat inputImage = cv::imread("/Users/sohamghosh/photo.jpg");
	std::printf("size(%d, %d)\n", inputImage.rows, inputImage.cols);
	cv::cvtColor(inputImage, inputImage, CV_RGB2GRAY, 0);
	inputImage.convertTo(inputImage, CV_32FC1);
	cv::Mat w = Wavelet(inputImage, Wavelet::DWT).getResult();
//	w.convertTo(w, CV_8UC1);
//	std::cout << w.rowRange(0, w.rows/2).colRange(0, w.cols/2);
//	cv::imshow("Wavelet", w.rowRange(0, w.rows/2).colRange(0, w.cols/2));
//	Encoder e(inputImage);
//	e.encodeImage();
//	printf("done encoding\n");
//	Decoder d(inputImage.rows, inputImage.cols, e.getKeyPhi(), e.getnonkeyPhi(), e.getEncodedValues());
//	d.decodeImage();
//	std::cout << d.getDecodedImage();
//	cv::imshow("Output", d.getDecodedImage());
//	cv::waitKey(0);
	return 0;
}
