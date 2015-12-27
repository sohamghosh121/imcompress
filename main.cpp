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
//	std::printf("size(%d, %d)\n", inputImage.rows, inputImage.cols);
	cv::cvtColor(inputImage, inputImage, CV_RGB2GRAY, 0);
	inputImage.convertTo(inputImage, CV_32FC1);
	cv::Mat d = Wavelet(inputImage, Wavelet::DWT).getResult();
	cv::Mat r = Wavelet(d, Wavelet::IDWT).getResult();
//	for (int i = 0; i < d.rows; i++){
//		for (int j = 0; j < d.cols; j++){
//			std::cout << d.at<float>(i, j) << " ";
//		}
//		std::cout << std::endl;
//	}
	r.convertTo(r, CV_8UC1);
//	printf("r (%d, %d) \n", r.rows, r.cols);
	inputImage.convertTo(inputImage, CV_8UC1);
	cv::imshow("Wavelet Reconstruction", r);
//	Encoder e(inputImage);
//	e.encodeImage();
//	printf("done encoding\n");
//	Decoder d(inputImage.rows, inputImage.cols, e.getKeyPhi(), e.getnonkeyPhi(), e.getEncodedValues());
//	d.decodeImage();
//	std::cout << d.getDecodedImage();
//	cv::imshow("Output", d.getDecodedImage());
	cv::waitKey(0);
	return 0;
}
