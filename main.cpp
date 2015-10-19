#include <opencv2/opencv.hpp>
#include "Encoder.h"
#include "Decoder.h"
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
	std::printf("%s\n", argv[1]);
	cv::Mat inputImage = cv::imread(argv[1]);
	std::printf("size(%d, %d)\n", inputImage.rows, inputImage.cols);

//	cv::imshow("Input Image", inputImage);

	Encoder e(inputImage);
	e.encodeImage();
	Decoder d(inputImage.rows, inputImage.cols, e.getKeyPhi(), e.getnonkeyPhi(), e.getEncodedValues());
	d.decodeImage();
	std::cout << d.getDecodedImage();
	cv::imshow("Output", d.getDecodedImage());
	cv::waitKey(0);
	return 0;
}
