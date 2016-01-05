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
	cv::Mat inputImage = cv::imread("/Users/sohamghosh/src/imcompress/photos/photo.jpg");
	clock_t startTime = clock();
	Encoder e(inputImage);
	std::cout << "Encoding image: " << double( clock() - startTime ) / (double)CLOCKS_PER_SEC << "s" << std::endl;
	e.encodeImage();
	Decoder d(inputImage.rows, inputImage.cols, e.getKeyPhi(), e.getnonkeyPhi(), e.getEncodedValues());
	startTime = clock();
	d.decodeImage();
	std::cout << "Decoding image: " << double( clock() - startTime ) / (double)CLOCKS_PER_SEC << "s" << std::endl;

	cv::imshow("Output", d.getDecodedImage());
	cv::imwrite("/Users/sohamghosh/src/imcompress/photos/output.jpg", d.getDecodedImage());
	cv::waitKey(0);
	return 0;
}
