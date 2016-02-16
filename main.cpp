#include <opencv2/opencv.hpp>
#include "Encoder.h"
#include "Decoder.h"
#include "Options.h"
#include<thread>
using namespace std;

int main(int argc, char** argv) {
	Options::parseOptionsFile("/Users/sohamghosh/src/imcompress/input/options");
	std::cout << "Average measurement rate: " << (Options::Mk + Options::Mw * (Options::M * Options::M - 1))/(Options::M * Options::M) << "\n";
	cv::Mat inputImage = cv::imread(argv[1]);
	cv::cvtColor(inputImage, inputImage, CV_RGB2GRAY);
	cv::resize(inputImage, inputImage, cv::Size(1216, 384)); //
	clock_t startTime = clock();
	Encoder e(inputImage);
	e.encodeImage();
	std::cout << "Encoding image: " << double( clock() - startTime ) / (double)CLOCKS_PER_SEC << "s" << std::endl;
//
//	Decoder d(inputImage.rows, inputImage.cols, e.getKeyPhi(), e.getnonkeyPhi(), e.getEncodedValues());
//	d.decodeImage();
//	std::cout << "Decoding image: " << double( clock() - startTime ) / (double)CLOCKS_PER_SEC << "s" << std::endl;
//	std::cout << "PSNR: " << cv::PSNR(inputImage, d.getDecodedImage()) << "dB" << std::endl;
	return 0;
}
