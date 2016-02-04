#include <opencv2/opencv.hpp>
#include "Encoder.h"
#include "Decoder.h"
#include "Wavelet.h"
#include "SBHE.h"
#include "SpaRSA_noSI.h"
#include "Options.h"
#include <cmath>
using namespace std;

int main(int argc, char** argv) {
//	Options::parseOptionsFile("/Users/sohamghosh/src/imcompress/input/options");
	Options::dumpOptions();
	std::cout << "-------------------\n";
	std::cout << "Average measurement rate: " << (Options::Mk + Options::Mw * (Options::M * Options::M - 1))/(Options::M * Options::M) << "\n";
	cv::Mat inputImage = cv::imread("/Users/sohamghosh/src/imcompress/photos/lena.jpg");
	cv::cvtColor(inputImage, inputImage, CV_RGB2GRAY);
	clock_t startTime = clock();
	Encoder e(inputImage);
	e.encodeImage();
	std::cout << "Encoding image: " << double( clock() - startTime ) / (double)CLOCKS_PER_SEC << "s" << std::endl;

	Decoder d(inputImage.rows, inputImage.cols, e.getKeyPhi(), e.getnonkeyPhi(), e.getEncodedValues());
	d.decodeImage();
	std::cout << "Decoding image: " << double( clock() - startTime ) / (double)CLOCKS_PER_SEC << "s" << std::endl;
	std::cout << "PSNR: " << cv::PSNR(inputImage, d.getDecodedImage()) << "dB" << std::endl;
	cv::imwrite("/Users/sohamghosh/src/imcompress/photos/lena_out.jpg", d.getDecodedImage());
	cv::imshow("Output", d.getDecodedImage());
	cv::waitKey(0);
	return 0;
}
