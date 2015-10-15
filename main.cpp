#include <opencv2/opencv.hpp>
#include "Encoder.h"

int main(int argc, char** argv) {
	std::printf("%s\n", argv[1]);
	cv::Mat inputImage = cv::imread(argv[1]);
	std::printf("size(%d, %d)\n", inputImage.rows, inputImage.cols);

	cv::imshow("Input Image", inputImage);

	Encoder e(inputImage);
	e.encodeImage();


	cv::waitKey(0);
	return 0;
}
