#pragma once
#include "define.h"

extern std::vector<RotatedRect> all_box;
class RCI_solver
{
public:
	std::pair<Mat,Mat> image_process(std::string address, std::string output_path);
	void target_detection(Mat processed, Mat src, std::string address, int window_size, std::string output_path);
};

