#include "RCI_solver.h"
#include <filesystem> // C++17 文件系统库
#include <opencv2/dnn.hpp> // OpenCV DNN 模块
#include <random>
#include <algorithm>
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> dis(0.0, 1.0);
float probability = 0.02;


using namespace cv::ml;
namespace fs = std::filesystem;

#define stride_size 4
#define block_size 16
#define blockstride 8
#define cell_size 8
#define nbins 9

// 计算灰度直方图
double calculateGrayHistogramStdDev(const Mat& roi) {
    int histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange = { range };
    Mat hist;
    calcHist(&roi, 1, 0, Mat(), hist, 1, &histSize, &histRange);

    // 计算灰度直方图的均值和标准差
    Scalar mean, stddev;
    meanStdDev(hist, mean, stddev);

    return stddev[0];
}

// 可视化 HOG 特征
void HOG_visualize(Mat& img, std::vector<float>& descriptors, std::string output_path, int &save_img_id, bool is_visualize = 0) {
    
    //计算图像梯度
     Mat gx, gy;
     Sobel(img, gx, CV_32F, 1, 0, 1);
     Sobel(img, gy, CV_32F, 0, 1, 1);

     //计算梯度幅值和方向
     Mat magnitude, angle;
     cartToPolar(gx, gy, magnitude, angle, 1);

     float max_magnitude = 0.0f;

     //构建梯度直方图
     int nCellsX = img.cols / cell_size;
     int nCellsY = img.rows / cell_size;
     std::vector<std::vector<std::vector<float>>> hist(nCellsY, std::vector<std::vector<float>>(nCellsX, std::vector<float>(nbins, 0.0f)));
     for (int i = 0; i < img.rows; i++) {
         for (int j = 0; j < img.cols; j++) {
             int cellX = j / cell_size;
             int cellY = i / cell_size;
             float bin = angle.at<float>(i, j) / (360.0 / nbins);
             int bin0 = static_cast<int>(bin) % nbins;
             int bin1 = (bin0 + 1) % nbins;
             float weight0 = 1.0f - (bin - bin0);
             float weight1 = bin - bin0;
             hist[cellY][cellX][bin0] += magnitude.at<float>(i, j) * weight0;
             hist[cellY][cellX][bin1] += magnitude.at<float>(i, j) * weight1;
             max_magnitude = max(max_magnitude, abs(magnitude.at<float>(i, j)));
         }
     }

     //可视化
     if (is_visualize) {
         Mat visualize = Mat::zeros(img.rows * 9, img.cols * 9, CV_8U);
         for (int i = 0; i < img.rows; i++) {
             for (int j = 0; j < img.cols; j++) {
                 for (int y = 0; y < 9; y++) {
                     for (int x = 0; x < 9; x++) {
                         visualize.at<uchar>(i * 9 + y, j * 9 + x) = img.at<uchar>(i, j);
                     }
                 }
                 float current_angle = angle.at<float>(i, j);
                 float current_magnitude = magnitude.at<float>(i, j);
                 float current_angle_rad = current_angle * CV_PI / 180.0;
                 Point current_center(j * 9 + 4, i * 9 + 4);
                 Point current_end(current_center.x + cos(current_angle_rad) * current_magnitude / max_magnitude * 40, current_center.y - sin(current_angle_rad) * current_magnitude / max_magnitude * 40);
                 arrowedLine(visualize, current_center, current_end, Scalar(0, 0, 255), 1);
             }
         }
         std::string visualize_path = output_path + "/" + std::to_string(save_img_id) + "_visualize.jpg";
         imwrite(visualize_path, visualize);
     }
     else {
         //构建block描述子
         int nBlocksX = nCellsX - 1;
         int nBlocksY = nCellsY - 1;
         for (int i = 0; i < nBlocksY; i++) {
             for (int j = 0; j < nBlocksX; j++) {
                 std::vector<float> block_descriptor;
                 for (int y = 0; y < 2; y++) {
                     for (int x = 0; x < 2; x++) {
                         for (int k = 0; k < nbins; k++) {
                             block_descriptor.push_back(hist[i + y][j + x][k]);
                         }
                     }
                 }
                 //归一化
                 float norm = 0.0f;
                 for (int k = 0; k < block_descriptor.size(); k++) {
                     norm += block_descriptor[k] * block_descriptor[k];
                 }
                 norm = sqrt(norm);
                 for (int k = 0; k < block_descriptor.size(); k++) {
                     block_descriptor[k] /= norm;
                 }
                 descriptors.insert(descriptors.end(), block_descriptor.begin(), block_descriptor.end());
             }
         }
     }
}

// 计算两个矩形框的 IoU
float computeIoU(const Rect& box1, const Rect& box2) {
    int x1 = std::max(box1.x, box2.x);
    int y1 = std::max(box1.y, box2.y);
    int x2 = std::min(box1.x + box1.width, box2.x + box2.width);
    int y2 = std::min(box1.y + box1.height, box2.y + box2.height);

    int intersectionArea = std::max(0, x2 - x1) * std::max(0, y2 - y1);
    int box1Area = box1.width * box1.height;
    int box2Area = box2.width * box2.height;

    return static_cast<float>(intersectionArea) / (box1Area + box2Area - intersectionArea);
}

// 非极大值抑制
void manualNMS(const std::vector<Rect>& boxes, const std::vector<float>& confidences, float confidences_threshold, float area_threshold, std::vector<int>& indices) {
    std::vector<int> sortedIndices(boxes.size());
    for(int i = 0; i < boxes.size(); i++) sortedIndices[i] = i;

    // 按置信度从高到低排序
    std::sort(sortedIndices.begin(), sortedIndices.end(), [&confidences](int a, int b) {
        return confidences[a] > confidences[b];
        });

    std::vector<bool> suppressed(boxes.size(), false);

    for (size_t i = 0; i < sortedIndices.size(); ++i) {
        int idx = sortedIndices[i];
        if (suppressed[idx] || confidences[idx] < confidences_threshold) continue;

        indices.push_back(idx);
        for (size_t j = i + 1; j < sortedIndices.size(); ++j) {
            int jdx = sortedIndices[j];
            if (suppressed[jdx]) continue;

            if (computeIoU(boxes[idx], boxes[jdx]) > area_threshold) {
                suppressed[jdx] = true;
            }
        }
    }
}

// 目标检测
void RCI_solver::target_detection(Mat processed, Mat src, std::string address, int window_size, std::string output_path) {
    HOGDescriptor hog{
        Size(32, 32), // winSize
        Size(16, 16), // blockSize
        Size(8, 8),   // blockStride
        Size(8, 8),   // cellSize
        9             // nbins
    };

    Ptr<SVM> svm = SVM::load("asset/" + std::to_string(window_size) + "_svm_model.xml");

    std::string hog_path = output_path + "/hog_visualize";
    fs::create_directory(hog_path);
    int hog_id = 0;

    std::vector<Rect> found;
    std::vector<float> confidences;
    std::vector<float> stddev;
    for (int i = 0; i < src.rows; i += stride_size) {
        for (int j = 0; j < src.cols; j += stride_size) {
            if (processed.at<Vec3b>(i, j) == Vec3b(0, 0, 0)) {
                continue; // 跳过黑色像素
            }
            if (j - window_size / 2 < 0 || i - window_size / 2 < 0 || j + window_size / 2 > src.cols || i + window_size / 2 > src.rows) continue;
            Rect roi(j - window_size / 2, i - window_size / 2, window_size, window_size);
            Mat gray = src(roi);
            cvtColor(gray, gray, COLOR_BGR2GRAY);//转换为灰度图像
            std::vector<float> roi_hog; // HOG 描述子
            if (dis(gen) <= probability) {
                HOG_visualize(gray, roi_hog, hog_path, ++hog_id, 1);
            }
            hog.compute(gray, roi_hog);//计算HOG描述子
            Mat roi_hog_mat(1, roi_hog.size(), CV_32FC1, roi_hog.data());//将HOG描述子转换为 Mat 类型

            if (roi_hog_mat.cols != svm->getVarCount()) {
                std::cout << "Feature vector length: " << roi_hog_mat.cols << std::endl;
                std::cout << "Var count: " << svm->getVarCount() << std::endl; // 输出 "Var count: 1764
                std::cerr << "Error: Feature vector length does not match var_count." << std::endl;
                continue;
            }//检查特征向量长度是否匹配

            float response = svm->predict(roi_hog_mat);//使用 SVM 进行预测
            if (response == 1) {
                found.push_back(roi);
                
                double minVal;
                Point center(roi.width / 2, roi.height / 2); // 计算中心点坐标（x, y）（列，行）（宽，高）（j，i）（x，y）
                Point minLoc;
                minMaxLoc(gray, &minVal, nullptr, &minLoc);
                double distance = norm(center - minLoc);
                float factor1 = 1.0f - static_cast<float>(minVal) / 255.0f;
                float factor2 = 1.0f - static_cast<float>(distance) / roi.width / sqrt(2);
                float factor3 = 1.0f - calculateGrayHistogramStdDev(gray) / 255.0f;
                stddev.push_back(factor3);
                float alpha = 0.55, beta = 0.15;
                float confidence = alpha * factor1 + beta * factor2 + (1 - alpha - beta) * factor3; //经验方程
                if(confidence < 0 || confidence > 1) {
					std::cerr << "Error: Confidence value out of range: " << confidence << std::endl;
					continue;
				}
                confidences.push_back(confidence);//计算置信度
            }//如果预测结果为正，则将当前窗口添加到 found 中
        }
    }

    // 使用 NMS 方法对检测结果进行过滤
    std::vector<int> indices;
    float nms_threshold = 0.005; // NMS 阈值
    manualNMS(found, confidences, 0.5, nms_threshold, indices);//非极大值抑制
    Mat result = src.clone();

    for(size_t i = 0; i < indices.size(); i++) {
        Point center(found[indices[i]].x + found[indices[i]].width / 2, found[indices[i]].y + found[indices[i]].height / 2);
        Point roi_center(src.cols / 2, src.rows / 2);
        double distance = norm(center - roi_center);
        double roi_distance = sqrt(src.cols * src.cols + src.rows * src.rows) / 2;
        if(distance / roi_distance > 0.4) {
			indices.erase(indices.begin() + i);//删除距离中心点过远的检测结果
			i--;
		}
	}

    // 绘制经过 NMS 筛选的结果（红色）
    for (size_t i = 0; i < min((int)indices.size(), 3); i++) {
        Rect r = found[indices[i]];
        rectangle(result, r.tl(), r.br(), Scalar(0, 0, 255), 2);//绘制矩形框

    }

    // 将处理后的图像保存到本地
    std::string result_path = output_path + "/result.jpg";
    imwrite(result_path, result);

    waitKey(0);
    destroyAllWindows();
}