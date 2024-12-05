#include "RCI_solver.h"

std::vector<RotatedRect> all_box;

#define threshold_ratio 0.25

double ellipse_area_ratio(const RotatedRect& box, Mat& cluster) {
    // 创建一个掩码，掩码区域为box
    Mat mask = Mat::zeros(cluster.size(), CV_8UC1);
    ellipse(mask, box, Scalar(255), -1);

    // 计算box区域内的非零像素个数
    int nonZeroPixels = countNonZero(cluster & mask);

    // 计算box区域的像素总数
    int totalPixels = box.size.area();

    // 计算非零像素占比
    double ratio = (double)nonZeroPixels / totalPixels;

    return ratio;
}

bool edge_detection(const Mat& cluster) {
    // 定义边缘宽度
    int border_width = 10;
    int total_border_pixels = 0;
    int colored_border_pixels = 0;

    // 上边缘
    for (int i = 0; i < border_width; i++) {
        for (int j = 0; j < cluster.cols; j++) {
            total_border_pixels++;
            if (cluster.at<uchar>(i, j) > 0) {
                colored_border_pixels++;
            }
        }
    }

    // 下边缘
    for (int i = cluster.rows - border_width; i < cluster.rows; i++) {
        for (int j = 0; j < cluster.cols; j++) {
            total_border_pixels++;
            if (cluster.at<uchar>(i, j) > 0) {
                colored_border_pixels++;
            }
        }
    }

    // 左边缘
    for (int i = 0; i < cluster.rows; i++) {
        for (int j = 0; j < border_width; j++) {
            total_border_pixels++;
            if (cluster.at<uchar>(i, j) > 0) {
                colored_border_pixels++;
            }
        }
    }

    // 右边缘
    for (int i = 0; i < cluster.rows; i++) {
        for (int j = cluster.cols - border_width; j < cluster.cols; j++) {
            total_border_pixels++;
            if (cluster.at<uchar>(i, j) > 0) {
                colored_border_pixels++;
            }
        }
    }

    // 计算边缘有色像素所占比例
    double edge_ratio = static_cast<double>(colored_border_pixels) / total_border_pixels;

    // 判断该比例是否超过设定的阈值
    return edge_ratio <= threshold_ratio;
}

std::pair<Mat, Mat>RCI_solver::image_process(std::string address, std::string output_path) {
    Mat src = imread(address, IMREAD_COLOR);  // 读取彩色图像
    Mat src_gray = imread(address, IMREAD_GRAYSCALE);  // 读取灰度图像

    Mat otsu_thresold;
    double binary_thershold = threshold(src_gray, otsu_thresold, 0, 255, THRESH_OTSU | THRESH_BINARY_INV);  // 对src_gray进行otsu二值化
    binary_thershold = binary_thershold * 0.975;
    threshold(src_gray, otsu_thresold, binary_thershold, 255, THRESH_BINARY_INV);  // 对src_gray进行otsu二值化

    Mat hsv;
    cvtColor(src, hsv, COLOR_BGR2HSV);  // 转换为HSV颜色空间
    medianBlur(hsv, hsv, 17);  // 进行中值模糊进行降噪
    Mat data;
    hsv.convertTo(data, CV_32F);
    data = data.reshape(1, data.total());  // 将图像数据转换为2D浮点型矩阵
    int k_means_clusters = 4;
    Mat labels, centers;
    TermCriteria k_means_criteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0);  // 设置K-means参数
    kmeans(data, k_means_clusters, labels, k_means_criteria, k_means_clusters, KMEANS_PP_CENTERS, centers);  // 执行K-means聚类

    Mat* clusters = new Mat[k_means_clusters];
    for (int i = 0; i < k_means_clusters; i++) clusters[i] = Mat(src.size(), src.type());
    Mat kmeans_result(src.size(), src.type());
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            int cluster_idx = labels.at<int>(i * src.cols + j);
            auto cur_centers = Vec3b((uchar)centers.at<float>(cluster_idx, 0), (uchar)centers.at<float>(cluster_idx, 1), (uchar)centers.at<float>(cluster_idx, 2));
            auto pure_black = Vec3b(0, 0, 0);
            kmeans_result.at<Vec3b>(i, j) = cur_centers;
            for (int k = 0; k < k_means_clusters; k++)  clusters[k].at<Vec3b>(i, j) = (k == cluster_idx) ? cur_centers : pure_black;
        }
    }  // 将聚类结果映射回图像
    int selected_cluster = -1;
    double max_area_ratio = 0;
    RotatedRect selected_BOX;

    for (int i = 0; i < k_means_clusters; i++) {
        Mat resize_imgae;
        resize_imgae = clusters[i].clone();
        resize(resize_imgae, resize_imgae, Size(), 0.5, 0.5);
        //imshow("resize" + std::to_string(i) + address, resize_imgae);
        Mat element = getStructuringElement(MORPH_RECT, Size(7, 7));
        Mat cur_cluster, otsu_thresold;
        cur_cluster = clusters[i].clone();
        cvtColor(cur_cluster, cur_cluster, COLOR_BGR2GRAY);
        morphologyEx(cur_cluster, otsu_thresold, MORPH_CLOSE, element);  // 闭运算
        threshold(otsu_thresold, otsu_thresold, 0, 255, THRESH_BINARY | THRESH_OTSU);  // OTSU二值化

        std::vector<std::vector<Point> > contours;
        std::vector<Vec4i> hierarchy;
        findContours(otsu_thresold, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);  // 寻找轮廓
        std::vector<RotatedRect> boxes;
        for (int j = 0; j < contours.size(); j++) {
            int cur_contours_size = contours[j].size();
            if (cur_contours_size < 100) continue;
            RotatedRect box = fitEllipse(contours[j]);
            boxes.push_back(box);
        }  // 绘制椭圆
        sort(boxes.begin(), boxes.end(), [](const RotatedRect& a, const RotatedRect& b) {
			return a.size.area() > b.size.area();
			});//
        double cur_ratio = ellipse_area_ratio(boxes[0], cur_cluster);
        if (cur_ratio > max_area_ratio && edge_detection(cur_cluster)) {
			max_area_ratio = cur_ratio;
			selected_cluster = i;
            if(!boxes.empty()) selected_BOX = boxes[0];//选取最大的椭圆
		}
    }//对每个聚类进行椭圆拟合
    delete[] clusters;

    if (selected_BOX.size.area() > 3e5) {
        double squtze_ratio = sqrt(3e5 / selected_BOX.size.area());
        selected_BOX.size.width *= squtze_ratio;
        selected_BOX.size.height *= squtze_ratio;
    }//对椭圆进行缩放
    //all_box.push_back(selected_BOX);//将所有的box存入all_box
    //std::cout << address+ "BOX_area: " << selected_BOX.size.area() << std::endl;

    Mat mask = Mat::zeros(src.size(), CV_8UC1);
    ellipse(mask, selected_BOX, Scalar(255), -1);
    Mat new_mask;
    bitwise_and(mask, otsu_thresold, new_mask);
    Mat smoothed_mask;
    GaussianBlur(new_mask, smoothed_mask, Size(7, 7), 0);// 高斯模糊
    Mat result;
    src.copyTo(result, smoothed_mask);//将原图像与mask相乘

    //储存结果
    std::string mk_path = output_path + "/mask.jpg";
    imwrite(mk_path, result);
    std::string src_path = output_path + "/src.jpg";
    imwrite(src_path, src);
    return { result, src };
}