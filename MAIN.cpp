#include "RCI_solver.h"
#include <filesystem>
#include <iostream>

using namespace cv;

namespace fs = std::filesystem;

void clear_output_directory(const std::string& path) {
    try {
        if (fs::exists(path)) {
            for (const auto& entry : fs::directory_iterator(path)) {
                fs::remove_all(entry.path());
            }
        }
    }
    catch (const fs::filesystem_error& e) {
        std::cerr << "Filesystem error: " << e.what() << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "General error: " << e.what() << std::endl;
    }
}


int main() {
    RCI_solver solver;
    fs::create_directories("output");
    clear_output_directory("output");
    std::string directory = "asset/bg-310/test/";
    int try_case = 0;
    try {
        for (const auto& entry : fs::directory_iterator(directory)) {
            if (entry.is_regular_file()) {
                try_case++;
                int cur_window_size = 128;
                std::string file_path = entry.path().string();
                std::cout << "Processing file: " << file_path << std::endl;
                std::string output_path = "output/" + std::to_string(try_case);
                fs::create_directory(output_path);

                //window size 64
                std::string path_64 = output_path + "/64";
                fs::create_directory(path_64);
                std::pair<Mat, Mat> processed = solver.image_process(file_path, path_64);
                solver.target_detection(processed.first, processed.second, file_path, 64, path_64);

                //window size 128
                std::string path_128 = output_path + "/128";
                fs::create_directory(path_128);
                processed = solver.image_process(file_path, path_128);
                solver.target_detection(processed.first, processed.second, file_path, 128, path_128);

                std::cout << "Finished processing file: " << file_path << std::endl;
            }
        }
    }
    catch (const fs::filesystem_error& e) {
        std::cerr << "Filesystem error: " << e.what() << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "General error: " << e.what() << std::endl;
    }

    //Mat image = solver.image_process("asset/bg-310/all/238.2.png");
    //solver.target_detection(image, "asset/bg-310/all/238.2.png", 32);
    //for (auto i : to_be_solved_file) {
    //    std::cout << "Processed: " << i << "\n";
    //    solver.image_process(directory + i + ".png");
    //}

 //   std::cout << "finished" << std::endl;
 //   for(auto i : all_box){
 //       std::cout << "i_area:" << i.size.area() << std::endl;
	//}
    return 0;
}