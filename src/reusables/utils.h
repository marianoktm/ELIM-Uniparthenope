#ifndef OPENCVELIM_UTILS_H
#define OPENCVELIM_UTILS_H

#include <string>
#include <opencv2/opencv.hpp>

void imshowWrapper(std::string const& winname, cv::Mat& mat) {
    cv::imshow(winname, mat);
    cv::waitKey(0);
    cv::destroyWindow(winname);
}

cv::Mat imreadWrapper(int argc, char ** argv, int type = cv::IMREAD_COLOR) {
    if (argc == 1) {
        std::cerr << "No img path parameter found." << std::endl;
        exit(-1);
    }

    cv::Mat inputImg = cv::imread(argv[1], type);
    if (inputImg.empty()) {
        std::cerr << "The img is empty" << std::endl;
        exit(-2);
    }

    return inputImg;
}

#endif //OPENCVELIM_UTILS_H
