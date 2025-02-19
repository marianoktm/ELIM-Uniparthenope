#include <opencv2/opencv.hpp>
#include <queue>
#include "./reusables/utils.h"

bool inRange(cv::Mat & img, cv::Point neigh) {
    return neigh.x >= 0 and neigh.x < img.cols and neigh.y >= 0 and neigh.y < img.rows;
}

bool isSimilar(cv::Mat & img, cv::Point seed, cv::Point neigh, int similTH) {
    int seedIntensity = img.at<uchar>(seed);
    int currIntensity = img.at<uchar>(neigh);
    return std::abs(seedIntensity - currIntensity) < similTH;
}

cv::Mat region_growing(cv::Mat & input, int similTH, cv::Point seed = cv::Point(0, 0)) {
    cv::Mat img = input.clone();

    cv::Mat out = cv::Mat::zeros(img.size(), CV_8U);
    std::queue<cv::Point> pixelQueue;
    pixelQueue.push(seed);
    while (!pixelQueue.empty()) {
        cv::Point currentPx = pixelQueue.front();
        pixelQueue.pop();

        if (out.at<uchar>(currentPx) == 0) {
            out.at<uchar>(currentPx) = 255;

            cv::Rect roi(currentPx.x - 1, currentPx.y - 1, 3, 3);
            for (int roi_x = roi.x; roi_x < roi.x + roi.height; ++roi_x) {
                for (int roi_y = roi.y; roi_y < roi.y + roi.width; ++roi_y) {
                    cv::Point neighPx(roi_x, roi_y);
                    if (inRange(img, neighPx) and isSimilar(img, seed, neighPx, similTH)) {
                        pixelQueue.push(neighPx);
                    }
                }
            }
        }
    }

    return out;
}

int main(int argc, char ** argv) {
    cv::Mat inputImg = imreadWrapper(argc, argv, cv::IMREAD_GRAYSCALE);
    imshowWrapper("inputImg", inputImg);

    int seedx = 20;
    int seedy = 40;
    cv::Point seed(seedx, seedy);
    int similTH = 50;

    cv::Mat rgImg = region_growing(inputImg, similTH, seed);
    imshowWrapper("rgImg", rgImg);

    return 0;
}