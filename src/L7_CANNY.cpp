#include <opencv2/opencv.hpp>
#include "./reusables/utils.h"

/** CANNY EDGE DETECTOR **/
cv::Mat canny(cv::Mat & input, int cannyTHL, int cannyTHH, int blurSize = 3, float blurSigma = 0.5) {
    cv::Mat img = input.clone();

    // Step 1: Apply Gaussian blur to reduce noise.
    cv::GaussianBlur(img, img, cv::Size(blurSize, blurSize), blurSigma, blurSigma);

    // Step 2: Compute gradient, magnitude, and phase.
    cv::Mat dx, dy;
    cv::Sobel(img, dx, CV_32F, 1, 0 );
    cv::Sobel(img, dy, CV_32F, 0, 1);

    cv::Mat mag = cv::abs(dx) + cv::abs(dy);
    cv::normalize(mag, mag, 0, 255, cv::NORM_MINMAX, CV_8U);

    cv::Mat phase;
    cv::phase(dx, dy, phase);

    // Step 3: Non-maximum suppression to retain local maximum gradient values.
    cv::Mat nms = mag.clone();
    uchar px1, px2;
    for (int y = 1; y < mag.rows - 1; ++y) {
        for (int x = 1; x < mag.cols - 1; ++x) {
            float angle = phase.at<float>(cv::Point(x, y));

            if ((angle >= 360-22.5 and angle <= 22.5) or (angle >= 360-22.5+180 and angle <= 22.5+180)) {
                px1 = mag.at<uchar>(cv::Point(x+1, y));
                px2 = mag.at<uchar>(cv::Point(x-1, y));
            }
            else if ((angle >= 22.5 and angle <= 22.5+45) or (angle >= 22.5+180 and angle <= 22.5+45+180)) {
                px1 = mag.at<uchar>(cv::Point(x-1, y+1));
                px2 = mag.at<uchar>(cv::Point(x+1, y-1));
            }
            else if ((angle >= 22.5+45 and angle <= 22.5+90) or (angle >= 22.5+45+180 and angle <= 22.5+90+180)) {
                px1 = mag.at<uchar>(cv::Point(x+1, y));
                px2 = mag.at<uchar>(cv::Point(x-1, y));
            }
            else {
                px1 = mag.at<uchar>(cv::Point(x+1, y-1));
                px2 = mag.at<uchar>(cv::Point(x-1, y+1));
            }

            uchar curmag = mag.at<uchar>(cv::Point(x, y));
            if (curmag < px1 or curmag < px2)
                nms.at<uchar>(cv::Point(x, y)) = 0;
        }
    }

    // Step 4: Hysteresis thresholding to identify edges.
    cv::Mat out = cv::Mat::zeros(nms.rows, nms.cols, nms.type());
    for (int y = 0; y < nms.rows; ++y) {
        for (int x = 0; x < nms.cols; ++x) {
            if (nms.at<uchar>(cv::Point(x, y)) > cannyTHH) {
                out.at<uchar>(cv::Point(x, y)) = 255;

                // Define a region of interest (ROI) for local thresholding.
                cv::Rect roi(x - 1, y - 1, 3, 3);
                for (int roi_y = roi.y; roi_y < roi.y + roi.height; ++roi_y) {
                    for (int roi_x = roi.x; roi_x < roi.x + roi.width ; ++roi_x) {
                        if (nms.at<uchar>(cv::Point(roi_x, roi_y)) > cannyTHL and nms.at<uchar>(cv::Point(roi_x, roi_y)) < cannyTHH)
                            out.at<uchar>(cv::Point(roi_x, roi_y)) = 255;
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

    int sigma = 3;
    int threshL = 20;
    int threshH = 30;
    cv::Mat myCannyImg = canny(inputImg, threshL, threshH);
    imshowWrapper("myCannyImg", myCannyImg);
    return 0;
}