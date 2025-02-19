#include <opencv2/opencv.hpp>
#include "../reusables/utils.h"

/**
 * Applies the Canny edge detection algorithm to an input image.
 *
 * This function performs the following steps:
 * 1. Gaussian blur to reduce noise.
 * 2. Gradient computation, including magnitude and phase.
 * 3. Non-maximum suppression to retain local maximum gradient values.
 * 4. Hysteresis thresholding to identify edges based on high and low threshold values.
 *
 * @param input     Input image (grayscale).
 * @param cannyTHL  Lower threshold for hysteresis thresholding.
 * @param cannyTHH  Upper threshold for hysteresis thresholding.
 * @param blurSize  Size of the Gaussian filter kernel (default is 3).
 * @param blurSigma Standard deviation for Gaussian blur (default is 0.5).
 *
 * @return Binary image with detected edges (edge pixels set to 255, others to 0).
 */
cv::Mat canny(cv::Mat & input, int cannyTHL, int cannyTHH, int blurSize = 3, float blurSigma = 0.5) {
    cv::Mat img = input.clone();

    // Step 1: Apply Gaussian blur to reduce noise.
    cv::GaussianBlur(img, img, cv::Size(blurSize, blurSize), blurSigma, blurSigma);

    // Step 2: Compute gradient, magnitude, and phase.
    cv::Mat x_gradient, y_gradient;
    cv::Sobel(img, x_gradient, CV_32F, 1, 0 );
    cv::Sobel(img, y_gradient, CV_32F, 0, 1);

    cv::Mat magnitude = cv::abs(x_gradient) + cv::abs(y_gradient);
    cv::normalize(magnitude, magnitude, 0, 255, cv::NORM_MINMAX, CV_8U);

    cv::Mat phase;
    cv::phase(x_gradient, y_gradient, phase);

    // Step 3: Non-maximum suppression to retain local maximum gradient values.
    cv::Mat nonMaximaSuppressed = magnitude.clone();
    uchar pixel1, pixel2;
    for (int y = 1; y < magnitude.rows - 1; ++y) {
        for (int x = 1; x < magnitude.cols - 1; ++x) {
            float angle = phase.at<float>(cv::Point(x, y));

            if ((angle >= 360-22.5 and angle <= 22.5)
            or (angle >= 360-22.5+180 and angle <= 22.5+180)) {
                pixel1 = magnitude.at<uchar>(cv::Point(x + 1, y));
                pixel2 = magnitude.at<uchar>(cv::Point(x - 1, y));
            }
            else if ((angle >= 22.5 and angle <= 22.5+45)
            or (angle >= 22.5+180 and angle <= 22.5+45+180)) {
                pixel1 = magnitude.at<uchar>(cv::Point(x - 1, y + 1));
                pixel2 = magnitude.at<uchar>(cv::Point(x + 1, y - 1));
            }
            else if ((angle >= 22.5+45 and angle <= 22.5+90)
            or (angle >= 22.5+45+180 and angle <= 22.5+90+180)) {
                pixel1 = magnitude.at<uchar>(cv::Point(x + 1, y));
                pixel2 = magnitude.at<uchar>(cv::Point(x - 1, y));
            }
            else {
                pixel1 = magnitude.at<uchar>(cv::Point(x + 1, y - 1));
                pixel2 = magnitude.at<uchar>(cv::Point(x - 1, y + 1));
            }

            uchar currentMagnitude = magnitude.at<uchar>(cv::Point(x, y));
            if (currentMagnitude < pixel1 or currentMagnitude < pixel2)
                nonMaximaSuppressed.at<uchar>(cv::Point(x, y)) = 0;
        }
    }

    // Step 4: Hysteresis thresholding to identify edges.
    cv::Mat edgesImg = cv::Mat::zeros(nonMaximaSuppressed.rows, nonMaximaSuppressed.cols, nonMaximaSuppressed.type());
    for (int y = 0; y < nonMaximaSuppressed.rows; ++y) {
        for (int x = 0; x < nonMaximaSuppressed.cols; ++x) {
            if (nonMaximaSuppressed.at<uchar>(cv::Point(x, y)) > cannyTHH) {
                edgesImg.at<uchar>(cv::Point(x, y)) = 255;

                // Define a region of interest (ROI) for local thresholding.
                cv::Rect regionOfInterest(x - 1, y - 1, 3, 3);
                for (int roi_y = regionOfInterest.y; roi_y < regionOfInterest.y + regionOfInterest.height; ++roi_y) {
                    for (int roi_x = regionOfInterest.x; roi_x < regionOfInterest.x + regionOfInterest.width ; ++roi_x) {
                        if (nonMaximaSuppressed.at<uchar>(cv::Point(roi_x, roi_y)) > cannyTHL and nonMaximaSuppressed.at<uchar>(cv::Point(roi_x, roi_y)) < cannyTHH)
                            edgesImg.at<uchar>(cv::Point(roi_x, roi_y)) = 255;
                    }
                }
            }
        }
    }

    return edgesImg;
}

int main(int argc, char ** argv) {
    cv::Mat inputImg = imreadWrapper(argc, argv, cv::IMREAD_GRAYSCALE);
    imshowWrapper("Input Img", inputImg);

    int cannyTHL = 5;
    int cannyTHH = 20;
    int blurSize = 21;

    cv::Mat cannyImg = canny(inputImg, cannyTHL, cannyTHH, blurSize);
    imshowWrapper("Canny Img", cannyImg);
    return 0;
}
