#include <opencv2/opencv.hpp>
#include "../reusables/utils.h"

/**
 * Applies the Hough Lines Detection algorithm to an input image.
 *
 * This function performs the following steps:
 * 1. Applies Gaussian blur to reduce noise for Canny edge detector.
 * 2. Performs Canny edge detection to highlight edges.
 * 3. Computes Hough Transform to detect lines.
 * 4. Draws detected lines on the input image.
 *
 * @param input     Input image (grayscale).
 * @param houghTH   Threshold for line detection in the Hough space.
 * @param cannyTHL  Lower threshold for Canny edge detection.
 * @param cannyTHH  Upper threshold for Canny edge detection.
 * @param blurSize  Size of the Gaussian filter kernel for smoothing.
 * @param blurSigma Standard deviation for Gaussian blur.
 *
 * @return An image with detected lines drawn on it.
 */
cv::Mat hough_lines(cv::Mat & input, int houghTH, int cannyTHL, int cannyTHH, int blurSize, float blurSigma) {
    cv::Mat img = input.clone();

    // Step 1: Apply Gaussian blur to reduce noise.
    cv::GaussianBlur(img, img, cv::Size(blurSize, blurSize), blurSigma);

    // Step 2: Perform Canny edge detection to highlight edges.
    cv::Canny(img, img, cannyTHL, cannyTHH);

    // Step 3: Compute Hough Transform to detect lines.
    int diagonalLenght = cvRound(std::hypot(img.rows, img.cols));
    int maxTheta = 180;
    cv::Mat votes = cv::Mat::zeros(diagonalLenght * 2, maxTheta, CV_8U);

    for (int y = 0; y < img.rows; ++y) {
        for (int x = 0; x < img.cols; ++x) {
            if (img.at<uchar>(cv::Point(x, y)) == 255) {
                for (int theta = 0; theta < maxTheta; ++theta) {
                    int rho = cvRound(x * std::cos(theta) + y * std::sin(theta));
                    int rhoIndex = rho + diagonalLenght;
                    votes.at<uchar>(rhoIndex, theta)++;
                }
            }
        }
    }

    // Step 4: Draw detected lines on the input image.
    int lineOffset = diagonalLenght * 2;
    cv::Mat lineImg = input.clone();
    for (int rhoIndex = 0; rhoIndex < votes.rows; ++rhoIndex) {
        for (int theta = 0; theta < votes.cols; ++theta) {
            if (votes.at<uchar>(rhoIndex, theta) > houghTH) {
                int rho = rhoIndex - diagonalLenght;

                // Finding two points to draw the line
                int x0 = cvRound(rho * std::cos(theta));
                int y0 = cvRound(rho * std::sin(theta));

                cv::Point point1;
                point1.x = cvRound(x0 + lineOffset * (-std::sin(theta)));
                point1.y = cvRound(y0 + lineOffset * std::cos(theta));

                cv::Point point2;
                point2.x = cvRound(x0 - lineOffset * (-std::sin(theta)));
                point2.y = cvRound(y0 - lineOffset * std::cos(theta));

                cv::line(lineImg, point1, point2, cv::Scalar(0), 2, 0);
            }
        }
    }

    return lineImg;
}

int main(int argc, char ** argv) {
    cv::Mat inputImg = imreadWrapper(argc, argv, cv::IMREAD_GRAYSCALE);
    imshowWrapper("inputImg" ,inputImg);

    int houghTH = 150;
    int cannyTHL  = 40;
    int cannyTHH  = 80;
    int blurSize  = 1;
    float blurSigma  = 0.0;

    cv::Mat linesImg = hough_lines(inputImg, houghTH, cannyTHL, cannyTHH, blurSize, blurSigma);
    imshowWrapper("Hough Lines", linesImg);

    return 0;
}

