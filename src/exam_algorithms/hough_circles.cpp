#include <opencv2/opencv.hpp>
#include "../reusables/utils.h"

/**
 * Applies the Hough Circles Detection algorithm to an input image.
 *
 * This function performs the following steps:
 * 1. Applies Gaussian blur to reduce noise for Canny edge detector.
 * 2. Performs Canny edge detection to highlight edges.
 * 3. Computes Hough Transform to detect circles.
 * 4. Draws detected circles on the input image.
 *
 * @param input     Input image (grayscale).
 * @param houghTH   Threshold for circle detection in the Hough space.
 * @param radiusMin    Minimum radius of circles to detect.
 * @param radiusMax    Maximum radius of circles to detect.
 * @param cannyTHL  Lower threshold for Canny edge detection.
 * @param cannyTHH  Upper threshold for Canny edge detection.
 * @param blurSize  Size of the Gaussian filter kernel for smoothing (default is 3).
 * @param blurSigma Standard deviation for Gaussian blur (default is 0.5).
 *
 * @return An image with detected circles drawn on it.
 */
cv::Mat hough_circles(cv::Mat & input, int houghTH, int radiusMin, int radiusMax, int cannyTHL, int cannyTHH, int blurSize = 3, float blurSigma = 0.5) {
    cv::Mat img = input.clone();

    // Step 1: Apply Gaussian blur to reduce noise.
    cv::GaussianBlur(img, img, cv::Size(blurSize, blurSize), blurSigma);

    // Step 2: Perform Canny edge detection to highlight edges.
    cv::Canny(img, img, cannyTHL, cannyTHH);

    // Step 3: Compute Hough Transform to detect circles.
    int radiusOffset = radiusMax - radiusMin + 1;
    int sizes[] = {img.cols, img.rows, radiusOffset};
    cv::Mat votes = cv::Mat(3, sizes, CV_8U, cv::Scalar(0));

    for (int y = 0; y < img.rows; ++y) {
        for (int x = 0; x < img.cols; ++x) {
            if (img.at<uchar>(cv::Point(x, y)) == 255) {
                for (int radius = radiusMin; radius < radiusMax; ++radius) {
                    for (int thetaDegrees = 0; thetaDegrees < 360; ++thetaDegrees) {
                        double thetaRadiants = thetaDegrees * CV_PI / 180;

                        // Calculate the center coordinates (alpha, beta) of the potential circle.
                        int alpha = cvRound(x - radius * std::cos(thetaRadiants));
                        int beta = cvRound(y - radius * std::sin(thetaRadiants));

                        // Ensure the center coordinates are within the image bounds.
                        if (alpha >= 0 and alpha < img.cols and beta >= 0 and beta < img.rows)
                            votes.at<uchar>(alpha, beta, radius - radiusMin)++;
                    }
                }
            }
        }
    }

    // Step 4: Draw detected circles on the input image.
    cv::Mat out = input.clone();
    for (int radius = radiusMin; radius < radiusMax; ++radius) {
        for (int alpha = 0; alpha < img.cols; ++alpha) {
            for (int beta = 0; beta < img.rows; ++beta) {
                if (votes.at<uchar>(alpha, beta, radius - radiusMin) > houghTH) {
                    // Draw a circle with the detected radius and center.
                    cv::circle(out, cv::Point(alpha, beta), radius, cv::Scalar(0), 2, 8);
                }
            }
        }
    }
    return out;
}

int main(int argc, char ** argv) {
    cv::Mat inputImg = imreadWrapper(argc, argv, cv::IMREAD_GRAYSCALE);
    imshowWrapper("Input Img", inputImg);

    int houghTH = 190;
    int radMin = 20;
    int radMax = 70;
    int cannyTHL = 40;
    int cannyTHH = 80;
    int blurSize = 1;
    float blurSigma  = 0.0;

    cv::Mat circlesImg = hough_circles(inputImg, houghTH, radMin, radMax, cannyTHL, cannyTHH, blurSize, blurSigma);
    imshowWrapper("Hough Circles", circlesImg);

    return 0;
}