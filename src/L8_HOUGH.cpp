#include <opencv2/opencv.hpp>
#include "./reusables/utils.h"
#include <iostream>

cv::Mat hough_lines(cv::Mat & input, int houghTH, int cannyTHL, int cannyTHH, int blurSize = 3, float blurSigma = 0.5) {
    cv::Mat img = input.clone();

    // Blurring and Canny
    cv::GaussianBlur(img, img, cv::Size(blurSize, blurSize), blurSigma, blurSigma);
    cv::Canny(img, img, cannyTHL, cannyTHH);

    // Accumulator
    int diagLen = std::ceil(std::hypot(img.rows, img.cols));
    int maxTheta = 180;
    cv::Mat votes = cv::Mat::zeros(diagLen * 2, maxTheta, CV_8U);

    // Calculating votes
    for (int x = 0; x < img.rows; ++x) {
        for (int y = 0; y < img.cols; ++y) {
            if (img.at<uchar>(x, y) == 255) {
                for (int theta = 0; theta < maxTheta; ++theta) {
                    int rho = (int) std::ceil((y * std::cos(theta) + x * std::sin(theta))) + diagLen;
                    votes.at<uchar>(rho, theta)++;
                }
            }
        }
    }

    // Drawing lines
    int alpha = 1000;
    int x0, y0;
    cv::Point p1, p2;
    cv::Mat out = input.clone();
    for (int rhoIdx = 0; rhoIdx < votes.rows; ++rhoIdx) {
        for (int theta = 0; theta < votes.cols; ++theta) {
            if (votes.at<uchar>(rhoIdx, theta) >= houghTH) {
                int rho = rhoIdx - diagLen;

                x0 = cvRound(rho * std::cos(theta));
                y0 = cvRound(rho * std::sin(theta));

                p1.x = cvRound(x0 + alpha * (-std::sin(theta)));
                p1.y = cvRound(y0 + alpha * std::cos(theta));

                p2.x = cvRound(x0 - alpha * (-std::sin(theta)));
                p2.y = cvRound(y0 - alpha * std::cos(theta));

                cv::line(out, p1, p2, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
            }
        }
    }

    return out;
}

cv::Mat hough_circles(cv::Mat & input, int houghTH, int radMin, int radMax, int cannyTHL, int cannyTHH, int blurSize = 3, float blurSigma = 0.5) {
    cv::Mat img = input.clone();

    // Blurring and Canny
    cv::GaussianBlur(img, img, cv::Size(blurSize, blurSize), blurSigma, blurSigma);
    cv::Canny(img, img, cannyTHL, cannyTHH);

    // Accumulator
    int radiusOffset = radMax - radMin + 1;
    int sizes[] = {img.rows, img.cols, radiusOffset};
    cv::Mat votes = cv::Mat(3, sizes, CV_8U, cv::Scalar(0));

    // Calculating votes
    for (int x = 0; x < img.rows; ++x) {
        for (int y = 0; y < img.cols; ++y) {
            if (img.at<uchar>(x, y) == 255) {
                for (int radius = radMin; radius < radMax; ++radius) {
                    for (int thetaDeg = 0; thetaDeg < 360; ++thetaDeg) {
                        double thetaRad = thetaDeg * CV_PI / 180;

                        int alpha = x - radius * std::cos(thetaRad);
                        int beta = y - radius * std::sin(thetaRad);

                        if (alpha >= 0 and alpha < img.rows and beta >= 0 and beta < img.cols) {
                            votes.at<uchar>(alpha, beta, radius - radMin)++;
                        }
                    }
                }
            }
        }
    }

    // Drawing circles
    cv::Mat out = input.clone();
    for (int radius = radMin; radius < radMax; ++radius) {
        for (int alpha = 0; alpha < img.rows; ++alpha) {
            for (int beta = 0; beta < img.cols; ++beta) {
                if (votes.at<uchar>(alpha, beta, radius - radMin) > houghTH) {
                    cv::circle(out, cv::Point(beta, alpha), 2, cv::Scalar(0), 2, 8, 0);
                    cv::circle(out, cv::Point(beta, alpha), radius, cv::Scalar(0), 2, 8, 0);
                }
            }
        }
    }

    return out;
}

int main(int argc, char ** argv) {
    // Test Hough Lines
    cv::Mat inputImg = imreadWrapper(argc, argv, cv::IMREAD_GRAYSCALE);
    imshowWrapper("inputImg" ,inputImg);

    int houghTH_l = 150;
    int cannyLTH_l  = 40;
    int cannyHTH_l  = 80;
    int blurSize_l  = 1;
    float blurSigma_l  = 0.0;


    cv::Mat linesImg = hough_lines(inputImg, houghTH_l , cannyLTH_l , cannyHTH_l , blurSize_l , blurSigma_l);
    imshowWrapper("Hough Lines", linesImg);

    // Test Hough Circles
    argv[1] = argv[2];
    cv::Mat inputImg2 = imreadWrapper(argc, argv, cv::IMREAD_GRAYSCALE);
    imshowWrapper("inputImg2", inputImg2);

    int houghTH_c = 190;
    int radMin_c = 20;
    int radMax_c = 70;
    int cannyLTH_c  = 40;
    int cannyHTH_c  = 80;
    int blurSize_c  = 1;
    float blurSigma_c  = 0.0;

    cv::Mat circlesImg = hough_circles(inputImg2, houghTH_c, radMin_c, radMax_c, cannyLTH_c, cannyHTH_c, blurSize_c, blurSigma_c);
    imshowWrapper("Hough Circles", circlesImg);

    return 0;
}