#include <opencv2/opencv.hpp>
#include <vector>
#include "./reusables/utils.h"

using namespace std;
using namespace cv;

cv::Mat otsu(cv::Mat & input, int blurSize = 3, float blurSigma = 0.5) {
    cv::Mat img = input.clone();

    std::vector<double> hist(256, 0.0);
    for (int x = 0; x < img.rows; ++x) {
        for (int y = 0; y < img.cols; ++y) {
            hist.at(img.at<uchar>(x,y))++;
        }
    }
    int pxNumber = img.rows * img.cols;
    for (double & ni : hist) {
        ni /= pxNumber;
    }

    double globCumMean = 0.0;
    for (int i = 0; i < hist.size(); ++i) {
        globCumMean += i * hist.at(i);
    }

    double prob = 0.0;
    double cumMean = 0.0;
    double bwcVar = 0.0;
    double maxVar = 0.0;
    int optimalTH = 0;
    for (int i = 0; i < hist.size(); ++i) {
        prob += hist.at(i);
        cumMean += i * hist.at(i);

        bwcVar = std::pow(globCumMean * prob - cumMean, 2) / (prob * (1 - prob));
        if (bwcVar > maxVar) {
            maxVar = bwcVar;
            optimalTH = i;
        }
    }

    cv::GaussianBlur(img, img, cv::Size(blurSize, blurSize), blurSigma, blurSigma);

    cv::Mat out;
    cv::threshold(img, out, optimalTH, 255, cv::THRESH_BINARY);

    return out;
}


Mat otsu2k (Mat & input_img, int bsize = 3, double bsigma = 2.0) {
    Mat img = input_img.clone();

    // Hist
    vector<double> hist(256, 0.0);
    for (size_t y = 0; y < img.rows; y++)
        for (size_t x = 0; x < img.cols; x++)
            hist.at(input_img.at<uchar>(Point(x, y)))++;

    // normalization
    int pixelNo = img.rows * img.cols;
    for (size_t i = 0; i < hist.size(); i++)
        hist.at(i) /= pixelNo;

    // globCumMean
    double globCumMean = 0.0;
    for (size_t i = 0; i < hist.size(); i++)
        globCumMean += i * hist.at(i);

    // Otsu 2k method
    vector<double> prob(3, 0.0);
    vector<double> cumMean(3, 0.0);
    double maxVar = 0.0;
    vector<int> optTh(2, 0);
    for (size_t k1 = 0; k1 < hist.size(); k1++) {
        prob.at(0) += hist.at(k1);
        cumMean.at(0) += k1 * hist.at(k1);

        for (size_t k2 = k1+1; k2 < hist.size()-1; k2++) {
            prob.at(1) += hist.at(k2);
            cumMean.at(1) += k2 * hist.at(k2);

            for (size_t k = k2+1; k < hist.size()-2; k++) {
                prob.at(2) += hist.at(k);
                cumMean.at(2) += k * hist.at(k);

                double bwcVar = 0.0;
                for (size_t i = 0; i < 3; i++) {
                    bwcVar += prob.at(i) * pow((cumMean.at(i)/prob.at(i) - globCumMean), 2);
                }

                if (bwcVar > maxVar) {
                    maxVar = bwcVar;
                    optTh.at(0) = k1;
                    optTh.at(1) = k2;
                }
            }
            prob.at(2) = 0.0;
            cumMean.at(2) = 0.0;
        }
        prob.at(1) = 0.0;
        cumMean.at(1) = 0.0;
    }

    // smoothing
    GaussianBlur(img, img, Size(bsize, bsize), bsigma, bsigma);

    // double threshold
    Mat thresh = Mat::zeros(img.size(), CV_8U);

    for (size_t y = 0; y < img.rows; y++) {
        for (size_t x = 0; x < img.cols; x++) {
            uchar pxCur = img.at<uchar>(Point(x,y));
            if (pxCur > optTh.at(1)) {
                thresh.at<uchar>(Point(x,y)) = 255;
            }
            else if (pxCur > optTh.at(0)) {
                thresh.at<uchar>(Point(x,y)) = (255+1)/2;
            }
        }
    }

    return thresh;
}

int main(int argc, char ** argv) {
    cv::Mat inputImg = imreadWrapper(argc, argv, cv::IMREAD_GRAYSCALE);
    imshowWrapper("inputImg", inputImg);

    cv::Mat otsuImg = otsu(inputImg);
    imshowWrapper("otsuImg", otsuImg);

    cv::Mat otsu2kImg = otsu2k(inputImg);
    imshowWrapper("otsu2kImg", otsu2kImg);

    return 0;
}
