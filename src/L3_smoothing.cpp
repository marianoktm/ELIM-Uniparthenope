#include <opencv2/opencv.hpp>
#include "./reusables/utils.h"

/*
void imshowWrapper(std::string const& winname, cv::Mat& mat) {
    cv::imshow(winname, mat);
    cv::waitKey(0);
    cv::destroyWindow(winname);
}
 */

int main(int argc, char** argv) {
    // Input img 1 reading
    if (argc == 1)
        return -1;
    cv::Mat inputImg_1 = cv::imread(argv[1], -1);
    if (inputImg_1.empty())
        return -2;

    // Grayscaling inputImg_1 for some functions
    cv::Mat grayscaleInputImg_1;
    cv::cvtColor(inputImg_1, grayscaleInputImg_1, cv::COLOR_BGR2GRAY);

    /* BLURRING */

    // Kernel allocation
    int maskSize = 25;
    // Si esegue una divisione per normalizzare il filtro. Il filtro di media richiede la normalizzazione in modo che la somma di tutti gli elementi nel kernel sia 1.
    cv::Mat meanFilterKernel = cv::Mat::ones(maskSize, maskSize, CV_32F) / (float)(maskSize*maskSize);

    // Filter2D with correlation
    cv::Mat filteredImg_correlation;
    cv::filter2D(inputImg_1, filteredImg_correlation, inputImg_1.type(), meanFilterKernel);

    imshowWrapper("filteredImg_correlation", filteredImg_correlation);

    // Filter2D with convolution
    cv::Mat meanFilterKernel_rotated;
    cv::rotate(meanFilterKernel, meanFilterKernel_rotated, cv::ROTATE_180);

    cv::Mat filteredImg_convolution;
    cv::filter2D(inputImg_1, filteredImg_convolution, inputImg_1.type(), meanFilterKernel_rotated);

    imshowWrapper("filteredImg_convolution", filteredImg_convolution);

    // Kernel size definition for other functions
    cv::Size maskCVSize(maskSize, maskSize);

    // cv::blur filter
    cv::Mat blurredBlur;
    blur(inputImg_1, blurredBlur, maskCVSize);

    imshowWrapper("blurredBlur", blurredBlur);

    // cv::boxFilter filter
    cv::Mat blurredBoxFilter;
    cv::boxFilter(inputImg_1, blurredBoxFilter, inputImg_1.type(), maskCVSize);

    imshowWrapper("blurredBoxFilter", blurredBoxFilter);

    // cv::medianBlur filter
    // NOTE: medianBlur needs a grayscale image!
    cv::Mat blurredMedianBlur;
    cv::medianBlur(grayscaleInputImg_1, blurredMedianBlur, maskSize);

    imshowWrapper("blurredMedianBlur", blurredMedianBlur);

    // cv::GaussianBlur
    cv::Mat blurredGaussianBlur;
    cv::GaussianBlur(inputImg_1, blurredGaussianBlur, maskCVSize, 0, 0);

    imshowWrapper("blurredGaussianBlur", blurredGaussianBlur);

    /* THRESHOLDING */
    // Input img 2 reading
    if (argc < 3)
        return -3;
    cv::Mat inputImg_2 = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);
    if (inputImg_2.empty())
        return -4;

    // Threshold value
    int threshold = 170;

    /*
    // maxval 1 Thresholding
    cv::Mat thresholded1;
    cv::threshold(inputImg_2, thresholded1, threshold, 1, cv::THRESH_BINARY);

    imshowWrapper("thresholded1", thresholded1);
    */

    // maxval 255 Thresholding
    cv::Mat thresholded255;
    cv::threshold(inputImg_2, thresholded255, threshold, 255, cv::THRESH_OTSU);

    cv::imshow("og img", inputImg_2);
    cv::imshow("th img", thresholded255);
    cv::waitKey(0);

    return 0;
}