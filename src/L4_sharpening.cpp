#include <opencv2/opencv.hpp>
#include "./reusables/utils.h"
#include <iostream>

int main(int argc, char** argv) {
    if (argc == 1)
        return -1;

    cv::Mat inputImg = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    if (inputImg.empty())
        return -2;

    imshowWrapper("Original Image", inputImg);
    auto ogImgType = inputImg.type();
    std::cout << "og img type: " << ogImgType << std::endl;

    // Laplacian kernel
    cv::Mat laplacianKernel4 = (cv::Mat_<int>(3, 3) << 0, 1, 0, 1, -4, 1, 0, 1, 0);
    std::cout << "kernel 4:" << std::endl << laplacianKernel4 << std::endl;

    cv::Mat laplacianKernel8 = (cv::Mat_<int>(3,3) << 1, 1, 1, 1, -8, 1, 1, 1, 1);
    std::cout << "kernel 8:" << std::endl << laplacianKernel8 << std::endl;

    // Smoothing for better results
    cv::Mat smoothedImg;
    cv::medianBlur(inputImg, smoothedImg, 1);

    imshowWrapper("Smoothed Img", smoothedImg);

    // Thresholding for better results
    /*
    cv::Mat thresholdedImg;
    cv::threshold(smoothedImg, thresholdedImg, 0, 255, cv::THRESH_OTSU);

    imshowWrapper("Thresholded Img", thresholdedImg);
    */

    // Applying the filter
    cv::Mat filteredImg;
    cv::filter2D(smoothedImg, filteredImg, ogImgType, laplacianKernel4);

    imshowWrapper("Laplacian Img", filteredImg);

    // opencv Laplacian filter
    cv::Mat opencvLaplacianImg;
    cv::Laplacian(smoothedImg, opencvLaplacianImg, ogImgType);

    imshowWrapper("Laplacian Img (opencv)", opencvLaplacianImg);

    // Normalizing
    cv::Mat normalizedImg;
    cv::normalize(filteredImg, normalizedImg, 0, 255, cv::NORM_MINMAX, ogImgType);

    imshowWrapper("Laplacian Img normalized", normalizedImg);

    // Sharpening with laplacian subtraction
    cv::Mat subtractedImg;

    inputImg.convertTo(inputImg, ogImgType);
    filteredImg.convertTo(filteredImg, ogImgType);
    cv::subtract(inputImg, filteredImg, subtractedImg);

    imshowWrapper("Original Image", inputImg);
    imshowWrapper("Sharpened Img", subtractedImg);

    // Unsharp Masking
    cv::Mat blurredImg;
    cv::medianBlur(inputImg, blurredImg, 25);

    imshowWrapper("Blurred Img", inputImg);

    cv::Mat unsharpMask = inputImg - blurredImg;

    imshowWrapper("Mask", unsharpMask);

    cv::Mat unshMaskSharpenedImg = inputImg + unsharpMask;

    imshowWrapper("Original Image", inputImg);
    imshowWrapper("Sharpened with unsharp masking", unshMaskSharpenedImg);

    // Custom Sobel
    cv::Mat sobelGX = (cv::Mat_<float>(3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);
    cv::Mat sobelGY = (cv::Mat_<float>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);

    cv::Mat gradX;
    cv::filter2D(inputImg, gradX, CV_32F, sobelGX);
    cv::Mat gradY;
    cv::filter2D(inputImg, gradY, CV_32F, sobelGY);

    cv::Mat sobelMagnitude;
    cv::magnitude(gradX, gradY, sobelMagnitude);

    cv::normalize(sobelMagnitude, sobelMagnitude, 0, 255, cv::NORM_MINMAX, CV_8U);
    imshowWrapper("Sobel Magnitude", sobelMagnitude);

    cv::Mat sobelSharpenedImg = inputImg + sobelMagnitude;

    imshowWrapper("Original Image", inputImg);
    imshowWrapper("Sobel Sharpened Image", sobelSharpenedImg);

    return 0;
}

