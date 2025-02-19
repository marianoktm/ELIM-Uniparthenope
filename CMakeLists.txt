cmake_minimum_required(VERSION 3.19)
project(opencvelim)

find_package(OpenCV)

include_directories(${OpenCV_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 14)

add_executable(canny src/exam_algorithms/canny.cpp src/reusables/utils.h)
target_link_libraries(canny  ${OpenCV_LIBS})

add_executable(harris src/exam_algorithms/harris.cpp src/reusables/utils.h)
target_link_libraries(harris  ${OpenCV_LIBS})

add_executable(hough_lines src/exam_algorithms/hough_lines.cpp src/reusables/utils.h)
target_link_libraries(hough_lines  ${OpenCV_LIBS})

add_executable(hough_circles src/exam_algorithms/hough_circles.cpp src/reusables/utils.h)
target_link_libraries(hough_circles  ${OpenCV_LIBS})

add_executable(otsu src/exam_algorithms/otsu.cpp src/reusables/utils.h)
target_link_libraries(otsu  ${OpenCV_LIBS})

add_executable(otsu2k src/exam_algorithms/otsu2k.cpp src/reusables/utils.h)
target_link_libraries(otsu2k  ${OpenCV_LIBS})

add_executable(region_growing src/exam_algorithms/region_growing.cpp src/reusables/utils.h)
target_link_libraries(region_growing  ${OpenCV_LIBS})

add_executable(split_and_merge src/exam_algorithms/split_and_merge.cpp src/reusables/utils.h)
target_link_libraries(split_and_merge  ${OpenCV_LIBS})

add_executable(kmeans_gray src/exam_algorithms/kmeans_gray.cpp src/reusables/utils.h)
target_link_libraries(kmeans_gray  ${OpenCV_LIBS})

#=======================================================================

add_executable(L1_imgwindow src/L1_imgwindow.cpp src/reusables/utils.h)
target_link_libraries(L1_imgwindow ${OpenCV_LIBS})

add_executable(L2_paddingand3x3 src/L2_padding.cpp src/reusables/utils.h)
target_link_libraries(L2_paddingand3x3  ${OpenCV_LIBS})

add_executable(L3_filtering src/L3_smoothing.cpp src/reusables/utils.h)
target_link_libraries(L3_filtering  ${OpenCV_LIBS})

add_executable(L4_sharpening src/L4_sharpening.cpp src/reusables/utils.h)
target_link_libraries(L4_sharpening  ${OpenCV_LIBS})

add_executable(L5_color src/L5_color.cpp src/reusables/utils.h)
target_link_libraries(L5_color  ${OpenCV_LIBS})

add_executable(L6_segmentation src/L6_segmentation.cpp src/reusables/utils.h)
target_link_libraries(L6_segmentation  ${OpenCV_LIBS})

add_executable(L7_CANNY src/L7_CANNY.cpp src/reusables/utils.h)
target_link_libraries(L7_CANNY  ${OpenCV_LIBS})

add_executable(L7_HARRIS src/L7_HARRIS.cpp src/reusables/utils.h)
target_link_libraries(L7_HARRIS  ${OpenCV_LIBS})

add_executable(L8_HOUGH src/L8_HOUGH.cpp src/reusables/utils.h)
target_link_libraries(L8_HOUGH  ${OpenCV_LIBS})

add_executable(L9_thresholding src/L9_OTSU.cpp src/reusables/utils.h)
target_link_libraries(L9_thresholding  ${OpenCV_LIBS})

add_executable(L10_REGIONGROWING src/L10_REGIONGROWING.cpp src/reusables/utils.h)
target_link_libraries(L10_REGIONGROWING  ${OpenCV_LIBS})