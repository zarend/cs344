#ifndef HW2_H
#define HW2_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

extern cv::Mat imageInputRGBA;
extern cv::Mat imageOutputRGBA;

extern uchar4 *d_inputImageRGBA__;
extern uchar4 *d_outputImageRGBA__;

extern float *h_filter__;

size_t numRows();
size_t numCols();
void preProcess(uchar4 **h_inputImageRGBA, uchar4 **h_outputImageRGBA,
                uchar4 **d_inputImageRGBA, uchar4 **d_outputImageRGBA,
                unsigned char **d_redBlurred,
                unsigned char **d_greenBlurred,
                unsigned char **d_blueBlurred,
                float **h_filter, int *filterWidth,
                const std::string &filename);
void postProcess(const std::string& output_file, uchar4* data_ptr);
void cleanUp(void);
void generateReferenceImage(std::string input_file, std::string reference_file, int kernel_size);

#endif
