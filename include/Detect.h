#ifndef YOLO_DETECT_H
#define YOLO_DETECT_H

#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <algorithm>
#include <iostream>
#include <utility>
#include <time.h>
#define GREEN   "\033[32m"
#define YELLOW  "\033[33m"
#include "utils.h"
using namespace std;

class YOLO
{
public:
    YOLO();
    ~YOLO();
    void GetImage(cv::Mat& RGB);
    void ClearImage();
    bool Detect();
    void ClearArea();
    cv::Mat convertTo3Channels(const cv::Mat& binImg);
    vector<cv::Rect2i> mvPersonArea = {};
    vector<torch::Tensor> non_max_suppression(torch::Tensor preds, float score_thresh=0.5, float iou_thresh=0.5);

public:
    cv::Mat mRGB;
    torch::jit::script::Module mModule;
    std::vector<std::string> mClassnames;
    Detection res;
    std::vector<Detection> dect;
    vector<string> mvDynamicNames;
    vector<cv::Rect2i> mvDynamicArea;


};


#endif //YOLO_DETECT_H