#include <YOLOV5.h>
using namespace cv;

YOLOV5::YOLOV5():device_(torch::kCUDA)
{
    try {

        //module_ = torch::jit::load("/home/xin/turtlebot3_ws/src/ORB_SLAM3_PRO/weights/best_home.torchscript_gpu.pt");
        module_ = torch::jit::load("/home/xin/project_ws/src/ORB_SLAM3_YOLOv5_gpu/weights/yolov5s_gpu.pt");
    }
    catch (const c10::Error& e) {
        std::cerr << "Error loading the model!\n";
        std::exit(EXIT_FAILURE);
    }

    half_ = (device_ != torch::kCPU);
    module_.to(device_);

    if (half_) {
        module_.to(torch::kHalf);
    }
    module_.eval();

    std::ifstream f("/home/xin/project_ws/src/ORB_SLAM3_YOLOv5_gpu/weights/coco.names");
    std::string name = "";
    while (std::getline(f, name))
    {
        mClassnames.push_back(name);
    }

    mvDynamicNames = {"person", "bicycle", "motorbike", "bus", "car"};
    mvStaicNames = {"car","aeroplane","traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","sofa","pottedplant","bed","diningtable","toilet","tvmonitor","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"};

}


YOLOV5::~YOLOV5()
{

}

bool YOLOV5::Detect()
{
    cv::Mat img;
    if(mRGB.empty())
    {
        std::cout << "Read RGB failed!" << std::endl;
        return false;
    }
    // Preparing input tensor
    cv::resize(mRGB, img, cv::Size(image_width, image_height));
    int N = img.channels();
    if(N==1)
    {
        img = convertTo3Channels(img);
    }
    auto result = YOLOV5::Run(img, 0.4, 0.5);
    if (result.size() > 0)
    {
        for(auto res = result.begin(); res != result.end(); res++)
        {
            mmDetectMap = *res;
            for(auto det = res->begin(); det != res->end(); det ++)
            {
                cv::Rect2i DetectArea = det->bbox;
                float score = det->score;
                int classID = det->class_idx;
                /// 判断80多个类别中等于自己定义的动态区间内的类别
                if (count(mvDynamicNames.begin(), mvDynamicNames.end(), mClassnames[classID]))
                {
                    cv::Rect2i DynamicArea = det->bbox;
                    mvDynamicArea.push_back(DynamicArea);
                }
//                cout <<GREEN<< "#--------------------------------------------#" << endl;
//                cout <<YELLOW<< "# 成功检测到目标 ：        "<< mClassnames[classID]<< endl;
//                cout <<GREEN<< "#--------------------------------------------#" << endl;
//                cout  << endl;

            }

        }
        if (mvDynamicArea.size() == 0)
        {
            cv::Rect2i tDynamicArea(1, 1, 1, 1);
            mvDynamicArea.push_back(tDynamicArea);
        }
    }
    return true;
}

void YOLOV5::GetImage(cv::Mat &RGB)
{
    mRGB = RGB;
}
std::vector<std::vector<Detection>>
YOLOV5::Run(const cv::Mat& img, float conf_threshold, float iou_threshold) {
    torch::NoGradGuard no_grad;
    // keep the original image for visualization purpose
    cv::Mat img_input = img.clone();

    std::vector<float> pad_info = LetterboxImage(img_input, img_input, cv::Size(640, 640));
    const float pad_w = pad_info[0];
    const float pad_h = pad_info[1];
    const float scale = pad_info[2];
    cv::cvtColor(img_input, img_input, cv::COLOR_BGR2RGB);  // BGR -> RGB
    img_input.convertTo(img_input, CV_32FC3, 1.0f / 255.0f);  // normalization 1/255
    auto tensor_img = torch::from_blob(img_input.data, {1, img_input.rows, img_input.cols, img_input.channels()}).to(device_);

    tensor_img = tensor_img.permute({0, 3, 1, 2}).contiguous();  // BHWC -> BCHW (Batch, Channel, Height, Width)

    if (half_) {
        tensor_img = tensor_img.to(torch::kHalf);
    }
    std::vector<torch::jit::IValue> inputs;
    inputs.emplace_back(tensor_img);
    // inference
    torch::jit::IValue output = module_.forward(inputs);

    ///end = std::chrono::high_resolution_clock::now();
    ///duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    // It should be known that it takes longer time at first time
    ///std::cout << "inference takes : " << duration.count() << " ms" << std::endl;

    /*** Post-process ***/

    ///start = std::chrono::high_resolution_clock::now();
    auto detections = output.toTuple()->elements()[0].toTensor();

    // result: n * 7
    // batch index(0), top-left x/y (1,2), bottom-right x/y (3,4), score(5), class id(6)
    auto result = PostProcessing(detections, pad_w, pad_h, scale, img.size(), conf_threshold, iou_threshold);
    ///end = std::chrono::high_resolution_clock::now();
    ///duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    // It should be known that it takes longer time at first time
    ///std::cout << "post-process takes : " << duration.count() << " ms" << std::endl;
    return result;
}


std::vector<float> YOLOV5::LetterboxImage(const cv::Mat& src, cv::Mat& dst, const cv::Size& out_size) {
    auto in_h = static_cast<float>(src.rows);
    auto in_w = static_cast<float>(src.cols);
    float out_h = out_size.height;
    float out_w = out_size.width;

    float scale = std::min(out_w / in_w, out_h / in_h);

    int mid_h = static_cast<int>(in_h * scale);
    int mid_w = static_cast<int>(in_w * scale);

    cv::resize(src, dst, cv::Size(mid_w, mid_h));

    int top = (static_cast<int>(out_h) - mid_h) / 2;
    int down = (static_cast<int>(out_h)- mid_h + 1) / 2;
    int left = (static_cast<int>(out_w)- mid_w) / 2;
    int right = (static_cast<int>(out_w)- mid_w + 1) / 2;

    cv::copyMakeBorder(dst, dst, top, down, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    std::vector<float> pad_info{static_cast<float>(left), static_cast<float>(top), scale};
    return pad_info;
}


std::vector<std::vector<Detection>> YOLOV5::PostProcessing(const torch::Tensor& detections,
                                                                  float pad_w, float pad_h, float scale, const cv::Size& img_shape,
                                                                  float conf_thres, float iou_thres) {
    constexpr int item_attr_size = 5;
    int batch_size = detections.size(0);
    // number of classes, e.g. 80 for coco dataset
    auto num_classes = detections.size(2) - item_attr_size;

    // get candidates which object confidence > threshold
    auto conf_mask = detections.select(2, 4).ge(conf_thres).unsqueeze(2);

    std::vector<std::vector<Detection>> output;
    output.reserve(batch_size);

    // iterating all images in the batch
    for (int batch_i = 0; batch_i < batch_size; batch_i++) {
        // apply constrains to get filtered detections for current image
        auto det = torch::masked_select(detections[batch_i], conf_mask[batch_i]).view({-1, num_classes + item_attr_size});

        // if none detections remain then skip and start to process next image
        if (0 == det.size(0)) {
            continue;
        }

        // compute overall score = obj_conf * cls_conf, similar to x[:, 5:] *= x[:, 4:5]
        det.slice(1, item_attr_size, item_attr_size + num_classes) *= det.select(1, 4).unsqueeze(1);

        // box (center x, center y, width, height) to (x1, y1, x2, y2)
        torch::Tensor box = xywh2xyxy(det.slice(1, 0, 4));

        // [best class only] get the max classes score at each result (e.g. elements 5-84)
        std::tuple<torch::Tensor, torch::Tensor> max_classes = torch::max(det.slice(1, item_attr_size, item_attr_size + num_classes), 1);

        // class score
        auto max_conf_score = std::get<0>(max_classes);
        // index
        auto max_conf_index = std::get<1>(max_classes);

        max_conf_score = max_conf_score.to(torch::kFloat).unsqueeze(1);
        max_conf_index = max_conf_index.to(torch::kFloat).unsqueeze(1);

        // shape: n * 6, top-left x/y (0,1), bottom-right x/y (2,3), score(4), class index(5)
        det = torch::cat({box.slice(1, 0, 4), max_conf_score, max_conf_index}, 1);

        // for batched NMS
        constexpr int max_wh = 4096;
        auto c = det.slice(1, item_attr_size, item_attr_size + 1) * max_wh;
        auto offset_box = det.slice(1, 0, 4) + c;

        std::vector<cv::Rect> offset_box_vec;
        std::vector<float> score_vec;

        // copy data back to cpu
        auto offset_boxes_cpu = offset_box.cpu();
        auto det_cpu = det.cpu();
        const auto& det_cpu_array = det_cpu.accessor<float, 2>();

        // use accessor to access tensor elements efficiently
        Tensor2Detection(offset_boxes_cpu.accessor<float,2>(), det_cpu_array, offset_box_vec, score_vec);

        // run NMS
        std::vector<int> nms_indices;
        cv::dnn::NMSBoxes(offset_box_vec, score_vec, conf_thres, iou_thres, nms_indices);

        /// vector<Detection> det_vec
        std::vector<Detection> det_vec;
        for (int index : nms_indices) {
            Detection t;
            const auto& b = det_cpu_array[index];
            t.bbox =
                    cv::Rect(cv::Point(b[Det::tl_x], b[Det::tl_y]),
                             cv::Point(b[Det::br_x], b[Det::br_y]));
            t.score = det_cpu_array[index][Det::score];
            t.class_idx = det_cpu_array[index][Det::class_idx];
            t.color_r =  color_box(t.class_idx).z();
            t.color_g =  color_box(t.class_idx).y();
            t.color_b =  color_box(t.class_idx).x();
            det_vec.emplace_back(t);
        }

        ScaleCoordinates(det_vec, pad_w, pad_h, scale, img_shape);

        // save final detection for the current image
        output.emplace_back(det_vec);
    } // end of batch iterating

    return output;
}


void YOLOV5::ScaleCoordinates(std::vector<Detection>& data,float pad_w, float pad_h,
                                     float scale, const cv::Size& img_shape) {
    auto clip = [](float n, float lower, float upper) {
        return std::max(lower, std::min(n, upper));
    };

    std::vector<Detection> detections;
    for (auto & i : data) {
        float x1 = (i.bbox.tl().x - pad_w)/scale;  // x padding
        float y1 = (i.bbox.tl().y - pad_h)/scale;  // y padding
        float x2 = (i.bbox.br().x - pad_w)/scale;  // x padding
        float y2 = (i.bbox.br().y - pad_h)/scale;  // y padding

        x1 = clip(x1, 0, img_shape.width);
        y1 = clip(y1, 0, img_shape.height);
        x2 = clip(x2, 0, img_shape.width);
        y2 = clip(y2, 0, img_shape.height);

        i.bbox = cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2));
    }
}


torch::Tensor YOLOV5::xywh2xyxy(const torch::Tensor& x) {
    auto y = torch::zeros_like(x);
    // convert bounding box format from (center x, center y, width, height) to (x1, y1, x2, y2)
    y.select(1, Det::tl_x) = x.select(1, 0) - x.select(1, 2).div(2);
    y.select(1, Det::tl_y) = x.select(1, 1) - x.select(1, 3).div(2);
    y.select(1, Det::br_x) = x.select(1, 0) + x.select(1, 2).div(2);
    y.select(1, Det::br_y) = x.select(1, 1) + x.select(1, 3).div(2);
    return y;
}


void YOLOV5::Tensor2Detection(const at::TensorAccessor<float, 2>& offset_boxes,
                                     const at::TensorAccessor<float, 2>& det,
                                     std::vector<cv::Rect>& offset_box_vec,
                                     std::vector<float>& score_vec) {

    for (int i = 0; i < offset_boxes.size(0) ; i++) {
        offset_box_vec.emplace_back(
                cv::Rect(cv::Point(offset_boxes[i][Det::tl_x], offset_boxes[i][Det::tl_y]),
                         cv::Point(offset_boxes[i][Det::br_x], offset_boxes[i][Det::br_y]))
        );
        score_vec.emplace_back(det[i][Det::score]);
    }
}


Mat YOLOV5::convertTo3Channels(const Mat& binImg)
{
    Mat three_channel = Mat::zeros(binImg.rows,binImg.cols,CV_8UC3);
    vector<Mat> channels;
    for (int i=0;i<3;i++)
    {
        channels.push_back(binImg);
    }
    merge(channels,three_channel);
    return three_channel;
}

Eigen::Vector3i YOLOV5::color_box(int& class_id)
{
    int x = 0; int y = 0; int z = 0;
    //https://blog.csdn.net/qq_51985653/article/details/113392665?
    if (mClassnames[class_id] == "person"){x = 160; y =32; z = 240;}  //紫色
    if (mClassnames[class_id] == "chair"){x =115; y =74; z = 18;}     /// 棕土棕色
    if (mClassnames[class_id] == "book"){x = 218; y =112; z = 214;}   /// 淡紫色
    if (mClassnames[class_id] == "car"){x = 1; y =255; z = 255;}      //青色
    if (mClassnames[class_id] == "keyboard"){x = 135; y =38; z = 87;}  /// 草莓色
    if (mClassnames[class_id] == "cup"){x = 255; y =97; z = 0;}    ///橙色
    if (mClassnames[class_id] == "laptop"){x = 163; y =148; z = 128;} //金属色
    if (mClassnames[class_id] == "tvmonitor"){x = 163; y =148; z = 128;}  /// 金属色
    if (mClassnames[class_id] == "knife"){x = 51; y =161; z = 201;}
    if (mClassnames[class_id] == "sofa"){x = 227; y =168; z = 105;}
    if (mClassnames[class_id] == "bed"){x = 188; y =143; z = 143;}
    if (mClassnames[class_id] == "bicycle"){x = 25; y =25; z = 112;}
    if (mClassnames[class_id] == "bear"){x = 139; y =69; z = 19;} /// 马赫棕色
    if (mClassnames[class_id] == "motorbike"){x = 218; y =112; z = 214;}  // 淡紫色
    if (mClassnames[class_id] == "bus"){x = 176; y =224; z = 230;}  //浅灰蓝色
    if (mClassnames[class_id] == "mouse"){x = 255; y = 215; z = 1;}  /// 金黄色
    if (mClassnames[class_id] == "clock"){x = 255; y =127; z = 80;}
    if (mClassnames[class_id] == "refrigerator"){x = 189; y =252; z = 201;}
    if (mClassnames[class_id] == "teddy bear"){x = 139; y =69; z = 19;} /// 紫色
    if (mClassnames[class_id] == "handbag"){x = 128; y =128; z = 105;}
    if (mClassnames[class_id] == "backpack"){x = 255; y =192; z = 203;} //很正的粉色
    if (mClassnames[class_id] == "bottle"){x = 0; y =199; z = 140;}  //土耳其玉色
    if (mClassnames[class_id] == "wine glass"){x = 176; y =23; z = 31;}
    if (mClassnames[class_id] == "truck"){x = 163; y =148; z = 128;}
    if (mClassnames[class_id] == "train"){x = 189; y =252; z = 201;}
    if (mClassnames[class_id] == "vase"){x = 127; y =255; z = 212;}
    if (mClassnames[class_id] == "cell phone"){x = 255; y =250; z = 250;}  ///雪白色
    if (mClassnames[class_id] == "pottedplant"){x = 61; y =145; z = 64;} ///盆栽植物

    // 未给特殊颜色的
    if (mClassnames[class_id] == "aeroplane"){x = 160; y =32; z = 240;}  //紫色
    if (mClassnames[class_id] == "boat"){x = 94; y =38; z = 18;}     // 乌贼墨棕色
    if (mClassnames[class_id] == "traffic light"){x = 218; y =112; z = 214;}   //淡紫色
    if (mClassnames[class_id] == "fire hydrant"){x = 0; y =255; z = 255;}      //青色
    if (mClassnames[class_id] == "stop sign"){x = 128; y =42; z = 42;}  // 棕色
    if (mClassnames[class_id] == "parking meter"){x = 3; y =168; z = 158;}
    if (mClassnames[class_id] == "bench"){x = 220; y =220; z = 220;}  //金属色
    if (mClassnames[class_id] == "bird"){x = 128; y =138; z = 125;}  //冷灰
    if (mClassnames[class_id] == "cat"){x = 139; y =69; z = 19;}
    if (mClassnames[class_id] == "dog"){x = 139; y =69; z = 19;}
    if (mClassnames[class_id] == "horse"){x = 188; y =143; z = 143;}
    if (mClassnames[class_id] == "sheep"){x = 25; y =25; z = 112;}
    if (mClassnames[class_id] == "cow"){x = 115; y =74; z = 18;}
    if (mClassnames[class_id] == "elephant"){x = 218; y =112; z = 214;}  // 淡紫色
    if (mClassnames[class_id] == "zebra"){x = 176; y =224; z = 230;}  //浅灰蓝色
    if (mClassnames[class_id] == "mouse"){x = 0; y =255; z = 127;}
    if (mClassnames[class_id] == "giraffe"){x = 255; y =127; z = 80;}
    if (mClassnames[class_id] == "backpack"){x = 189; y =252; z = 201;}
    if (mClassnames[class_id] == "umbrella"){x = 210; y =7180; z = 240;}
    if (mClassnames[class_id] == "tie"){x = 128; y =128; z = 105;}
    if (mClassnames[class_id] == "suitcase"){x = 255; y =192; z = 203;} //很正的粉色
    if (mClassnames[class_id] == "frisbee"){x = 135; y =38; z = 37;}  //草莓色
    if (mClassnames[class_id] == "skis"){x = 176; y =23; z = 31;}
    if (mClassnames[class_id] == "snowboard"){x = 163; y =148; z = 128;}
    if (mClassnames[class_id] == "sports ball"){x = 189; y =252; z = 201;}
    if (mClassnames[class_id] == "kite"){x = 127; y =255; z = 212;}
    if (mClassnames[class_id] == "baseball bat"){x = 160; y =32; z = 240;}  //紫色
    if (mClassnames[class_id] == "baseball glove"){x = 94; y =38; z = 18;}     // 乌贼墨棕色
    if (mClassnames[class_id] == "skateboard"){x = 218; y =112; z = 214;}   //淡紫色
    if (mClassnames[class_id] == "surfboard"){x = 0; y =255; z = 255;}      //青色
    if (mClassnames[class_id] == "tennis racket"){x = 128; y =42; z = 42;}  // 棕色
    if (mClassnames[class_id] == "fork"){x = 3; y =168; z = 158;}
    if (mClassnames[class_id] == "spoon"){x = 220; y =220; z = 220;}  //金属色
    if (mClassnames[class_id] == "bowl"){x = 128; y =138; z = 125;}  //冷灰
    if (mClassnames[class_id] == "banana"){x = 51; y =161; z = 201;}
    if (mClassnames[class_id] == "apple"){x = 227; y =168; z = 105;}
    if (mClassnames[class_id] == "sandwich"){x = 188; y =143; z = 143;}
    if (mClassnames[class_id] == "orange"){x = 25; y =25; z = 112;}
    if (mClassnames[class_id] == "broccoli"){x = 115; y =74; z = 18;}
    if (mClassnames[class_id] == "carrot"){x = 218; y =112; z = 214;}  // 淡紫色
    if (mClassnames[class_id] == "hot dog"){x = 176; y =224; z = 230;}  //浅灰蓝色
    if (mClassnames[class_id] == "pizza"){x = 0; y =255; z = 127;}
    if (mClassnames[class_id] == "donut"){x = 255; y =127; z = 80;}
    if (mClassnames[class_id] == "cake"){x = 189; y =252; z = 201;}
    if (mClassnames[class_id] == "diningtable"){x = 128; y =128; z = 105;}
    if (mClassnames[class_id] == "toilet"){x = 255; y =192; z = 203;} //很正的粉色
    if (mClassnames[class_id] == "remote"){x = 135; y =38; z = 37;}  //草莓色
    if (mClassnames[class_id] == "microwave"){x = 163; y =148; z = 128;}
    if (mClassnames[class_id] == "oven"){x = 189; y =252; z = 201;}
    if (mClassnames[class_id] == "toaster"){x = 127; y =255; z = 212;}
    if (mClassnames[class_id] == "sink"){x = 25; y =25; z = 112;}
    if (mClassnames[class_id] == "scissors"){x = 115; y =74; z = 18;}
    if (mClassnames[class_id] == "hair drier"){x = 218; y =112; z = 214;}  // 淡紫色
    if (mClassnames[class_id] == "toothbrush"){x = 176; y =224; z = 230;}  //浅灰蓝

    Eigen::Vector3i color_rgb(x,y,z);
    return color_rgb;
}