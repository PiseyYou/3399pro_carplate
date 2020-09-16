#ifndef __LISENCE_PLATE_PRO_HPP__
#define __LISENCE_PLATE_PRO_HPP__

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <sys/time.h>
#include <unistd.h>
#include "map"
#include "rknn_api.h"
#include "plate_utils.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#define NUM_CLASSES 2

#define VEHICLE_NUM_RESULTS    294
#define PLATE_NUM_RESULTS      2500
#define PLATECHAR_NUM_RESULTS  2500
#define CHAR_CLS_NUM           72
#define MAX_NUM_RESULTS        2500

using namespace std;




class LP_Recognition
{   public:
    rknn_context Vehicle_detModel = 0;
    //const int VEHICLEDET_NUM_RESULTS = 294;
    float vehcile_boxPriors[4*VEHICLE_NUM_RESULTS];

    rknn_context Plate_detModel = 0;
    const int PLATEDET_NUM_RESULTS = 2500;
    float plate_boxPriors[4*PLATE_NUM_RESULTS];

    rknn_context Platechar_detModel = 0;
    //const int PLATECHARDET_NUM_RESULTS = 2500;
    float platechar_boxPriors[4*PLATECHAR_NUM_RESULTS];

    rknn_context Char_recogModel = 0;
    std::vector<string> charlabels;
   
    int nLoc_idcr = 1; // indicator for locating

    std::string color = "未知";
    std::string layer_type = "single";
    std::string stPlate_type = "未知";
    std::string plate_type = "未知";
    int nCnChar_c = 0;
    
    
    LP_Recognition();
    ~LP_Recognition();
     
    int process_init();

    void LP_main_process(
         cv::Mat& imgsrc
     );

     std::vector<cv::Rect> RknnDetInference_Process(
     cv::Mat& imgsrc, 
     rknn_context& RknnModel, 
     int img_width,
     int img_height,
     int img_channels,
     float* boxPriors,
     int NUM_RESULTS
     );

     void locate_plateDetArea(
     cv::Mat& imgsrc,
     cv::Rect& VehicleBoxes,
     cv::Mat& crop_rgb
     );
    
     void locate_charDetArea(
     cv::Mat& crop_rgb,
     cv::Mat& crop_gray,
     cv::Rect platebox, 
     std::pair<cv::Mat,cv::Mat>& matPlate,
     std::array<int,2>& plate_tpbtlim
     );
     
     void plateColor_Recog(
     cv::Mat& plate_img,
     cv::Rect& platebox
     );

     std::vector<cv::Rect> Error_Correction_pre(
     std::vector<cv::Rect> charboxes,
     std::array<int,2>& plate_tpbtlim
     );

     void char_Recog(
     cv::Mat& matCharimg,
     rknn_context& RknnModel, 
     int img_width,
     int img_height,
     int img_channels,
     int char_clsnum,
     std::vector<std::pair<float, int>>& top_results 
     );

     void char_RecogPro(
     cv::Mat& matPlatimg,
     std::vector<cv::Rect>& Charboxes,
     std::vector<std::vector<std::pair<float, int>>>& array_result
     );

     std::string Error_Correction_aft(
     std::vector<cv::Rect> charboxes_sort,
     std::vector<std::vector<std::pair<float, int>>>& array_result
     );

};


LP_Recognition::LP_Recognition()
{
   process_init();
}
LP_Recognition::~LP_Recognition()
{
   rknn_destroy (Vehicle_detModel);
   rknn_destroy (Plate_detModel);
   rknn_destroy (Platechar_detModel);
   rknn_destroy (Char_recogModel);
}


int LP_Recognition::process_init()
{
    int ret = 0;
    const char *vehicle_model_path = "./Model/vehicle_DetRkmodel0113_qu.rknn";
    const char *vehicle_box_priors_path = "./Model/Vehicle_DetBox_1219.txt";
    loadCoderOptions_my(vehicle_box_priors_path, vehcile_boxPriors, VEHICLE_NUM_RESULTS);
    std::pair<void*, int> vehicle_modelres = load_model(vehicle_model_path);
    if (vehicle_modelres.first == nullptr)
    {
        printf("vehicle_modelres load fail! ret=%d\n", ret);
        return -1;
    }
    ret = rknn_init(&Vehicle_detModel, vehicle_modelres.first, vehicle_modelres.second, RKNN_FLAG_PRIOR_MEDIUM);
    if(ret < 0) {
        printf("vehicle_detModel rknn_init fail! ret=%d\n", ret);
        return -1;
    }

    const char *plate_model_path = "./Model/plate_DetRkmodel_qu1227.rknn";
    const char *plate_box_priors_path = "./Model/Plate_DetBox1224.txt";
    loadCoderOptions_my(plate_box_priors_path, plate_boxPriors, PLATE_NUM_RESULTS);
    std::pair<void*, int> plate_modelres = load_model(plate_model_path);
    if (plate_modelres.first == nullptr)
    {
        printf("plate_modelres load fail! ret=%d\n", ret);
        return -1;
    }
    ret = rknn_init(&Plate_detModel, plate_modelres.first, plate_modelres.second, RKNN_FLAG_PRIOR_MEDIUM);
    if(ret < 0) {
        printf("plate_detModel rknn_init fail! ret=%d\n", ret);
        return -1;
    }

    const char *platechar_model_path = "./Model/platechar_DetRkmodel_qu0106.rknn";
    const char *platechar_box_priors_path = "./Model/Platechar_DetBox0106.txt";
    loadCoderOptions_my(platechar_box_priors_path, platechar_boxPriors, PLATECHAR_NUM_RESULTS);
    std::pair<void*, int> platechar_modelres = load_model(platechar_model_path);
    if (platechar_modelres.first == nullptr)
    {
        printf("platechar_modelres load fail! ret=%d\n", ret);
        return -1;
    }
    ret = rknn_init(&Platechar_detModel, platechar_modelres.first, platechar_modelres.second, RKNN_FLAG_PRIOR_MEDIUM);
    if(ret < 0) {
        printf("Platechar_detModel rknn_init fail! ret=%d\n", ret);
        return -1;
    }

    const char *charRecog_model_path = "./Model/charrecog0102.rknn";
    const char *char_lable = "./Model/labels.txt";
    size_t label_count;
    ReadLabelsFile(char_lable, &charlabels, &label_count);
    std::pair<void*, int> charRecog_modelres = load_model(charRecog_model_path);
    if (charRecog_modelres.first == nullptr)
    {
        printf("charRecog_modelres load fail! ret=%d\n", ret);
        return -1;
    }
    ret = rknn_init(&Char_recogModel, charRecog_modelres.first, charRecog_modelres.second, RKNN_FLAG_PRIOR_MEDIUM);
    if(ret < 0) {
        printf("charRecog_modelres rknn_init fail! ret=%d\n", ret);
        return -1;
    }
    
    return 0;
}


std::vector<cv::Rect> LP_Recognition::RknnDetInference_Process(
     cv::Mat& imgsrc, 
     rknn_context& RknnModel, 
     int img_width,
     int img_height,
     int img_channels,
     float* boxPriors,
     int NUM_RESULTS
)
{

    //const int img_width = 100;
    //const int img_height = 100;
    //const int img_channels = 3;
    
    cv::Mat img_res;
    if(imgsrc.cols != img_width || imgsrc.rows != img_height)
        cv::resize(imgsrc, img_res, cv::Size(img_width, img_height), (0, 0), (0, 0), cv::INTER_LINEAR);

    // Start Inference
    int ret = 0;
    rknn_input inputs[1];
    rknn_output outputs[2];
    rknn_tensor_attr outputs_attr[2];

    outputs_attr[0].index = 0;
    ret = rknn_query(RknnModel, RKNN_QUERY_OUTPUT_ATTR, &(outputs_attr[0]), sizeof(outputs_attr[0]));
    if(ret < 0) {
        printf("rknn_query fail! ret=%d\n", ret);
    }

    outputs_attr[1].index = 1;
    ret = rknn_query(RknnModel, RKNN_QUERY_OUTPUT_ATTR, &(outputs_attr[1]), sizeof(outputs_attr[1]));
    if(ret < 0) {
        printf("rknn_query fail! ret=%d\n", ret);
    }

    inputs[0].index = 0;
    inputs[0].buf = img_res.data;
    inputs[0].size = img_width * img_height * img_channels;
    inputs[0].pass_through = false;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    ret = rknn_inputs_set(RknnModel, 1, inputs);
    if(ret < 0) {
        printf("rknn_input_set fail! ret=%d\n", ret);
    }

    ret = rknn_run(RknnModel, nullptr);
    if(ret < 0) {
        printf("rknn_run fail! ret=%d\n", ret);
    }

    outputs[0].want_float = true;
    outputs[0].is_prealloc = false;
    outputs[1].want_float = true;
    outputs[1].is_prealloc = false;
    ret = rknn_outputs_get(RknnModel, 2, outputs, nullptr);
    if(ret < 0) {
        printf("rknn_outputs_get fail! ret=%d\n", ret);
    }

    // Process output
    std::vector<cv::Rect> validbox ;
    if(outputs[0].size == outputs_attr[0].n_elems*sizeof(float) && outputs[1].size == outputs_attr[1].n_elems*sizeof(float))
    {
        //float boxPriors[4][NUM_RESULTS];
        string labels[2] = {"0","1"};

        /* load label and boxPriors */
        //const char *label_path = "./tmp/license_plate/coco_labels_list.txt";
        //loadLabelName(label_path, labels);
        

        float* predictions = (float*)outputs[0].buf;
        float* outputClasses = (float*)outputs[1].buf;

        int output[2*NUM_RESULTS];

        /* transform */
        decodeCenterSizeBoxes(predictions, boxPriors, NUM_RESULTS);

        int validCount = scaleToInputSize(outputClasses, output, NUM_CLASSES, NUM_RESULTS);
        //printf("validCount: %d\n", validCount);

        if (validCount < 100) {
            /* detect nest box */
            nms(validCount, predictions, output, NUM_RESULTS);
            /* get valid box*/
            validbox = get_target(imgsrc, output, predictions,validCount, NUM_RESULTS);
            //cv::Mat rgba = draw_rect(imgsrc, labels, validbox);
            //imshow("out", rgba);
            //waitKey(0);
            //cv::imwrite("./tmp/outx.jpg", rgba);
            //getchar();
            //printf("write out.jpg succ!\n");
        } else {
            printf("validCount too much!\n");
        }
    }
    else
    {
        printf("rknn_outputs_get fail! get outputs_size = [%d, %d], but expect [%lu, %lu]!\n",
            outputs[0].size, outputs[1].size, outputs_attr[0].n_elems*sizeof(float), outputs_attr[1].n_elems*sizeof(float));
    }

    rknn_outputs_release(RknnModel, 2, outputs);
    return validbox;

}

void LP_Recognition::locate_plateDetArea(
     cv::Mat& imgsrc,
     cv::Rect& vehicleBox,
     cv::Mat& crop_rgb
     )
{
    std::array<int,2> arLoc_args = {1,1};
    if (this->nLoc_idcr == -1)arLoc_args = {0,2};
    cv::Rect crop_rect;
    crop_rect.width = vehicleBox.width;
    crop_rect.height = vehicleBox.height;

    crop_rect.x = vehicleBox.x + crop_rect.width/8;
    crop_rect.y = vehicleBox.y + crop_rect.height*arLoc_args[0]/2;

    crop_rect.height = crop_rect.height*arLoc_args[1]/2;
    crop_rect.width = crop_rect.width*6/8;

    crop_rect.x = fixEdge_W(crop_rect.x, imgsrc.cols);
    crop_rect.y = fixEdge_H(crop_rect.y, imgsrc.rows);
     
    crop_rect.width = fixEdge_W(crop_rect.x + crop_rect.width, imgsrc.cols) - crop_rect.x;
    crop_rect.height = fixEdge_H(crop_rect.y + crop_rect.height, imgsrc.rows) - crop_rect.y;

    crop_rgb = imgsrc(crop_rect);
}

void LP_Recognition::locate_charDetArea(
     cv::Mat& crop_rgb,
     cv::Mat& crop_gray,
     cv::Rect platebox,
     std::pair<cv::Mat,cv::Mat>& matPlate,
     std::array<int,2>& plate_tpbtlim
     )
{
     
     float plate_ratio = platebox.width*1.0/platebox.height;
 
     this->layer_type = (plate_ratio < 2.6)?"double":"single";


     std::array<float,4> rat_box = {0.1,1.3,0.1,1.3}; 
     if (this->layer_type == "double")rat_box = {0.1,0.4,0.1,0.4};
     
     //std::cout << rat_box[0] << " " << rat_box[1] << " " << rat_box[2] << " " << rat_box[3] << std::endl;
     cv::Rect pl_exRect;
     int plate_xmax = platebox.x + platebox.width;
     int plate_ymax = platebox.y + platebox.height;

     pl_exRect.x = fixEdge_W((platebox.x - rat_box[0]*platebox.width*1.0), crop_gray.cols);
     pl_exRect.y = fixEdge_H((platebox.y - rat_box[1]*platebox.height*1.0), crop_gray.rows);
     
     pl_exRect.width = fixEdge_W((plate_xmax + rat_box[2]*platebox.width*1.0), crop_gray.cols) - pl_exRect.x;
     pl_exRect.height = fixEdge_H((plate_ymax + rat_box[3]*platebox.height*1.0), crop_gray.rows) - pl_exRect.y;
     //std::cout << pl_exRect.x << " " << pl_exRect.y << " " << pl_exRect.width << " " << pl_exRect.height << " " << std::endl;
     //cv::Mat plateimg = crop_gray(pl_exRect);
     //cv::imwrite("./tmp/xxxxxx.jpg", plateimg);

     cv::Mat temp_img = crop_gray;
     int expand_h = 1.5*platebox.height;
     int plate_ct_y = pl_exRect.y + pl_exRect.height/2;

     if (plate_ct_y < expand_h && this->layer_type == "single")
     {
        //std::cout<< "1" << std::endl;
        cv::Mat pad_img = cv:: Mat::zeros(expand_h, crop_gray.cols, CV_8UC1);
        cv::vconcat(pad_img,crop_gray,crop_gray);
        pl_exRect.y = 0;
        pl_exRect.height += expand_h;
        platebox.y += expand_h;

     }  
    
     if (plate_ct_y > crop_gray.rows - expand_h && this->layer_type == "single")
     {
        //std::cout << "2" << std::endl;
        cv::Mat pad_img = cv::Mat::zeros(expand_h, crop_gray.cols, CV_8UC1);
        cv::vconcat(crop_gray,pad_img,crop_gray);
        //cv::imwrite("./tmp/outxconcat.jpg", crop_gray);
        pl_exRect.height += expand_h;
        //platebox.height += expand_h;
     }
     //std::cout << pl_exRect.y + pl_exRect.height << " " << crop_gray.rows << std::endl;
     matPlate.first = crop_gray(pl_exRect);
     //matPlate.second = crop_rgb(pl_exRect);
     
     platebox.x = platebox.x - pl_exRect.x;
     platebox.y = platebox.y - pl_exRect.y;
     //std::cout << platebox.x << " " << platebox.y << std::endl;
     plate_tpbtlim = {platebox.y, platebox.y + platebox.height};

}

void LP_Recognition::plateColor_Recog(
     cv::Mat& plate_img,
     cv::Rect& platebox
     )
{
     //std::string test = "123456789";
     //std::cout << strlen(test.c_str()) << std::endl; 
     cv::Mat matCalcolor ;
     cv::Mat matCrop = plate_img(platebox);
     matCrop.copyTo(matCalcolor) ;
     //cv::imwrite("./tmp/color.jpg", matCalcolor);
     cv::cvtColor(matCalcolor,matCalcolor,cv::COLOR_BGR2HSV);
   
     cv::Mat matColormask, matBlackMask, matWhiteMask;
     cv::inRange(matCalcolor,cv::Scalar(0,43,46),cv::Scalar(179,254,254),matColormask);
     cv::inRange(matCalcolor,cv::Scalar(0,0,0),cv::Scalar(179,254,46),matBlackMask);
     cv::inRange(matCalcolor,cv::Scalar(0,0,46),cv::Scalar(179,30,254),matWhiteMask);

     float nblack_r = cal_MaskCount(matBlackMask);
     float nwhite_r = cal_MaskCount(matWhiteMask);
     //std::cout << nblack_r << "  " << nwhite_r << std::endl;
     if (nblack_r > 0.15f && nwhite_r >0.3)
     {  
         this->color = "黑";
         return;
     }
    
     cv::Rect recMask_l(0,0,matCalcolor.cols*1/5,matCalcolor.rows);
     cv::Mat matColormask_l = matColormask(recMask_l);

     cv::Mat matH_hist;
     int nHistSize = 18;
     float flRange[] = { 0,180 };
     const float *cflHistRanges = { flRange };

     std::vector<cv::Mat> hsv_planes;
     cv::split(matCalcolor, hsv_planes);
     cv::calcHist(&hsv_planes[0],1,0,matColormask,matH_hist,1,&nHistSize,&cflHistRanges,true,false);
     //std::cout << matH_hist.rows << " " << matH_hist.cols << std::endl;
     double minVal = 0;
     double maxVal = 0;
     cv::Point minLoc(0,0);
     cv::Point maxLoc(0,0);
     cv::minMaxLoc(
        matH_hist,
        &minVal,
        &maxVal,
        &minLoc,
        &maxLoc
     );
     //std::cout << maxLoc.y << " " << maxVal << endl;
     int color_bin[] = {3,6,11};
     std::map <int,std::string> color_map = {{0,"黄"},{1,"绿"},{2,"蓝"}};

     auto temp_f = [&color_bin,&color_map](int a){
          int min_val = 1000;
          int min_idx = 1;
          
          for(int k=0;k<3;k++)
          {  

             int diff = abs(a-color_bin[k]);
             if (diff < min_val)
             {
                 min_idx = k;
                 min_val = diff;
             }
 
          }
          return color_map.at(min_idx);
     };
     std::string color_res = temp_f(maxLoc.y);
     //std::cout << maxVal << std::endl;

     if (nwhite_r >0.3 && color_res!="蓝" && maxVal < 500 || nwhite_r >0.5)
     {  
         this->color = "白";
         return;
     }
     //std::cout << "3: " << color_res << std::endl; 
     this->color = color_res;
     if (color_res == "绿")
     {  
        cv::Mat matCalcolor_l = matCalcolor(recMask_l);
        cv::Mat matH_hist_l;
        std::vector<cv::Mat> hsv_planes_l;
        cv::split(matCalcolor_l, hsv_planes_l);
        cv::calcHist(&hsv_planes_l[0],1,0,matColormask_l,matH_hist_l,1,&nHistSize,&cflHistRanges,true,false);
        cv::minMaxLoc(matH_hist_l,&minVal,&maxVal,&minLoc,&maxLoc);
        color_res = temp_f(maxLoc.y);
        //std::cout << "4: " << color_res << std::endl;
        if(color_res == "黄")this->color = "黄绿";
     }

     
     //std::cout << this->color << std::endl;
}

std::vector<cv::Rect> LP_Recognition::Error_Correction_pre(
     std::vector<cv::Rect> charboxes,
     std::array<int,2>& plate_tpbtlim
     )
{
     //delete the rect of crossing boundary and sort the charboxes
     std::vector<cv::Rect> charboxes_sort;
     charboxes_sort = (this->layer_type == "single")?sort_charbox(charboxes,plate_tpbtlim):sort_Doublebox(charboxes);


     return charboxes_sort;
}




void LP_Recognition::char_Recog(
     cv::Mat& matCharimg,
     rknn_context& RknnModel, 
     int img_width,
     int img_height,
     int img_channels,
     int char_clsnum,
     std::vector<std::pair<float, int>>& top_results
     )
{

     //const char *lable_path = "./tmp/license_plate/labels.txt";
     const int input_index = 0;      // node name "input"
     const int output_index = 0;     // node name "MobilenetV1/Predictions/Reshape_1"
     int output_elems = char_clsnum;
     cv::Mat matImg_res;
     
     cv::resize(matCharimg, matImg_res, cv::Size(img_width, img_height), (0, 0), (0, 0), cv::INTER_LINEAR);

     // Start Inference
     rknn_input inputs[1];
     rknn_output outputs[1];
     rknn_tensor_attr output0_attr;

     int ret = 0;

     output0_attr.index = 0;
     ret = rknn_query(RknnModel, RKNN_QUERY_OUTPUT_ATTR, &output0_attr, sizeof(output0_attr));
     if(ret < 0) {
        printf("rknn_query fail! ret=%d\n", ret);
     }

     inputs[0].index = input_index;
     inputs[0].buf = matImg_res.data;
     inputs[0].size = img_width * img_height * img_channels;
     inputs[0].pass_through = false;
     inputs[0].type = RKNN_TENSOR_UINT8;
     inputs[0].fmt = RKNN_TENSOR_NHWC;
     ret = rknn_inputs_set(RknnModel, 1, inputs);
     if(ret < 0) {
        printf("rknn_input_set fail! ret=%d\n", ret);
     }

     ret = rknn_run(RknnModel, nullptr);
     if(ret < 0) {
        printf("rknn_run fail! ret=%d\n", ret);
     }

     outputs[0].want_float = true;
     outputs[0].is_prealloc = false;
     ret = rknn_outputs_get(RknnModel, 1, outputs, nullptr);
     if(ret < 0) {
        printf("rknn_outputs_get fail! ret=%d\n", ret);
     }
     // Process output
     //std::vector<std::pair<float, int>> top_results;
     if(outputs[0].size == output0_attr.n_elems * sizeof(float))
     {
        const size_t num_results = 2;
        const float threshold = 0.000f;

        
        get_top_n<float>((float*)outputs[0].buf, output_elems,
                           num_results, threshold, &top_results, true);

        //std::vector<string> labels;
        /*for (const auto& result : top_results) {
                const float confidence = result.first;
                const int index = result.second;
                std::cout << charlabels[index] << ": " << confidence << "\n";
            }*/
        
     }
     else
     {
        printf("rknn_outputs_get fail! get output_size = [%d], but expect %u!\n",
            outputs[0].size, (uint32_t)(output0_attr.n_elems * sizeof(float)));
     }
     
     rknn_outputs_release(RknnModel, 1, outputs);
     //return top_results;
}


void LP_Recognition::char_RecogPro(
     cv::Mat& matPlatimg,
     std::vector<cv::Rect>& Charboxes,
     std::vector<std::vector<std::pair<float, int>>>& array_result
     )
{
   
     this->nCnChar_c = 0;
     for (int i = 0 ; i < Charboxes.size(); i++)
     {
        Charboxes[i].x = fixEdge_W(Charboxes[i].x, matPlatimg.cols);
        Charboxes[i].y = fixEdge_H(Charboxes[i].y, matPlatimg.rows);
     
        Charboxes[i].width = fixEdge_W(Charboxes[i].x+Charboxes[i].width, matPlatimg.cols) -  Charboxes[i].x;
        Charboxes[i].height = fixEdge_H(Charboxes[i].y+Charboxes[i].height, matPlatimg.rows) -  Charboxes[i].y;
         
         cv::Mat matChar = matPlatimg(Charboxes[i]);
         //char savename[32];
         //sprintf(savename, "./tmp/char_%d.jpg", i);
         //cv::imwrite(savename, matChar);
         std::vector<std::pair<float, int>> top_results;
         char_Recog(matChar, Char_recogModel,24,24,1,CHAR_CLS_NUM,top_results);
         if (top_results[0].second >= 33)this->nCnChar_c++;
         array_result.push_back(top_results);
         
     } 
     //std::cout << " the result is " << plate_result << std::endl;
}

std::string LP_Recognition::Error_Correction_aft(
     std::vector<cv::Rect> charboxes_sort,
     std::vector<std::vector<std::pair<float, int>>>& array_result
     )
{

    
    //fix char in designated pos, fix the color and output final plate type

    std::vector<int> last_sepChar = {34,50,63,67};
    std::vector<int> black_char = {34,50};
    std::vector<int> veFirst_inv = {34,50,63,67,71};
    std::vector<std::string> spe_color= {"黑","白"};
    std::vector<std::string> veNorColor= {"黄","绿","蓝"};
    std::string plate_result="";
    int nResult_len = array_result.size();
    for (int i = 0 ;i < nResult_len; i++)
    {  
        int nchar_key = array_result[i][0].second;
        //std::cout << i << std::endl;
        if (i == 0 && nchar_key <=33 && !check_element(spe_color,this->color))
        {   
            //std::cout << "NO.1" << std::endl;
            array_result[i][0] = array_result[i][2];
        }
        if (i == 0 && check_element(veNorColor,this->color) && check_element(veFirst_inv,nchar_key))
        {   
            //std::cout << "NO.1" << std::endl;
            array_result[i][0] = array_result[i][3];
        }
        if (i == 1 && nchar_key < 10)
        {
            //std::cout << "NO.2" << std::endl;
            array_result[i][0] = array_result[i][5];
        }
        if (i != 6 && check_element(last_sepChar,nchar_key))
        {
            //std::cout << "NO.3" << std::endl;
            array_result[i][0] = array_result[i][3];
        }
        if ((i > 0 && i < 6 && nchar_key > 33 && layer_type == "single") ||
            (i == 6 && nchar_key > 33 && !check_element(last_sepChar,nchar_key)))
        {
           //std::cout << "NO.4" << std::endl;
           array_result[i][0] = array_result[i][4];
        }

        // output type of license plate
        if (i==6 && nchar_key==36)
        {
            plate_result += "挂";
            this->plate_type = "挂车牌";
            this->color = "黄";
            continue;
        }

        if (i==6 && check_element(black_char,nchar_key))
        {
            plate_result += charlabels[array_result[i][0].second];
            this->plate_type = "港澳牌";
            this->color = "黑";
            continue;
        }
      
        if (i==6 && nchar_key == 70 && nResult_len == 8)
        {
            plate_result += "应急";
            this->plate_type = "应急牌";
            this->color = "白";
            i++;
            continue;
        }

        if (i==0 && nchar_key == 71 && this->color == "绿")
        {
            plate_result += "民航";
            this->plate_type = "民航牌";
            this->color = "绿";
            i++;
            continue;
        }
        
        if (i ==0 && this->layer_type == "single" && array_result[0][0].second == 30 && array_result[0][0].second == 18)
        {
            plate_result = "WJ";
            this->plate_type = "武警牌";
            this->color = "白";
            i++;
            continue;
        }

        if (i==2 && nchar_key>33 && this->layer_type == "double")
        {
            plate_result ="WJ" + charlabels[array_result[i][0].second];
            
            this->plate_type = "武警牌";
            this->color = "白";
            continue;
        }

        if (i==6 && nchar_key ==63)
        {
            plate_result += "学";
            this->plate_type = "教练牌";
            this->color = "黄";
            continue;

        }
         
        if (i==6 && nchar_key ==67)
        {
            plate_result += "警";
            this->plate_type = "警牌";
            this->color = "白";
            continue;

        }
      
        plate_result += charlabels[array_result[i][0].second];

    }
    //std::cout << this->color << std::endl;

    if (nResult_len ==8 && this->layer_type == "single" &&this->plate_type=="未知")
    {
        this->plate_type = "新能源牌";
        this->color = (this->color == "黄绿" || this->color == "黄")?"黄绿":"绿";
    }
    if (nResult_len ==7 && this->color == "蓝" &&this->plate_type=="未知")
    {
        this->plate_type = "普通蓝牌";
    }
    if (nResult_len ==7 && this->color == "黄" &&this->plate_type=="未知")
    {
        this->plate_type = "普通黄牌";
    }

    std::string stlayer = (this->layer_type == "single")?"单层":"双层";

    this->stPlate_type = stlayer + "-" + this->plate_type + "-" + this->color;
    //std::cout << "the plate_type is " << this->stPlate_type << std::endl;
    //std::cout << "the result is " << plate_result << std::endl;
    return plate_result;

}


void LP_Recognition::LP_main_process(cv::Mat& imgsrc)
{
    std::vector<cv::Rect> Vehicleboxes; 
    //printf("vehicle det!!\n");
    Vehicleboxes = RknnDetInference_Process(imgsrc, Vehicle_detModel, 100, 100, 3, vehcile_boxPriors, VEHICLE_NUM_RESULTS); //detect vehicle
    for (auto vehiclebox : Vehicleboxes)
    {
     cv::Mat matCroprgb;
     //printf("locate_plateDetArea!!\n");
     locate_plateDetArea(imgsrc,vehiclebox,matCroprgb);
     cv::Mat matCropgray;
     cv::cvtColor(matCroprgb,matCropgray,cv::COLOR_BGR2GRAY);
        
     std::vector<cv::Rect> Plateboxes;
     //printf("plate det!!\n");
     Plateboxes = RknnDetInference_Process(matCropgray, Plate_detModel, 200, 200, 1, plate_boxPriors, PLATE_NUM_RESULTS); //detect vehicle

     for (auto platebox:Plateboxes)
     {   

         platebox.x = fixEdge_W(platebox.x, matCroprgb.cols);
         platebox.y = fixEdge_H(platebox.y, matCroprgb.rows);
     
         platebox.width = fixEdge_W(platebox.x+platebox.width, matCroprgb.cols)-platebox.x;
         platebox.height = fixEdge_H(platebox.y+platebox.height, matCroprgb.rows)-platebox.y;
    
         if (platebox.width < 10 || platebox.height < 10)continue;
      
         //printf("color recog!!\n");
         plateColor_Recog(matCroprgb,platebox);
   
         std::pair<cv::Mat,cv::Mat> matPlate;// first is gray, second is rgb
         std::array<int,2>plate_tpbtlim;
         //printf("locate_charDetArea!!\n");
         locate_charDetArea(matCroprgb,matCropgray,platebox,matPlate,plate_tpbtlim);

         std::vector<cv::Rect> Charboxes;
         //printf("char det!!\n");
         Charboxes = RknnDetInference_Process(matPlate.first, Platechar_detModel, 100, 100, 1, platechar_boxPriors, PLATECHAR_NUM_RESULTS); //detect char

         //printf("correct_pre\n");
         std::vector<cv::Rect> charbox_Res_pre;
         charbox_Res_pre = Error_Correction_pre(Charboxes,plate_tpbtlim);
 
         //printf("char recog!!\n");
         std::vector<std::vector<std::pair<float, int>>> array_result;
         char_RecogPro(matPlate.first, charbox_Res_pre, array_result);

         //printf("correct_aft!!\n");
         Error_Correction_aft(charbox_Res_pre,array_result);

    
     }
      
    }
}


/*
int main()
{

  const char *img_path = "./tmp/license_plate/46.jpg";
  cv::Mat img = cv::imread(img_path, 1);
    if(!img.data) {
        printf("cv::imread %s fail!\n", img_path);
        return -1;
    }
  LP_Recognition a = LP_Recognition();


  std::vector<cv::String>image_file;
  //std::string pattern = "/home/supernode/lisk_data/test_data/plate_sfz/test_1/*.jpg";
  //std::string pattern = "/home/supernode/lisk_data/test_data/plate_sfz/test_3/*.jpg"; 
  std::string pattern = "/home/supernode/lisk_data/test_data/plate_sfz/color/*.jpg";
  //std::string pattern = "/home/supernode/lisk_dpproject/c_lisence_platepro/rknn_api_sdk/wr/*.jpg";
  cv::glob(pattern, image_file);
  
  
  LP_Recognition a = LP_Recognition();
  int img_num = 0;
  for (auto val : image_file)
  {   
      img_num++;
      cout << val << endl;
      cv::Mat imgsrc = cv::imread(val, 1);
      a.LP_main_process(imgsrc);
      imgsrc.release();
      getchar();
  }
  printf("img_count is %d",img_num);
  //a.LP_main_process(img);
  return 0;

}

*/

#endif
























