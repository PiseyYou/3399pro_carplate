#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <sys/time.h>
#include <unistd.h>

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

    std::string layer_type = "single";


    
    
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
     cv::Rect& platebox, 
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
     std::vector<cv::Rect>& Charboxes
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
   
    cv::Rect crop_rect;
    crop_rect.width = vehicleBox.width;
    crop_rect.height = vehicleBox.height;

    crop_rect.x = vehicleBox.x + crop_rect.width/9;
    crop_rect.y = vehicleBox.y + crop_rect.height/2;

    crop_rect.height = crop_rect.height/2;
    crop_rect.width = crop_rect.width*7/9;

    crop_rect.x = fixEdge_W(crop_rect.x, imgsrc.cols);
    crop_rect.y = fixEdge_H(crop_rect.y, imgsrc.rows);
     
    crop_rect.width = fixEdge_W(crop_rect.x + crop_rect.width, imgsrc.cols) - crop_rect.x;
    crop_rect.height = fixEdge_H(crop_rect.y + crop_rect.height, imgsrc.rows) - crop_rect.y;

    crop_rgb = imgsrc(crop_rect);
}

void LP_Recognition::locate_charDetArea(
     cv::Mat& crop_rgb,
     cv::Mat& crop_gray,
     cv::Rect& platebox,
     std::pair<cv::Mat,cv::Mat>& matPlate,
     std::array<int,2>& plate_tpbtlim
     )
{

     float plate_ratio = platebox.width*1.0/platebox.height;
 
     this->layer_type = (plate_ratio < 2.6)?"double":"single";


     std::array<float,4> rat_box = {0.1,1.3,0.1,1.3}; 
     if (this->layer_type == "double")rat_box = {0.1,0.4,0.1,0.4};

     cv::Rect pl_exRect;
     int plate_xmax = platebox.x + platebox.width;
     int plate_ymax = platebox.y + platebox.height;

     pl_exRect.x = fixEdge_W((platebox.x - rat_box[0]*platebox.width*1.0), crop_gray.cols);
     pl_exRect.y = fixEdge_H((platebox.y - rat_box[1]*platebox.height*1.0), crop_gray.rows);
     
     pl_exRect.width = fixEdge_W((plate_xmax + rat_box[2]*platebox.width*1.0), crop_gray.cols) - pl_exRect.x;
     pl_exRect.height = fixEdge_H((plate_ymax + rat_box[3]*platebox.height*1.0), crop_gray.rows) - pl_exRect.y;
     std::cout << pl_exRect.x << " " << pl_exRect.y << " " << pl_exRect.width << " " << pl_exRect.height << " " << std::endl;

     cv::Mat temp_img = crop_gray;
     int expand_h = 1.5*platebox.height;
     int plate_ct_y = pl_exRect.y + pl_exRect.height/2;

     if (plate_ct_y < expand_h)
     {
        std::cout<< "1" << std::endl;
        cv::Mat pad_img = cv:: Mat::zeros(expand_h, crop_gray.cols, CV_8UC1);
        cv::vconcat(pad_img,crop_gray,crop_gray);
        pl_exRect.y = 0;
        pl_exRect.height += expand_h;
        platebox.y += expand_h;

     }  
    
     if (plate_ct_y > crop_gray.rows - expand_h)
     {
        std::cout << "2" << std::endl;
        cv::Mat pad_img = cv::Mat::zeros(expand_h, crop_gray.cols, CV_8UC1);
        cv::vconcat(crop_gray,pad_img,crop_gray);
        //cv::imwrite("./tmp/outxconcat.jpg", crop_gray);
        pl_exRect.height += expand_h;
        //platebox.height += expand_h;
     }
     std::cout << pl_exRect.y + pl_exRect.height << " " << crop_gray.rows << std::endl;
     matPlate.first = crop_gray(pl_exRect);
     //matPlate.second = crop_rgb(pl_exRect);
     
     platebox.x = platebox.x - pl_exRect.x;
     platebox.y = platebox.y - pl_exRect.y;
     std::cout << platebox.x << " " << platebox.y << std::endl;
     plate_tpbtlim = {platebox.y, platebox.y + platebox.height};

}

void LP_Recognition::plateColor_Recog(
     cv::Mat& plate_img,
     cv::Rect& platebox
     )
{
 
     cv::Mat matCalcolor = plate_img(platebox);
     //cv::imwrite("./tmp/color.jpg", matCalcolor);
     cv::cvtColor(matCalcolor,matCalcolor,cv::COLOR_BGR2HSV);
   
     cv::Mat matColormask;
     cv::inRange(matCalcolor,cv::Scalar(0,43,46),cv::Scalar(179,254,254),matColormask);
    
     cv::Rect recMask_l(0,0,matCalcolor.cols/4,matCalcolor.rows);
     cv::Mat matColormask_l = matColormask(recMask_l);

     cv::Mat matH_hist;
     int nHistSize = 18;
     float flRange[] = { 0,180 };
     const float *cflHistRanges = { flRange };

     std::vector<cv::Mat> hsv_planes;
     cv::split(matCalcolor, hsv_planes);
     cv::calcHist(&hsv_planes[0],1,0,matColormask,matH_hist,1,&nHistSize,&cflHistRanges,true,false);
     std::cout << matH_hist.rows << " " << matH_hist.cols << std::endl;

     std::pair<int,int>paHistres = find_maxIndexforHist(matH_hist);

     std::cout << paHistres.first << " " << paHistres.second << endl;


}

std::vector<cv::Rect> LP_Recognition::Error_Correction_pre(
     std::vector<cv::Rect> charboxes,
     std::array<int,2>& plate_tpbtlim
     )
{
     //delete the rect of crossing boundary and sort the charboxes
     std::vector<cv::Rect> charboxes_sort;
     charboxes_sort = sort_charbox(charboxes,plate_tpbtlim);


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
        const float threshold = 0.001f;

        
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
 
     //return top_results;
}


void LP_Recognition::char_RecogPro(
     cv::Mat& matPlatimg,
     std::vector<cv::Rect>& Charboxes
     )
{
   
     std::string plate_result="";  
     for (int i = 0 ; i < Charboxes.size(); i++)
     {
        Charboxes[i].x = fixEdge_W(Charboxes[i].x, matPlatimg.cols);
        Charboxes[i].y = fixEdge_H(Charboxes[i].y, matPlatimg.rows);
     
        Charboxes[i].width = fixEdge_W(Charboxes[i].x+Charboxes[i].width, matPlatimg.cols) -  Charboxes[i].x;
        Charboxes[i].height = fixEdge_H(Charboxes[i].y+Charboxes[i].height, matPlatimg.rows) -  Charboxes[i].y;
         
         cv::Mat matChar = matPlatimg(Charboxes[i]);
         char savename[32];
         sprintf(savename, "./tmp/char_%d.jpg", i);
         //cv::imwrite(savename, matChar);
         std::vector<std::pair<float, int>> top_results;
         char_Recog(matChar, Char_recogModel,24,24,1,CHAR_CLS_NUM,top_results);
         plate_result += charlabels[top_results[0].second];
     } 
     std::cout << " the result is " << plate_result << std::endl;
}


void LP_Recognition::LP_main_process(cv::Mat& imgsrc)
{
    std::vector<cv::Rect> Vehicleboxes; 
    printf("vehicle det!!\n");
    Vehicleboxes = RknnDetInference_Process(imgsrc, Vehicle_detModel, 100, 100, 3, vehcile_boxPriors, VEHICLE_NUM_RESULTS); //detect vehicle
    for (auto vehiclebox : Vehicleboxes)
    {
     cv::Mat matCroprgb;
     printf("locate_plateDetArea!!\n");
     locate_plateDetArea(imgsrc,vehiclebox,matCroprgb);
     cv::Mat matCropgray;
     cv::cvtColor(matCroprgb,matCropgray,cv::COLOR_BGR2GRAY);
        
     std::vector<cv::Rect> Plateboxes;
     printf("plate det!!\n");
     Plateboxes = RknnDetInference_Process(matCropgray, Plate_detModel, 200, 200, 1, plate_boxPriors, PLATE_NUM_RESULTS); //detect vehicle

     for (auto platebox:Plateboxes)
     {   

         platebox.x = fixEdge_W(platebox.x, matCroprgb.cols);
         platebox.y = fixEdge_H(platebox.y, matCroprgb.rows);
     
         platebox.width = fixEdge_W(platebox.x+platebox.width, matCroprgb.cols) -  platebox.x;
         platebox.height = fixEdge_H(platebox.y+platebox.height, matCroprgb.rows) -  platebox.y;

         printf("color recog!!\n");
         plateColor_Recog(matCroprgb,platebox);
   
         std::pair<cv::Mat,cv::Mat> matPlate;// first is gray, second is rgb
         std::array<int,2>plate_tpbtlim;
         printf("locate_charDetArea!!\n");
         locate_charDetArea(matCroprgb,matCropgray,platebox,matPlate,plate_tpbtlim);

         std::vector<cv::Rect> Charboxes;
         printf("char det!!\n");
         Charboxes = RknnDetInference_Process(matPlate.first, Platechar_detModel, 100, 100, 1, platechar_boxPriors, PLATECHAR_NUM_RESULTS); //detect char

         printf("correct_pre\n");
         std::vector<cv::Rect> charbox_Res_pre;
         charbox_Res_pre = Error_Correction_pre(Charboxes,plate_tpbtlim);
 
         printf("char recog!!\n");
         char_RecogPro(matPlate.first, charbox_Res_pre);

    
     }
      
    }
}




int main()
{
/*
  const char *img_path = "./tmp/license_plate/46.jpg";
  cv::Mat img = cv::imread(img_path, 1);
    if(!img.data) {
        printf("cv::imread %s fail!\n", img_path);
        return -1;
    }
  LP_Recognition a = LP_Recognition();
*/

  std::vector<cv::String>image_file;
  //std::string pattern = "/home/toybrick/lisk_project/c_test_pro/2020_2_10_cplate/wr/*.jpg"; 
  std::string pattern = "/home/toybrick/lisk_project/0110_platepro/test_1/*.jpg"; 
  cv::glob(pattern, image_file);
  
  
  LP_Recognition a = LP_Recognition();
  double sumtime = 0.f;
  int nimg_num = 0;
  cv::Mat imgsrc;
  for (auto val : image_file)
  {
      //nimg_num++;
      cout << val << endl;
      //clock_t start = clock();
      imgsrc = cv::imread(val, 1);
     
      nimg_num++;
      clock_t start = clock();
      a.LP_main_process(imgsrc);
      clock_t end = clock();
      std::cout << "time cost " << (double)(end - start)/CLOCKS_PER_SEC << std::endl;
      sumtime += (double)(end - start)/CLOCKS_PER_SEC;
      std::cout << nimg_num << " avg cost " << sumtime/nimg_num << std::endl;

      //imgsrc.release();
      //getchar();
  }
  //a.LP_main_process(img);
  return 0;

}




























