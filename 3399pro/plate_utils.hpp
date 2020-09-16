#ifndef __PLATE_UTILS_HPP
#define __PLATE_UTILS_HPP

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <queue>

#define Y_SCALE  10.0f
#define X_SCALE  10.0f
#define H_SCALE  5.0f
#define W_SCALE  5.0f

using namespace std;


inline int fixEdge_W(int p_val, int img_w)
{
    if (p_val<=0)p_val = 1;
    if (p_val>=img_w)p_val = img_w - 1;
    return p_val;
}

inline int fixEdge_H(int p_val, int img_h)
{
    if (p_val<=0)p_val = 1;
    if (p_val>=img_h)p_val = img_h - 1;
    return p_val;
}



template <class T>
void get_top_n(T* prediction, int prediction_size, size_t num_results,
               float threshold, std::vector<std::pair<float, int>>* top_results,
               bool input_floating) {
  // Will contain top N results in ascending order.
  std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>,
                      std::greater<std::pair<float, int>>>
      top_result_pq,top_result_cn,top_result_en;

  const long count = prediction_size;  // NOLINT(runtime/int)
  for (int i = 0; i < count; ++i) {
    float value;
    if (input_floating)
      value = prediction[i];
    else
      value = prediction[i] / 255.0;
    // Only add it if it beats the threshold and has a chance at being in
    // the top N.
    if (value < threshold) {
      continue;
    }

    top_result_pq.push(std::pair<float, int>(value, i));
  
    // If at capacity, kick the smallest value out.
    if (top_result_pq.size() > num_results) {
      top_result_pq.pop();
    }


    if (i>=33)
    {
       top_result_cn.push(std::pair<float, int>(value, i));
       if (top_result_cn.size() > 2) {
          top_result_cn.pop();
       }
    }
    if (i<33)
    {
       top_result_en.push(std::pair<float, int>(value, i));
       if (top_result_en.size() > 2) {
          top_result_en.pop();
       }
    }


  }

  // Copy to output vector and reverse into descending order.
  while (!top_result_en.empty()) {
    top_results->push_back(top_result_en.top());
    top_result_en.pop();
  }
  while (!top_result_cn.empty()) {
    top_results->push_back(top_result_cn.top());
    top_result_cn.pop();
  }
  while (!top_result_pq.empty()) {
    top_results->push_back(top_result_pq.top());
    top_result_pq.pop();
  }
  std::reverse(top_results->begin(), top_results->end());
}

int ReadLabelsFile(const string& file_name,
                            std::vector<string>* result,
                            size_t* found_label_count) {
  std::ifstream file(file_name);
  if (!file) {
    std::cerr << "Labels file " << file_name << " not found\n";
    return -1;
  }
  result->clear();
  string line;
  while (std::getline(file, line)) {
    result->push_back(line);
  }
  *found_label_count = result->size();
  const int padding = 16;
  while (result->size() % padding) {
    result->emplace_back();
  }
  return 0;
}








float MIN_SCORE = 0.4f;

float NMS_THRESHOLD = 0.3f;

cv::Scalar colorArray[10] = {
        cv::Scalar(139,   0,   0, 255),
        cv::Scalar(139,   0, 139, 255),
        cv::Scalar(  0,   0, 139, 255),
        cv::Scalar(  0, 100,   0, 255),
        cv::Scalar(139, 139,   0, 255),
        cv::Scalar(209, 206,   0, 255),
        cv::Scalar(  0, 127, 255, 255),
        cv::Scalar(139,  61,  72, 255),
        cv::Scalar(  0, 255,   0, 255),
        cv::Scalar(255,   0,   0, 255),
};

void getimages(std::string pattern, std::vector<cv::String>& image_file)
{   
    glob(pattern, image_file);              //必须为CV的String  
}


int loadLabelName(string locationFilename, string* labels) {
    ifstream fin(locationFilename);
    string line;
    int lineNum = 0;
    while(getline(fin, line))
    {
        labels[lineNum] = line;
        lineNum++;
    }
    return 0;
}

std::pair<void*, int> load_model(const char* path)
{
    std::pair<void*, int> result(nullptr, 0);
    FILE *fp = fopen(path, "rb");
    if(fp == NULL) {
        printf("fopen %s fail!\n", path);
        return result;
    }
    fseek(fp, 0, SEEK_END);
    int model_len = ftell(fp);
    void *model = malloc(model_len);
    fseek(fp, 0, SEEK_SET);
    if(model_len != fread(model, 1, model_len, fp)) {
        printf("fread %s fail!\n", path);
        free(model);
        return result;
    }

    result.first = model;
    result.second = model_len;
    return result;

}
int loadCoderOptions_my(string locationFilename, float *boxPriors, int NUM_RESULTS)
{
    ifstream fin(locationFilename);
    string line;
    int lineNum = 0;
    int priorIndex = 0;
    float temp[4]={0.f};
    while(getline(fin, line))
    {
        char *line_str = const_cast<char *>(line.c_str());
        float number = static_cast<float>(atof(line_str));
        //std::cout << number << std::endl;
        temp[lineNum] = number;
        if (lineNum == 3)
        {
           boxPriors[0*NUM_RESULTS+priorIndex] = (temp[0] + temp[2]) / 2; //cx
           boxPriors[1*NUM_RESULTS+priorIndex] = (temp[1] + temp[3]) / 2; //cy
           boxPriors[2*NUM_RESULTS+priorIndex] = temp[2] - temp[0];       //w
           boxPriors[3*NUM_RESULTS+priorIndex] = temp[3] - temp[1];       //h
           //std::cout << boxPriors[3][priorIndex] << " " << std::endl;
           
           priorIndex++;
           lineNum = 0;
        }
        else lineNum++;
        
    }
    return 0;

}

float CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0, float xmin1, float ymin1, float xmax1, float ymax1) {
    float w = max(0.f, min(xmax0, xmax1) - max(xmin0, xmin1));
    float h = max(0.f, min(ymax0, ymax1) - max(ymin0, ymin1));
    float i = w * h;
    float u = (xmax0 - xmin0) * (ymax0 - ymin0) + (xmax1 - xmin1) * (ymax1 - ymin1) - i;
    return u <= 0.f ? 0.f : (i / u);
}

float expit(float x) {
    return (float) (1.0 / (1.0 + exp(-x)));
}
float expit2(float x) {
    return (float) exp(x);
}

void decodeCenterSizeBoxes(float* predictions, float *boxPriors, int NUM_RESULTS) {

    for (int i = 0; i < NUM_RESULTS; ++i) {
        float xcenter = predictions[i*4+0] / Y_SCALE * boxPriors[2*NUM_RESULTS+i] + boxPriors[0*NUM_RESULTS+i];
        float ycenter = predictions[i*4+1] / X_SCALE * boxPriors[3*NUM_RESULTS+i] + boxPriors[1*NUM_RESULTS+i];
        float w = (float) exp(predictions[i*4 + 2] / H_SCALE) * boxPriors[2*NUM_RESULTS+i];
        float h = (float) exp(predictions[i*4 + 3] / W_SCALE) * boxPriors[3*NUM_RESULTS+i];

        float ymin = ycenter - h / 2.0f;
        float xmin = xcenter - w / 2.0f;
        float ymax = ycenter + h / 2.0f;
        float xmax = xcenter + w / 2.0f;

        predictions[i*4 + 0] = xmin;
        predictions[i*4 + 1] = ymin;
        predictions[i*4 + 2] = xmax;
        predictions[i*4 + 3] = ymax;
    }
}

int scaleToInputSize(float * outputClasses, int *output, int numClasses, int NUM_RESULTS)
{
    int validCount = 0;
    // Scale them back to the input size.
    for (int i = 0; i < NUM_RESULTS; ++i) {
        float topClassScore = static_cast<float>(-1000.0);
        int topClassScoreIndex = -1;

        float obj_conf = expit2(outputClasses[i*numClasses+1]);
        float sum_score = expit2(outputClasses[i*numClasses])+obj_conf;
        topClassScore = obj_conf / sum_score;
        
        if (topClassScore >= MIN_SCORE) {
            output[0*NUM_RESULTS+validCount] = i;
            output[1*NUM_RESULTS+validCount] = 1;
            ++validCount;
        }
    }

    return validCount;
}

int nms(int validCount, float* outputLocations, int *output, int NUM_RESULTS)
{
    for (int i=0; i < validCount; ++i) {
        if (output[0*NUM_RESULTS+i] == -1) {
            continue;
        }
        int n = output[0*NUM_RESULTS+i];
        for (int j=i + 1; j<validCount; ++j) {
            int m = output[0*NUM_RESULTS+j];
            if (m == -1) {
                continue;
            }
            float xmin0 = outputLocations[n*4 + 0];
            float ymin0 = outputLocations[n*4 + 1];
            float xmax0 = outputLocations[n*4 + 2];
            float ymax0 = outputLocations[n*4 + 3];

            float xmin1 = outputLocations[m*4 + 0];
            float ymin1 = outputLocations[m*4 + 1];
            float xmax1 = outputLocations[m*4 + 2];
            float ymax1 = outputLocations[m*4 + 3];

            float iou = CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1);

            if (iou >= NMS_THRESHOLD) {
                output[0*NUM_RESULTS+j] = -1;
            }
        }
    }

    return 0;
}

std::vector<cv::Rect> get_target(cv::Mat& img, int* output, float* predictions, int validCount, int NUM_RESULTS)
{
   std::vector<cv::Rect> validbox;
   /* box valid detect target */
   for (int i = 0; i < validCount; ++i) {
       if (output[0*NUM_RESULTS+i] == -1) {
                    continue;
        }
       int n = output[0*NUM_RESULTS+i];
       int topClassScoreIndex = output[1*NUM_RESULTS+i];

       int x1 = static_cast<int>(predictions[n * 4 + 0] * img.cols);
       int y1 = static_cast<int>(predictions[n * 4 + 1] * img.rows);
       int x2 = static_cast<int>(predictions[n * 4 + 2] * img.cols);
       int y2 = static_cast<int>(predictions[n * 4 + 3] * img.rows);   
       
       cv::Rect vbox(x1,y1,x2-x1,y2-y1);
       validbox.push_back(vbox);
   }  

   return validbox;
}


cv::Mat draw_rect(cv::Mat& img, string* labels, std::vector<cv::Rect>& validbox)
{
   cv::Mat rgba = img.clone();
   //cv::resize(rgba, rgba, cv::Size(1200, 1200), (0, 0), (0, 0), cv::INTER_LINEAR);

   for(auto val : validbox)
   {
       int x1 = val.x;
       int y1 = val.y;
       int x2 = val.x + val.width;
       int y2 = val.y + val.height;
       string label = labels[1];

       std::cout << label << "\t@ (" << x1 << ", " << y1 << ") (" << x2 << ", " << y2 << ")" << "\n";

       rectangle(rgba, cv::Point(x1, y1), cv::Point(x2, y2), colorArray[1%10], 3);
       putText(rgba, label, cv::Point(x1, y1 - 12), 1, 2, cv::Scalar(0, 255, 0, 255));
   }


   return rgba;

}


float cal_MaskCount(cv::Mat& matMask)
{
    cv::Mat_<uchar> matIm = matMask;
    int nCount = 0;
    for (int i = 0; i < matMask.rows; i++)
        for (int j =0 ; j < matMask.cols; j++)
            if(matIm(i,j)==255)nCount++;
    float res = nCount*1.0/(matMask.rows*matMask.cols);
    return res;
}



std::pair<int,int> find_maxIndexforHist(cv::Mat& matHist)
{
    cv::Mat_<uchar> matIm = matHist;
    int nMax = 0;
    int nindex = 0;
    for (int i=0;i<matHist.rows;i++)
    {
         if (matIm(i,0) > nMax)
         {
             nMax = matIm(i,0);
             nindex = i;
         }

    }

    return std::pair<int,int>(nMax,nindex);

}


struct CharBoxCmp
{
    bool operator()(cv::Rect &a, cv::Rect &b) const
    {
        
        return a.x > b.x;
    }
};

std::vector<cv::Rect> sort_charbox(
     std::vector<cv::Rect>& charboxes,
     std::array<int,2> tpbt_lim = {0,100000}
     )
{
     std::priority_queue<
          cv::Rect, 
          std::vector<cv::Rect>, 
          CharBoxCmp> sort_box;
    
     for (auto charbox:charboxes)
     {
           int cy = charbox.y + charbox.width/2;
           if (cy <= tpbt_lim[0] or cy >=tpbt_lim[1])continue;
           sort_box.push(charbox);
     }
     std::vector<cv::Rect> charbox_sort;

     while (!sort_box.empty()) {
          charbox_sort.push_back(sort_box.top());
          sort_box.pop();
     }

    //for (auto cbox:charbox_sort)std::cout << cbox.x << " ";
    
    return charbox_sort;

}

std::vector<cv::Rect> sort_Doublebox(
     std::vector<cv::Rect>& veCharboxes     
     )
{
     int nMax_temp = 0;
     int nMin_temp = 10000;
     int nMax_Diff = 0;

     for (auto charbox:veCharboxes)
     {
         if (charbox.y > nMax_temp)nMax_temp = charbox.y;
         if (charbox.y < nMin_temp)nMin_temp = charbox.y;
         if (charbox.x > nMax_Diff)nMax_Diff = charbox.x;
     }
     int nTop_num = 0;
     for (auto& charbox:veCharboxes)
     {
         if (charbox.y < 0.5*(nMax_temp + nMin_temp))
         {
             charbox.x-= nMax_Diff;
             nTop_num ++;
         }
     }
     std::vector<cv::Rect> veBox_sort = sort_charbox(veCharboxes);
     std::for_each(veBox_sort.begin(),veBox_sort.begin()+nTop_num,[&](cv::Rect& a){a.x += nMax_Diff;});
     return veBox_sort;
}


template <class T>
bool check_element(
     std::vector<T>& spe_list,
     T value
     )
{

     auto iter = std::find(std::begin(spe_list), std::end(spe_list), value);
     return (iter != std::end (spe_list))?true:false;

}



float cal_iou(
      cv::Rect reRect_1,
      cv::Rect reRect_2
      )
{

     auto cal_max = [](int a,int b){return (a>b)?a:b;};
     auto cal_min = [](int a,int b){return (a<b)?a:b;};
     
     int nCross_W = cal_min(reRect_1.x+reRect_1.width,reRect_2.x+reRect_2.width)-
                    cal_max(reRect_1.x,reRect_2.x);
     int nCross_H = cal_min(reRect_1.y+reRect_1.height,reRect_2.y+reRect_2.height)-
                    cal_max(reRect_1.y,reRect_2.y);

     if (nCross_W <=0 || nCross_H <=0)return 0.f;
     

     int nArea_1 = reRect_1.width*reRect_1.height;
     int nArea_2 = reRect_2.width*reRect_2.height;
     int nCross_area = nCross_W * nCross_H;

     return nCross_area*1.0/(nArea_1+nArea_2-nCross_area);

}

int cal_resultcount(
    std::vector<std::string>& vePlate_result,
    std::string& stResult_plate
    )
{
    int nCount = 0;
    for (auto plres : vePlate_result)
        if(plres == stResult_plate)nCount++;
    return nCount++;
}

inline bool isIndeSend_buff(
    std::string s,
    std::deque<std::string>& deSend_buff
    )
{
    for (auto it=deSend_buff.begin();it<deSend_buff.end();it++)
    {
        if (s == *it)
        {
           //deSend_buff.erase(it);
           //std::cout << "delete" <<std::endl;
           //deSend_buff.push_back(s);
           return true;
        }
    }

    return false;

}




#endif
