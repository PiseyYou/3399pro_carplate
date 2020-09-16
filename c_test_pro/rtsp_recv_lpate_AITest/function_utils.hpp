
#ifndef __FUNCTION_UTILS_INCLUDED__
#define __FUNCTION_UTILS_INCLUDED__

#include <stdio.h>
#include <string>

void string_split(std::string& stString, std::vector<std::string>& veResult, std::string stSplit)
{
     string::size_type pos1,pos2;
     pos2 = stString.find(stSplit);
     pos1 = 0;

     while(string::npos != pos2)
     {
           veResult.push_back(stString.substr(pos1,pos2 - pos1));
           pos1 = pos2 + stSplit.size();
           pos2 = stString.find(stSplit,pos1);
     }
     if(pos1 != stString.length())veResult.push_back(stString.substr(pos1));

}

inline bool check_plateID(std::string& stRegres,std::vector<std::string>& veTargetID)
{
     
     for(auto plate_id:veTargetID)
     {  
         if (stRegres == plate_id)return true;
     }

     return false;
}



std::vector<std::string> Search_ID(std::vector<std::string>& veTarget)
{

     std::vector<std::string> veResult;
     ifstream ifFile("result_log.txt");

     std::string stS;
     while(getline(ifFile,stS))
     {
          std::vector<std::string> veSpResult;
          string_split(stS, veSpResult,"-");
          std::string stPlate_ID = veSpResult[veSpResult.size()-1];
          std::for_each(veTarget.begin(),veTarget.end(),[&](std::string stTs)
          {
               if (stTs == stPlate_ID) veResult.push_back(stS);
               
          });
     }
     return veResult;
}

std::string get_Ntp()
{
     struct timeval curTime;
     gettimeofday(&curTime,NULL);
     int milli = curTime.tv_usec/1000;

     char buffer[80] = {0};
     struct tm nowTime;
     localtime_r(&curTime.tv_sec, &nowTime);
     strftime(buffer,sizeof(buffer),"%Y-%m-%d %H:%M:%S:",&nowTime);

     char currentTime[84] = {0};
     snprintf(currentTime,sizeof(currentTime),"%s%03d",buffer,milli);
     //cv::putText(matImage,currentTime,cv::Point(100,200),cv::FONT_HERSHEY_PLAIN,2.0,cv::Scalar(255,255,255),4,8);

     std::string stNtpTime(currentTime);

     return stNtpTime;


}

#endif
