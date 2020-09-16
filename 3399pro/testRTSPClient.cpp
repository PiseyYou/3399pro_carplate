#include "liveMedia.hh"
//#include "RTSPClient.hh"
#include "BasicUsageEnvironment.hh"
#include "stdio.h"
#include <stdio.h>
#include <iostream>
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <bitset>
#include <string>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#include <time.h>
#include <fstream>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>

//*********************
#include "thread"
#include "mutex"
#include "condition_variable"
#include <locale.h>
#include <wchar.h> 
#include <fstream>
#include <codecvt>
#include <string>
//*********************
#include "lisence_plate_proceess.hpp"
#include "trackpro.hpp"
//
#include <stdlib.h>
#include <sys/time.h>
//#include "rkdrm.h"
#include "mppdecoder.h"
#include "arm_neon.h"

//rga head files
#include <RockchipRga.h>
#include <RockchipFileOps.h>
#include "omp.h"
//*********************
//#include <object.h>
//********************
#include <zmq.hpp>
#include "zhelpers.hpp"
#include "zmq_req.hpp"
#include "uart.h"
#include <iconv.h> 
#include <fcntl.h>  
//*********************
//#define Is_Record_Datas 1
//#define Is_Record_H264 1
// Forward function definitions:
#define TCP_PORT "5562"
#define IMAGE_TCP_PORT "5562"
using namespace cv;
using namespace std;
using namespace chrono;
int IMAGE_WIDTH = 1920;//1088;//1920;
int IMAGE_HEIGHT = 1080;//1920;//1080;
/* FPS calculator */
RK_U64 fps_ms2,fps_ms3;
RK_U64 start_ms,end_ms;
RK_U32 fps_counter2,fps_counter3;
float fps2 = 0.0,fps3 = 0.0;

int record_count = 0;
int record_flag = 1;
string dir11;
string dir22;

int VEHICLE_NUM = 0;

std::ofstream fw_remove;
bool is_begin_write = false;

extern int is_restart_client;
extern mutex mutex_basedata;
extern condition_variable cond_basedata;
extern bool gFlagNewBasedata;
RK_U8* base_data;
unsigned char *bgr_data;

#define PACK_SIZE 20
#define BUFFER_SIZE 3
std::string serial_dev = "/dev/ttyS4";
int bardrate_dev = 115200;
int uart_fd;
std::deque<std::string> deSend_buff;
std::mutex muLock;
bool IS_Receive_Message = false;
unsigned char ReceiveBuffer[21];
unsigned char NoDataBuffer[PACK_SIZE];
unsigned char Sendbuffer[5+16*BUFFER_SIZE];
bool isInitCamera = false;

void thread_rtsp_mpp();
void thread_uart();
// RTSP 'response handlers':
void continueAfterDESCRIBE(RTSPClient* rtspClient, int resultCode, char* resultString);
void continueAfterSETUP(RTSPClient* rtspClient, int resultCode, char* resultString);
void continueAfterPLAY(RTSPClient* rtspClient, int resultCode, char* resultString);

// Other event handler functions:
void subsessionAfterPlaying(void* clientData); // called when a stream's subsession (e.g., audio or video substream) ends
void subsessionByeHandler(void* clientData); // called when a RTCP "BYE" is received for a subsession
void streamTimerHandler(void* clientData);
  // called at the end of a stream's expected duration (if the stream has not already signaled its end using a RTCP "BYE")

// The main streaming routine (for each "rtsp://" URL):
void openURL(UsageEnvironment& env, char const* progName, char const* rtspURL);

// Used to iterate through each stream's 'subsessions', setting up each one:
void setupNextSubsession(RTSPClient* rtspClient);

// Used to shut down and close a stream (including its "RTSPClient" object):
void shutdownStream(RTSPClient* rtspClient, int exitCode = 1);

// A function that outputs a string that identifies each stream (for debugging output).  Modify this if you wish:
UsageEnvironment& operator<<(UsageEnvironment& env, const RTSPClient& rtspClient) {
  return env << "[URL:\"" << rtspClient.url() << "\"]: ";
}

// A function that outputs a string that identifies each subsession (for debugging output).  Modify this if you wish:
UsageEnvironment& operator<<(UsageEnvironment& env, const MediaSubsession& subsession) {
  return env << subsession.mediumName() << "/" << subsession.codecName();
}


//*************************************************************************************//


std::map <std::string,unsigned char> color_code = {{"蓝",0x00},{"黄",0x01},{"黑",0x02},{"白",0x03},{"绿",0x04},{"黄绿",0x05}};


int code_convert(char *from_charset, char *to_charset, char *inbuf, size_t inlen,  
        char *outbuf, size_t outlen) 
{  
    iconv_t cd;  
    char **pin = &inbuf;  
    char **pout = &outbuf;  
  
    cd = iconv_open(to_charset, from_charset);  
    if (cd == 0)  
        return -1;  
    memset(outbuf, 0, outlen);  
    if (iconv(cd, pin, &inlen, pout, &outlen) == -1)  
        return -1;  
    iconv_close(cd);  
    *pout = "\0";  
  
    return 0;  
}  
  
int u2g(char *inbuf, size_t inlen, char *outbuf, size_t outlen) {  
    return code_convert("utf-8", "gb2312", inbuf, inlen, outbuf, outlen);  
}  
  
int g2u(char *inbuf, size_t inlen, char *outbuf, size_t outlen) {  
    return code_convert("gb2312", "utf-8", inbuf, inlen, outbuf, outlen);  
}  


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




struct Content {
			uchar img_data[2764800];
			char timestamp[32];
		};
void send_image(cv::Mat& frame, zmq::socket_t& publisher,std::string zmq_saveName)
{
	
	auto start = chrono::steady_clock::now();
	
		size_t frameSize = frame.step[0] * frame.rows;

		Content msgs;

		//string number = std::to_string(img_time);
		strcpy(msgs.timestamp,zmq_saveName.c_str());

		memcpy(&msgs.img_data,  frame.data,frameSize );
		//(msgs)->img_data = frame.data;

		zmq::message_t message( sizeof(Content) );

		memcpy(message.data(),  &msgs,sizeof(Content));

		s_sendmore (publisher, "Image");
		publisher.send(message);
         

}



void save_Imgresult(cv::Mat& saveImg,zmq::socket_t& publisher,char* result,long int timpstamp = 0)
{
    const char* basePath = "/home/toybrick/lisk_project/mnt/Data/";
    char tmp[64];
    char tmp_2[32];
    time_t timep;
    time(&timep);
    strftime(tmp,sizeof(tmp),"%Y%m%d%H%M%S",localtime(&timep));
    strftime(tmp_2,sizeof(tmp_2),"%Y%m%d",localtime(&timep));

    char* fileName = tmp_2;
    string filepathstr = string(basePath) + string(fileName);
    const char* fileNamePath = filepathstr.c_str();
/*
    struct stat file_stat;
    stat(basePath,&file_stat);
    if(errno == ENOENT)mkdir(basePath,0775);
    stat(fileNamePath,&file_stat);
    if(errno == ENOENT)mkdir(fileNamePath,0775);
*/

    std::string timestr(tmp);    
    std::string res(result);
  
    std::string zmq_saveName = /*timestr + "_" +*/ res;
    send_image(saveImg,publisher,zmq_saveName);

    std::string savePath = filepathstr + "/" + timestr + "_" + res + ".jpg";
    //cv::imwrite(savePath,saveImg);

    std::string saveCmpPath = "../Save_img/";
    //std::string fsave_path = saveCmpPath + std::to_string(timpstamp) + "_" + res + "_" + std::to_string(VEHICLE_NUM)+".jpg";
    std::string fsave_path = saveCmpPath + timestr + "_" + res + "_" + std::to_string(VEHICLE_NUM)+".jpg";
    //cv::imwrite(fsave_path,saveImg);

}

void ComBineBuffer_init(
     unsigned char* Sendbuf
)
{

    Sendbuf[0] = 0xaa;
    Sendbuf[1] = 0x13;
    Sendbuf[2] = 0xe1;
    Sendbuf[3] = 0x00;
    for(int k=4; k < 53;k++) Sendbuf[k] = 0x00;


}

void ComBineMessage(
    std::deque<std::string>& debuffer,
    unsigned char* Sendbuf,
    int &ndix
    )
{

    unsigned char* Sendbuf_temp = Sendbuf;
    Sendbuf += 4; //skip to the license index
    int nSizeBuf = debuffer.size();
    while(nSizeBuf!=0)
    {
       Sendbuf_temp[3]++;
       std::string message = debuffer[0];//.back();
       std::vector<std::string> veResut;
       string_split(message,veResut,"-");
       std::string stColor = veResut[2];
       std::string stLicense = veResut[3];
       if (stLicense != "无车牌")
       {
           char *buf = const_cast<char*>(stLicense.c_str());
	   char bfff[12];
           u2g(buf, strlen(buf), bfff, sizeof(bfff));  
           std::cout << bfff << std::endl;
           //printf("%02x%02x\n",bfff[0],bfff[1]);
			
           strcpy((char*)(Sendbuf),static_cast<char*>(bfff));
           Sendbuf += 12;
           *Sendbuf=color_code.at(stColor);
       }
       else
       {
           memset((char*)(Sendbuf),0,13);
           Sendbuf += 12;
       } 
       ndix++;
       Sendbuf += 4;
       debuffer.pop_front();
       if (ndix==nSizeBuf)break;
       std::cout<< "ndix " << ndix <<std::endl;
    }

    
    if(ndix == 0)  //if they send the request while the buf having no plate,send the non_plate result
    {
       Sendbuf += 16;
       ndix = 1;
       Sendbuf_temp[3] = 0x01;
    }
    else if(ndix == 1)
    {
       Sendbuf_temp[1] = 0x13;
    }
    else
    {
       Sendbuf_temp[1] = 0x23;
    }

     Sendbuf_temp += 1;
     while(Sendbuf!=Sendbuf_temp)
     {
         *Sendbuf = (*Sendbuf)^(*Sendbuf_temp);
         Sendbuf_temp++;
     }


}

//*****************************************************


void usage(UsageEnvironment& env, char const* progName) {
  env << "Usage: " << progName << " <rtsp-url-1> ... <rtsp-url-N>\n";
  env << "\t(where each <rtsp-url-i> is a \"rtsp://\" URL)\n";
}

char eventLoopWatchVariable = 0;
RTSPClient* rtspClient_copy;
int init_mpp = 0;

#define DEBUG
int RECORD = 1;
//*************************************************************************************//

int main(int argc, char** argv) { 

        
        cpu_set_t mask;
        CPU_ZERO(&mask);
        CPU_SET(0,&mask);
        CPU_SET(1,&mask);
        CPU_SET(2,&mask);
        CPU_SET(3,&mask);
        if (sched_setaffinity(0,sizeof(mask),&mask) == -1){
            printf("error_set!\n");
        }

	//1. 打开串口设备  
	uartdev *udev = new uartdev((const char *)(serial_dev.data()), bardrate_dev); //新建一个串口设备
	uart_fd = udev->open_dev();  //打开串口设备

	if (uart_fd>0 && udev->set_parity(uart_fd,8,1,'N')== -1)  // 8位数据，非两位的停止位，不使用奇偶校验 ，不允许输入奇偶校验
	{
	  printf("Set Parity Error\n");
	  return 0;
	}
 	cout << "set Parity success." << endl;
	for(int i=0; i < PACK_SIZE;i++) ReceiveBuffer[i] = 0x00;
	NoDataBuffer[0] = 0xaa;
	NoDataBuffer[1] = 0x12;
        NoDataBuffer[2] = 0xe1;
	for(int i=3; i < PACK_SIZE;i++) NoDataBuffer[i] = 0x00;
	for(int i=1;i<19;i++)
	{
		NoDataBuffer[19] = NoDataBuffer[19] ^ NoDataBuffer[i];
	}
        ComBineBuffer_init(Sendbuffer);

	sleep(1);
	// Begin by setting up our usage environment:
        std::cout << "1111" << std::endl;
	TaskScheduler* scheduler = BasicTaskScheduler::createNew();
	UsageEnvironment* env = BasicUsageEnvironment::createNew(*scheduler);
        std::cout << "2222" << std::endl;
	// We need at least one "rtsp://" URL argument:
	if (argc < 2)
	{
		usage(*env, argv[0]);
		return 1;
	}
	//sleep(5);
        
	thread rtsp_mpp(thread_rtsp_mpp);
        thread uart_thread(thread_uart);
        
	//************************************************//
	while(1)
	{
	// There are argc-1 URLs: argv[1] through argv[argc-1].  Open and start streaming each one:		
		for (int i = 1; i <= argc-1; ++i) 
		{			
			openURL(*env, argv[0], argv[i]);
			
		}

		// All subsequent activity takes place within the event loop:             
		env->taskScheduler().doEventLoop(&eventLoopWatchVariable);
		// This function call does not return, unless, at some point in time, "eventLoopWatchVariable" gets set to something non-zero.
		if(is_restart_client == 1) 
		{
			shutdownStream(rtspClient_copy);
		}
		isInitCamera = false;
		sleep(30);
	}

	
	//*****************************************************//	
	rtsp_mpp.join();
	uart_thread.join();
	return 0;
}

// uart通道数据接收函数
void thread_uart()
{	
	unsigned char buff[PACK_SIZE];
	unsigned char *temp;
	int nread =0;
	int rcvcount =0;
	//UART_DW_BUF_CB uart_cb;
	//uart_cb.RxState = IDLE;
	int buffcount = 0;
	double t3;
	while(true)
	{	
			
		nread = read(uart_fd,buff,10);
		/*msleep(300);*/

                nread = 5;
		//simulate client reauest cmd, lock the uart_buff, 
                buff[0]=0xaa;buff[1]=0x03;buff[2]=0xf1;buff[3]=0x00;buff[4]=0xf2;



		if(nread > 0)
		{
			//cout << "read buff:"  << buff << endl;
			//cout << "nread:" << nread << endl;
			if(nread == 5)
			{
				if(buff[0]==0xaa && buff[1]==0x03 && buff[2]==0xf0 && buff[3]==0x00 && buff[4]==0xf3)
				{
					if(isInitCamera)
					{
						unsigned char rcvBuff[5];
						rcvBuff[0] = 0xaa;
						rcvBuff[1] = 0x03;
						rcvBuff[2] = 0xe0;
						rcvBuff[3] = 0x01;
						rcvBuff[4] = 0xe2;
						int bits = 0;
		      				bits = write(uart_fd,rcvBuff,5);  //在串口上发送数据，发送读命令
					}
					else
					{
						unsigned char rcvBuff[5];
						rcvBuff[0] = 0xaa;
						rcvBuff[1] = 0x03;
						rcvBuff[2] = 0xe0;
						rcvBuff[3] = 0x00;
						rcvBuff[4] = 0xe3;
						int bits = 0;
		      				bits = write(uart_fd,rcvBuff,5);  //在串口上发送数据，发送读命令
					}
				}
                                /*
				else if(buff[0]==0xaa && buff[1]==0x03 && buff[2]==0xf1 && buff[3]==0x00 && buff[4]==0xf2)
				{
					if(IS_Receive_Message)
					{
						int bits = 0;
		      				bits = write(uart_fd,ReceiveBuffer,20);  //在串口上发送数据，发送读命令
						//IS_Receive_Message = false;
					}
					else
					{
						int bits = 0;
		      				bits = write(uart_fd,NoDataBuffer,20);  //在串口上发送数据，发送读命令
					}
				}
                                */
                                else if(buff[0]==0xaa && buff[1]==0x03 && buff[2]==0xf1 && buff[3]==0x00 && buff[4]==0xf2)
                                {
                                       muLock.lock();
                                       ComBineBuffer_init(Sendbuffer);
                                       int ndix = 0;
                                       ComBineMessage(deSend_buff,Sendbuffer,ndix);
                                       muLock.unlock();
                                       int bits = 0;
                                       std::cout << "send info: ";
                                       for(int i=0;i<5+ndix*16;i++) 
        		               {
	    			           printf("%02x",Sendbuffer[i]);
	
        		               }
                                       printf("\n");
		      		       bits = write(uart_fd,Sendbuffer,5+ndix*16);  //在串口上发送数据，发送读命令
                                       sleep(20);
                                }
			}
			else usleep(10);
		}
		else usleep(10);
	}
}

void thread_rtsp_mpp()
{       

        cpu_set_t mask;
        CPU_ZERO(&mask);

        //CPU_SET(5,&mask);
        //CPU_SET(2,&mask);
        //CPU_SET(3,&mask);
        CPU_SET(5,&mask);
        CPU_SET(4,&mask);
        std::cout << "xxxx" << std::endl;
        
        int nReg_w = 1280;//720;//1280;
        int nReg_h = 720;//1280;//720;
        //uchar *CArrays = new uchar[nReg_w*nReg_h*3];
	//创建车牌检测与车牌确认线程		
	fps_ms2 = current_ms();
    	fps_counter2 = 0;
	bool first_init = true;
	bgr_data = (unsigned char*)malloc(sizeof(unsigned char) * IMAGE_WIDTH * IMAGE_HEIGHT * 3);
	
	//rga src init
	int ret = 0;
	int srcFormat;
	int dstFormat;
	Mat imgSrc;
	bo_t bo_src, bo_dst;
	
	srcFormat = RK_FORMAT_YCrCb_420_SP;  //NV12
	dstFormat = RK_FORMAT_RGB_888;
	
	RockchipRga rkRga;
	rkRga.RkRgaInit();	
	//int vReadNum=0;
	//int srcHeight = 290;
	//int dstWidth = 1280;
    	//int dstHeight = 480;
	//int framesNum=0;


        zmq::context_t context(1);
	zmq::socket_t Reqer(context, ZMQ_PUB);
	string tcp_s = "tcp://192.168.3.100:5555";
        Reqer.connect(tcp_s);
        //req_queue(Reqer);
        while(1)
        {
           //if (req_img(Reqer))break;
           //else sleep(10);
           break;
        }


        if (pthread_setaffinity_np(pthread_self(),sizeof(mask),&mask) < 0){
            perror("pthead_setaffinity_np");
        }

        cv::Mat pad_img = cv:: Mat::zeros(1280, 280, CV_8UC3);
        cv::Mat save_img(nReg_h,nReg_w,CV_8UC3);
        cv::Mat matCropPl;

        char* res;
        long int timestamp;
        //LP_Recognition a = LP_Recognition();
        trackSystem_pro b = trackSystem_pro();
        std::cout << "begin" << std::endl;
        //clock_t start = clock();
        //sleep(5);
        //clock_t end = clock();
        //std::cout << "time cost " << (double)(end - start)/200000 << std::endl;
        
        //long int last_timestamp = 0;
        //char* last_chRect_res = "0,0,0,0";
        //cv::Scalar last_scMean = {0,0,0};
        int nSend_c = 0;
        long int liTimestamp_save = 0;
        //std::deque<std::tuple<long int,char*,cv::Scalar>> deSend_buff;
        bool Ispush = false;
	while(1)
	{	
                //nSend_c++;
		unique_lock<mutex> lock_basedata(mutex_basedata);
		cond_basedata.wait(lock_basedata,[](){return gFlagNewBasedata; });
		base_data = decoder_frame(timestamp);
		gFlagNewBasedata = false;
		lock_basedata.unlock();

                isInitCamera = true;
		RK_U32 frame_width    = nReg_w;
      		RK_U32 frame_height   = nReg_h;
     		RK_U32 h_stride = IMAGE_WIDTH;
      		RK_U32 v_stride = IMAGE_HEIGHT;
		//lock_basedata.unlock();

		rga_info_t src;
    		rga_info_t dst;
		
    		memset(&src, 0, sizeof(rga_info_t));
    		src.fd = -1;
    		src.mmuFlag = 1;
		src.virAddr = (unsigned char*)(base_data + 8*1920);
		
    		memset(&dst, 0, sizeof(rga_info_t));
    		dst.fd = -1;
    		dst.mmuFlag = 1;
		dst.virAddr = bgr_data;
		
		rga_set_rect(&src.rect, 0,0,IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_WIDTH,IMAGE_HEIGHT,srcFormat);
		rga_set_rect(&dst.rect, 0,0,nReg_w,nReg_h,nReg_w,nReg_h,dstFormat);

		ret = rkRga.RkRgaBlit(&src, &dst, NULL);
		cv::Mat img(nReg_h,nReg_w,CV_8UC3,bgr_data);
                
                //clock_t start = clock();
                //a.LP_main_process(img);
                
                //cv::hconcat(pad_img,img,img);
                //cv::hconcat(img,pad_img,img);
		
                std::string stResx = b.track_pro(img);
                //clock_t end = clock();
                //std::cout << "time cost " << (double)(end - start)/200000 << std::endl;
                /*
                cv::Scalar scMean = cv::mean(img);
                char* chRect_res = const_cast<char*>(stResx.c_str());
                //std::cout << chRect_res << std::endl;
                //send_result(Reqer, timestamp, chRect_res, scMean);
                
                if (deSend_buff.size()==2)
                {
                    //deSend_buff[0]
                    last_timestamp = std::get<0>(deSend_buff[0]);
                    last_chRect_res = std::get<1>(deSend_buff[0]);
                    last_scMean = std::get<2>(deSend_buff[0]);
                    send_result(Reqer, last_timestamp, last_chRect_res, last_scMean);
                    deSend_buff.pop_front();
                    
                }
                deSend_buff.push_back(std::tuple<long int,char*,cv::Scalar>(timestamp,chRect_res,scMean));
                */
                //std::string stRes = "Null";
                
                if (stResx != "Null")
                {

                    auto pos = stResx.find_first_of("-");
                    std::string stLayer_type = (pos!=std::string::npos)?stResx.substr(0,pos):"";
                    //std::cout << stLayer_type << std::endl;
                    if (stResx == "save_img")
                    {
                        img.copyTo(save_img);
                        liTimestamp_save = timestamp;
                        //matCropPl = save_img(b.reCrop_plate_save);
                        //cout << b.reCrop_plate_save.x << " " << b.reCrop_plate_save.y << " " << b.reCrop_plate_save.width << " " << b.reCrop_plate_save.height << endl;
                    }
                    
                    else 
                    {
                        //std::cout<<"the result is "<<stResx<<std::endl;                

			ReceiveBuffer[0] = 0xaa;
			ReceiveBuffer[1] = 0x13;
                        ReceiveBuffer[2] = 0xe1;
                        ReceiveBuffer[3] = 0x01;
                        std::vector<std::string> veResut;
                        string_split(stResx,veResut,"-");
                        std::string stColor = veResut[2];
                        std::string stLicense = veResut[3];
                        std::string stCurrent = veResut[4];
                        //std::cout<< stColor << "  " << stLicense <<std::endl;
                        //stLicense = "无车牌";
                        if (stLicense != "无车牌")
                        {
                           char *buf = const_cast<char*>(stLicense.c_str());
	                   char bfff[12];
                           u2g(buf, strlen(buf), bfff, sizeof(bfff));  
                           //std::cout << bfff << std::endl;
                           //printf("%02x%02x\n",bfff[0],bfff[1]);
                           strcpy((char*)(ReceiveBuffer+4),static_cast<char*>(bfff));

                           ReceiveBuffer[16]=color_code.at(stColor);
                        }
                        else
                        {
                           memset((char*)(ReceiveBuffer+4),0,13);
                        }

			ReceiveBuffer[17] = 0x00;
			ReceiveBuffer[18] = 0x00;
			ReceiveBuffer[19] = 0x00;
			ReceiveBuffer[20] = 0x00;
                     
			for(int i=1;i<20;i++)
			{       
				ReceiveBuffer[20] = ReceiveBuffer[20] ^ ReceiveBuffer[i];
			} 
            
			//unsigned char *buf = "粤";
			//printf("%02x%02x\n",buf[0],buf[1]);
                        //ReceiveBuffer[3] = 0xd4;
                        //ReceiveBuffer[4] = 0xc1;
                        printf("*******************************\n");
			for(int i=0;i<21;i++) 
        		{
	    			printf("%02x",ReceiveBuffer[i]);
	
        		}
			printf("\n");
			start_ms = current_ms();

			if(!IS_Receive_Message) IS_Receive_Message = true;
                       
                        muLock.lock();
                        /*
                        if (!isIndeSend_buff(stResx,deSend_buff))
                        {
                            if (deSend_buff.size()==BUFFER_SIZE)deSend_buff.pop_front();
                            deSend_buff.push_back(stResx);
                        }
                        */
                        if(Ispush || deSend_buff.empty())
                        {
                           std::cout << "push" << std::endl;
                           Ispush = false;
                           if (deSend_buff.size()==BUFFER_SIZE)deSend_buff.pop_front();
                           deSend_buff.push_back(stResx);
                        }
                        else
                        {
                           std::cout << "change" << std::endl;
                           deSend_buff.pop_back();
                           deSend_buff.push_back(stResx);
                        }
                        

                        if (stCurrent == "1") Ispush = true;                  
   
                        muLock.unlock();

                        //test combinebuffer
                        /*
                        ComBineBuffer_init(Sendbuffer);
                        std::cout << "init done" << std::endl;
                        int nidx = 0;
                        ComBineMessage(deSend_buff,Sendbuffer,nidx);
                        std::cout << "combine done" << std::endl;
                        
                        for(int i=0;i<5+16*BUFFER_SIZE;i++) 
        		{
	    			printf("%02x",Sendbuffer[i]);
	
        		}
                        printf("\n");
                        */
                        std::cout << "buf" << std::endl;
                        for (auto buf:deSend_buff)std::cout << buf << std::endl;

                        //VEHICLE_NUM++;
                        //send_result(Reqer, timestamp, recs);
                        
                        //cv::imshow("result",save_img);
                        /*
                        if (matCropPl.rows >10)
                        {
                            cv::namedWindow("matCropPl",CV_WINDOW_NORMAL);
                            cv::imshow("matCropPl",matCropPl);
                        }
                        */
                        //char* chRes = const_cast<char*>(stResx.c_str());
                        //save_Imgresult(save_img,Reqer,chRes,timestamp);                       

                    }
                }
		end_ms = current_ms();
                /*
                if (IS_Receive_Message && b.boIsleave == true && (end_ms-start_ms) > 1000)
                {
                    std::cout<<"vehicle has left" << std::endl;
		    IS_Receive_Message = false; 
                    b.boIsleave == false;
                }
   		*/
		imshow("bgr",img);
		waitKey(1);		
		double t1 = 0;
		int nPlateNum = 0;
		fps_counter2 ++;
		RK_U64 diff, now = current_ms();
                if ((diff = now - fps_ms2) >= 1000.0) 
		{
                    fps2 = fps_counter2 / (diff / 1000.0);
                    fps_counter2 = 0;
                    fps_ms2 = now;
                    printf("bgr resize FPS = %3.2f\n", fps2);		
		   		
		   
                }
	        //img.release();
		//********************************************************************
		
        }

	free(bgr_data);
	bgr_data = NULL;	
	std::cerr<<"finish."<<std::endl;	
	//return 0;
}


// Define a class to hold per-stream state that we maintain throughout each stream's lifetime:

class StreamClientState {
public:
  StreamClientState();
  virtual ~StreamClientState();

public:
  MediaSubsessionIterator* iter;
  MediaSession* session;
  MediaSubsession* subsession;
  TaskToken streamTimerTask;
  double duration;
};

// If you're streaming just a single stream (i.e., just from a single URL, once), then you can define and use just a single
// "StreamClientState" structure, as a global variable in your application.  However, because - in this demo application - we're
// showing how to play multiple streams, concurrently, we can't do that.  Instead, we have to have a separate "StreamClientState"
// structure for each "RTSPClient".  To do this, we subclass "RTSPClient", and add a "StreamClientState" field to the subclass:

class ourRTSPClient: public RTSPClient {
public:
  static ourRTSPClient* createNew(UsageEnvironment& env, char const* rtspURL,
				  int verbosityLevel = 0,
				  char const* applicationName = NULL,
				  portNumBits tunnelOverHTTPPortNum = 0);

protected:
  ourRTSPClient(UsageEnvironment& env, char const* rtspURL,
		int verbosityLevel, char const* applicationName, portNumBits tunnelOverHTTPPortNum);
    // called only by createNew();
  virtual ~ourRTSPClient();

public:
  StreamClientState scs;
};

// Define a data sink (a subclass of "MediaSink") to receive the data for each subsession (i.e., each audio or video 'substream').
// In practice, this might be a class (or a chain of classes) that decodes and then renders the incoming audio or video.
// Or it might be a "FileSink", for outputting the received data into a file (as is done by the "openRTSP" application).
// In this example code, however, we define a simple 'dummy' sink that receives incoming data, but does nothing with it.

class DummySink: public MediaSink {
public:
  static DummySink* createNew(UsageEnvironment& env,
			      MediaSubsession& subsession, // identifies the kind of data that's being received
			      char const* streamId = NULL); // identifies the stream itself (optional)

private:
  DummySink(UsageEnvironment& env, MediaSubsession& subsession, char const* streamId);
    // called only by "createNew()"
  virtual ~DummySink();

  static void afterGettingFrame(void* clientData, unsigned frameSize,
                                unsigned numTruncatedBytes,
				struct timeval presentationTime,
                                unsigned durationInMicroseconds);
  void afterGettingFrame(unsigned frameSize, unsigned numTruncatedBytes,
			 struct timeval presentationTime, unsigned durationInMicroseconds);

private:
  // redefined virtual functions:
  virtual Boolean continuePlaying();

private:
  u_int8_t* fReceiveBuffer;
  MediaSubsession& fSubsession;
  char* fStreamId;
};

#define RTSP_CLIENT_VERBOSITY_LEVEL 1 // by default, print verbose output from each "RTSPClient"

static unsigned rtspClientCount = 0; // Counts how many streams (i.e., "RTSPClient"s) are currently in use.

void openURL(UsageEnvironment& env, char const* progName, char const* rtspURL) {
  // Begin by creating a "RTSPClient" object.  Note that there is a separate "RTSPClient" object for each stream that we wish
  // to receive (even if more than stream uses the same "rtsp://" URL).
  RTSPClient* rtspClient = ourRTSPClient::createNew(env, rtspURL, RTSP_CLIENT_VERBOSITY_LEVEL, progName);
  rtspClient_copy = rtspClient;
  if (rtspClient == NULL) {
    env << "Failed to create a RTSP client for URL \"" << rtspURL << "\": " << env.getResultMsg() << "\n";
    return;
  }

  ++rtspClientCount;

  // Next, send a RTSP "DESCRIBE" command, to get a SDP description for the stream.
  // Note that this command - like all RTSP commands - is sent asynchronously; we do not block, waiting for a response.
  // Instead, the following function call returns immediately, and we handle the RTSP response later, from within the event loop:
  rtspClient->sendDescribeCommand(continueAfterDESCRIBE); 
}


// Implementation of the RTSP 'response handlers':

void continueAfterDESCRIBE(RTSPClient* rtspClient, int resultCode, char* resultString) {
  do {
    UsageEnvironment& env = rtspClient->envir(); // alias
    StreamClientState& scs = ((ourRTSPClient*)rtspClient)->scs; // alias

    if (resultCode != 0) {
      env << *rtspClient << "Failed to get a SDP description: " << resultString << "\n";
      is_restart_client = 2;
      delete[] resultString;
      break;
    }

    char* const sdpDescription = resultString;
    env << *rtspClient << "Got a SDP description:\n" << sdpDescription << "\n";

    // Create a media session object from this SDP description:
    scs.session = MediaSession::createNew(env, sdpDescription);
    delete[] sdpDescription; // because we don't need it anymore
    if (scs.session == NULL) {
      env << *rtspClient << "Failed to create a MediaSession object from the SDP description: " << env.getResultMsg() << "\n";
      break;
    } else if (!scs.session->hasSubsessions()) {
      env << *rtspClient << "This session has no media subsessions (i.e., no \"m=\" lines)\n";
      break;
    }

    // Then, create and set up our data source objects for the session.  We do this by iterating over the session's 'subsessions',
    // calling "MediaSubsession::initiate()", and then sending a RTSP "SETUP" command, on each one.
    // (Each 'subsession' will have its own data source.)
    scs.iter = new MediaSubsessionIterator(*scs.session);
    setupNextSubsession(rtspClient);
    return;
  } while (0);

  // An unrecoverable error occurred with this stream.
  shutdownStream(rtspClient);
}

// By default, we request that the server stream its data using RTP/UDP.
// If, instead, you want to request that the server stream via RTP-over-TCP, change the following to True:
#define REQUEST_STREAMING_OVER_TCP true

void setupNextSubsession(RTSPClient* rtspClient) {
  UsageEnvironment& env = rtspClient->envir(); // alias
  StreamClientState& scs = ((ourRTSPClient*)rtspClient)->scs; // alias
  
  scs.subsession = scs.iter->next();
  if (scs.subsession != NULL) {
    if (!scs.subsession->initiate()) {
      env << *rtspClient << "Failed to initiate the \"" << *scs.subsession << "\" subsession: " << env.getResultMsg() << "\n";
      setupNextSubsession(rtspClient); // give up on this subsession; go to the next one
    } else {
      env << *rtspClient << "Initiated the \"" << *scs.subsession << "\" subsession (";
      if (scs.subsession->rtcpIsMuxed()) {
	env << "client port " << scs.subsession->clientPortNum();
      } else {
	env << "client ports " << scs.subsession->clientPortNum() << "-" << scs.subsession->clientPortNum()+1;
      }
      env << ")\n";

      // Continue setting up this subsession, by sending a RTSP "SETUP" command:
	
      rtspClient->sendSetupCommand(*scs.subsession, continueAfterSETUP, False, REQUEST_STREAMING_OVER_TCP);
    }
    return;
  }

  // We've finished setting up all of the subsessions.  Now, send a RTSP "PLAY" command to start the streaming:
  if (scs.session->absStartTime() != NULL) {
    // Special case: The stream is indexed by 'absolute' time, so send an appropriate "PLAY" command:
    rtspClient->sendPlayCommand(*scs.session, continueAfterPLAY, scs.session->absStartTime(), scs.session->absEndTime());
  } else {
    scs.duration = scs.session->playEndTime() - scs.session->playStartTime();
    rtspClient->sendPlayCommand(*scs.session, continueAfterPLAY);
  }
}

void continueAfterSETUP(RTSPClient* rtspClient, int resultCode, char* resultString) {
  do {
    UsageEnvironment& env = rtspClient->envir(); // alias
    StreamClientState& scs = ((ourRTSPClient*)rtspClient)->scs; // alias
	
    if (resultCode != 0) {
	cout<<"******resultCode*****"<<resultCode<<"\n"<<endl;
      env << *rtspClient << "Failed to set up the \"" << *scs.subsession << "\" subsession: " << resultString << "\n";
      break;
    }

    env << *rtspClient << "Set up the \"" << *scs.subsession << "\" subsession (";
    if (scs.subsession->rtcpIsMuxed()) {
      env << "client port " << scs.subsession->clientPortNum();
    } else {
      env << "client ports " << scs.subsession->clientPortNum() << "-" << scs.subsession->clientPortNum()+1;
    }
    env << ")\n";

    // Having successfully setup the subsession, create a data sink for it, and call "startPlaying()" on it.
    // (This will prepare the data sink to receive data; the actual flow of data from the client won't start happening until later,
    // after we've sent a RTSP "PLAY" command.)

    scs.subsession->sink = DummySink::createNew(env, *scs.subsession, rtspClient->url());
      // perhaps use your own custom "MediaSink" subclass instead
    if (scs.subsession->sink == NULL) {
      env << *rtspClient << "Failed to create a data sink for the \"" << *scs.subsession
	  << "\" subsession: " << env.getResultMsg() << "\n";
      break;
    }

    env << *rtspClient << "Created a data sink for the \"" << *scs.subsession << "\" subsession\n";
    scs.subsession->miscPtr = rtspClient; // a hack to let subsession handler functions get the "RTSPClient" from the subsession 
    scs.subsession->sink->startPlaying(*(scs.subsession->readSource()),
				       subsessionAfterPlaying, scs.subsession);
    // Also set a handler to be called if a RTCP "BYE" arrives for this subsession:
    if (scs.subsession->rtcpInstance() != NULL) {
      scs.subsession->rtcpInstance()->setByeHandler(subsessionByeHandler, scs.subsession);
    }
  } while (0);
  delete[] resultString;

  // Set up the next subsession, if any:
  setupNextSubsession(rtspClient);
}

void continueAfterPLAY(RTSPClient* rtspClient, int resultCode, char* resultString) {
  Boolean success = False;

  do {
    UsageEnvironment& env = rtspClient->envir(); // alias
    StreamClientState& scs = ((ourRTSPClient*)rtspClient)->scs; // alias

    if (resultCode != 0) {
      env << *rtspClient << "Failed to start playing session: " << resultString << "\n";
      break;
    }

    // Set a timer to be handled at the end of the stream's expected duration (if the stream does not already signal its end
    // using a RTCP "BYE").  This is optional.  If, instead, you want to keep the stream active - e.g., so you can later
    // 'seek' back within it and do another RTSP "PLAY" - then you can omit this code.
    // (Alternatively, if you don't want to receive the entire stream, you could set this timer for some shorter value.)
    if (scs.duration > 0) {
      unsigned const delaySlop = 2; // number of seconds extra to delay, after the stream's expected duration.  (This is optional.)
      scs.duration += delaySlop;
      unsigned uSecsToDelay = (unsigned)(scs.duration*1000000);
      scs.streamTimerTask = env.taskScheduler().scheduleDelayedTask(uSecsToDelay, (TaskFunc*)streamTimerHandler, rtspClient);
    }
	
    env << *rtspClient << "Started playing session";
    if (scs.duration > 0) {
      env << " (for up to " << scs.duration << " seconds)";
    }
    env << "...\n";

    success = True;
  } while (0);
  delete[] resultString;

  if (!success) {
    // An unrecoverable error occurred with this stream.
    shutdownStream(rtspClient);
  }
}


// Implementation of the other event handlers:

void subsessionAfterPlaying(void* clientData) {
  MediaSubsession* subsession = (MediaSubsession*)clientData;
  RTSPClient* rtspClient = (RTSPClient*)(subsession->miscPtr);

  // Begin by closing this subsession's stream:
  Medium::close(subsession->sink);
  subsession->sink = NULL;

  // Next, check whether *all* subsessions' streams have now been closed:
  MediaSession& session = subsession->parentSession();
  MediaSubsessionIterator iter(session);
  while ((subsession = iter.next()) != NULL) {
    if (subsession->sink != NULL) return; // this subsession is still active
  }

  // All subsessions' streams have now been closed, so shutdown the client:
  shutdownStream(rtspClient);
}

void subsessionByeHandler(void* clientData) {
  MediaSubsession* subsession = (MediaSubsession*)clientData;
  RTSPClient* rtspClient = (RTSPClient*)subsession->miscPtr;
  UsageEnvironment& env = rtspClient->envir(); // alias

  env << *rtspClient << "Received RTCP \"BYE\" on \"" << *subsession << "\" subsession\n";

  // Now act as if the subsession had closed:
  subsessionAfterPlaying(subsession);
}

void streamTimerHandler(void* clientData) {
  ourRTSPClient* rtspClient = (ourRTSPClient*)clientData;
  StreamClientState& scs = rtspClient->scs; // alias

  scs.streamTimerTask = NULL;

  // Shut down the stream:
  shutdownStream(rtspClient);
}

void shutdownStream(RTSPClient* rtspClient, int exitCode) {
  UsageEnvironment& env = rtspClient->envir(); // alias
  StreamClientState& scs = ((ourRTSPClient*)rtspClient)->scs; // alias

  // First, check whether any subsessions have still to be closed:
  if (scs.session != NULL) { 
    Boolean someSubsessionsWereActive = False;
    MediaSubsessionIterator iter(*scs.session);
    MediaSubsession* subsession;

    while ((subsession = iter.next()) != NULL) {
      if (subsession->sink != NULL) {
	Medium::close(subsession->sink);
	subsession->sink = NULL;

	if (subsession->rtcpInstance() != NULL) {
	  subsession->rtcpInstance()->setByeHandler(NULL, NULL); // in case the server sends a RTCP "BYE" while handling "TEARDOWN"
	}

	someSubsessionsWereActive = True;
      }
    }

    if (someSubsessionsWereActive) {
      // Send a RTSP "TEARDOWN" command, to tell the server to shutdown the stream.
      // Don't bother handling the response to the "TEARDOWN".
      rtspClient->sendTeardownCommand(*scs.session, NULL);
    }
  }

  env << *rtspClient << "Closing the stream.\n";
  Medium::close(rtspClient);
    // Note that this will also cause this stream's "StreamClientState" structure to get reclaimed.

  if (--rtspClientCount == 0) {
    // The final stream has ended, so exit the application now.
    // (Of course, if you're embedding this code into your own application, you might want to comment this out,
    // and replace it with "eventLoopWatchVariable = 1;", so that we leave the LIVE555 event loop, and continue running "main()".)
    //exit(exitCode);
	;
  }
}


// Implementation of "ourRTSPClient":

ourRTSPClient* ourRTSPClient::createNew(UsageEnvironment& env, char const* rtspURL,
					int verbosityLevel, char const* applicationName, portNumBits tunnelOverHTTPPortNum) {
  return new ourRTSPClient(env, rtspURL, verbosityLevel, applicationName, tunnelOverHTTPPortNum);
}

ourRTSPClient::ourRTSPClient(UsageEnvironment& env, char const* rtspURL,
			     int verbosityLevel, char const* applicationName, portNumBits tunnelOverHTTPPortNum)
  : RTSPClient(env,rtspURL, verbosityLevel, applicationName, tunnelOverHTTPPortNum, -1) {
}

ourRTSPClient::~ourRTSPClient() {
}


// Implementation of "StreamClientState":

StreamClientState::StreamClientState()
  : iter(NULL), session(NULL), subsession(NULL), streamTimerTask(NULL), duration(0.0) {
}

StreamClientState::~StreamClientState() {
  delete iter;
  if (session != NULL) {
    // We also need to delete "session", and unschedule "streamTimerTask" (if set)
    UsageEnvironment& env = session->envir(); // alias

    env.taskScheduler().unscheduleDelayedTask(streamTimerTask);
    Medium::close(session);
  }
}


// Implementation of "DummySink":

// Even though we're not going to be doing anything with the incoming data, we still need to receive it.
// Define the size of the buffer that we'll use:
#define DUMMY_SINK_RECEIVE_BUFFER_SIZE 100000

DummySink* DummySink::createNew(UsageEnvironment& env, MediaSubsession& subsession, char const* streamId) {
  return new DummySink(env, subsession, streamId);
}

DummySink::DummySink(UsageEnvironment& env, MediaSubsession& subsession, char const* streamId)
  : MediaSink(env),
    fSubsession(subsession) {
  fStreamId = strDup(streamId);
  fReceiveBuffer = new u_int8_t[DUMMY_SINK_RECEIVE_BUFFER_SIZE];
}

DummySink::~DummySink() {
  delete[] fReceiveBuffer;
  delete[] fStreamId;
}

void DummySink::afterGettingFrame(void* clientData, unsigned frameSize, unsigned numTruncatedBytes,
				  struct timeval presentationTime, unsigned durationInMicroseconds) {
  DummySink* sink = (DummySink*)clientData;
  sink->afterGettingFrame(frameSize, numTruncatedBytes, presentationTime, durationInMicroseconds);
  
}

// If you don't want to see debugging output for each received frame, then comment out the following line:
//#define DEBUG_PRINT_EACH_RECEIVED_FRAME 1
void DummySink::afterGettingFrame(unsigned frameSize, unsigned numTruncatedBytes,
				  struct timeval presentationTime, unsigned /*durationInMicroseconds*/) {
  // We've just received a frame of data.  (Optionally) print out information about it:
#ifdef DEBUG_PRINT_EACH_RECEIVED_FRAME
  if (fStreamId != NULL) envir() << "Stream \"" << fStreamId << "\"; ";
  envir() << fSubsession.mediumName() << "/" << fSubsession.codecName() << ":\tReceived " << frameSize << " bytes\n";
  //envir() << "fReceiveBuffer: " << fReceiveBuffer << "\n";
  if (numTruncatedBytes > 0) envir() << " (with " << numTruncatedBytes << " bytes truncated)";
  char uSecsStr[6+1]; // used to output the 'microseconds' part of the presentation time
  sprintf(uSecsStr, "%06u", (unsigned)presentationTime.tv_usec);
  envir() << ".\tPresentation time: " << (int)presentationTime.tv_sec << "." << uSecsStr;
  if (fSubsession.rtpSource() != NULL && !fSubsession.rtpSource()->hasBeenSynchronizedUsingRTCP()) {
    envir() << "!"; // mark the debugging output to indicate that this presentation time is not RTCP-synchronized
  }
#ifdef DEBUG_PRINT_NPT
  envir() << "\tNPT: " << fSubsession.getNormalPlayTime(presentationTime);
#endif
  envir() << "\n";
#endif
  // Then continue, to request the next frame of data:
  continuePlaying();
}

Boolean DummySink::continuePlaying() {
  if (fSource == NULL) return False; // sanity check (should not happen)

  // Request the next frame of data from our input source.  "afterGettingFrame()" will get called later, when it arrives:
  fSource->getNextFrame(fReceiveBuffer, DUMMY_SINK_RECEIVE_BUFFER_SIZE,
                        afterGettingFrame, this,
                        onSourceClosure, this);
  return True;
}
