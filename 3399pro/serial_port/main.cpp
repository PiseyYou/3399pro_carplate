#include <stdio.h>
#include <iostream>
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <bitset>
#include <string>
#include "IPlateIDRecog.h"
#include <pthread.h>
#include <unistd.h>
#include "tools.h"
#include "serial.h"
//#include "UART_DW_BUF_CB.h"
//#include "rkdrm.h"
extern "C"
{
#include "rtspprotocolutil.h"
#include "mppdecoder.h"
}

#define PACK_SIZE 20
#define FULL_SIZE (2*PACK_SIZE)
//#define IMAGE_WIDTH 2048
//#define IMAGE_HEIGHT 1536
int IMAGE_WIDTH = 1920;
int IMAGE_HEIGHT = 1080;
bool is_Thread_1_Finish = false;
bool is_Thread_2_Finish = true;
bool iswrite1 = false;
bool iswrite2 = false;
cv::Mat bgr1,bgr2;


std::string serial_dev = "/dev/ttyUSB0";
int bardrate_dev = 115200;
int uart_fd;
bool IS_Receive_Message = false;
bool IS_START_DETECT = false;
unsigned char readbufferNew1[PACK_SIZE];
unsigned char readbufferNew2[PACK_SIZE];
bool isNew1 = false;
bool isNew2 = false;

// uart通道数据接收函数
void *uart_Datarcv_Th(void *arg)
{	
	unsigned char buff[FULL_SIZE];
	unsigned char gBuff[FULL_SIZE*2];
	unsigned char *temp;
	temp = gBuff;	
	int nread =0;
	int rcvcount =0;
	//UART_DW_BUF_CB uart_cb;
	//uart_cb.RxState = IDLE;
	int buffcount = 0;
	double t3;
	while(true)
	{	
			
		nread = read(uart_fd,buff,10);
		//int bits = 0;
		//bits=write(uart_fd,readbufferNew1,PACK_SIZE);  //在串口上发送数据，发送读命令
		/*if(isNew2)
   		for(int i=0;i<PACK_SIZE;i++) 
	        {
		    printf("%02x",readbufferNew1[i]);
		
	        }
		printf("\n");/
		/*msleep(300);*/
		//if(nread > 0) printf("nread:%d,IS_START_DETECT = %d\n",nread,IS_START_DETECT);
	
		if(!IS_START_DETECT && rcvcount == 0 && buff[0]==0xaa)
	        {	    
	            //t3 = (double)cv::getTickCount();//开始时间
		    IS_START_DETECT = true;
		    rcvcount= rcvcount + nread;
		    memcpy(temp,buff,nread);
		    temp = gBuff + rcvcount;
		    if(rcvcount == 5)
		    {
			if(gBuff[0]==0xaa && gBuff[1]==0x03 && gBuff[2]==0xf1 && gBuff[3]==0x00 && gBuff[4]==0xf2)
		   	{
		      		rcvcount = 0;
		      		temp = gBuff;
		      		if(!IS_Receive_Message) IS_Receive_Message = true;
		      		IS_START_DETECT = false;
		      		for(int i =0;i<5;i++) gBuff[i] = 0x00;
		      		int bits = 0;
		      		bits=write(uart_fd,readbufferNew1,PACK_SIZE);  //在串口上发送数据，发送读命令
				/*if(isNew2)
		      		for(int i=0;i<PACK_SIZE;i++) 
	        		{
		    			printf("%02x",readbufferNew1[i]);
		
	        		}
				printf("\n");*/
		     		//buffcount ++;
		      		//printf("buffcount:%d\n",buffcount);
		   	} 
			else 
			{
				rcvcount = 0;
				IS_START_DETECT = false;
			}
		    }
		    //printf("1:%d\n",rcvcount);
		}
		else if(IS_START_DETECT && rcvcount < 5)
		{
		    rcvcount= rcvcount + nread;
		    memcpy(temp,buff,nread);
		    temp = gBuff + rcvcount;
		    if(rcvcount == 5)
		    {
			if(gBuff[0]==0xaa && gBuff[1]==0x03 && gBuff[2]==0xf1 && gBuff[3]==0x00 && gBuff[4]==0xf2)
		   	{
		      		rcvcount = 0;
		      		temp = gBuff;
		      		if(!IS_Receive_Message) IS_Receive_Message = true;
		      		IS_START_DETECT = false;
		      		for(int i =0;i<5;i++) gBuff[i] = 0x00;
		      		int bits = 0;
		      		bits=write(uart_fd,readbufferNew1,PACK_SIZE);  //在串口上发送数据，发送读命令
				/*if(isNew2)
		      		for(int i=0;i<PACK_SIZE;i++) 
	        		{
		    			printf("%02x",readbufferNew1[i]);
		
	        		}
				printf("\n");*/
		     		//buffcount ++;
		      		//printf("buffcount:%d\n",buffcount);
		   	} 
			else 
			{
				rcvcount = 0;
				IS_START_DETECT = false;
			}
		    }
		    //printf("2:%d",rcvcount);
                    //printf("\n");
		}
		else if(IS_START_DETECT && rcvcount >= 5)
		{
		    //printf("3:%d\n",rcvcount);
		    if(gBuff[0]==0xaa && gBuff[1]==0x03 && gBuff[2]==0xf1 && gBuff[3]==0x00 && gBuff[4]==0xf2)
		   {
		      rcvcount = 0;
		      temp = gBuff;
		      if(!IS_Receive_Message) IS_Receive_Message = true;
		      IS_START_DETECT = false;
		      for(int i =0;i<5;i++) gBuff[i] = 0x00;
		      int bits = 0;
		      bits=write(uart_fd,readbufferNew1,PACK_SIZE);  //在串口上发送数据，发送读命令
		      /*if(isNew2)
		      for(int i=0;i<PACK_SIZE;i++) 
	              {
		    	printf("%02x",readbufferNew1[i]);
		
	              }
		      printf("\n");*/
		      //buffcount ++;
		      //printf("buffcount:%d\n",buffcount);
		   } 
		   else 
		   {
			rcvcount = 0;
			IS_START_DETECT = false;
		   }
		}
		usleep(1);
	}
}

void *sendMessage(void *arg)
{
	while(true)
	{
	    int bits =0;	
	    if(IS_Receive_Message)	
	    {  
		//if(isNew1)
		{
		   iswrite1 = true;
		   bits=write(uart_fd,readbufferNew1,PACK_SIZE);  //在串口上发送数据，发送读命令
		   IS_Receive_Message = false;
                   iswrite1 = false;
		   //printf("new1:%x ",readbufferNew1[0]);
		   
		}
	
		//IS_Receive_Message = false;
	    }
	    msleep(1);
	}
   	
}

void *create2(void *arg)
{
    cv::Mat img,rgb,rgb1;
    double sum = 0.0;
    //创建车牌识别接口
	IPlateIDRecog* pPlateIDRecog = IPlateIDRecog::CreateRecogCtrl();

	//< 车牌识别初始化
	PLATE_RECOG_PARAM cParam;
	cParam.nImageFormat = 1;
	cParam.fIsNight = 0;
	cParam.fIsFieldImage = 0;
	cParam.nImageWidth = IMAGE_WIDTH;
	cParam.nImageHeight = IMAGE_HEIGHT;
	pPlateIDRecog->Init(&cParam);
	int writecount = 1;
	//车牌识别
    while(true)
    {
        RK_U8* base = decoder_frame();
        if (NULL == base) {
            msleep(8);
	    //printf("frame is NULL.\n");
            continue;
        }

	int nSize = (IMAGE_HEIGHT*3>>1);
        cv::Mat src(IMAGE_HEIGHT*3>>1,IMAGE_WIDTH,CV_8UC1,(unsigned char*)base);
        cv::Mat dst;//(frame_height,frame_width,CV_8UC3);
	cvtColor(src,bgr1,CV_YUV2BGR_NV12);
	 
#if 0	
	cv::resize (bgr1,bgr2,cv::Size(512,384));
	imshow("img",bgr2);
	//cv::waitKey(10);
if(cv::waitKey(10) > 0 && writecount < 100) 
        {
	    char buf[50];
	    sprintf(buf,"bgr%d.jpg",writecount);
            cv::imwrite(buf,bgr1);
	    writecount++;
	    //iswrite = true;
        }
#endif
	double t1 = 0;
        int nPlateNum = 0;
        double t3 = (double)cv::getTickCount();//开始时间
        PLATE_ID_INFO cPlateInfo[10];
	bool fDetRes = pPlateIDRecog->RecogPlateID(&bgr1, &nPlateNum, cPlateInfo);
        t3 = (double)cv::getTickCount() - t3;
        t3 = (int)(t3*1000 / cv::getTickFrequency()*100)/100.00;

        if(fDetRes == false)
        {
           printf("license detect error\n");
        }
        else
        {
	    //printf("time: %4.2f\n",t3);
            
	    //unsigned char writebuffer[PACK_SIZE] ={0x5a,0x5a,0x00,0x00,0x00,0x04,0x00,0x00,0x00,0x01,0x05,0xaa};
	    unsigned char readbuffer[PACK_SIZE]; //= {0xaa,0x12,0xe1,0xd4,0xc1,0x41,0x30,0x30,0x30,0x30,0x31,0x00,0x00,0x00,0x00,0x00,0x01,0x03,0x20,0xb4};
	    readbuffer[0] = 0xaa;
	    readbuffer[1] = 0x12;
	    readbuffer[2] = 0xe1;
	    for(int i = 3; i < 15; i++)
	    {
		readbuffer[i] = cPlateInfo[0].license[i-3];
		//if(i>4) printf("%x ",cPlateInfo[0].license[i-3]);
	    }	
	    readbuffer[15] = (char)cPlateInfo[0].nColor;
	    readbuffer[16] = 0x00;
	    readbuffer[17] = 0x00;
	    readbuffer[18] = 0x00;
	    readbuffer[19] = 0x00;
 	    for(int i=1;i<19;i++)
	    {
		readbuffer[19] = readbuffer[19]^readbuffer[i];
	    }
	    isNew2 = false;
            if(!iswrite1)
	    for(int i=0;i<PACK_SIZE;i++) 
	    {
		readbufferNew1[i] = readbuffer[i];
	    }
            
	    isNew2 = true;
	    //printf("nPlateNum: %d license: %s\n",nPlateNum,cPlateInfo[0].license);
	    /*int bits =0;	
	    if(IS_Receive_Message)	
	    {  
		bits=write(uart_fd,readbuffer,PACK_SIZE);  //在串口上发送数据，发送读命令
		IS_Receive_Message = false;
		printf("nPlateNum: %d license: %s\n",nPlateNum,cPlateInfo.license);
	    }*/
	    msleep(1);
        }
	
     }
     if(NULL != pPlateIDRecog)
	{		
		pPlateIDRecog->Destroy();
		pPlateIDRecog = NULL;
	
	}
}

static void *
       thread_rtsp_mpp(void *arg)
{
    (void)(arg);
    int flagno = 0;
    while (1) {
        // Check the RTSP connection
        if (!isStart()) {
            msleep(1000);
            continue;
        }
        // 尝试从socket获取数据
        rtsp_read();
        // 获取RTSP中的H264数据
        if (rtsp_packet() > 0) {
	    flagno = 0;
            // 解码
            decoder_routine();
        }
        else {
            msleep(8);
	    flagno ++;
	    if(flagno > 100) 
	    {
		reStart();
		//syslog(LOG_USER|LOG_DEBUG,"restart.\n");
		flagno = 0;
		//printf("rtsp_packet() = %d\n",rtsp_packet());
	    }
	    //printf("rtsp_packet() is %d.\n",rtsp_packet());
        }  
    }
    return NULL;
}


int main()
{
    //1. 打开串口设备  
    uartdev *udev = new uartdev((const char *)(serial_dev.data()), bardrate_dev); //新建一个串口设备
    uart_fd = udev->open_dev();  //打开串口设备
    /*if (uart_fd>0)
	 udev->set_speed(uart_fd);   // 设置波特率
    else
    {
	  printf("Can't Open Serial Port!\n");
	  return 0;
    }*/
    if (uart_fd>0 && udev->set_parity(uart_fd,8,1,'N')== -1)  // 8位数据，非两位的停止位，不使用奇偶校验 ，不允许输入奇偶校验
    {
         printf("Set Parity Error\n");
	  return 0;
    }

    // Initialise RTSP client
    if (RtspProtocolUtil_init("rtsp://192.168.1.88/av0_0")) {
        fprintf(stderr, "RTSP initialise error.\n");
        exit(-1);
    } 

    // Initialise MPP decoder 
    mppDecoder();

    pthread_t thread;
    pthread_create(&thread, NULL, &thread_rtsp_mpp, NULL);

    pthread_t tidp1,tidp2,tidp3,tidp4;
    int rc1,rc2,rc3,rc4;
    IMAGE_WIDTH = 2048;
    IMAGE_HEIGHT = 1536;
    rc1 = pthread_create(&tidp1,NULL,create2,NULL);
    if(rc1!=0)
    {
         std::cerr << "pthread_create is not created..." << std::endl;
	 return -1;
    }
    //std::cerr << "Thread1 is created..." << std::endl;

    msleep(3000);
    /*rc2 = pthread_create(&tidp2,NULL,create2,NULL);
    if(rc2!=0)
    {
         std::cerr << "create2 is not created..." << std::endl;
	 return -1;
    }
    std::cerr << "create2 is created..." << std::endl;*/
  
    rc3 = pthread_create(&tidp3,NULL,uart_Datarcv_Th,NULL);
    if(rc3!=0)
    {
         std::cerr << "uart_Datarcv_Th is not created..." << std::endl;
	 return -1;
    }
    //std::cerr << "uart_Datarcv_Th is created..." << std::endl;
   
    /*rc4 = pthread_create(&tidp4,NULL,sendMessage,NULL);
    if(rc4!=0)
    {
         std::cerr << "sendMessage is not created..." << std::endl;
	 return -1;
    }
    std::cerr << "sendMessage is created..." << std::endl;*/

    while(1)
    {
	if(is_Thread_1_Finish && is_Thread_2_Finish && is_Thread_1_Finish && is_Thread_2_Finish) break; 
	//cv::waitKey(100);
	sleep(1);
    }
}
