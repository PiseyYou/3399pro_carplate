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

#include <stdlib.h>
#include <sys/time.h>
//#include "rkdrm.h"
#include "mppdecoder.h"
#include "arm_neon.h"

//rga head files
//#include <RockchipRga.h>
//#include <RockchipFileOps.h>
//#include "omp.h"
extern "C" {
#include <rockchip/rockchip_rga.h>
}
#include <queue>

//*********************
#include <Python.h>
#include <object.h>
#include <numpy/arrayobject.h>
//********************
#include <zmq.hpp>
#include "zhelpers.hpp"
//*********************
//#define Is_Record_Datas 1
//#define Is_Record_H264 1
// Forward function definitions:
#define TCP_PORT "5563"
#define IMAGE_TCP_PORT "5562"
using namespace cv;
using namespace std;
using namespace chrono;
int IMAGE_WIDTH = 1920;
int IMAGE_HEIGHT = 1080;

/* FPS calculator */
RK_U64 fps_ms2,fps_ms3;
RK_U32 fps_counter2,fps_counter3;
float fps2 = 0.0,fps3 = 0.0;

int record_count = 0;
int record_flag = 1;
string dir11;
string dir22;

std::ofstream fw_remove;
bool is_begin_write = false;

extern mutex mutex_basedata;
extern condition_variable cond_basedata;
extern bool gFlagNewBasedata;
RK_U8* base_data;
unsigned char *bgr_data;

void thread_rtsp_mpp();
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
PyObject* mat2numpy(cv::Mat& img,uchar *CArrays)
{

        import_array();
        PyObject *ArgList = PyTuple_New(1);
 
 
        auto sz = img.size();
        int x = sz.width;
        int y = sz.height;
        int z = img.channels();
        //std::cout<<x<<" "<<y<<std::endl;
        //uchar *CArrays = new uchar[x*y*z];//这一行申请的内存需要释放指针，否则存在内存泄漏的问题
        int iChannels = img.channels();
        int iRows = img.rows;
        int iCols = img.cols * iChannels;
        if (img.isContinuous())
        {
            iCols *= iRows;
            iRows = 1;
        }
 
        uchar* p;
        int id = -1;
        for (int i = 0; i < iRows; i++)
        {
            // get the pointer to the ith row
            p = img.ptr<uchar>(i);
            // operates on each pixel
            for (int j = 0; j < iCols; j++)
            {
                CArrays[++id] = p[j];//连续空间
            }
        }
        //std::cout << "is here" << std::endl;
        npy_intp Dims[3] = { y, x, z}; //注意这个维度数据！
        PyObject *PyArray = PyArray_SimpleNewFromData(3, Dims, NPY_UBYTE, CArrays);
        PyTuple_SetItem(ArgList, 0, PyArray);
        //PyObject *pReturn = PyObject_CallObject(pf,ArgList);
        //delete []CArrays ;
        return ArgList;
}
//*************************************************************************************//
struct Content {
			uchar img_data[1843200];
			char timestamp[24];
		};
void send_image(cv::Mat& frame, zmq::socket_t& publisher,double img_time)
{
	
	auto start = chrono::steady_clock::now();
	
		size_t frameSize = frame.step[0] * frame.rows;

		Content msgs;

		string number = std::to_string(img_time);
		strcpy(msgs.timestamp,number.c_str());

		memcpy(&msgs.img_data,  frame.data,frameSize );
		//(msgs)->img_data = frame.data;

		zmq::message_t message( sizeof(Content) );

		memcpy(message.data(),  &msgs,sizeof(Content));

		s_sendmore (publisher, "Image");
		publisher.send(message);

}
//*****************************************************8

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
	sleep(1);
	// Begin by setting up our usage environment:
	TaskScheduler* scheduler = BasicTaskScheduler::createNew();
	UsageEnvironment* env = BasicUsageEnvironment::createNew(*scheduler);

	// We need at least one "rtsp://" URL argument:
	if (argc < 2)
	{
		usage(*env, argv[0]);
		return 1;
	}
	sleep(5);
	thread rtsp_mpp(thread_rtsp_mpp);

        cpu_set_t mask;
        CPU_ZERO(&mask);
        CPU_SET(0,&mask);
        CPU_SET(1,&mask);
        CPU_SET(2,&mask);
        CPU_SET(3,&mask);
        if (sched_setaffinity(0,sizeof(mask),&mask) == -1){
            printf("error_set!\n");
        }
       

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
		shutdownStream(rtspClient_copy);
		sleep(60);
	}

	
	//*****************************************************//	
	rtsp_mpp.join();
	return 0;
}


void thread_rtsp_mpp()
{       

        cpu_set_t mask;
        CPU_ZERO(&mask);
        //CPU_SET(0,&mask);
        CPU_SET(5,&mask);
        CPU_SET(4,&mask);
        
        
        //c++_python init
        Py_Initialize();
        PyRun_SimpleString("import sys"); 
        PyRun_SimpleString("sys.argv=['']"); 
        PyRun_SimpleString("sys.path.append('./')");
        std::cout << "Import Module" << std::endl;
        PyObject *pModule = PyImport_ImportModule("platePro_3399pro_1015_onclass");
        std::cout << "GetDict" << std::endl;
        PyObject *pDict = PyModule_GetDict(pModule);
        std::cout << "GetItemString" << std::endl;
        PyObject *pClass = PyDict_GetItemString(pDict,"plate_pro");
        std::cout << "InstanceMethod_New" << std::endl;
        PyObject *pInstance = PyObject_CallFunctionObjArgs(pClass,NULL);
	RockchipRga *rga;
	rga = RgaCreate();
	if (!rga) 
	{
		std::cout << "rgaCreate error!\n" << std::endl;
		return ;
	}
        int nReg_w = 1920;
        int nReg_h = 1080;
        uchar *CArrays = new uchar[nReg_w*nReg_h*3];
	//创建车牌检测与车牌确认线程		
	fps_ms2 = current_ms();
    	fps_counter2 = 0;
	bool first_init = true;
	bgr_data = (unsigned char*)malloc(sizeof(unsigned char) * IMAGE_WIDTH * IMAGE_HEIGHT * 3);
	
	//rga src init
        int ret;
	int width = 1920, height = 1080;
	int resize_w = 1920, resize_h = 1080;
	static int frame_size = 0;
	unsigned char *frame_rgb = NULL;
	rga->ops->initCtx(rga);
	rga->ops->setRotate(rga, RGA_ROTATE_NONE);
	rga->ops->setSrcFormat(rga, V4L2_PIX_FMT_NV12, width, height);
	rga->ops->setDstFormat(rga, V4L2_PIX_FMT_RGB24, resize_w, resize_h);
        cv::Mat img(resize_h , resize_w , CV_8UC3, bgr_data);
	rga->ops->setDstBufferPtr(rga, bgr_data);
        if (pthread_setaffinity_np(pthread_self(),sizeof(mask),&mask) < 0){
            perror("pthead_setaffinity_np");
        }
        zmq::context_t context(1);
	zmq::socket_t publisher(context, ZMQ_PUB);
	string tcp_s = "tcp://*:";
	tcp_s.append(TCP_PORT);
	publisher.bind(tcp_s);

        cv::Mat save_img(nReg_h,nReg_w,CV_8UC3);
        char* res="Null";
	while(1)
	{	

		unique_lock<mutex> lock_basedata(mutex_basedata);
		cond_basedata.wait(lock_basedata,[](){return gFlagNewBasedata; });
		base_data = decoder_frame();
		gFlagNewBasedata = false;
		lock_basedata.unlock();


		RK_U32 frame_width    = nReg_w;
      		RK_U32 frame_height   = nReg_h;
     		RK_U32 h_stride = IMAGE_WIDTH;
      		RK_U32 v_stride = IMAGE_HEIGHT;
		//lock_basedata.unlock();


		rga->ops->setSrcBufferPtr(rga, base_data);

		ret = rga->ops->go(rga);
                high_resolution_clock::time_point t_temp = high_resolution_clock::now();
	        double img_time   = duration_cast<microseconds>(t_temp.time_since_epoch()).count();
		cv::Mat img(nReg_h,nReg_w,CV_8UC3,bgr_data);
                
		if (!ret) 
		{    
                     PyObject* ArgList = mat2numpy(img,CArrays);
                     PyObject* result = PyObject_CallMethod(pInstance,"track_pro","O",ArgList);

                     PyArg_Parse(result,"s",res);
                     if (strcmp(res,"Null")!=0)
                     { 
                         if (strcmp(res,"save_img")==0)
                         {
                              img.copyTo(save_img);
                         }
                         else
                         {
                              std::cout<<"the result is "<<res<<std::endl;
                              cv::imshow("result",save_img);
                         }
                     }
                     


                     Py_DECREF(ArgList);
                     Py_DECREF(result);
                     imshow("bgr",img);
		     waitKey(1);
		}		
				
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
	
		//********************************************************************
		
        }
        delete []CArrays;
        pthread_exit(NULL);
        Py_Finalize();
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
