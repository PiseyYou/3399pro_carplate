
#include "stdio.h"
#include <stdio.h>
#include <iostream>
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <bitset>
#include <string>
#include "Util.h"

#include <time.h>
#include <fstream>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include <stdlib.h>
#include <locale.h>
#include <wchar.h>
#include <iostream>  
#include <fstream>
#include <codecvt>


#include <sys/time.h>



#include <zmq.hpp>
#include <string>

#include "zhelpers.hpp"
#include "thread"
#include "mutex"

// ---- PARAMETERS ----



#define TCP_PORT "5563"
#define PUB_IP "192.168.3.105"
#define IMAGE_TCP_PORT "5562"
#define IMAGE_PUB_IP "192.168.3.105"
#define MIN_TIME_DIF 50000      // minimal time difference to consider two frames the same in microseconds


// --------------------


#include "safequeue.cpp"
using namespace std;
using namespace cv;
using namespace chrono;


//###########################
void image_subscriber_thread();
void subscriber_thread();
void parse_detection(string& det, vector<string>& dets);
cv::Rect parse_rect(string& rec_s);
void video_show();
SafeQueue<vector<string>> subscriber_queue;
SafeQueue<tuple<cv::Mat,long int>> img_queue;
//#######################
				struct Content {
			uchar img_data[1843200];
			char timestamp[24];
		};
int main(int argc, char** argv) 
{	
  
    thread img_sub(image_subscriber_thread);
	//thread sub(subscriber_thread);
	//thread show(video_show);

    img_sub.join();
	//sub.join();
//	show.join();

    
    return 0;
}



/*
    Thread that subscribes to the computation publisher and prints the results
*/
void subscriber_thread()
{
    zmq::context_t context(1);
    zmq::socket_t subscriber (context, ZMQ_SUB); //zqm tcp subscriber
    string tcp_s = "tcp://";
    tcp_s.append(PUB_IP);
    tcp_s.append(":");
    tcp_s.append(TCP_PORT);
    subscriber.connect(tcp_s);
    subscriber.setsockopt( ZMQ_SUBSCRIBE, "LicensePlate", 1);
    cout << "initialized sub" << endl;

    while(true)
    {
	 std::string address = s_recv (subscriber);// waits for a message
        //  Read message contents
        std::string contents = s_recv (subscriber);
        vector<string> results;
        parse_detection(contents,results);
        subscriber_queue.push(results);
        std::cout << "[" << address << "] " << contents << std::endl;
    }

    
}

/*
    Analyses the puiblished detection and separates each invidual part, the detection is expected to be:
    (double img_timestamp ;string Location(x,y,width,height) ; string License_plate)
*/
void parse_detection(string& det, vector<string>& dets)
{
	auto start = 0U;
    auto end = det.find(";");
    string delim = ";";
    //separetes the detection by ";"
    while (end != std::string::npos)
    {
	    std::string part = det.substr(start, end - start);
        dets.push_back(part);
        start = end + delim.length();
        end = det.find(delim, start);
    }
	std::string part = det.substr(start, end - start);
	dets.push_back(part);

}

/*
    This function stores the images and detections in vectors and matches the detections to the correct frames
    and shows them on the screen. 
*/
void video_show()
{
	namedWindow("img",CV_WINDOW_NORMAL);
	namedWindow("d_img",CV_WINDOW_NORMAL);
//		VideoWriter video("out.avi",CV_FOURCC('M','J','P','G'),20, Size(1280,480));

    vector<cv::Mat> img_vec ;
    vector<long int> time_vec;
	vector<long int> detection_time;
	vector<Rect> detection_vec;

  while(true)
  {
    //cout << "det size : " << detection_vec.size() << endl << "img size : " << img_vec.size() << endl;
    if(img_queue.empty())
      continue;
    //reads image queue and stores them along with the timestamp into vectors
    tuple<cv::Mat,long int> d1;
    bool suc = img_queue.next(d1);
    if(suc)
    {
        Mat img =  get<0>(d1);
        long int t = get<1>(d1);

        img_vec.push_back(img);
        time_vec.push_back(t);
    }
    //reads detection queue 
	if(not subscriber_queue.empty())
	{
		vector<string> data;
		suc = subscriber_queue.next(data);
		if(suc){
			long int r_time = stol(data[0]);
			Rect rec = parse_rect(data[1]);
			detection_vec.push_back(rec);
			detection_time.push_back(r_time);
	}
	}
    //if there's no detections to match just shows the image
    if(detection_vec.size() == 0){
    	if(img_vec.size() > 100){
			imshow("img",img_vec[0]);
			//video.write(img_vec[0]);
			waitKey(1);
        	img_vec.erase(img_vec.begin());
        	time_vec.erase(time_vec.begin());
			}
	}
	else
	{
        //if there's detections to be matched tries to match them
		long int c_time = time_vec[0];
		int min_dif = 100000000000;
		int min_idx = -1;
		int idx = 0;
		for (auto s : detection_time) 
			{
				if(abs(s-c_time) < min_dif)
				{
				min_dif = abs(s-c_time);
				min_idx = idx;
				}
				idx++;
			}
        //if the difference in timestamps is less then MIN_TIME_DIF microseconds we consider a match
		if(min_dif  ==  0)
			{
				Mat imgNow = img_vec[0];
				Rect rec = detection_vec[min_idx];
				rectangle(imgNow, rec, Scalar(255,0,0));
				imshow("d_img",imgNow);
				waitKey(1);
				//video.write(imgNow);
				detection_time.erase(detection_time.begin() + min_idx);
        		detection_vec.erase(detection_vec.begin() + min_idx);//deletes the matches
				img_vec.erase(img_vec.begin());
        		time_vec.erase(time_vec.begin());
			}
		else
		{
            //if no match is found the image is shown
			Mat imgNow = img_vec[0];
			imshow("img",imgNow);
			waitKey(1);
			//video.write(imgNow);
			img_vec.erase(img_vec.begin());
        	time_vec.erase(time_vec.begin());
		}
    if(detection_vec.size() > 300)
    {
      for(unsigned int j=0;j < 100 ; j++)
      {
          detection_vec.erase(detection_vec.begin());
      }
     }
  }
  //cleans old detections

  }

}

/*
    Parses a opencv rect in a string in the format (x,y,width,height) to an cv::Rect
*/
cv::Rect parse_rect(string& rec_s)
{
  auto start = 0U;
  auto end = rec_s.find(",");
  string delim = ",";
  vector<string> dets; 
  while (end != std::string::npos)
    {
	      std::string part = rec_s.substr(start, end - start);
        dets.push_back(part);
        start = end + delim.length();
        end = rec_s.find(delim, start);
    }
std::string part = rec_s.substr(start, end - start);
	dets.push_back(part);

  int x = stoi(dets[0].substr(0U + 1) );
  int y = stoi(dets[1]);
  int width = stoi(dets[2]);
  end = dets[3].find(")");
  int height = stoi(dets[3].substr(0U,end));

  return cv::Rect(x,y,width,height);
}

/*
    Thread that subscribes to the computation publisher and prints the results
*/
void image_subscriber_thread()
{
    zmq::context_t context(1);
    zmq::socket_t subscriber (context, ZMQ_SUB); //zqm tcp subscriber
    string tcp_s = "tcp://";
    tcp_s.append(IMAGE_PUB_IP);
    tcp_s.append(":");
    tcp_s.append(IMAGE_TCP_PORT);
    subscriber.connect(tcp_s);
    subscriber.setsockopt( ZMQ_SUBSCRIBE, "Image", 1);

    cout << "initialized Image Sub" << endl;

	int bytes = 0;

    // change the last loop to below statement
    while(true)
    {
        auto start = chrono::steady_clock::now();
        zmq::message_t reply;
        subscriber.recv(&reply);
	    subscriber.recv(&reply);
        auto start1 = chrono::steady_clock::now();
        Content* input = reinterpret_cast<Content*>(reply.data());
        auto end1 = chrono::steady_clock::now();
		double ms1 = std::chrono::duration_cast<
  		std::chrono::duration<double> >(end1 - start1).count();
		std::cout << "cast: " << ms1 << " ms" << endl;
        // store the reply data into an image structure
        cv::Mat img(480, 1280, CV_8UC3, (input)->img_data);
        long int img_time = stol((input)->timestamp);

        tuple<cv::Mat,long int> data;
        data = make_tuple(img.clone(),img_time);
        img_queue.push(data);
        auto end = chrono::steady_clock::now();
		double ms = std::chrono::duration_cast<
  		std::chrono::duration<double> >(end - start).count();
		std::cout << "Elapsed time camera thread: " << ms << " ms" << endl;
        imshow("live", img);
        waitKey(1);

     }
}

