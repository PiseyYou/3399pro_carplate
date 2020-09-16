#ifndef __ZMQ_REQ_HPP_INCLUDED__
#define __ZMQ_REQ_HPP_INCLUDED__

//#include <zmq.hpp>
//#include "zhelpers.hpp"


#include "function_utils.hpp"
#include <time.h>

#define IMG_W 608
#define IMG_H 358

struct Content
{
   char timestamp[24]; 
   char imagelenght[24];
   uchar img_data[500000];

};

struct Ai_Ui_Content
{
   char chStr_info[1024];
   uchar img_data[2764800];
};

struct Content_yuv
{
   char timestamp[64]; 
   char imagelenght[24];
   uchar img_data[IMG_W*IMG_H*3/2];


};


int req_begin(zmq::socket_t& Reqer)
{

        std::cout << "send the begin state" << std::endl;
        std::string name = "name=detector1;action=connecting";
        zmq::message_t message( strlen(name.c_str()) );
        memcpy((void*)message.data(),name.c_str(),strlen(name.c_str()));
	Reqer.send(message);
        std::cout << "send " << name <<" "<< strlen(name.c_str()) <<std::endl;
        zmq::message_t reply;
        Reqer.recv(&reply);
        //char* remsg = reinterpret_cast<char*>(reply.data());
        std::string remsg = std::string(static_cast<char*>(reply.data()),reply.size());
        std::cout << "get the reply: " << remsg << std::endl;

        return 1;


}

std::string req_info(zmq::socket_t& Reqer,std::vector<std::string>& vePlate_target, std::string& rmtp_address)
{
        std::cout << "xxxxx" << std::endl;
        std::string name = "name=detector1;action=getinfo";
        zmq::message_t message( strlen(name.c_str()) );
        memcpy((void*)message.data(),name.c_str(),strlen(name.c_str()));
	Reqer.send(message);
        std::cout << "send " << name <<" "<< strlen(name.c_str()) <<std::endl;
        zmq::message_t reply;
        Reqer.recv(&reply);
        //char* remsg = reinterpret_cast<char*>(reply.data());
        std::string remsg = std::string(static_cast<char*>(reply.data()),reply.size());
        std::cout << "get the info: " << remsg << std::endl;
        
        if (remsg == "" || remsg=="noinfo")return remsg;

        //split the msg:
        std::vector<std::string> veResult;
        string_split(remsg,veResult,",");

        rmtp_address = veResult[1];
        std::string stTrack_type = veResult[2];
        std::string stPlate_ids = veResult[3];
        std::vector<std::string> vePlate_ids;
        string_split(stPlate_ids,vePlate_ids,";");

        for (int i=0;i<vePlate_ids.size();i++)
        {   
       
            std::string stPlate_id = vePlate_ids[i];
            if (i==0)
            {
                stPlate_id = vePlate_ids[i].substr(1);

            }
            if(i == vePlate_ids.size()-1)
            {
                int nStlen = vePlate_ids[i].length();
                stPlate_id = vePlate_ids[i].substr(0,nStlen-1);
            }
            std::cout << stPlate_id << std::endl;
            vePlate_target.push_back(stPlate_id);
        }

        //return (stTrack_type=="0")?0:1;// 0 nothing 1 get the current result 2 get the both of history
        //sleep(5);
        return remsg;

}

std::string req_port(zmq::socket_t& Reqer)
{
        std::cout << "xxxxx" << std::endl;
        std::string name = "name=detector1;action=get_rtsp_url";
        zmq::message_t message( strlen(name.c_str()) );
        memcpy((void*)message.data(),name.c_str(),strlen(name.c_str()));
	Reqer.send(message);
        std::cout << "send " << name <<" "<< strlen(name.c_str()) <<std::endl;
        zmq::message_t reply;
        Reqer.recv(&reply);
        //char* remsg = reinterpret_cast<char*>(reply.data());
        std::string remsg = std::string(static_cast<char*>(reply.data()),reply.size());
        std::cout << "get the port: " << remsg << std::endl;

        return remsg;

}
std::string req_rtsp_url(zmq::socket_t& Reqer)
{
        std::cout << "xxxxx" << std::endl;
        std::string name = "name=detector1;action=get_rtsp_url";
        zmq::message_t message( strlen(name.c_str()) );
        memcpy((void*)message.data(),name.c_str(),strlen(name.c_str()));
	Reqer.send(message);
        std::cout << "send " << name <<" "<< strlen(name.c_str()) <<std::endl;
        zmq::message_t reply;
        Reqer.recv(&reply);
        //char* remsg = reinterpret_cast<char*>(reply.data());
        std::string remsg = std::string(static_cast<char*>(reply.data()),reply.size());
        std::cout << "get the rtsp_url: " << remsg << std::endl;

        return remsg;

}



int req_queue(zmq::socket_t& Reqer)
{

        std::string name = "name=detector1;action=requestqueue";
        zmq::message_t message( strlen(name.c_str()) );
        memcpy((void*)message.data(),name.c_str(),strlen(name.c_str()));
	Reqer.send(message);
        std::cout << "send " << name <<" "<< strlen(name.c_str()) <<std::endl;
        zmq::message_t reply;
        Reqer.recv(&reply);
        //char* remsg = reinterpret_cast<char*>(reply.data());
        std::string remsg = std::string(static_cast<char*>(reply.data()),reply.size());
        std::cout << remsg << std::endl;

        return 0;

}

int req_img(zmq::socket_t& Reqer)
{
        clock_t start = clock();
        char* name = "name=detector1;action=getnextframe";
        zmq::message_t message( strlen(name) );
        memcpy(message.data(),name,strlen(name));
	Reqer.send(message);
       // std::cout << "send " << name << std::endl;

        zmq::message_t reply;
        Reqer.recv(&reply);
        clock_t end = clock();
        std::cout << "zmq cost " << (double)(end - start)/CLOCKS_PER_SEC << std::endl;
        char* remsg = reinterpret_cast<char*>(reply.data());
        std::cout << remsg << std::endl;
        std::string msg(remsg);
        if (msg.substr(0,2) == "ok")return 1;
        //if (strcmp(remsg, "ok")==0)return 1;
        else return 0;

}


int get_img(zmq::socket_t& subscriber,Content* input,int& nRelySize)
{
    zmq::message_t reply;
    auto res = subscriber.recv(&reply);

    if (res==0)
    {
        std::cout << "get the image time out"<<std::endl;
        return 0;
    }
    if (reply.size()<100)return 0;
    input = reinterpret_cast<Content*>(reply.data()); 
    nRelySize = reply.size();
    return 1;
}

int send_result(zmq::socket_t& Reqer, std::string timpstamp, std::string result)
{   
    std::cout << "send result" << std::endl;
    //clock_t start = clock();
    //std::string res(result);
    std::string name = "name=detector1;";
    //std::string timesp = "timestamp=" + timpstamp+";";
    std::string event = "event=Plate_detect;";
    std::string plat_res = "plate_result=" + result;
 
    std::string msg_res = name + timpstamp + event + plat_res;

    zmq::message_t message( strlen(msg_res.c_str()) );
    memcpy((void*)message.data(),msg_res.c_str(),strlen(msg_res.c_str()));
    Reqer.send(message);

    zmq::message_t reply;
    Reqer.recv(&reply);
    //char* remsg = reinterpret_cast<char*>(reply.data());
    std::string remsg = std::string(static_cast<char*>(reply.data()),reply.size());
    //clock_t end = clock();
    //std::cout << "zmq cost " << (double)(end - start)/CLOCKS_PER_SEC << std::endl;

    std::cout << msg_res.c_str() << std::endl;
    std::cout << remsg << std::endl;
    
    return (remsg == "novideo")?0:1;

}

int send_sy(zmq::socket_t& Reqer, std::string timpstamp)
{   
    //std::cout << "send result" << std::endl;
    //clock_t start = clock();
    //std::string res(result);
    std::string name = "name=detector1;";
    //std::string timesp = "timestamp=" + timpstamp+";";
    std::string event = "event=work1";
    //std::string plat_res = "plate_result=" + result;
 
    std::string msg_res = name + timpstamp + event;

    zmq::message_t message( strlen(msg_res.c_str()) );
    memcpy((void*)message.data(),msg_res.c_str(),strlen(msg_res.c_str()));
    Reqer.send(message);

    zmq::message_t reply;
    auto res = Reqer.recv(&reply);
    if (res==0)
    {
       std::cout << "get the reply time out" << std::endl;
       return 0;
    }
    //char* remsg = reinterpret_cast<char*>(reply.data());
    std::string remsg = std::string(static_cast<char*>(reply.data()),reply.size());
    //clock_t end = clock();
    //std::cout << "zmq cost " << (double)(end - start)/CLOCKS_PER_SEC << std::endl;

    //std::cout << msg_res.c_str() << std::endl;
    //std::cout << remsg << std::endl;
    
    return 1;

}
//search plate_ID function
int get_searchId(zmq::socket_t& subscriber, std::vector<std::string>& veTarget)
{
    
    zmq::message_t getmsg;
    subscriber.recv(&getmsg);
    std::string remsg = std::string(static_cast<char*>(getmsg.data()),getmsg.size());
    std::cout << "get the search demand: " << remsg << std::endl;
    
    //split the msg:
    std::vector<std::string> veResult_1;
    string_split(remsg,veResult_1,";");
    std::string stPara = veResult_1[2];
    //split the Para:
    std::vector<std::string> veResult_2;
    string_split(stPara,veResult_2,",");
    std::string stTrack_type = veResult_2[3];

    for (int i=4;i<veResult_2.size();i++)
    {   
       
        std::string stPlate_id;
        if (i==4)
        {
            stPlate_id = veResult_2[i].substr(1);

        }
        if(i == veResult_2.size()-1)
        {
            int nStlen = veResult_2[i].length();
            stPlate_id = veResult_2[i].substr(0,nStlen-1);
        }
        veTarget.push_back(stPlate_id);
    }

    return (stTrack_type=="0")?0:1;// 0 nothing 1 get the current result 2 get the both of history

}

int send_trackResult(
    zmq::socket_t& requester, 
    std::string& track_id, 
    int nFrameIndex,
    int nDetectIndex,
    std::string& stState,
    std::string& stReTime,
    cv::Rect& rePlate,
    cv::Mat& matImage,
    std::string stIsSend
)
{

        std::string name = "name=AI;event=1;";
        std::string stPara = "para="+ std::to_string(nFrameIndex)+","
                                    + std::to_string(nDetectIndex)+","
                                    + stState + "," + stReTime+",";

        std::string track_plateInfo = track_id  + "," + std::to_string(rePlate.x) + "," +
                                      std::to_string(rePlate.y) + "," +
                                      std::to_string(rePlate.width) + "," +
                                      std::to_string(rePlate.height);

        std::string stObjBoxesInfo = "[1,"+track_plateInfo+"]" + ";";
        
        std::string stSend_msg = name+stPara+stObjBoxesInfo;
        std::cout << "send " << stSend_msg <<" "<< strlen(name.c_str()) <<std::endl;
        //return 0;

        Ai_Ui_Content msg_content;

        //int nlen = stSend_msg.length();
        //msg_content.chStr_info = (char*)malloc((nlen+1)*sizeof(char));
    
        //stSend_msg.copy(msg_content.chStr_info,nlen,0);
        strcpy(msg_content.chStr_info,stSend_msg.c_str());
        //msg_content.chStr_info = stSend_msg;
        //std::cout << msg_content.chStr_info << "!!!!" << endl;
        //size_t frameSize = matImage.step[0] * matImage.rows;
        //memcpy(&msg_content.img_data,matImage.data,frameSize );




        if (stIsSend == "1")
        {
          std::vector<uchar> veTmp;
          cv::imencode(".jpg", matImage, veTmp);
          for (int i = 0;i<veTmp.size();i++)msg_content.img_data[i]=veTmp[i];
        }


       
        zmq::message_t message( sizeof(msg_content));
        memcpy(message.data(),&msg_content,sizeof(msg_content));
	requester.send(message);
        
        zmq::message_t reply;
        requester.recv(&reply);
        //char* remsg = reinterpret_cast<char*>(reply.data());
        std::string remsg = std::string(static_cast<char*>(reply.data()),reply.size());
        std::cout << "get the reply: " << remsg << std::endl;
        
        //free(msg_content.chStr_info);
        //msg_content.chStr_info = nullptr;
        return 0;
}


#endif
