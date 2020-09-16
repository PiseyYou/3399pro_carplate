#ifndef __TRACKPRO_HPP__
#define __TRACKPRO_HPP__

#include "lisence_plate_proceess.hpp"


class trackSystem_pro:private LP_Recognition
{

    public:
    trackSystem_pro();
    //using LP_Recognition::LP_Recognition;
    bool boIsleave = false;

    std::string track_pro(
    cv::Mat& matImg_src    
    );

    private:
    int nRect_num = 0;
    std::vector<std::array<int,8>> veRect_preboxes;
    std::vector<std::array<int,8>> veRect_aftboxes;

  
    int nTrack_miss = 0;
    int nPlate_state = 0;
    int nTrack_count = 0;
    int nNone_plate_ct = 0;
    int nVehicle_count = 0;
  



    std::vector<std::string> vePlate_result;
    int nPlate_result_lock = -1;
    int nAp_max = 0;
    std::string stPlateMax_res = "";
    cv::Rect reCrop_plate;
    cv::Rect reCrop_plate_save;
    std::string stLast_res = "";

    int nSaveDif_min = 10000;

    int nType_max_num = 0;
    std::vector<std::string> veType_list_type;
    std::string stType_max_type = "";

    void cal_rectpro(
    std::vector<cv::Rect>& Vehicleboxes,
    cv::Mat& matImg_src    
    );

    void init_trackoutput();

    std::string output_platepro(
    cv::Mat& matImg_src
    );
    void plate_recogByvehicleDeted( 
    cv::Rect& vehicle_box,
    cv::Mat& matImg_src,
    std::vector<std::pair<std::string,std::string>>& stResult_plate
    );
};

trackSystem_pro::trackSystem_pro()
{

    this->veRect_preboxes.reserve(10);
    this->veRect_aftboxes.reserve(10);
    this->vePlate_result.reserve(25);
    
}


void trackSystem_pro::cal_rectpro(
    std::vector<cv::Rect>& Vehicleboxes,
    cv::Mat& matImg_src    
    )
{
    this->nRect_num = Vehicleboxes.size();
    this->veRect_aftboxes = this->veRect_preboxes;


    if (this->veRect_aftboxes.size()>=5)this->veRect_aftboxes.clear(); // if size > 5 ,clear the buffer


    //track -1
    //for (auto it=veRect_aftboxes.begin();it!=veRect_aftboxes.end();it++)
    //     (*it)[4]--;
    for (auto& aftbox:this->veRect_aftboxes)aftbox[4]--;

    for (auto reRect_1:Vehicleboxes)
    {
         bool boNewRect = false;
         for (int j=0;j<this->veRect_preboxes.size();j++)
         {
             cv::Rect reRect_2(
             this->veRect_preboxes[j][0],
             this->veRect_preboxes[j][1],
             this->veRect_preboxes[j][2],
             this->veRect_preboxes[j][3]
             );
          
             float flIou = cal_iou(reRect_1,reRect_2);
             //std::cout <<"iou: "<< flIou << " num: " << veRect_preboxes.size() <<std::endl;
             if (flIou > 0.35)
             {
                this->veRect_aftboxes[j][6] ++;
                this->veRect_aftboxes[j][4] = 0;
                this->veRect_aftboxes[j][0] =reRect_1.x;
                this->veRect_aftboxes[j][1] =reRect_1.y;
                this->veRect_aftboxes[j][2] =reRect_1.width;
                this->veRect_aftboxes[j][3] =reRect_1.height;
                //this->veRect_aftboxes[j] = this->veRect_preboxes[j];
               
                boNewRect = true;
                break;
             }

        }
   
        if (boNewRect) continue;

        std::array<int,8> arNewRect = {reRect_1.x,reRect_1.y,reRect_1.width,reRect_1.height,0,0,0,0};
        this->veRect_aftboxes.push_back(arNewRect);
        

    }
   
    for (auto aftbox: this->veRect_aftboxes)
    {
        if (aftbox[6]>=5)
        {
            int x1 = aftbox[0];
            int y1 = aftbox[1];
            int x2 = x1 + aftbox[2];
            int y2 = y1 + aftbox[3];
            rectangle(matImg_src, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(139, 139,   0, 255), 3);
        }

    }


}


void trackSystem_pro::init_trackoutput()
{
     this->nAp_max = 0;
     this->nNone_plate_ct = 0;
     this->stPlateMax_res = "";
     this->nPlate_result_lock = -1;
     this->vePlate_result.clear();
    
     this->nType_max_num = 0;
     this->stType_max_type = "";
     this->veType_list_type.clear();
     
     this->reCrop_plate_save.x = 0;
     this->reCrop_plate_save.y = 0;
     this->reCrop_plate_save.width = 0;
     this->reCrop_plate_save.height = 0;

     this->nSaveDif_min = 10000;

     this->nLoc_idcr = 1;
}


std::string trackSystem_pro::output_platepro(
    cv::Mat& matImg_src
    )
{
	std::string stFinal_result = "Null";
	//this->boIsleave = false;
	int i = -1;
	for (auto it=this->veRect_aftboxes.begin();it!=this->veRect_aftboxes.end();it++)
	{
		i++;
		//delete miss vehicle
		if ((*it)[4]<=-5) 
		{
			if (this->nAp_max >=2 && (*it)[5] >= 2 /*&& this->stPlateMax_res != this->stLast_res*/)
			{
				std::string rect_res = "-" +std::to_string(this->reCrop_plate_save.x) + "," +
				std::to_string(this->reCrop_plate_save.y) + "," +
				std::to_string(this->reCrop_plate_save.width) + "," +
				std::to_string(this->reCrop_plate_save.height);
				stFinal_result = (stPlateMax_res!="")? this->stType_max_type+"-"+this->stPlateMax_res+"-1" /*+ rect_res*/:"Null"; // get the final result
				this->stLast_res = this->stPlateMax_res;
				(*it)[5] = -1;
				cout<< "!!!" << endl;
				init_trackoutput();//init
			}
			if ((*it)[5] == -1)this->boIsleave = true;

			it = veRect_aftboxes.erase(it);

			if (this->nPlate_result_lock == i) this->nPlate_result_lock = -1;
			if (this->nPlate_result_lock > i) this->nPlate_result_lock --;
		}
		if(it==veRect_aftboxes.end())break;

		cv::Rect vehicle_box((*it)[0],(*it)[1],(*it)[2],(*it)[3]);
		int nMiss_time = (*it)[4];
		int nRecog_time = (*it)[5];
		int nTrack = (*it)[6];
		int nRect_cy = vehicle_box.y + vehicle_box.height/2;
		int nRect_cx = vehicle_box.x + vehicle_box.width/2;

		if (nTrack >=2 && vehicle_box.x < matImg_src.cols/2 && vehicle_box.y+vehicle_box.height > matImg_src.rows/5 &&
		nRecog_time != -1 && nMiss_time ==0)
		{
			//printf("!!!!here!!\n");
			if(this->nAp_max < 1 && nNone_plate_ct < 1 && vehicle_box.y+vehicle_box.height > 5*matImg_src.rows/6)
			{
				printf("continue 1\n");
				continue;
			}
			if (this->nPlate_result_lock != -1 && this->nPlate_result_lock != i)
			{
				printf("continue 2\n");
				continue;
			}
			if (nRecog_time == 0)init_trackoutput(); //init
			if (nRecog_time == 3)this->nPlate_result_lock = i; //lock the plate result buffer

			std::vector<std::pair<std::string,std::string>> stResult_plate;
			plate_recogByvehicleDeted(vehicle_box,matImg_src,stResult_plate);
			//std::cout << stResult_plate[0].first << std::endl;
			if (stResult_plate.size()==1)
			{
				(*it)[5]++;
				int nAp_num = -1;
				if (stResult_plate[0].first!="无车牌")
				{
					this->vePlate_result.push_back(stResult_plate[0].first);
					nAp_num = cal_resultcount(vePlate_result,stResult_plate[0].first);
					this->veType_list_type.push_back(stResult_plate[0].second);
					int nType_num = cal_resultcount(veType_list_type,stResult_plate[0].second);
					if (nType_num > this->nType_max_num)
					{
						this->nType_max_num = nType_num;
						this->stType_max_type = stResult_plate[0].second;
					}					
					stFinal_result = stResult_plate[0].second + "-" + stResult_plate[0].first + "-0"; // 20200601 lisk add
					//std::cout<<"111111111111111111111111"<<stFinal_result<<std::endl;
				}
				else
				{
					  this->nNone_plate_ct++;
					  this->nLoc_idcr *= -1;
					  stFinal_result = "/-/-/-无车牌-0"; // 20200601 lisk add
					  //std::cout<<"22222222222222222222222222222"<<stFinal_result<<std::endl;
				}
				stFinal_result = (stPlateMax_res!="")? this->stType_max_type+"-"+this->stPlateMax_res + "-0":"/-/-/-无车牌-0";
				std::cout<<"111111111111111111111111:::"<<stFinal_result<<std::endl;
				if (nAp_num > this->nAp_max)
				{
					this->nAp_max = nAp_num;
					this->stPlateMax_res = stResult_plate[0].first;
				}
				if ((nRecog_time == 100 /*!!!*/|| this->nAp_max >=7 || vehicle_box.y+vehicle_box.height > 4*matImg_src.rows/5)/*&&this->stPlateMax_res != this->stLast_res*/)
				//if (((*it)[5] == 1))//2020 5 27 lisk add
				{
					std::cout<<"@@@@@@@@ nRecog_time:"<<nRecog_time<<", nAp_max:"<<nAp_max<<", outView: "
					<<(vehicle_box.y+vehicle_box.height > 4*matImg_src.rows/5)<<"@@@@@@"<<std::endl;
					//printf("xxxxxx!!!\n");
					std::string rect_res = "-" +std::to_string(this->reCrop_plate_save.x) + "," +
					std::to_string(this->reCrop_plate_save.y) + "," +
					std::to_string(this->reCrop_plate_save.width) + "," +
					std::to_string(this->reCrop_plate_save.height);

					if(this->nNone_plate_ct >= 5 && this->nAp_max < 2 && vehicle_box.y+vehicle_box.height > 4*matImg_src.rows/5)  //situation for none-plate
					//if(this->nNone_plate_ct >= 1)//2020 5 27 lisk add
					{
						this->stPlateMax_res = "无车牌";
						this->stType_max_type = "/-/-/"; 
						rect_res = "-0,0,0,0";
					}	          
					stFinal_result = (stPlateMax_res!="")? this->stType_max_type+"-"+this->stPlateMax_res + "-1"/*+ rect_res*/:"Null"; // get the final result
					std::cout<<"222222222222222222222222222:::"<<stFinal_result<<std::endl;
					// 20200601 lisk add
					(*it)[5] = -1; //set the symbol for output
					//(*it)[5] = 0; //2020 5 27 lisk add
					if (this->stPlateMax_res!="" && this->stPlateMax_res!="无车牌")this->stLast_res = this->stPlateMax_res;		  
					init_trackoutput();//init
				}
				if (stFinal_result == "Null" )
				{
					int nDif_y = abs(nRect_cy - matImg_src.rows/2) + abs(nRect_cx - matImg_src.cols/2);
					//std::cout << nRect_cy << " " << matImg_src.rows/2 << std::endl;  
					if(nDif_y < this->nSaveDif_min && this->nAp_max <= 2 || stResult_plate[0].first == this->stPlateMax_res)
					{
						//std::cout << "123123123" << std::endl;
						this->nSaveDif_min = nDif_y;
						stFinal_result = "save_img";
						std::cout<<"333333333333333333333333333:::"<<stFinal_result<<std::endl;
						if(this->reCrop_plate.width > 0 && this->reCrop_plate.height >0)this->reCrop_plate_save = this->reCrop_plate;
					}
				}
			}

		}
	}
	this->veRect_preboxes = this->veRect_aftboxes;

	return stFinal_result;
}


void trackSystem_pro::plate_recogByvehicleDeted( 
     cv::Rect& vehicle_box,
     cv::Mat& matImg_src,
     std::vector<std::pair<std::string,std::string>>& stResult_plate
     )
{

	//printf("begin recog!!!\n");
	this->color = "未知";
	this->layer_type = "single";
	this->stPlate_type = "未知";
	this->plate_type = "未知";

	this->reCrop_plate.x = 0;
	this->reCrop_plate.y = 0;
	this->reCrop_plate.width = 0;
	this->reCrop_plate.height = 0;
	std::string stC_result = "无车牌";

	cv::Mat matCroprgb;
	//printf("locate_plateDetArea!!\n");
	locate_plateDetArea(matImg_src,vehicle_box,matCroprgb);

	cv::Mat matCropgray;
	cv::cvtColor(matCroprgb,matCropgray,cv::COLOR_BGR2GRAY);

	std::vector<cv::Rect> Plateboxes;
	//printf("plate det!!\n");
	Plateboxes = RknnDetInference_Process(matCropgray, Plate_detModel, 200, 200, 1, plate_boxPriors, PLATE_NUM_RESULTS); //detect vehicle

	for (auto platebox:Plateboxes)
	{   
		//cout << Plateboxes.size() << endl;
		rectangle(matCroprgb, platebox, cv::Scalar(0,0, 255),3);
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
		if (Charboxes.size()<=6)
		{
			stResult_plate.push_back(std::pair<std::string,std::string>(stC_result,this->stPlate_type));
			continue;
		}
		//printf("correct_pre\n");
		std::vector<cv::Rect> charbox_Res_pre;
		charbox_Res_pre = Error_Correction_pre(Charboxes,plate_tpbtlim);

		//printf("char recog!!\n");
		std::vector<std::vector<std::pair<float, int>>> array_result;
		char_RecogPro(matPlate.first, charbox_Res_pre, array_result);
		if (this->nCnChar_c>=4)
		{
			stResult_plate.push_back(std::pair<std::string,std::string>(stC_result,this->stPlate_type));
			continue;
		}

		//printf("correct_aft!!\n");
		stC_result = Error_Correction_aft(charbox_Res_pre,array_result);
		stResult_plate.push_back(std::pair<std::string,std::string>(stC_result,this->stPlate_type));
		//get the rect of plate
		this->reCrop_plate = platebox;
		int nArg = (this->nLoc_idcr == 1)?1:0;
		this->reCrop_plate.x = fixEdge_W(platebox.x  + vehicle_box.width/8 + vehicle_box.x - 0.2*platebox.width, matImg_src.cols);
		this->reCrop_plate.y = fixEdge_H(platebox.y  + vehicle_box.height*nArg/2 + vehicle_box.y - 0.5*platebox.height, matImg_src.rows);//(this->nLoc_idcr == 1)?vehicle_box.height/2:0;

		this->reCrop_plate.width = fixEdge_W(this->reCrop_plate.x + this->reCrop_plate.width + 0.4*platebox.width, matImg_src.cols) - this->reCrop_plate.x;
		this->reCrop_plate.height = fixEdge_H(this->reCrop_plate.y + this->reCrop_plate.height + platebox.height, matImg_src.rows) - this->reCrop_plate.y;

		//cout << reCropPl.x << " " << reCropPl.y << " " << reCropPl.width << " " << reCropPl.height << " " << matVehicle.cols << " " << matVehicle.rows << endl; 

		if (this->reCrop_plate.width <= 0 || this->reCrop_plate.height <=0)continue;
		//cv::Mat matPlate_show = matVehicle(reCropPl);
		//rectangle(matImg_src, this->reCrop_plate, cv::Scalar(139, 139,   0, 255), 3);
		cv::imshow("plate",matCroprgb);
	}


	if (stResult_plate.size()>1)
	{
		for (auto it = stResult_plate.begin();it!=stResult_plate.end();it++)
		{
			if((*it).first=="无车牌")it=stResult_plate.erase(it);
			if(it==stResult_plate.end())break;
		}
	}
		if (stResult_plate.size()==0)stResult_plate.push_back(std::pair<std::string,std::string>(stC_result,this->stPlate_type));
}


std::string trackSystem_pro::track_pro(cv::Mat& matImg_src )
{
	//usleep(30000);
	//printf("123123\n");
	std::vector<cv::Rect> Vehicleboxes;
	Vehicleboxes = RknnDetInference_Process(matImg_src, Vehicle_detModel, 176, 176, 3, vehcile_boxPriors, VEHICLE_NUM_RESULTS); //detect vehicle
	cal_rectpro(Vehicleboxes,matImg_src);
	std::string result = output_platepro(matImg_src);
	//std::string result = "Null";

	/*
	std::string rect_res = "0,0,0,0";
	for (auto vebox:Vehicleboxes)
	{
	rect_res = std::to_string(vebox.x) + "," +
	std::to_string(vebox.y) + "," +
	std::to_string(vebox.width) + "," +
	std::to_string(vebox.height);
	break;
	}
	*/
	return result;
	//return rect_res;

}






#endif
