#include "mppdecoder.h"
#include "mutex"
#include "condition_variable"
#include "thread"

#if DEBUG
#define mpp_log printf
#define mpp_err printf
void mpp_dump(uint8_t* data, int sz)
{
   int i = 0;
   for (i = 0; i < sz; i++) {
       printf("%02x ", data[i]);
       if ((i & 0x0f) == 0x0f) printf("\n");
   }
   if ((sz & 0x0f) != 0) printf("\n");
}
#else
#define mpp_log
#define mpp_err 
#define mpp_dump
#endif

using namespace std;
using namespace cv;

mutex mutex_basedata;
condition_variable cond_basedata;
bool gFlagNewBasedata = false;

MppCodingType   type = MPP_VIDEO_CodingAVC;
MppCtx          ctx;
MppApi          *mpi;
/* end of stream flag when set quit the loop */
RK_U32          eos;
RK_U32          pkt_eos;
/* buffer for stream data */
uint8_t         *packet_buffer;
uint32_t        packet_wpos = 0;

//uint8_t         *packet_buffer_out;
//int        packet_wpos_out = 0;
/* input and output */
MppBufferGroup  frm_grp;
MppBufferGroup  pkt_grp;
MppPacket       packet;
size_t          _packet_size  = SZ_4K*2;
MppFrame        frame[MAX_BUFFER_FRAMES];
MppBuffer 	buffer[MAX_BUFFER_FRAMES];
RK_U8 		*base[MAX_BUFFER_FRAMES];
RK_U8 		*base2[MAX_BUFFER_FRAMES];
RK_U64          frame_count = 0;
RK_U64          frame_discards = 0;
RK_U64          frame_err = 0;
long int timestamp1,timestamp2 = -1;
/* FPS calculator */
RK_U64 fps_ms;
RK_U32 fps_counter;
float fps = 0.0;
/* Output buffer */
pthread_mutex_t frames_lock;
frame_st* frames[MAX_BUFFER_FRAMES];

RK_U32  frames_r,frames_w;
int w_flag = 0;
int Is_out_frame = 0,is_begin_record = 0;

static void frame_out(RK_U64 ms);
#ifdef Record_H264
void thread_files_remove();
	int writecount = 0;
	int writeseconds = 90;//one h264 file record time
    	bool isNextDir = false;
   	int filenums = 96; //number of file
	int dirnums = 60; // number of dir
	int nCount_h264 = 1,nRecord_h264 = 0;
	string dir = "/home/toybrick/lisk_project/mnt/";//"/mnt/";
        string dir3 = dir + "video_record/";
	string dir33,dir_now_remove; 
	FILE *fp_h264;
    	struct timeval tv_h264;
    	struct timeval ts_h264;
	mutex mutex_rm_h264;
	condition_variable cond_rm_h264;
	bool gFlagRmH264 = false;
#endif
int mkdir_video(string dir)
{
    if(access(dir.c_str(),0) == -1)
    {
	//cout << dir << " is not existing" << endl;
        int flag = mkdir(dir.c_str(),0777);
	if(flag == 0) 
        {
	    //cout << "make successfully." << endl;
	    return 1;
	}
	else
	{
	   cout << "make" << dir << "errorly." << endl;
 	   return 0;
	}
    }
    return 1;
}


#ifdef Record_H264
void thread_files_remove()
{
    string dir_rm;
    printf("thread_files_remove begin.\n");
    while(1)
    {
	unique_lock<mutex> lock_rm_h264(mutex_rm_h264);
	cond_rm_h264.wait(lock_rm_h264,[](){return gFlagRmH264; });
	gFlagRmH264 = false;
	lock_rm_h264.unlock();
	dir_rm = dir_now_remove + "*.h264";
	std::vector<cv::String> files;
    	cv::glob(dir_rm,files,true);
	//printf("dir_rm = %s,size = %d\n",dir_rm.c_str(),files.size());
    	for(int i=0; i<files.size(); i++)
    	{
	     if(remove(files[i].c_str()) != 0)
	     {
		printf("remove file failed.\n");
		continue;
	     }
	     //printf("remove %s\n",files[i].c_str());
    	}
	gFlagRmH264 = false;

    }
}

void Write_H264_Datas()
{
    if(writecount == 0)
   {
		    if(packet_buffer[0] == 0x00 && packet_buffer[1] == 0x00 && packet_buffer[2] == 0x00 && packet_buffer[3] == 0x01 && packet_buffer[4] == 0x67)
	            {
			timespec time;
			clock_gettime(CLOCK_REALTIME,&time);
			tm nowTime;
			localtime_r(&time.tv_sec,&nowTime);
			gettimeofday(&tv_h264,NULL);
			char buf[1024];	
			sprintf(buf,"%s%04d%02d%02d%02d%02d%02d.h264",dir33.c_str(),nowTime.tm_year+1900,nowTime.tm_mon+1,
			        nowTime.tm_mday,nowTime.tm_hour,nowTime.tm_min,nowTime.tm_sec);
			fp_h264 = fopen( buf,"wb");
			if(fp_h264 == NULL)  
                	{
		    		printf("could not record video.\n");
		    		return;
			}
			nRecord_h264 ++;
	     		if(nRecord_h264 >= filenums)
	     		{
		   		//nCount_h264 = nCount_h264 + 1;
	    			//if(nCount_h264 > dirnums) nCount_h264 = 1;
		    		dir33 = dir3 + to_string(nCount_h264) + "/";
		    		if(mkdir_video(dir33) == 1)
		    		{
		    			nRecord_h264 = 0;
					nCount_h264 = nCount_h264 + 1;
	    				if(nCount_h264 > dirnums) nCount_h264 = 1;
					dir_now_remove = dir3 + to_string(nCount_h264) + "/";
					if(mkdir_video(dir_now_remove) == 1)
		    			{
		    				unique_lock<mutex> lock_rm_h264(mutex_rm_h264);
		    				gFlagRmH264 = true;
						lock_rm_h264.unlock();
						cond_rm_h264.notify_all();
		    			}
		    		}
	     		}
			fwrite(packet_buffer, packet_wpos, 1, fp_h264);
		    	fflush(fp_h264);
		    	writecount ++;
		    }

	  }
	  else
	  {
		
	            fwrite(packet_buffer, packet_wpos, 1, fp_h264);
		    fflush(fp_h264);
		    writecount ++;
		gettimeofday(&ts_h264,NULL);
		if((ts_h264.tv_sec-tv_h264.tv_sec) >= writeseconds)
		{
		     writecount = 0;
		     fclose(fp_h264);
		     fp_h264 = NULL;
		}
	   }
}
#endif
int mppDecoder()
{
    type = MPP_VIDEO_CodingAVC;

    MPP_RET ret         = MPP_OK;
    MpiCmd mpi_cmd      = MPP_CMD_BASE;
    MppParam param      = NULL;
    RK_U32 need_split   = 1;

    packet_buffer = (uint8_t*)malloc(_packet_size);
    //packet_buffer_out = (uint8_t*)malloc(_packet_size);
    ret = mpp_packet_init(&packet, packet_buffer, _packet_size);
    if (MPP_OK != ret) {
        mpp_err("mpi->control failed\n");
        return -1;
    }

    ret = mpp_create(&ctx, &mpi);
    if (MPP_OK != ret) {
        mpp_err("mpi->control failed\n");
        return -1;
    }

    mpi_cmd = MPP_DEC_SET_PARSER_SPLIT_MODE;
    param = &need_split;
    ret = mpi->control(ctx, mpi_cmd, param);
    if (MPP_OK != ret) {
        mpp_err("mpi->control failed\n");
        return -1;
    }

    ret = mpp_init(ctx, MPP_CTX_DEC, type);
    if (MPP_OK != ret) {
        mpp_err("mpp_init failed\n");
        return -1;
    }

    fps_ms = current_ms();
    fps_counter = 0;

    pthread_mutex_init(&frames_lock, NULL);
    frames_r = frames_w = 0;
    memset(frames, 0, sizeof(frames));
#ifdef Record_H264
       //thread files_remove(thread_files_remove);
        if(mkdir_video(dir3) == 1)
	{
	    std::vector<cv::String> files;
  	    string dir31 = dir3 + "*.h264";
  	    cv::glob(dir31,files,true);
	    //cout << dir31 << " files.size():" << files.size() << endl;
	    nCount_h264 = files.size()/filenums + 1;
	    if(nCount_h264 > dirnums) nCount_h264 = 1;
	    printf("nCount_h264 = %d,files.size() =%d,nRecord_h264 = %d\n",nCount_h264,files.size(),nRecord_h264);
	}
        dir33 = dir3 + to_string(nCount_h264) + "/";
	if(mkdir_video(dir33) == 1)
	{
	    std::vector<cv::String> files;
  	    string dir34 = dir33 + "*.h264";
  	    cv::glob(dir34,files,true);
	    if(files.size() > 0)
	    {
		nCount_h264 = nCount_h264 + 1;
		if(nCount_h264 > dirnums) nCount_h264 = 1;
		dir33 = dir3 + to_string(nCount_h264) + "/";
		if(mkdir_video(dir33) == 1)
		{
		    std::vector<cv::String> h264_files;
		    string dir35 = dir33 + "*.h264";
		    cv::glob(dir35,h264_files,true);
    		    for(int i=0; i<h264_files.size(); i++)
    		    {
	     		if(remove(h264_files[i].c_str()) != 0) 
			{
				printf("remove file failed.\n");
				continue;
			}
    		    }
		    nRecord_h264 = 0;
		    nCount_h264 = nCount_h264 + 1;
		    if(nCount_h264 > dirnums) nCount_h264 = 1;
		    dir_now_remove = dir3 + to_string(nCount_h264) + "/";
		    if(mkdir_video(dir_now_remove) == 1)
		    {
			unique_lock<mutex> lock_rm_h264(mutex_rm_h264);
		    	gFlagRmH264 = true;
			lock_rm_h264.unlock();
			cond_rm_h264.notify_all();
		    }
		  
		}

	    }
	    else 
	    {
		nRecord_h264 = 0;
		nCount_h264 = nCount_h264 + 1;
		if(nCount_h264 > dirnums) nCount_h264 = 1;
		dir_now_remove = dir3 + to_string(nCount_h264) + "/";
		if(mkdir_video(dir_now_remove) == 1)
		{
		    unique_lock<mutex> lock_rm_h264(mutex_rm_h264);
		    gFlagRmH264 = true;
		    lock_rm_h264.unlock();
		    cond_rm_h264.notify_all();	    
		}
	    }
	    printf("nCount_h264 = %d,files.size() =%d,nRecord_h264 = %d\n",nCount_h264,files.size(),nRecord_h264);
	}
	
#endif
    /*for(int i=0;i<MAX_BUFFER_FRAMES;i++)
    {
	frames[i] = (frame_st*)malloc(sizeof(frame_st));
    }*/
    return 0;
}

void decoder_routine(long int timestamp)
{
    int ret;
    RK_U32 pkt_done = 0;
    timestamp1 = timestamp;
    if(timestamp2 == -1) timestamp2 = timestamp;
    else if(timestamp2 != timestamp1)
    {
	//printf("timestamp:%ld\n",timestamp2);
	timestamp2 = timestamp1;
    }
#ifdef Record_H264
    Write_H264_Datas();
#endif
    // write data to packet
    mpp_packet_write(packet, 0, packet_buffer, packet_wpos);
    // reset pos and set valid length
    mpp_packet_set_pos(packet, packet_buffer);
    mpp_packet_set_length(packet, packet_wpos);
    packet_wpos = 0;
    base[frames_r] = NULL;
    // setup eos flag
    if (pkt_eos)
        mpp_packet_set_eos(packet);

    do {

        RK_S32 times = 5;
        // send the packet first if packet is not done
        if (!pkt_done) {
            ret = mpi->decode_put_packet(ctx, packet);
            if (MPP_OK == ret)
                pkt_done = 1;
        }

        // then get all available frame and release
        do {
            RK_S32 get_frm = 0;
            RK_U32 frm_eos = 0;

        try_again:	
            ret = mpi->decode_get_frame(ctx, &frame[frames_r]);
            if (MPP_ERR_TIMEOUT == ret) {
                if (times > 0) {
                    times--;
                    msleep(MPP_H264_DECODE_TIMEOUT);
                    goto try_again;
                }
                mpp_err("decode_get_frame failed too much time\n");
            }
            if (MPP_OK != ret) {
                mpp_err("decode_get_frame failed ret %d\n", ret);
                break;
            }

            if (frame[frames_r]) {
                if (mpp_frame_get_info_change(frame[frames_r])) {
                    RK_U32 width = mpp_frame_get_width(frame[frames_r]);
                    RK_U32 height = mpp_frame_get_height(frame[frames_r]);
                    RK_U32 hor_stride = mpp_frame_get_hor_stride(frame[frames_r]);
                    RK_U32 ver_stride = mpp_frame_get_ver_stride(frame[frames_r]);

                    //mpp_log("decode_get_frame get info changed found\n");
                    //mpp_log("decoder require buffer w:h [%d:%d] stride [%d:%d]\n",
                    //        width, height, hor_stride, ver_stride);

                    /*
                     * NOTE: We can choose decoder's buffer mode here.
                     * There are three mode that decoder can support:
                     *
                     * Mode 1: Pure internal mode
                     * In the mode user will NOT call MPP_DEC_SET_EXT_BUF_GROUP
                     * control to decoder. Only call MPP_DEC_SET_INFO_CHANGE_READY
                     * to let decoder go on. Then decoder will use create buffer
                     * internally and user need to release each frame they get.
                     *
                     * Advantage:
                     * Easy to use and get a demo quickly
                     * Disadvantage:
                     * 1. The buffer from decoder may not be return before
                     * decoder is close. So memroy leak or crash may happen.
                     * 2. The decoder memory usage can not be control. Decoder
                     * is on a free-to-run status and consume all memory it can
                     * get.
                     * 3. Difficult to implement zero-copy display path.
                     *
                     * Mode 2: Half internal mode
                     * This is the mode current test code using. User need to
                     * create MppBufferGroup according to the returned info
                     * change MppFrame. User can use mpp_buffer_group_limit_config
                     * function to limit decoder memory usage.
                     *
                     * Advantage:
                     * 1. Easy to use
                     * 2. User can release MppBufferGroup after decoder is closed.
                     *    So memory can stay longer safely.
                     * 3. Can limit the memory usage by mpp_buffer_group_limit_config
                     * Disadvantage:
                     * 1. The buffer limitation is still not accurate. Memory usage
                     * is 100% fixed.
                     * 2. Also difficult to implement zero-copy display path.
                     *
                     * Mode 3: Pure external mode
                     * In this mode use need to create empty MppBufferGroup and
                     * import memory from external allocator by file handle.
                     * On Android surfaceflinger will create buffer. Then
                     * mediaserver get the file handle from surfaceflinger and
                     * commit to decoder's MppBufferGroup.
                     *
                     * Advantage:
                     * 1. Most efficient way for zero-copy display
                     * Disadvantage:
                     * 1. Difficult to learn and use.
                     * 2. Player work flow may limit this usage.
                     * 3. May need a external parser to get the correct buffer
                     * size for the external allocator.
                     *
                     * The required buffer size caculation:
                     * hor_stride * ver_stride * 3 / 2 for pixel data
                     * hor_stride * ver_stride / 2 for extra info
                     * Total hor_stride * ver_stride * 2 will be enough.
                     *
                     * For H.264/H.265 20+ buffers will be enough.
                     * For other codec 10 buffers will be enough.
                     */
                    ret = mpp_buffer_group_get_internal(&frm_grp, MPP_BUFFER_TYPE_DRM);
                    if (ret) {
                        mpp_err("get mpp buffer group  failed ret %d\n", ret);
                        break;
                    }
                    mpi->control(ctx, MPP_DEC_SET_EXT_BUF_GROUP, frm_grp);
                    mpi->control(ctx, MPP_DEC_SET_INFO_CHANGE_READY, NULL);

                } else {
                    RK_U32 err_info = mpp_frame_get_errinfo(frame[frames_r]) | mpp_frame_get_discard(frame[frames_r]);
                    if (err_info) {
                        frame_err++;
                        mpp_log("decoder_get_frame get err info:%d discard:%d.\n",
                                mpp_frame_get_errinfo(frame[frames_r]), mpp_frame_get_discard(frame[frames_r]));
                    }
                    else {
                        /* FPS calculation */
                        fps_counter++;
                        frame_count++;
                        RK_U64 diff, now = current_ms();
                        if ((diff = now - fps_ms) >= 1000) {
                            fps = fps_counter / (diff / 1000.0);
                            fps_counter = 0;
                            fps_ms = now;
                            //printf("decode_get_frame get frame %llu, error %llu, discard %llu, FPS = %3.2f\n", frame_count, frame_err, frame_discards, fps);
                        }

                        /** Got a frame */
			//Is_out_frame = 1;
                        frame_out(now);
			
                    }
                }
                frm_eos = mpp_frame_get_eos(frame[frames_r]);
                mpp_frame_deinit(&frame[frames_r]);
                frame[frames_r] = NULL;
		frames_r ++;
		if (frames_r >= MAX_BUFFER_FRAMES)
            		frames_r = 0;
                get_frm = 1;
            }

            // if last packet is send but last frame is not found continue
            if (pkt_eos && pkt_done && !frm_eos) {
                //msleep(MPP_H264_DECODE_TIMEOUT);
                continue;
            }

            if (frm_eos) {
                mpp_log("found last frame\n");
		mpp_buffer_group_put(frm_grp);
                break;
            }

            if (!get_frm)
                break;
        } while (1);

        if (pkt_done)
            break;

        /*
         * why sleep here:
         * mpi->decode_put_packet will failed when packet in internal queue is
         * full,waiting the package is consumed .
         */
        msleep(MPP_H264_DECODE_TIMEOUT);

    } while (1);
}

void frame_out(RK_U64 ms)
{
#if 1
    //if(gFlagNewBasedata && w_flag == frames_w) {printf("oops\n");return;}
    RK_U32 width    = 0;
    RK_U32 height   = 0;
    RK_U32 h_stride = 0;
    RK_U32 v_stride = 0;
    MppFrameFormat fmt  = MPP_FMT_YUV420SP;
    buffer[frames_r] = NULL;
    base[frames_w] = NULL;

    if (NULL == frame[frames_r])
    {
	if(frames_r > 0) frames_r -= 1;
	else if(frames_r == 0) frames_r = MAX_BUFFER_FRAMES - 1;
        return ;
    }

    fmt      = mpp_frame_get_fmt(frame[frames_r]);
    buffer[frames_r] = mpp_frame_get_buffer(frame[frames_r]);

    if (NULL == buffer)
    {
	if(frames_r > 0) frames_r -= 1;
	else if(frames_r == 0) frames_r = MAX_BUFFER_FRAMES - 1;
        return ;
    }
    //mpp_log("Frame width=%u, height=%u, h_stride=%u, v_stride=%u, buffer size=%lu.\n", width, height, h_stride, v_stride, mpp_buffer_get_size(buffer));
    
    base[frames_w] = (RK_U8 *)mpp_buffer_get_ptr(buffer[frames_r]);

    frames_w++;
    //if(frames_w >= MAX_BUFFER_FRAMES) frames_w = 0
    frames_w %= MAX_BUFFER_FRAMES;
    unique_lock<mutex> lock_basedata(mutex_basedata);
    gFlagNewBasedata = true;
     lock_basedata.unlock();

    cond_basedata.notify_all();
    switch (fmt) {
    case MPP_FMT_YUV420SP : // YUV NV12
    {
	;
        //put_frame(ms, base[frames_r], width, height, h_stride, v_stride, mpp_buffer_get_size(buffer[frames_r]));
	//printf("mpp_buffer_get_size(buffer):%d\n",mpp_buffer_get_size(buffer));
    }
        break;
    default :
        mpp_err("not supported format %d\n", fmt);
        break;
    }
#else
    (void)ms;
    rkdrm_display(frame[frames_r]);
    //mpp_log("picture of count: %u\n", mpp_frame_get_poc(frame));
#endif
}


RK_U8* decoder_frame(long int &timestamp)
{
    timestamp = timestamp2;
    if(frames_w > 0) 
    {
	//w_flag = frames_w - 1;
	//cond_basedata.notify_all();
	return base[frames_w - 1];
	
    }
    else
    {
	//w_flag = MAX_BUFFER_FRAMES - 1;
	//cond_basedata.notify_all();
	return base[MAX_BUFFER_FRAMES - 1];
    }
    
}

float decoder_fps() {
    return fps;
}
