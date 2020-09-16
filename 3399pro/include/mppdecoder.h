#ifndef MPPDECODER_H
#define MPPDECODER_H
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

#include <time.h>
#include <syslog.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <string.h>
#include <pthread.h>
#include "tools.h"
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
//#include "rkdrm.h"
extern "C"
{
#include "rockchip/rk_type.h"
#include "rockchip/rk_mpi.h"
#include "rockchip/mpp_buffer.h"
#include "rockchip/mpp_frame.h"
#include "rockchip/mpp_packet.h"
}

/*
 * Usually hardware decode one
 * frame which resolution is 1080p needs 2 ms
 */
#define MPP_H264_DECODE_TIMEOUT   (3)  // milliseconds
/*
 * Maxium frames in the queue
 */
#define MAX_BUFFER_FRAMES         (30)
#define MAX_BUFFER_FRAMES2         (200)
#define FRAME_SIZE                (1920*1088*2) /* h_stride*v_stride*2 */
//#define FRAME_SIZE                (2048*1536*2) /* h_stride*v_stride*2 */
//#define Record_H264 1
typedef struct _frame_st {
    uint64_t cap_ms;
    uint32_t width;
    uint32_t height;
    uint32_t h_stride;
    uint32_t v_stride;
    uint8_t  data[FRAME_SIZE];
} frame_st;


int mppDecoder();

void decoder_routine(long int timestamp);

float decoder_fps();

MppFrame get_current_frame();

/*
 * Get a frame from decoder
 * DON'T free the memory outside
 */
RK_U8* decoder_frame(long int &timestamp);

#endif // MPPDECODER_H
