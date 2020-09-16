#ifndef __SERIAL_H__
#define __SERIAL_H__

#include <stdio.h>      /*标准输入输出定义*/
#include <stdlib.h>     /*标准函数库定义*/
#include <unistd.h>     /*Unix标准函数定义*/
#include <sys/types.h>  /**/
#include <sys/stat.h>   /**/
#include <fcntl.h>      /*文件控制定义*/
#include <termios.h>    /*PPSIX终端控制定义*/
#include <errno.h>      /*错误号定义*/


class uartdev
{

   public:
     uartdev(const char *dev="/dev/ttyUSB0",int brad=115200);
     ~uartdev();
     void close(int fd);
     int open_dev();
     void set_speed(int fd);
     int set_parity(int fd,int databits,int stopbits,int parity);
   private:
     const char *dev;
     int dev_fd;
     int bardrate;

};
#endif
