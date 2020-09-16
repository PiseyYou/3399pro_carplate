#include "uart.h"
#include <string.h>
/***@brief  设置串口通信速率
 *@param  fd     类型 int  打开串口的文件句柄
 *@param  speed  类型 int  串口速度
 *@return  void*/

int speed_arr[] = { B115200,B38400, B19200, B9600, B4800, B2400, B1200, B300,
	B38400, B19200, B9600, B4800, B2400, B1200, B300, };
int name_arr[] = {115200, 38400,  19200,  9600,  4800,  2400,  1200,  300,
	38400,  19200,  9600, 4800, 2400, 1200,  300, };

uartdev::uartdev(const char *arg1,int arg2)
{
	dev = arg1;
	bardrate = arg2;
}

uartdev::~uartdev()
{
	//return;
}

void uartdev::close(int fd)
{
	close(fd);
	//return true;
}
void uartdev::set_speed(int fd)   //Linux 下串口USB等设备通信编程入门2中有终端属性的获取设置函数
{
        int speed = bardrate;
	unsigned int   i;
	int   status;
	struct termios   Opt,newio;
	tcgetattr(fd, &Opt);
	for ( i= 0;  i < sizeof(speed_arr) / sizeof(int);  i++)
	{
		if  (speed == name_arr[i])
		{
			printf("speed:%d\n",speed);
			tcflush(fd, TCIOFLUSH);//Update the options and do it NOW
			cfsetispeed(&newio, B115200);
			cfsetospeed(&newio, B115200);
			status = tcsetattr(fd, TCSANOW, &newio);
			if  (status != 0)
				perror("tcsetattr fd1");
			return;
		}
		tcflush(fd,TCIOFLUSH);
	}
}
/**
 *@brief   设置串口数据位，停止位和效验位
 *@param  fd     类型  int  打开的串口文件句柄*
 *@param  databits 类型  int 数据位   取值 为 7 或者8   数据位为7位或8位
 *@param  stopbits 类型  int 停止位   取值为 1 或者2*    停止位为1或2位
 *@param  parity  类型  int  效验类型 取值为N,E,O,,S     N->无奇偶校验，O->奇校验 E->为偶校验，
 */
int uartdev::set_parity(int fd,int databits,int stopbits,int parity)
{
	struct termios options,newoptions;
	if  ( tcgetattr( fd,&options)  !=  0)
	{
		perror("SetupSerial 1");
		return -1;
	}
	bzero(&newoptions,sizeof(newoptions));
	newoptions.c_cflag |= CLOCAL | CREAD;
	options.c_cflag &= ~CSIZE;
	switch (databits) /*设置数据位数*/
	{
		case 7:
			newoptions.c_cflag |= CS7;
			break;
		case 8:
			newoptions.c_cflag |= CS8;
			break;
		default:
			fprintf(stderr,"Unsupported data size\n");
			return -1;
	}
	switch (parity)
	{
		case 'n':
		case 'N':
			newoptions.c_cflag &= ~PARENB;   /* Clear parity enable */
			newoptions.c_iflag &= ~INPCK;     /* Enable parity checking */
			break;
		case 'o':
		case 'O':
			newoptions.c_cflag |= (PARODD | PARENB);  /* 设置为奇效验*/
			newoptions.c_iflag |= INPCK;             /* Disnable parity checking */
			break;
		case 'e':
		case 'E':
			newoptions.c_cflag |= PARENB;     /* Enable parity */
			newoptions.c_cflag &= ~PARODD;   /* 转换为偶效验*/
			newoptions.c_iflag |= INPCK;       /* Disnable parity checking */
			break;
		case 'S':
		case 's':  /*as no parity*/
			newoptions.c_cflag &= ~PARENB;
			newoptions.c_cflag &= ~CSTOPB;
			break;
		default:
			fprintf(stderr,"Unsupported parity\n");
			return -1;
	}
	switch(bardrate)
	{
		case 115200:
			printf("speed:%d\n",bardrate);
			cfsetispeed(&newoptions, B115200);
			cfsetospeed(&newoptions, B115200);
			break;
		default:
			fprintf(stderr,"Unsupported bardrate\n");
			return -1;
	}
	/* 设置停止位*/
	switch (stopbits)
	{
		case 1:
			newoptions.c_cflag &= ~CSTOPB;
			break;
		case 2:
			newoptions.c_cflag |= CSTOPB;
			break;
		default:
			fprintf(stderr,"Unsupported stop bits\n");
			return -1;
	}
	/* Set input parity option */
	//if (parity != 'n' || parity !='N')
	//	options.c_iflag |= INPCK;
	newoptions.c_cc[VTIME] = 1; // 15 seconds
	newoptions.c_cc[VMIN] = 0;

	tcflush(fd,TCIFLUSH); /* Update the options and do it NOW */
	if (tcsetattr(fd,TCSANOW,&newoptions) != 0)
	{
		perror("SetupSerial 3");
		return -1;
	}
	return 0;
}

//打开串口 
int uartdev::open_dev()
{
	const char *Dev = dev;
        //std::string Dev = dev;
	int	fd = open( Dev, O_RDWR );         //| O_NOCTTY | O_NDELAY
	dev_fd = fd;
	if (-1 == fd)
	{ 		
		perror("Can't Open Serial Port");
		return -1;
	}
	else
		return fd;

}
