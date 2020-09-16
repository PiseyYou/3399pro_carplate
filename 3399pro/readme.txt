ai3399pro运行车牌识别步骤

1)更新系统包,更新rknntoolkit
sudo dnf clean all
sudo dnf update

2)安装RGA,MPP,RTSP,opencv相关库

sudo dnf install -y cmake gcc gcc-c++ protobuf-devel protobuf-compiler lapack-devel opencv-devel
sudo dnf install -y python3-devel python3-opencv python3-numpy-f2py python3-h5py python3-lmdb  python3-grpcio
sudo dnf install librockchip_mpp-devel
sudo dnf install librockchip_rga-devel
sudo dnf install librockchip_rtsp-devel curl-devel
sudo dnf install libgnome-devel gnome-devel-docs
注:ai3399pro使用不了原厂rga库

3)cd build & make

注:
1.相关库在lib文件夹，包含发送库zmq，编译好的rga以及CMakeLists，其中zmq.h,zmq.hpp放入/usr/include/,其余放入/usr/lib64/
2.使用<numpy/arrayobject.h>需sudo dnf install python-numpy
3.c++调用python声明一个对象一定要用PyObject_CallFunctionObjArgs，不要使用PyInstanceMethod_New，否则无法通过self调用成员变量和函数
4.c++调用python的tensorflow要加一句PyRun_SimpleString("sys.argv=['']"); 传入一个空参数

4)设置开机自启动
1.将.desktop放入/etc/xdg/autostart/中
2.设置自动登入：
/etc/lxdm/lxdm.conf里面配置autologin。如下：

## uncomment and set autologin username to enable autologin
autologin=toybrick

## uncomment and set timeout to enable timeout autologin,
## the value should >=5
timeout=0

## default session or desktop used when no systemwide config
# session=/usr/bin/startlxde

## uncomment and set to set numlock on your keyboard
# numlock=0

## set this if you don't want to put xauth file at ~/.Xauthority
# xauth_path=/tmp

# not ask password for users who have empty password
# skip_password=1

注:rc-local.service设置的是后台运行，千万不要用multi-user模式,板子会变砖头！
