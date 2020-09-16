i=0
Max_temp=0
Max_date=0
while true
do
    ((i=i+1))
    if ((i%100000==0));then
       DATE=$(date "+%Y-%m-%d %H:%M:%S")
       TEMP=$(cat /sys/devices/virtual/thermal/thermal_zone0/temp)
       
       if (($TEMP>$Max_temp));then
          Max_temp=$TEMP
          Max_date=$DATE
       fi
       echo $DATE"--"$TEMP "the MaxTemp is :" $Max_date $Max_temp >> templog.txt
       echo $DATE"--"$TEMP "the MaxTemp is :" $Max_date $Max_temp
       i=0
    fi    



done
