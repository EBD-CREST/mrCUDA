#!/bin/bash

#k=0
#while [ $k -lt 15 ]
#do
#    num=`echo "2^$k" | bc`
#    j=0
#    while [ $j -lt 10 ]
#    do
#        memsize=`echo "2^(20+$j)" | bc`
#        i=0
#        while [ $i -lt  10 ]
#        do
#            echo "mrcuda $memsize $num"
#            taskset 1 ~/src/mrCUDA/scripts/mrCUDAExec -t IB -s rc015 --switch-threshold=1 -- ./memcpybw $memsize $num
#            i=`expr $i + 1`
#            sleep 1
#        done
#        j=`expr $j + 1`
#    done
#    k=`expr $k + 1`
#done

j=0
while [ $j -lt 20 ]
do
    memsize=`echo "2^($j)" | bc`
    i=0
    while [ $i -lt  10 ]
    do
        echo "mrcuda $memsize 1"
        taskset 1 ~/src/mrCUDA/scripts/mrCUDAExec -t IB -s rc015 --switch-threshold=1 -- ./memcpybw $memsize 1
        i=`expr $i + 1`
        sleep 1
    done
    j=`expr $j + 1`
done
