#!/bin/bash

#i=0
#while [ $i -lt  10 ]
#do
#    echo "nullker mrcuda $i"
#    taskset 1 ~/src/mrCUDA/scripts/mrCUDAExec -t IB -f ~/src/mrCUDA/scripts/sf.in -n 2 -- ./nullker
#    i=`expr $i + 1`
#done
#
#sleep 1
#
#i=0
#while [ $i -lt  10 ]
#do
#    echo "nullker native $i"
#    taskset 1 ./nullker
#    i=`expr $i + 1`
#done
#
#sleep 1

i=8
while [ $i -lt  10 ]
do
    echo "cudamemcpy mrcuda $i"
    taskset 1 ~/src/mrCUDA/scripts/mrCUDAExec -t IB -f ~/src/mrCUDA/scripts/sf.in -n 2 -- ./cudamemcpy
    i=`expr $i + 1`
done

sleep 1

i=0
while [ $i -lt  10 ]
do
    echo "cudamemcpy native $i"
    taskset 1 ./cudamemcpy
    i=`expr $i + 1`
done

