#!/bin/bash

# Ubuntu 运行脚本：逐个执行指定命令

echo "开始执行命�?.."

# 第一个命�?echo "正在执行第一个命�?.."
bash tools/dist.sh train projects/salma/configs/sa2va_1B_NoFilm_exp1.py 4

if [ $? -ne 0 ]; then
    echo "错误：第一个命令执行失败，停止后续执行�?
    exit 1
fi

# 第二个命�?echo "正在执行第二个命�?.."
bash tools/dist.sh train projects/salma/configs/sa2va_1B_NoFilm_exp2.py 4

if [ $? -ne 0 ]; then
    echo "错误：第二个命令执行失败，停止后续执行�?
    exit 1
fi

# 第三个命令（请替换为你实际的命令�?echo "正在执行第三个命�?.."
bash tools/dist.sh train projects/salma/configs/sa2va_1B_NoFilm_exp3.py 4

if [ $? -ne 0 ]; then
    echo "错误：第三个命令执行失败，停止后续执行�?
    exit 1
fi

# 第四个命令（请替换为你实际的命令�?echo "正在执行第四个命�?.."
bash tools/dist.sh train projects/salma/configs/sa2va_1B_NoFilm_exp4.py 4

if [ $? -ne 0 ]; then
    echo "错误：第四个命令执行失败，停止后续执行�?
    exit 1
fi

echo "所有命令执行完成！"
