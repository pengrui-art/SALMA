#!/bin/bash

# Ubuntu 运行脚本：逐个执行指定命令

echo "开始执行命�?.."

# 第一个命�?echo "正在执行第一个命�?.."
./projects/salma/evaluation/dist_test.sh projects/salma/evaluation/ref_vos_eval.py save_model/sa2va_1b_exp2 4 --work-dir eval/sa2va-1b-exp2/ref_vos/Davis --dataset DAVIS
python tools/eval/eval_davis.py eval/sa2va-1b-exp2/ref_vos/Davis/results.json

if [ $? -ne 0 ]; then
    echo "错误：第一个命令执行失败，停止后续执行�?
    exit 1
fi

# 第二个命�?echo "正在执行第二个命�?.."
./projects/salma/evaluation/dist_test.sh projects/salma/evaluation/ref_vos_eval.py save_model/sa2va_1b_exp3 4 --work-dir eval/sa2va-1b-exp3/ref_vos/Davis --dataset DAVIS
python tools/eval/eval_davis.py eval/sa2va-1b-exp3/ref_vos/Davis/results.json

if [ $? -ne 0 ]; then
    echo "错误：第二个命令执行失败，停止后续执行�?
    exit 1
fi

# 第三个命令（请替换为你实际的命令�?echo "正在执行第三个命�?.."
./projects/salma/evaluation/dist_test.sh projects/salma/evaluation/ref_vos_eval.py save_model/sa2va_1b_exp4 4 --work-dir eval/sa2va-1b-exp4/ref_vos/Davis --dataset DAVIS
python tools/eval/eval_davis.py eval/sa2va-1b-exp4/ref_vos/Davis/results.json

if [ $? -ne 0 ]; then
    echo "错误：第三个命令执行失败，停止后续执行�?
    exit 1
fi

# 第四个命令（请替换为你实际的命令�?echo "正在执行第四个命�?.."
./projects/salma/evaluation/dist_test.sh projects/salma/evaluation/ref_vos_eval.py save_model/sa2va_1b_exp5 4 --work-dir eval/sa2va-1b-exp5/ref_vos/Davis --dataset DAVIS
python tools/eval/eval_davis.py eval/sa2va-1b-exp5/ref_vos/Davis/results.json

if [ $? -ne 0 ]; then
    echo "错误：第四个命令执行失败，停止后续执行�?
    exit 1
fi

echo "所有命令执行完成！"
