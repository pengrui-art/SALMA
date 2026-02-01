#!/bin/bash

# Ubuntu 运行脚本：逐个执行指定命令

echo "开始执行命�?.."



# 第二个命�?echo "正在执行第二个命�?.."
python projects/salma/evaluation/refcoco_eval_splits.py \
    save_model/sa2va_1b_exp4 \
    --dataset refcoco_plus \
    --splits all \
    --work-dir eval/sa2va-1b-exp4/refcoco+ \
    --num-gpus 4

if [ $? -ne 0 ]; then
    echo "错误：第二个命令执行失败，停止后续执行�?
    exit 1
fi

# 第三个命令（请替换为你实际的命令�?echo "正在执行第三个命�?.."
python projects/salma/evaluation/refcoco_eval_splits.py \
    save_model/sa2va_1b_exp4 \
    --dataset refcocog \
    --splits all \
    --work-dir eval/sa2va-1b-exp4/refcocog \
    --num-gpus 4

if [ $? -ne 0 ]; then
    echo "错误：第三个命令执行失败，停止后续执行�?
    exit 1
fi

echo "所有命令执行完成！"
