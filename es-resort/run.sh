#!/bin/bash

<<'COMMENT'
es-resort服务部署流程

提前下载和部署 bert_as_service 服务

提前下载 utils-toolkit 工具包
git clone http://git.longhu.net/ningshixian/utils_toolkit.git

测试sbert服务是否可用
curl -X POST --header 'Content-Type: application/json' --header 'Accept: application/json' -d '{"text": "LXHBASE"}' 'http://10.231.9.140:8096/sbert'

# 虚拟环境搭建(非lhadmin用户)
conda create --name nsxenv python=3.6
source activate nsxenv
mkdir ningshixian/work/

# 下载es-resort项目
git clone http://git.longhu.net/ningshixian/es-resort.git
sudo chown -R ningshixian:ningshixian ./es-resort
cd es-resort/sample/
mkdir logs configs model data data/train data/dev data/test
cd data
vi 中文停用词表.text

# 修改config配置文件
vi ../configs/config.ini
# 增加pip配置文件
mkdir ~/.pip
vi ~/.pip/pip.conf

sudo yum install lsof
pip install -r requirements.txt
pip install -U scikit-learn==0.20.3


sbert依赖pytorch，且需要提前下载预训练好的多语模型
wget -c https://download.pytorch.org/whl/cpu/torch-1.6.0%2Bcpu-cp37-cp37m-linux_x86_64.whl
> pip install /data/software/torch-1.6.0+cpu-cp37-cp37m-linux_x86_64.whl
wget -c https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/v0.2/distilbert-multilingual-nli-stsb-quora-ranking.zip
> pip install torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
> pip install -U sentence-transformers==0.3.6

若无法翻墙，下载速度会很慢，可本地下载好后传给服务器（拖拽）
D:/#Pre-trained_Language_Model/weights/pytorch包 → 拖入/data/software/
D:/#Pre-trained_Language_Model/weights/多语模型 → 拖入/data/ningshixian/work/
↓

测试机
sh run.sh xxx
pkill -f crontab_retraining.py
nohup python crontab_retraining.py pro >logs/cronlog.txt 2>&1 &
nohup python crontab_retraining.py test >logs/cronlog.txt 2>&1 &


生产机
pkill -f "listen_resort.py"
nohup python listen_resort.py > logs/listen.log 2>&1 &
tail -f logs/listen.log

COMMENT

## ===================================安装脚本====================================== ##

if [ ! -n "$1" ] ;then

    python bert_embedding_util.py -jieya 
    printf "解压bert向量文件完成！\n"

    pkill -f "listen_resort.py"
    pkill -f "gunicorn step4_rerank:app"
    # pkill -f "gunicorn predict:app"

    nohup python listen_resort.py > logs/listen.log 2>&1 &
    # nohup uvicorn listen_resort:app --host 0.0.0.0 --port 8098 --workers 1 --limit-concurrency 10 > logs/listen.log 2>&1 &
    printf "8098端口点调重排序监听服务已启用\n"

    nohup gunicorn step4_rerank:app -b 0.0.0.0:8099 -w 1 --threads 50 --timeout 60 -k uvicorn.workers.UvicornWorker > logs/run.log 2>&1 &
    printf "8099重排序服务已启用\n"

    lsof -i:8099
    printf "\n"
    printf "脚本启动完毕\n"

else
    python bert_embedding_util.py -jieya 
    printf "解压bert向量文件完成！\n"

    python step1_prepare_data_A.py
    printf "数据收集完成！\n"
    python step2_feature_extraction_A.py
    printf "数据处理完成！\n"
    python step3_lgb_train_A.py -train
    printf "模型训练完成！\n" 

    # #执行linux清理内存命令
    # echo 1 > /proc/sys/vm/drop_caches

    python step1_prepare_data_B.py
    printf "数据收集完成！\n"
    python step2_feature_extraction_B.py
    printf "数据处理完成！\n"
    python step3_lgb_train_B.py -train
    printf "模型训练完成！\n" 

    pkill -f "gunicorn step4_rerank:app"
    nohup gunicorn step4_rerank:app -b 0.0.0.0:8099 -w 1 --threads 50 --timeout 60 -k uvicorn.workers.UvicornWorker > logs/run.log 2>&1 &

    sleep 5
    lsof -i:8099
    printf "8099重排序服务已启用\n"

    printf "\n"
    printf "脚本启动完毕\n"

    # 检测端口是否存在
    check_port() {
        echo "正在检测8099端口。。。"
        netstat -tlpn | grep 8099
    }
    if check_port ;then
        python step3_lgb_train_A.py -push -$1
        printf "模型A推送完成！\n" 
        python step3_lgb_train_B.py -push -$1
        printf "模型B推送完成！\n" 
    else
        echo "8099端口死亡"
        DATE_N=`date "+%Y-%m%d %H:%M:%S"`
        echo "时间：${DATE_N}"
    fi

fi


