# -*- coding: utf-8 -*-

# import joblib
from sklearn.externals import joblib
from sklearn.metrics.pairwise import cosine_similarity
from textdistance import cosine, jaccard
from collections import OrderedDict
import pickle as pkl
import codecs
import base64
import requests
import json
import time
import subprocess
from functools import wraps
from gensim.models.doc2vec import Doc2Vec
import traceback
import numpy as np
import pandas as pd
from typing import Optional
from fastapi import FastAPI
from starlette.requests import Request
from pydantic import BaseModel, validator
import lightgbm as lgb
import uvicorn
import setting


"""监听服务
用于接收模型更新信息，重载模型
启动命令: nohup python listen_resort.py > logs/listen.log 2>&1 &
"""

app = FastAPI()


class model_Item(BaseModel):
    model64: str
    trial: str
    iter: int

    stdsc: Optional[str] = None
    label_encoder: Optional[str] = None
    feature_name_all: list

    bucket: str
    category_id: Optional[str] = None
    ctr: Optional[str] = None
    tfidf_cos_sim: Optional[str] = None
    score_tfidf: Optional[str] = None
    score_bow: Optional[str] = None
    score_bm25: Optional[str] = None
    score_doc2vec: Optional[str] = None
    sen2docvec_dict: Optional[str] = None
    time_diff: Optional[str] = None


@app.post("/upload_model")
def upload_model(data: model_Item):
    try:
        json_data = data.dict()
        model64 = json_data.get("model64")
        best_trial_number = json_data.get("trial")
        best_iter = json_data.get("iter")
        
        # 根据A/B桶，更新bucketA/bucketB的数据
        bucket = json_data.get("bucket")
        if bucket=='A':
            # 模型相关的
            with codecs.open(setting.model_bucketA + setting.best_trial, "w", "utf-8") as f:
                f.write(best_trial_number)
            with codecs.open(
                setting.model_bucketA + setting.best_iteration.format(best_trial_number), "w", "utf-8"
            ) as f:
                f.write(str(best_iter))
            with open(setting.model_bucketA + setting.gbdt_model.format(best_trial_number), "wb") as f:
                f.write(base64.b64decode(model64))
            # 特征相关的
            feature_name_all = json_data.get("feature_name_all")
            print("所有可用特征: ", feature_name_all)
            with codecs.open(setting.data_bucketA + setting.feature_all_file, "w", "utf-8") as f:
                f.write("\n".join(feature_name_all))
            for feat in feature_name_all:
                if feat == "ctr":
                    with open(setting.data_bucketA + setting.ctr_file, "wb") as fin:
                        fin.write(base64.b64decode(json_data.get(feat)))
                elif feat == "tfidf_cos_sim":
                    with open(setting.data_bucketA + setting.tfidf_vec_file, "wb") as fin:
                        fin.write(base64.b64decode(json_data.get(feat)))
                elif feat == "time_diff":
                    with open(setting.data_bucketA + setting.time_feats_file, "wb") as fin:
                        fin.write(base64.b64decode(json_data.get(feat)))
        elif bucket=='B':
            # 模型相关的
            with codecs.open(setting.model_bucketB + setting.best_trial, "w", "utf-8") as f:
                f.write(best_trial_number)
            with codecs.open(
                setting.model_bucketB + setting.best_iteration.format(best_trial_number), "w", "utf-8"
            ) as f:
                f.write(str(best_iter))
            with open(setting.model_bucketB + setting.gbdt_model.format(best_trial_number), "wb") as f:
                f.write(base64.b64decode(model64))
            # 特征相关的
            feature_name_all = json_data.get("feature_name_all")
            print("所有可用特征: ", feature_name_all)
            with codecs.open(setting.data_bucketB + setting.feature_all_file, "w", "utf-8") as f:
                f.write("\n".join(feature_name_all))
            stdsc = json_data.get("stdsc")
            label_encoder = json_data.get("label_encoder")
            with open(setting.data_bucketB + setting.stdsc_file, "wb") as fin:
                fin.write(base64.b64decode(stdsc))
            with open(setting.data_bucketB + setting.label_encoder_file, "wb") as fin:
                fin.write(base64.b64decode(label_encoder))
            for feat in feature_name_all:
                if feat == "category_id":
                    with open(setting.data_bucketB + setting.cat_feat_file, "wb") as fin:
                        fin.write(base64.b64decode(json_data.get(feat)))
                elif feat == "score_tfidf":
                    with open(setting.data_bucketB + setting.tfidf_vec_file, "wb") as fin:
                        fin.write(base64.b64decode(json_data.get(feat)))
                elif feat == "score_bow":
                    with open(setting.data_bucketB + setting.bow_vec_file, "wb") as fin:
                        fin.write(base64.b64decode(json_data.get(feat)))
                elif feat == "score_bm25":
                    with open(setting.data_bucketB + setting.bm25_file, "wb") as fin:
                        fin.write(base64.b64decode(json_data.get(feat)))
                elif feat == "score_doc2vec":
                    with open(setting.data_bucketB + setting.doc2vec_file, "wb") as fin:
                        fin.write(base64.b64decode(json_data.get(feat)))
                    with open(setting.data_bucketB + setting.sen2docvec_dict_file, "wb") as fin:
                        fin.write(base64.b64decode(json_data.get("sen2docvec_dict")))
                elif feat == "time_diff":
                    with open(setting.data_bucketB + setting.time_feats_file, "wb") as fin:
                        fin.write(base64.b64decode(json_data.get(feat)))

            # 当A/B桶都更新完成后，重启服务
            time.sleep(5)
            loader = subprocess.Popen(["sh", "run.sh"])
            returncode = loader.wait()  # 阻塞直至子进程完成
            print("排序服务重启完成！")

    except Exception as e:
        print(e)
        return "upload model fail: " + str(e)


if __name__ == "__main__":
    """
    --limit-concurrency INTEGER 允许的最大并发连接或任务的最大数量
    --limit-max-requests 终止进程之前允许的最大服务请求数
    --timeout-keep-alive INTEGER 如果在此超时时间内未收到新数据，则关闭保Keep-Alive连接
    """
    uvicorn.run(
        app="listen_resort:app",  # core
        host="0.0.0.0",
        port=8098,
        backlog=2048,
        # workers=1,
        limit_concurrency=200,
        # reload=True,
        # debug=True,
    )
