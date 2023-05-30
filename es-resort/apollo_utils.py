# -*- encoding: utf-8 -*-
"""
@Author  : Wu Wenjie
@Contact : 406959268@qq.com
@Software: PyCharm
@File    : apollo_test.py
@Time    : 2021/1/13 16:09
@License : (C) Copyright 2019, by Wu Wenjie
@Desc    :

"""
import sys
import requests
import json

from config import setting
sys.path.append(r"../../")
import utils_toolkit as ut


def get_dbinfo_from_apollo(NAME_SPACE):
    CONFIG_SERVER_URL = setting.APOLLO_HOST
    APPID = setting.APPID
    CLUSTER_NAME = setting.CLUSTER
    TOKEN = setting.TOKEN
    decrypt_url = setting.DECRYPT_HOST
    api_key = setting.API_KEY
    
    # 从apollo获取NAME_SPACE的配置信息
    url = (
        "{config_server_url}/configfiles/json/{appId}/{clusterName}+{token}/"
        "{namespaceName}".format(
            config_server_url=CONFIG_SERVER_URL,
            appId=APPID,
            clusterName=CLUSTER_NAME,
            token=TOKEN,
            namespaceName=NAME_SPACE,
        )
    )

    res = requests.get(url=url)
    slot_config = json.loads(res.text)  # dict

    # -------------------------------------------------------------
    # -------------------------------------------------------------
    
    # apollo获取解密后的密码
    if NAME_SPACE in ["slot", "slot-kg"]:
        
        headers = {
            "Content-Type": "application/json",
            "X-Gaia-API-Key": api_key,
        }  # X-Gaia-API-Key为PaaS平台上申请的对应key

        with open("/etc/apollo/apollo_private_key", "r") as f:
            PRIVATE_KEY = f.read()

        body = {
            "privateKey": PRIVATE_KEY,
            "cipherText": [slot_config["passwd"]],
        }

        res = requests.post(url=decrypt_url, headers=headers, data=json.dumps(body))
        slot_config["passwd"] = json.loads(res.text)[0]

    return slot_config


# CFG = get_dbinfo_from_apollo(NAME_SPACE='slot-sbert')
# print(CFG)
