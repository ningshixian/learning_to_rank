import sys
import time
import datetime
from apscheduler.schedulers.blocking import BlockingScheduler
import subprocess
import setting

"""定时任务脚本
命令：
pkill -f crontab_retraining.py
nohup python crontab_retraining.py >logs/cronlog.txt 2>&1 &

注意：隔一段时间需要清理cronlog.txt日志文件
"""


def func():
    # 重启服务
    # loader = subprocess.Popen(["pkill", "-f", "crontab_retraining.py"])
    # returncode = loader.wait()  # 阻塞直至子进程完成
    print("【模型重训&服务重启】定时任务启动!")
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
    loader = subprocess.Popen(["sh", "run.sh", mode])
    returncode = loader.wait()  # 阻塞直至子进程完成
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
    print("【模型重训&服务重启】完成!")


def dojob():
    #创建调度器：BlockingScheduler
    scheduler = BlockingScheduler()
    #添加任务,时间:每周三凌晨2点
    scheduler.add_job(func, 'cron', day_of_week='wed', hour=2, minute=0, id='es-resort retraining') # 周三
    # scheduler.add_job(func, 'cron', day_of_week='thu', hour=18, minute=38, id='es-resort retraining')
    scheduler.start()


# 推送环境自动选择
# ['test', 'pro', 'local', '154']
mode = setting.mode
print(mode)
dojob()
