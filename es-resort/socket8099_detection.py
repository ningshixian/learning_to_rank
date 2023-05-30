import socket
import time
import os
import subprocess


while True:
    time.sleep(60)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1)
    ip = "127.0.0.1"  # ip对应服务器的ip地址
    port = 8099
    result = sock.connect_ex((ip, port))  # 返回状态值
    if result == 0:
        pass
    else:
        print("Port %d is not open" % port)
        loader = subprocess.Popen(
            [
                "nohup",
                "gunicorn",
                "step4_rerank:app",
                "-b", "0.0.0.0:8099",
                "-w", "1",
                "--threads", "50",
                "--backlog", "2048",
                "-k", "uvicorn.workers.UvicornWorker",
                # "> logs/log.txt 2>&1 &"
            ]
        )
        returncode = loader.wait()  # 阻塞直至子进程完成
        # print("returncode= %s" %(returncode)) ###打印子进程的返回码
    sock.close()
