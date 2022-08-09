import time
import os
print(f"定时任务：Process ID {os.getpid()}, Process Parent ID {os.getppid()}  --------------------\n")
schedule=(8,9,0,20) # 8月9日0点20分
# schedule=(8,8,22,17)
while True:
    
    cur = time.localtime()
    tmp = (cur.tm_mon,cur.tm_mday,cur.tm_hour,cur.tm_min)
    
    if tmp == schedule:
        # sh = 'bash projects/sh_test.sh'
        sh = 'bash projects/sh_run.sh'
        print(f'执行 {sh}')
        os.system(sh)
        break
    time.sleep(10)
    
print('脚本启动成功。程序结束。')