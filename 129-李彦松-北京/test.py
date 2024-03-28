import datetime
import time

def remind_me():
    while True:
        # 获取当前日期和时间
        current_date = datetime.datetime.now()
        # 检查是否是周日并且当前时间是9点
        if current_date.weekday() == 6 and current_date.hour == 9:
            print("现在是上午9点，今天是周日，记得上课！")
        # 每小时检查一次
        time.sleep(3600)


remind_me()