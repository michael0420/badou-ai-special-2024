import os
import re

from DATA.step1 import step1
from DATA.step2 import step2

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By


def drive(url):
    driver_path = Service(r"I:/Project/GPT_AI/driver/chromedriver.exe")
    driver = webdriver.Chrome(service=driver_path)

    driver.get(url)

    filename = driver.find_element(By.XPATH, '//span[@class="ItemDetailRed-header-row34-content ng-binding"]').text

    pb_time = driver.find_element(By.XPATH, '//span[@class="ItemDetailRed-header-row12-content ng-binding"]').text

    driver.quit()

    return pb_time, filename


dir_name = "State Administration of Foreign Exchange"

url = input("url:")

pb_time = input("pb_time:")

pb_time = pb_time.replace("年", "-").replace("月", "-").replace("日", "")

filename = input("pb_name:")

path = f"sources\\{dir_name}"

_path = path + "\\" + filename

os.makedirs(_path)

__path = path + "\\" + filename + "\\" + "content"

os.makedirs(__path)

step1()
step2()

with open("拆分/sample_res.txt", "r", encoding="utf-8") as fr:
    data = fr.read()

with open(f"{__path}\\1.txt", "w", encoding="utf-8") as fw:
    fw.write(filename + "\n")
    fw.write(url + "\n")
    fw.write("发布日期:" + pb_time + "\n")
    fw.write(data)
