# -*- coding: utf-8 -*-
# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import requests

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 Edg/122.0.0.0'}


def login():

    data = {
        'email': 'xxxxx@gmail.com',
        'passwd': 'xxxxx'
    }
    login_url = 'https://julang.site/auth/login'

    res = requests.post(url=login_url, data=data, headers=headers)
    cookie_jra = res.cookies
    cookies = cookie_jra.get_dict()
    return cookies


def sign_in(cookies):
    sign_url='https://julang.site/user/checkin'
    res = requests.post(url=sign_url, headers=headers, cookies=cookies)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    cookies = login()
    sign_in(cookies)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
