# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :mylearn
# @File     :__init__
# @Date     :2021/1/14 19:15
# @Author   :Jay_Lee
# @Software :PyCharm
-------------------------------------------------
"""
import conf.global_settings as settings
class Settings:
    def __init__(self,settings):
        for attr in dir(settings):
            if attr.isupper():
                setattr(self,attr,getattr(settings,attr))

settings = Settings(settings)
# print(settings)