# resources.py
import sys
import os
from PyQt5.QtCore import QResource

def get_resource_path(resource_name):
    """获取资源路径，支持打包和开发模式"""
    if hasattr(sys, '_MEIPASS'):
        # PyInstaller打包模式
        return os.path.join(sys._MEIPASS, resource_name)
    else:
        # 开发模式
        return os.path.join(os.path.dirname(__file__), resource_name)

def get_background_image_path():
    """获取背景图片路径"""
    return get_resource_path('data/guipic.png')