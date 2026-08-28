# -*- coding: utf-8 -*-
"""
下载 Qlib 官方中国A股日线数据（含 Alpha158/360 特征所需的基础数据）
数据落在项目目录 qlib_data/cn_data 下，不写到用户主目录。
用法:
    python scripts/download_qlib_data.py
"""
import os
import pathlib
from qlib.tests.data import GetData

# Qlib 官方数据在 Azure Blob；避免本机代理中断大文件下载。
os.environ["NO_PROXY"] = "*"

TARGET_DIR = pathlib.Path(__file__).resolve().parent.parent / "qlib_data" / "cn_data"
TARGET_DIR.mkdir(parents=True, exist_ok=True)

GetData().qlib_data(
    name="qlib_data",
    target_dir=str(TARGET_DIR),
    interval="1d",
    region="cn",
    exists_skip=True,
)
print(f"数据已下载到: {TARGET_DIR}")
