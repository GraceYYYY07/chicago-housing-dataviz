"""
cut_data.py
从原始 address_data.xlsx 提取坐标并清洗数据

运行方法（在项目根目录下）：
    python data/raw-data/cut_data.py

输出：
    data/raw-data/cleaned_addresses_full.csv
"""

import re
import pandas as pd
from pathlib import Path

# 路径
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw-data"
IN_PATH  = RAW_DIR / "address_data.xlsx"
OUT_PATH = RAW_DIR / "cleaned_addresses_full.csv"


def extract_lonlat(geom_str):
    """从 MULTIPOLYGON 字符串里提取第一个坐标点的 lon, lat"""
    nums = re.findall(r'-?\d+\.\d+', str(geom_str))
    if len(nums) >= 2:
        return float(nums[0]), float(nums[1])
    return None, None


print("=" * 60)
print("Step 1) 读取原始数据（请耐心等待）")
print("=" * 60)
df = pd.read_excel(IN_PATH)
print(f"原始行数: {len(df):,}")

print("\n" + "=" * 60)
print("Step 2) 从 the_geom 提取 lon / lat")
print("=" * 60)
df[["lon", "lat"]] = df["the_geom"].apply(
    lambda x: pd.Series(extract_lonlat(x))
)
print(f"坐标提取完成，空值数: {df['lon'].isna().sum():,}")

print("\n" + "=" * 60)
print("Step 3) 只保留芝加哥市范围内的坐标")
print("=" * 60)
df = df[
    (df["lat"] >= 41.60) &
    (df["lat"] <= 42.05) &
    (df["lon"] >= -87.95) &
    (df["lon"] <= -87.50)
].copy()
print(f"坐标过滤后行数: {len(df):,}")

print("\n" + "=" * 60)
print("Step 4) 过滤：只保留 F_ADD1 > 0 的行（有门牌号的建筑）")
print("=" * 60)
# 强制转为数字，非数字值变成 NaN
df["F_ADD1"] = pd.to_numeric(df["F_ADD1"], errors="coerce")
df = df[df["F_ADD1"] > 0].copy()
print(f"F_ADD1 > 0 的行数: {len(df):,}")

print("\n" + "=" * 60)
print("Step 5) 只保留必要列，删除坐标为空的行")
print("=" * 60)
keep_cols = [
    "BLDG_ID", "F_ADD1", "T_ADD1", "ST_NAME1", "ST_TYPE1",
    "NO_OF_UNIT", "NO_STORIES", "YEAR_BUILT", "lon", "lat"
]
df_out = df[keep_cols].dropna(subset=["lon", "lat"]).copy()
print(f"最终行数: {len(df_out):,}")
print(f"lat 范围: {df_out['lat'].min():.4f} ~ {df_out['lat'].max():.4f}")
print(f"lon 范围: {df_out['lon'].min():.4f} ~ {df_out['lon'].max():.4f}")

print("\n" + "=" * 60)
print("Step 6) 保存 CSV")
print("=" * 60)
df_out.to_csv(OUT_PATH, index=False)
print(f"✅ 保存完成: {OUT_PATH}")
print(f"   文件大小: {OUT_PATH.stat().st_size / 1024 / 1024:.1f} MB")
