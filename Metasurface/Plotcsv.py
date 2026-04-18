# -*- coding: utf-8 -*-
"""
批次處理所有子資料夾中的 ONA_1.csv（Lumerical/INTERCONNECT 匯出格式）：
- 以「Excel 視角」索引行數，抓 A40047-A80047 作 X、I40047-I80047 作 Y
- 自動忽略中間的 .HEADER / ..NAMES / .DATA 等標頭區塊與任何非數值行
- X(公尺) 自動轉成奈米（乘 1e9）
- 產生兩張圖（完整範圍、Peak 範圍），圖名=子資料夾名 與 子資料夾名_Peak
- 在 x=1270,1280,...,1340 nm 畫紅色虛線
- 圖片輸出到 主資料夾/DataOutput/；若同名檔存在會先刪除
"""

import os
import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ========== 可調整參數（字串） ==========
# 主資料夾（放著一堆像 1270nm/ 的子資料夾）
main_dir_str = "."

# 每個子資料夾內的 CSV 檔名
csv_filename_str = "ONA_1.csv"

# 以 Excel 視角指定行列區間（A/I 欄；40047~80047 列）
excel_row_start_str = "40047"  # 含
excel_row_end_str   = "80047"  # 含
excel_col_x_str     = "A"      # X 在 A 欄（1-based）
excel_col_y_str     = "I"      # Y 在 I 欄（1-based）

# 軸標籤（字串）
x_label_str = "Wavelength ( nm )"
y_label_str = "Gain ( dB )"

# 全局字體大小（字串）
font_size_str = "16"

# Peak 圖範圍（字串，以 center_nm=資料夾名中的數值）
# 例：X="center_nm-0.1, center_nm+0.1"；Y="-2, 0"
peak_x_range_str = "center_nm-0.1, center_nm+0.1"
peak_y_range_str = "-2, 0"


# 垂直虛線位置（nm；逗號分隔，字串）
vlines_nm_str = "1270,1280,1290,1300,1310,1320,1330,1340"

# 輸出圖片參數（字串）
dpi_str = "300"
figsize_w_str = "10"
figsize_h_str = "6"

# ======================================

def excel_col_to_index(col_letter: str) -> int:
    """Excel 欄位字母轉 0-based index（A->0, B->1, ...）"""
    col_letter = col_letter.strip().upper()
    val = 0
    for ch in col_letter:
        if not ('A' <= ch <= 'Z'):
            raise ValueError(f"無效的 Excel 欄位字母：{col_letter}")
        val = val * 26 + (ord(ch) - ord('A') + 1)
    return val - 1

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def delete_if_exists(p: Path):
    try:
        if p.exists():
            p.unlink()
    except Exception as e:
        print(f"[警告] 刪除已存在檔案失敗：{p}；原因：{e}")

def safe_eval(expr: str, center_nm: float) -> float:
    """允許中心值運算的簡單安全 eval：僅支持數字/center_nm/+ - * / () 與空白。"""
    allowed = set("0123456789.+-*/() eE")
    tmp = expr.replace("center_nm", str(center_nm))
    if not set(tmp) <= allowed:
        raise ValueError(f"不允許的字元出現在表達式：{expr}")
    return eval(tmp, {"__builtins__": None}, {})

def parse_range_pair(expr_pair: str, center_nm: float) -> tuple[float, float]:
    parts = [p.strip() for p in expr_pair.split(",")]
    if len(parts) != 2:
        raise ValueError(f"範圍字串需為兩個以逗號分隔的數值/表達式：{expr_pair}")
    a = safe_eval(parts[0], center_nm)
    b = safe_eval(parts[1], center_nm)
    return (min(a, b), max(a, b))

def extract_center_nm_from_folder(name: str, fallback: float) -> float:
    """從資料夾名抓數字（如 '1270nm' -> 1270.0），失敗就用 fallback。"""
    m = re.search(r"(\d+(\.\d+)?)", name)
    if m:
        try: return float(m.group(1))
        except: pass
    return float(fallback)

def load_xy_excel_view(csv_path: Path,
                       row_start_excel: int,
                       row_end_excel: int,
                       col_x_letter: str,
                       col_y_letter: str):
    """
    以「Excel 視角」擷取 A/I 欄與指定列區間的數據。
    - 逐行讀檔、用逗號分割；若欄數<所需欄位或無法轉為數字就忽略該行
    - X 視為公尺，轉成奈米
    回傳：np.ndarray X_nm, Y
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"找不到檔案：{csv_path}")

    lines = csv_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    jx = excel_col_to_index(col_x_letter)  # 0-based
    jy = excel_col_to_index(col_y_letter)

    i0 = max(0, row_start_excel - 1)
    i1 = min(len(lines), row_end_excel)  # 右界不含

    X_list, Y_list = [], []
    for idx in range(i0, i1):
        parts = lines[idx].split(",")
        # 需要至少到達 I 欄（index 8）
        if len(parts) <= max(jx, jy):
            continue
        try:
            x_m = float(parts[jx])      # A 欄：wavelength（公尺）
            y    = float(parts[jy])     # I 欄：gain（dB）
        except Exception:
            continue
        X_list.append(x_m * 1e9)        # 轉 nm
        Y_list.append(y)

    X = np.array(X_list, dtype=float)
    Y = np.array(Y_list, dtype=float)

    if X.size == 0:
        raise ValueError(f"{csv_path.name} 在 Excel 列 {row_start_excel}-{row_end_excel} 與欄 {col_x_letter}/{col_y_letter} 內沒有有效數據")

    return X, Y

def plot_one(ax, X, Y, title, x_label, y_label, font_size, vlines_nm):
    ax.plot(X, Y, linewidth=1.0)
    for xv in vlines_nm:
        ax.axvline(x=xv, linestyle="--", color="red", linewidth=1.0)
    ax.set_title(title, fontsize=font_size)
    ax.set_xlabel(x_label, fontsize=font_size)
    ax.set_ylabel(y_label, fontsize=font_size)
    ax.tick_params(axis="both", labelsize=font_size)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)

def main():
    main_dir = Path(main_dir_str).resolve()
    out_dir = main_dir / "DataOutput"
    ensure_dir(out_dir)

    row_start = int(excel_row_start_str)
    row_end   = int(excel_row_end_str)
    colX = excel_col_x_str.strip()
    colY = excel_col_y_str.strip()

    x_label = x_label_str
    y_label = y_label_str
    font_size = int(font_size_str)

    vlines_nm = [float(s.strip()) for s in vlines_nm_str.split(",") if s.strip()]
    dpi = int(dpi_str)
    figsize = (float(figsize_w_str), float(figsize_h_str))

    subfolders = [p for p in main_dir.iterdir() if p.is_dir() and p.name != "DataOutput"]
    if not subfolders:
        print(f"[提示] 主資料夾中沒有可處理的子資料夾：{main_dir}")
        return

    print(f"[資訊] 主資料夾：{main_dir}")
    print(f"[資訊] 子資料夾數量：{len(subfolders)}")
    print(f"[資訊] 產出目錄：{out_dir}")

    for folder in sorted(subfolders):
        csv_path = folder / csv_filename_str
        folder_name = folder.name  # 例如 "1270nm"

        try:
            X, Y = load_xy_excel_view(csv_path, row_start, row_end, colX, colY)
        except Exception as e:
            print(f"[跳過] {folder_name}: {e}")
            continue

        # 從資料夾名稱推 center_nm（找不到就用 X 的中位數）
        center_nm = extract_center_nm_from_folder(folder_name, fallback=float(np.median(X)))

        # 檔名與標題
        title_full = folder_name
        title_peak = f"{folder_name}_Peak"
        out_full = out_dir / f"{folder_name}.png"
        out_peak = out_dir / f"{folder_name}_Peak.png"

        # 先刪同名檔
        delete_if_exists(out_full)
        delete_if_exists(out_peak)

        # 完整範圍圖
        fig1, ax1 = plt.subplots(figsize=figsize)
        plot_one(ax1, X, Y, title_full, x_label, y_label, font_size, vlines_nm)
        ax1.set_xlim(1260,1350)
        fig1.tight_layout()
        fig1.savefig(out_full, dpi=dpi)
        plt.close(fig1)

        # Peak 範圍圖
        try:
            xlo, xhi = parse_range_pair(peak_x_range_str, center_nm)
            ylo, yhi = parse_range_pair(peak_y_range_str, center_nm)
        except Exception as e:
            print(f"[警告] Peak 範圍字串解析失敗，改用預設 (center±0.1, -2~0)：{e}")
            xlo, xhi = center_nm - 0.1, center_nm + 0.1
            ylo, yhi = -2, 0

        fig2, ax2 = plt.subplots(figsize=figsize)
        plot_one(ax2, X, Y, title_peak, x_label, y_label, font_size, vlines_nm)
        ax2.set_xlim(xlo, xhi)
        ax2.set_ylim(ylo, yhi)
        fig2.tight_layout()
        fig2.savefig(out_peak, dpi=dpi)
        plt.close(fig2)

        print(f"[完成] {folder_name}: 輸出 -> {out_full.name}, {out_peak.name}")

if __name__ == "__main__":
    main()
