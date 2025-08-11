#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
watermark_tiler.py
==================
可將圖片「貼滿」指定的文字或圖片作為防偽造浮水印（tiled watermark）。
支援：文字/圖片浮水印、旋轉、縮放、間距、透明度、隨機位移、邊緣抗鋸齒。
依賴：Pillow
sudo apt-get install -y fonts-noto-cjk fonts-noto-cjk-extra
python3 water.py --in test.jpeg --out out.png --mode text \
  --text "繁體真棒" \
  --font "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc" \
  --size 64 --color "#000000" --opacity 0.3 --angle 30 \
  --spacing 200 200 --jitter --blend normal

安裝：
    pip install pillow

用法（文字水印）：
    python watermark_tiler.py \
        --in in.jpg --out out.png \
        --mode text --text "CONFIDENTIAL" \
        --font "C:/Windows/Fonts/arial.ttf" \
        --size 64 --color "#FFFFFF" \
        --opacity 0.18 --angle 30 \
        --spacing 280 200 --jitter

用法（圖片水印）：
    python watermark_tiler.py \
        --in in.jpg --out out.png \
        --mode image --wm logo.png \
        --scale 0.5 --opacity 0.15 --angle 35 \
        --spacing 320 240 --jitter
"""
from __future__ import annotations

import argparse
import math
import os
import random
from typing import Tuple, Optional

from PIL import Image, ImageDraw, ImageFont, ImageChops


def _ensure_rgba(img: Image.Image) -> Image.Image:
    return img.convert("RGBA") if img.mode != "RGBA" else img


def _hex_to_rgba(s: str, default=(255, 255, 255, 255)) -> Tuple[int, int, int, int]:
    """接受 #RGB/#RRGGBB/#RRGGBBAA 或 'r,g,b[,a]'。"""
    s = s.strip()
    if s.startswith("#"):
        s = s[1:]
        if len(s) == 3:
            r, g, b = [int(c * 2, 16) for c in s]
            return (r, g, b, 255)
        if len(s) == 6:
            r = int(s[0:2], 16)
            g = int(s[2:4], 16)
            b = int(s[4:6], 16)
            return (r, g, b, 255)
        if len(s) == 8:
            r = int(s[0:2], 16)
            g = int(s[2:4], 16)
            b = int(s[4:6], 16)
            a = int(s[6:8], 16)
            return (r, g, b, a)
        return default
    parts = s.split(",")
    try:
        if len(parts) == 3:
            r, g, b = map(int, parts)
            return (r, g, b, 255)
        if len(parts) == 4:
            r, g, b, a = map(int, parts)
            return (r, g, b, a)
    except Exception:
        pass
    return default


def _diagonal_size(w: int, h: int) -> int:
    return int(math.ceil(math.hypot(w, h)))


def _make_text_tile(
    text: str,
    font_path: Optional[str],
    font_size: int,
    color_rgba: Tuple[int, int, int, int],
    padding_xy: Tuple[int, int],
    stroke_width: int = 0,
    stroke_fill: Optional[Tuple[int, int, int, int]] = None,
) -> Image.Image:
    """預先渲染一張透明背景的單字串 tile，居中擺放。"""
    try:
        if font_path and os.path.exists(font_path):
            font = ImageFont.truetype(font_path, font_size)
        else:
            font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    # 計算文字尺寸（使用 getbbox 可較準確）
    bbox = font.getbbox(text)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

    pad_x, pad_y = padding_xy
    # tile 大小 = 文字尺寸 + padding
    tile_w = max(1, tw + pad_x)
    tile_h = max(1, th + pad_y)

    tile = Image.new("RGBA", (tile_w, tile_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(tile)

    # 將文字置中
    x = (tile_w - tw) // 2 - bbox[0]
    y = (tile_h - th) // 2 - bbox[1]

    if stroke_width > 0 and stroke_fill is None:
        # 預設描邊顏色取 50% 透明黑
        stroke_fill = (0, 0, 0, color_rgba[3] // 2)

    draw.text(
        (x, y),
        text,
        font=font,
        fill=color_rgba,
        stroke_width=stroke_width,
        stroke_fill=stroke_fill,
    )
    return tile


def _make_image_tile(
    wm_img: Image.Image,
    scale: float,
    padding_xy: Tuple[int, int],
) -> Image.Image:
    """縮放水印圖片後置中到 tile。"""
    wm_img = _ensure_rgba(wm_img)
    w0, h0 = wm_img.size
    # 安全縮放界限
    scale = max(0.05, min(8.0, scale))
    w = max(1, int(w0 * scale))
    h = max(1, int(h0 * scale))
    wm_resized = wm_img.resize((w, h), Image.LANCZOS)

    pad_x, pad_y = padding_xy
    tile_w = max(1, w + pad_x)
    tile_h = max(1, h + pad_y)

    tile = Image.new("RGBA", (tile_w, tile_h), (0, 0, 0, 0))
    x = (tile_w - w) // 2
    y = (tile_h - h) // 2
    tile.alpha_composite(wm_resized, (x, y))
    return tile


def _tile_to_rotated_sheet(
    base_size: Tuple[int, int],
    tile: Image.Image,
    angle_deg: float,
    jitter: bool,
) -> Image.Image:
    """先把 tile 鋪滿一張「比底圖更大」的 square sheet，再整張旋轉，最後裁回底圖大小。"""
    W, H = base_size
    diag = _diagonal_size(W, H)
    sheet = Image.new("RGBA", (diag, diag), (0, 0, 0, 0))

    tw, th = tile.size

    # 為了加速，先把 tile 做成一次性快取（避免多次重新渲染）
    tile_cached = tile

    # 隨機起點，避免水印網格容易被「局部平移+裁切」一次去除
    ox = random.randint(0, tw - 1) if jitter and tw > 1 else 0
    oy = random.randint(0, th - 1) if jitter and th > 1 else 0

    # 平鋪
    y = oy - th
    while y < diag + th:
        x = ox - tw
        while x < diag + tw:
            sheet.alpha_composite(tile_cached, (int(x), int(y)))
            x += tw
        y += th

    # 整張旋轉（expand=False 讓尺寸維持 diag x diag）
    if angle_deg % 360 != 0:
        sheet = sheet.rotate(angle_deg, resample=Image.BICUBIC, expand=False)

    # 從中心裁回原圖大小
    cx = (diag - W) // 2
    cy = (diag - H) // 2
    sheet = sheet.crop((cx, cy, cx + W, cy + H))
    return sheet


def _apply_opacity(layer: Image.Image, opacity: float) -> Image.Image:
    """對整層調整透明度（0~1）。"""
    opacity = max(0.0, min(1.0, float(opacity)))
    if opacity >= 1.0:
        return layer
    if opacity <= 0.0:
        return Image.new("RGBA", layer.size, (0, 0, 0, 0))

    r, g, b, a = layer.split()
    # 以乘法縮放 alpha
    a = a.point(lambda v: int(v * opacity))
    out = Image.merge("RGBA", (r, g, b, a))
    return out


def _apply_blend_mode(base_rgba: Image.Image, wm_layer: Image.Image, mode: str) -> Image.Image:
    """
    簡易混合模式：normal/multiply/screen/overlay。
    注意：為了效率與可預測性，採用 RGBA 分離後再合成。
    """
    mode = (mode or "normal").lower()
    base = _ensure_rgba(base_rgba)
    wm = _ensure_rgba(wm_layer)
    if mode == "normal":
        out = base.copy()
        out.alpha_composite(wm)
        return out

    # 將水印的不透明部分擷取為一張 RGB 層（alpha 乘到顏色上）
    wr, wg, wb, wa = wm.split()
    # 先把 wm 轉到跟 base 同樣的大小 RGBA（已是）
    # 將 wm 的顏色預乘 alpha，得到「實際會覆蓋的顏色」
    wrp = ImageChops.multiply(wr, wa)
    wgp = ImageChops.multiply(wg, wa)
    wbp = ImageChops.multiply(wb, wa)

    br, bg, bb, ba = base.split()

    if mode == "multiply":
        mr = ImageChops.multiply(br, wrp)
        mg = ImageChops.multiply(bg, wgp)
        mb = ImageChops.multiply(bb, wbp)
    elif mode == "screen":
        inv_br = ImageChops.invert(br)
        inv_bg = ImageChops.invert(bg)
        inv_bb = ImageChops.invert(bb)
        mr = ImageChops.invert(ImageChops.multiply(inv_br, ImageChops.invert(wrp)))
        mg = ImageChops.invert(ImageChops.multiply(inv_bg, ImageChops.invert(wgp)))
        mb = ImageChops.invert(ImageChops.multiply(inv_bb, ImageChops.invert(wbp)))
    elif mode == "overlay":
        # overlay = combine(multiply for dark, screen for light) 的簡化版本
        # 用線性插值近似： out = base*(1-alpha) + blend(base,wm)*alpha
        # 為簡化，我們直接採用 multiply 與 screen 的混合近似
        mul_r = ImageChops.multiply(br, wrp)
        mul_g = ImageChops.multiply(bg, wgp)
        mul_b = ImageChops.multiply(bb, wbp)
        scr_r = ImageChops.invert(ImageChops.multiply(ImageChops.invert(br), ImageChops.invert(wrp)))
        scr_g = ImageChops.invert(ImageChops.multiply(ImageChops.invert(bg), ImageChops.invert(wgp)))
        scr_b = ImageChops.invert(ImageChops.multiply(ImageChops.invert(bb), ImageChops.invert(wbp)))
        # 以 base 的亮度作為權重（簡化）
        # 亮 -> screen 權重高；暗 -> multiply 權重高
        # 計算灰階作為權重
        gray = br.convert("L")
        mr = ImageChops.lighter(mul_r, ImageChops.blend(mul_r, scr_r, 0.7))
        mg = ImageChops.lighter(mul_g, ImageChops.blend(mul_g, scr_g, 0.7))
        mb = ImageChops.lighter(mul_b, ImageChops.blend(mul_b, scr_b, 0.7))
    else:
        # fallback normal
        out = base.copy()
        out.alpha_composite(wm)
        return out

    # 將 alpha 當作 mask 混回去（premultiplied 的簡易合成）
    # out_rgb = base_rgb*(1-a) + blend_rgb
    out_r = ImageChops.composite(mr, br, wa)
    out_g = ImageChops.composite(mg, bg, wa)
    out_b = ImageChops.composite(mb, bb, wa)
    out = Image.merge("RGBA", (out_r, out_g, out_b, ba))
    return out


def apply_tiled_text_watermark(
    in_image: Image.Image,
    text: str,
    font_path: Optional[str] = None,
    font_size: int = 48,
    color: str = "#FFFFFF",
    opacity: float = 0.15,
    angle: float = 30.0,
    spacing: Tuple[int, int] = (280, 200),
    stroke_width: int = 0,
    stroke_color: Optional[str] = None,
    jitter: bool = True,
    blend_mode: str = "normal",
) -> Image.Image:
    """
    回傳一張已貼滿文字浮水印的新圖。
    spacing: (dx, dy) 表示每個 tile 的額外 padding；tile 本身會以文字尺寸+padding 組成。
    stroke_width/ stroke_color 可增加輪廓，提升在淺/深色底圖上的可視度。
    """
    base = _ensure_rgba(in_image)
    color_rgba = _hex_to_rgba(color)
    stroke_rgba = _hex_to_rgba(stroke_color) if stroke_color else None

    pad_x = max(0, int(spacing[0]))
    pad_y = max(0, int(spacing[1]))
    tile = _make_text_tile(
        text=text,
        font_path=font_path,
        font_size=int(font_size),
        color_rgba=color_rgba,
        padding_xy=(pad_x, pad_y),
        stroke_width=int(stroke_width),
        stroke_fill=stroke_rgba,
    )

    sheet = _tile_to_rotated_sheet(base.size, tile, angle, jitter=jitter)
    sheet = _apply_opacity(sheet, opacity)
    out = _apply_blend_mode(base, sheet, blend_mode)
    return out


def apply_tiled_image_watermark(
    in_image: Image.Image,
    wm_image: Image.Image,
    scale: float = 0.5,
    opacity: float = 0.15,
    angle: float = 30.0,
    spacing: Tuple[int, int] = (280, 200),
    jitter: bool = True,
    blend_mode: str = "normal",
) -> Image.Image:
    """
    回傳一張已貼滿圖片浮水印的新圖。
    scale：水印圖片的縮放倍率（相對於原水印尺寸）。
    spacing：每個 tile 額外 padding（x, y）。
    """
    base = _ensure_rgba(in_image)
    pad_x = max(0, int(spacing[0]))
    pad_y = max(0, int(spacing[1]))
    tile = _make_image_tile(_ensure_rgba(wm_image), scale=scale, padding_xy=(pad_x, pad_y))

    sheet = _tile_to_rotated_sheet(base.size, tile, angle, jitter=jitter)
    sheet = _apply_opacity(sheet, opacity)
    out = _apply_blend_mode(base, sheet, blend_mode)
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="將圖片貼滿文字/圖片的防偽造浮水印")
    p.add_argument("--in", dest="inp", required=True, help="輸入圖片路徑")
    p.add_argument("--out", dest="out", required=True, help="輸出圖片路徑 (建議 .png)")
    p.add_argument("--mode", choices=["text", "image"], required=True, help="水印模式")
    p.add_argument("--angle", type=float, default=30.0, help="整體旋轉角度（度）")
    p.add_argument("--opacity", type=float, default=0.15, help="透明度 0~1")
    p.add_argument("--spacing", type=int, nargs=2, default=[280, 200], help="tile 間距 paddingX paddingY")
    p.add_argument("--jitter", action="store_true", help="啟用隨機起點偏移（更難一刀去除）")
    p.add_argument("--blend", choices=["normal", "multiply", "screen", "overlay"], default="normal",
                   help="混合模式（特殊情況可強化在深/淺底上的可視度）")

    # text mode
    p.add_argument("--text", type=str, help="水印文字")
    p.add_argument("--font", type=str, default=None, help="字型檔路徑（缺省用系統預設）")
    p.add_argument("--size", type=int, default=48, help="文字大小 px")
    p.add_argument("--color", type=str, default="#FFFFFF", help="文字顏色 (#RRGGBB 或 #RRGGBBAA 或 r,g,b[,a])")
    p.add_argument("--stroke", type=int, default=0, help="描邊寬度（像素）")
    p.add_argument("--stroke-color", type=str, default=None, help="描邊顏色（同 color 格式）")

    # image mode
    p.add_argument("--wm", type=str, help="水印圖片路徑")
    p.add_argument("--scale", type=float, default=0.5, help="水印圖片縮放倍率")

    return p.parse_args()


def main():
    args = _parse_args()

    base = Image.open(args.inp)
    if args.mode == "text":
        if not args.text:
            raise SystemExit("--mode text 需要 --text")
        out = apply_tiled_text_watermark(
            base,
            text=args.text,
            font_path=args.font,
            font_size=args.size,
            color=args.color,
            opacity=args.opacity,
            angle=args.angle,
            spacing=(args.spacing[0], args.spacing[1]),
            stroke_width=args.stroke,
            stroke_color=args.stroke_color,
            jitter=args.jitter,
            blend_mode=args.blend,
        )
    else:
        if not args.wm:
            raise SystemExit("--mode image 需要 --wm")
        wm_img = Image.open(args.wm)
        out = apply_tiled_image_watermark(
            base,
            wm_img,
            scale=args.scale,
            opacity=args.opacity,
            angle=args.angle,
            spacing=(args.spacing[0], args.spacing[1]),
            jitter=args.jitter,
            blend_mode=args.blend,
        )

    # 若輸出為 JPEG，會失去透明度；建議 PNG
    out.save(args.out)
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
