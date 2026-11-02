# -*- coding: utf-8 -*-
"""Numerical aperture study and exploratory raw-CCD intensity comparison.

No SLM SDK. Optical model is ideal 4f; CCD comparison is preliminary visual
analysis after manually selected square ROIs, NOT a modal purity measurement.
"""
from __future__ import annotations
import csv
import math
from pathlib import Path
import numpy as np
from PIL import Image
from slm_hologram_core import encode_existing_field
from diffraction_simulation import reconstruct

METHODS = ('phase_only', 'double_phase', 'compensated_double_phase')


def run_aperture_study(result, radii_mm, settings, methods=METHODS):
    """Sweep actual *millimeter* Fourier-aperture radius with fixed source field.
    settings=(f_mm, ignored_current_radius, external_tilt_enabled, tilt_angle_deg).
    Returns list of scalar dicts only; does not keep reconstructed complex fields.
    """
    f, _, tilt_on, tilt_deg = settings
    radii = np.asarray(list(radii_mm), dtype=float)
    if not 1 <= len(radii) <= 40:
        raise ValueError('扫描点数须为1～40')
    if not np.all(np.isfinite(radii)) or np.any(radii <= 0):
        raise ValueError('光阑半径必须为有限正数（mm）')
    if any(m not in METHODS for m in methods):
        raise ValueError('编码方法无效')
    rows=[]
    for method in methods:
        variant=encode_existing_field(result,method)
        for rad in radii:
            rr=reconstruct(variant,fourier_f_mm=f,aperture_radius_mm=float(rad),
                           use_external_tilt=tilt_on,tilt_x_deg=tilt_deg)
            rows.append({'method':method,'radius_mm':float(rad),
                         'fidelity_squared_ideal':rr.fidelity,
                         'intensity_correlation_ideal':rr.intensity_correlation,
                         'filtered_power_fraction_ideal':rr.efficiency,
                         'relative_complex_error_ideal':rr.relative_error,
                         'frequency_center_x_mm':rr.spectrum_x,
                         'frequency_center_y_mm':rr.spectrum_y,
                         'fourier_f_mm':f,'tilt_x_deg_sim_only':tilt_deg if tilt_on else 0.0})
    return rows


def pareto_rows(rows):
    """Mark nondominated points for fidelity-vs-filtered-power tradeoff."""
    marks=[]
    for r in rows:
        f,e = r['fidelity_squared_ideal'],r['filtered_power_fraction_ideal']
        dominated = any((s['fidelity_squared_ideal'] >= f and
                         s['filtered_power_fraction_ideal'] >= e and
                         (s['fidelity_squared_ideal'] > f or s['filtered_power_fraction_ideal'] > e))
                        for s in rows)
        marks.append(not dominated)
    return marks


def write_study_csv(rows,path):
    if not rows:raise ValueError('没有扫描结果')
    with Path(path).open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
        writer.writeheader();writer.writerows(rows)


def load_ccd_image(path):
    """Read single-frame camera image, preserve original gray data scale."""
    with Image.open(str(path)) as im:
        if getattr(im, 'n_frames', 1) > 1:
            im.seek(0)
        if im.mode in ('RGB','RGBA','P','CMYK'):
            raise ValueError('请导入原始单通道灰度CCD BMP/PNG/TIFF，不使用彩色截图')
        arr=np.asarray(im).astype(np.float64)
    if arr.ndim != 2 or min(arr.shape) < 16 or not np.all(np.isfinite(arr)):
        raise ValueError('CCD图像必须是有效的二维灰度阵列')
    return arr


def _normalize_roi(image):
    image=np.asarray(image,dtype=float)
    lo=np.percentile(image, 5)
    a=np.clip(image-lo,0,None)
    m=float(a.max())
    if m<=1e-12:raise ValueError('ROI信号全为零或为常量')
    return a/m


def compare_ccd_roi(camera_image, target_intensity, x0, y0, size_px, flip_x=False, flip_y=False):
    """Manual-square-ROI visual comparison. Exploratory metrics only.

    Target is cropped to its central square signal support, then both regions
    resized to 256x256 without clipping or image-driven shifts. No physical
    magnification calibration is implied.
    """
    a=np.asarray(camera_image,dtype=float)
    h,w=a.shape
    s=int(size_px);x=int(x0);y=int(y0)
    if s<16 or x<0 or y<0 or x+s>w or y+s>h:
        raise ValueError(f'CCD方形ROI不在图像范围内：图像{w}×{h}，ROI=({x},{y},{s})')
    camera_crop=a[y:y+s,x:x+s].copy()
    if flip_x:camera_crop=np.fliplr(camera_crop)
    if flip_y:camera_crop=np.flipud(camera_crop)
    camera=_normalize_roi(camera_crop)
    # Crop square target centered at grid center; include 99%-intensity support.
    t=np.asarray(target_intensity,dtype=float)
    yt,xt=np.where(t>float(t.max())*.01)
    if not len(xt):raise ValueError('理论目标强度为空')
    cx=(t.shape[1]-1)/2;cy=(t.shape[0]-1)/2
    half=max(np.max(np.abs(xt-cx)),np.max(np.abs(yt-cy))) * 1.18
    half=max(8,int(np.ceil(half)))
    xlo=max(0,int(cx-half));xhi=min(t.shape[1],int(cx+half+1))
    ylo=max(0,int(cy-half));yhi=min(t.shape[0],int(cy+half+1))
    target_crop=t[ylo:yhi,xlo:xhi]
    # Resizing both images is explicitly for preliminary morphology comparison.
    def resize(arr):
        v=_normalize_roi(arr)
        return np.asarray(Image.fromarray(v.astype(np.float32),mode='F').resize(
            (256,256),getattr(getattr(Image,'Resampling',Image),'BILINEAR')),dtype=float)
    measured=resize(camera)
    expected=resize(target_crop)
    va=measured.ravel()-measured.mean();vb=expected.ravel()-expected.mean()
    denom=np.linalg.norm(va)*np.linalg.norm(vb)
    corr=float(np.vdot(va,vb).real/denom) if denom>0 else 0.0
    rmse=float(np.sqrt(np.mean((measured-expected)**2)))
    return {'raw':a,'roi':camera_crop,'measured':measured,'target':expected,
            'residual':measured-expected,'correlation_exploratory':corr,
            'rmse_exploratory':rmse,'roi_box':(x,y,s),
            'caution':'ROI/理论支持域被独立缩放；仅供图样初筛，非模式纯度或定量物理配准'}
