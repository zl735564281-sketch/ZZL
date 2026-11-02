# -*- coding: utf-8 -*-
"""Ideal phase-only SLM, Fourier plane physically scaled in millimeters.

The exported gray map is converted using ideal gray*(2*pi/256). An optional
software Tilt overlay is applied ONLY in simulation; it is not added to BMP.
FFT and inverse FFT reconstruct an image of the SLM plane after an ideal 4f
spatial-filtering system. This is not a single-lens CCD plane prediction.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from slm_hologram_core import incident_gaussian

@dataclass
class Reconstruction:
    target: np.ndarray
    recovered: np.ndarray
    spectrum_log: np.ndarray
    spectrum_x: float               # millimeters
    spectrum_y: float               # millimeters
    aperture_radius: float          # millimeters
    fidelity: float
    relative_error: float
    efficiency: float
    intensity_correlation: float
    warnings: tuple
    spectrum_extent_mm: tuple
    spectrum_step_x_mm: float
    spectrum_step_y_mm: float


def _overlap(a,b):
    aa=float(np.vdot(a,a).real)
    bb=float(np.vdot(b,b).real)
    return float(abs(np.vdot(a,b))**2/(aa*bb)) if aa>0 and bb>0 else 0.


def _intensity_corr(a,b):
    a=np.asarray(a).ravel().astype(float)
    b=np.asarray(b).ravel().astype(float)
    a=a-a.mean();b=b-b.mean()
    return float(np.vdot(a,b).real / math.sqrt(float(np.vdot(a,a).real*np.vdot(b,b).real))) if np.any(a) and np.any(b) else 0.


def reconstruct(result, *, fourier_f_mm=300.0, aperture_radius_mm=0.7,
                use_external_tilt=True, tilt_x_deg=-0.680):
    """Return ideal coherent reconstruction of a selected Fourier order.

    Optical Fourier-plane positions are λ f fx with fx in cycles/m. A physical
    circular aperture radius is applied in mm (not FFT pixels). The external
    Tilt mimics phase addition by Pattern Generator and never alters result BMP.
    """
    c=result.config;p=c['physical'];h=c['hologram']
    ny,nx=result.hologram_gray.shape
    pitch=float(p['pitch_um'])*1e-6
    wavelength=float(p['wavelength_nm'])*1e-9
    f_mm=float(fourier_f_mm);r_mm=float(aperture_radius_mm)
    tilt=float(tilt_x_deg)
    if not(math.isfinite(f_mm) and f_mm>0 and math.isfinite(r_mm) and r_mm>0):
        raise ValueError('傅里叶透镜焦距与实体光阑半径必须为正值(mm)')
    if not math.isfinite(tilt) or abs(tilt)>=5:
        raise ValueError('此模型Tilt角度限定在±5度内')
    if h['use_lens']:
        lens_warn=True
    else:
        lens_warn=False
    incident=incident_gaussian(ny,nx,pitch,h['incident_waist_mm'])
    gray_phase=2*np.pi*result.hologram_gray.astype(np.float64)/256.0
    xx=(np.arange(nx,dtype=float)-(nx-1)/2)*pitch
    if use_external_tilt:
        f_ext=math.sin(math.radians(tilt))/wavelength  # cycles/m
        extra_phase=2*np.pi*f_ext*xx[None,:]
    else:
        f_ext=0.
        extra_phase=0.
    slm=incident*np.exp(1j*(gray_phase+extra_phase))
    spectrum=np.fft.fftshift(np.fft.fft2(slm))
    fx=np.fft.fftshift(np.fft.fftfreq(nx,d=pitch))
    fy=np.fft.fftshift(np.fft.fftfreq(ny,d=pitch))
    x_mm=wavelength*f_mm*fx
    y_mm=wavelength*f_mm*fy
    dx=float(x_mm[1]-x_mm[0]);dy=float(y_mm[1]-y_mm[0])
    # Center of target carrier: software Tilt + any embedded program grating.
    f_emb_x=f_emb_y=0.
    if h['use_carrier']:
        f_emb=h['carrier_sign']/(h['period_px']*pitch)
        if h['carrier_axis']=='x':f_emb_x=f_emb
        else:f_emb_y=f_emb
    total_x=f_ext+f_emb_x
    total_y=f_emb_y
    cx=wavelength*f_mm*total_x
    cy=wavelength*f_mm*total_y
    mask=((x_mm[None,:]-cx)**2+(y_mm[:,None]-cy)**2)<=r_mm**2
    if not mask.any():
        raise ValueError('所选目标载频超出FFT频谱范围；检查Tilt和光栅周期')
    filtered=spectrum*mask
    back=np.fft.ifft2(np.fft.ifftshift(filtered))
    back*=np.exp(-2j*np.pi*(total_x*xx[None,:]+total_y*((np.arange(ny)-(ny-1)/2)*pitch)[:,None]))
    target=result.field
    fidelity=_overlap(target,back)
    target2=float(np.vdot(target,target).real)
    back2=float(np.vdot(back,back).real)
    if target2>0 and back2>0:
        alpha=np.vdot(back,target)/back2
        error=float(np.linalg.norm((alpha*back-target).ravel())/math.sqrt(target2))
    else:error=float('nan')
    eff=float(np.vdot(filtered,filtered).real/np.vdot(spectrum,spectrum).real)
    I_tar=np.abs(target)**2; I_back=np.abs(back)**2
    corr=_intensity_corr(I_tar,I_back)
    F=np.log10(1+10000*np.abs(spectrum)/max(np.abs(spectrum).max(),1e-30))
    notes=[]
    if abs(total_x)<1e-8 and abs(total_y)<1e-8:
        notes.append('当前没有载频：零频滤波不代表离轴分级；实际零级反射无法通过该模型显示')
    if math.hypot(cx,cy) < 2*r_mm:
        notes.append('滤波圆孔与零频可能重叠，建议调整光阑或Tilt')
    if abs(cx)+r_mm > max(abs(x_mm.min()),abs(x_mm.max())) or abs(cy)+r_mm > max(abs(y_mm.min()),abs(y_mm.max())):
        notes.append('目标级接近频谱边界，有采样混叠风险')
    if lens_warn:
        notes.append('图案包含数字透镜；共轭面拟合误差不能代表最佳焦面的保真度')
    if h['encoding']!='phase_only':
        notes.append('棋盘双相位依赖理想低通；未模型化液晶像素串扰与像素孔径')
    if use_external_tilt:
        notes.append('厂家Tilt只在仿真叠加，不存在于输出BMP中；上机需在Pattern Generator另行叠加')
    notes.append('理想相位LUT与4f模型；非CCD实测、非严格衍射效率')
    return Reconstruction(target,back,F,float(cx),float(cy),r_mm,fidelity,error,eff,corr,
                          tuple(notes),
                          (float(x_mm[0]-dx/2),float(x_mm[-1]+dx/2),
                           float(y_mm[0]-dy/2),float(y_mm[-1]+dy/2)),dx,dy)
