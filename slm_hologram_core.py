# -*- coding: utf-8 -*-
"""Independent HG/LG/IG field and phase-only hologram computation.

Uses the accompanying physics_hg_lg_ig_refined.py as the authoritative modal
physics layer. No HOLOEYE SDK is imported or required.
"""
from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from physics_hg_lg_ig_refined import BeamConfig, GeneralizedPhysicsLayer


DEFAULT_CONFIG = {
    "physical": {"nx": 1920, "ny": 1080, "pitch_um": 8.0,
                 "wavelength_nm": 632.8, "w0_mm": 0.8},
    "geometry": {"z_mm": 0.0, "rotation_deg": 0.0,
                 "dx_px": 0.0, "dy_px": 0.0, "scale_x": 1.0, "scale_y": 1.0},
    "hologram": {"encoding": "phase_only", "use_carrier": False,
                 "period_px": 12.0, "carrier_axis": "x", "carrier_sign": 1,
                 "use_lens": False, "lens_f_mm": 300.0, "offset_deg": 0.0,
                 "incident_waist_mm": 2.5, "incident_threshold": 0.03},
    "normalize_coefficients": True,
    "modes": [
        {"family": "LG", "a": 0, "b": 10, "epsilon": 4.0,
         "parity": "even", "amplitude": 1.0, "phase_deg": 0.0}
    ],
}


@dataclass
class Result:
    config: dict
    field: np.ndarray
    intensity: np.ndarray
    hologram_phase: np.ndarray  # wrapped radians
    hologram_gray: np.ndarray   # uint8, 0..255
    integrated_power: float
    coeffs: np.ndarray
    notes: tuple


def _finite_float(value, name, positive=False):
    v = float(value)
    if not math.isfinite(v) or (positive and v <= 0):
        raise ValueError(f"{name}必须为{'正' if positive else '有限'}数")
    return v


def _int_value(value, name, minimum=0):
    if isinstance(value, bool):
        raise ValueError(f"{name}不能为布尔值")
    v = int(value)
    if float(value) != float(v) or v < minimum:
        raise ValueError(f"{name}必须为不小于{minimum}的整数")
    return v


def normalize_config(config):
    """Validate and canonicalize configuration; never mutate GUI state."""
    c = copy.deepcopy(config)
    p, g, h = c["physical"], c["geometry"], c["hologram"]
    p["nx"] = _int_value(p["nx"], "NX", 16)
    p["ny"] = _int_value(p["ny"], "NY", 16)
    for name in ("pitch_um", "wavelength_nm", "w0_mm"):
        p[name] = _finite_float(p[name], name, True)
    for name in ("z_mm", "rotation_deg", "dx_px", "dy_px"):
        g[name] = _finite_float(g[name], name)
    for name in ("scale_x", "scale_y"):
        g[name] = _finite_float(g[name], name, True)
    h["encoding"] = str(h["encoding"]).lower()
    if h["encoding"] not in ("phase_only", "double_phase", "compensated_double_phase"):
        raise ValueError("编码方式必须是phase_only、double_phase或compensated_double_phase")
    h["incident_waist_mm"] = _finite_float(h.get("incident_waist_mm", 2.5), "入射束腰(mm)", True)
    h["incident_threshold"] = _finite_float(h.get("incident_threshold", 0.03), "入射照明阈值")
    if not 0 < h["incident_threshold"] < 1:
        raise ValueError("入射振幅阈值需处于0至1之间")
    h["use_carrier"] = bool(h["use_carrier"])
    h["period_px"] = _finite_float(h["period_px"], "光栅周期", True)
    h["carrier_axis"] = str(h["carrier_axis"]).lower()
    if h["carrier_axis"] not in ("x", "y"):
        raise ValueError("光栅方向应是x或y")
    h["carrier_sign"] = int(h["carrier_sign"])
    if h["carrier_sign"] not in (-1, 1):
        raise ValueError("光栅方向符号应是+1或-1")
    h["use_lens"] = bool(h["use_lens"])
    h["lens_f_mm"] = _finite_float(h["lens_f_mm"], "数字透镜焦距")
    if h["use_lens"] and h["lens_f_mm"] == 0:
        raise ValueError("启用数字透镜时焦距不能为0")
    h["offset_deg"] = _finite_float(h["offset_deg"], "相位偏置")
    c["normalize_coefficients"] = bool(c.get("normalize_coefficients", True))
    if not c.get("modes"):
        raise ValueError("至少需要添加一个模式")
    for i, m in enumerate(c["modes"], 1):
        family = str(m["family"]).upper()
        if family not in ("HG", "LG", "IG"):
            raise ValueError(f"第{i}个模式类型必须是HG/LG/IG")
        m["family"] = family
        m["a"] = _int_value(m["a"], f"第{i}个模式索引a", 0)
        m["b"] = _int_value(m["b"], f"第{i}个模式索引b", -100000)
        m["epsilon"] = _finite_float(m.get("epsilon", 4.0), "IG椭圆率")
        m["parity"] = str(m.get("parity", "even")).lower()
        m["amplitude"] = _finite_float(m["amplitude"], f"第{i}个模式幅值")
        if m["amplitude"] < 0:
            raise ValueError("模式幅值系数必须非负，负号可以用180度相位表达")
        m["phase_deg"] = _finite_float(m["phase_deg"], f"第{i}个模式相位")
        if family == "HG" and m["b"] < 0:
            raise ValueError("HG m必须≥0")
        if family == "IG":
            if m["b"] < 0 or m["b"] > m["a"] or (m["a"] - m["b"]) % 2:
                raise ValueError("IG要求0≤m≤p，且p与m具有相同奇偶性")
            if m["parity"] not in ("even", "odd") or (m["parity"] == "odd" and m["b"] == 0):
                raise ValueError("IG奇偶性应为even/odd；odd模式要求m≥1")
            if m["epsilon"] < 0 or m["epsilon"] > 1000:
                raise ValueError("IG椭圆率epsilon需位于[0,1000]")
    if not any(m["amplitude"] > 0 for m in c["modes"]):
        raise ValueError("至少一个模式的幅值应大于0")
    return c


def modal_specs_and_coeffs(c):
    specs, coeffs = [], []
    for m in c["modes"]:
        family = m["family"]
        if family == "IG":
            idx = (m["a"], m["b"], m["epsilon"], m["parity"])
        else:
            idx = (m["a"], m["b"])
        specs.append((family, idx))
        coeffs.append(m["amplitude"] * np.exp(1j*np.deg2rad(m["phase_deg"])))
    coeffs = np.asarray(coeffs, dtype=np.complex128)
    if c["normalize_coefficients"]:
        # This is L2 normalization of the coefficient vector ONLY.
        # Mixed HG/LG/IG families need not be orthogonal.
        coeffs /= np.linalg.norm(coeffs)
    return specs, coeffs


def incident_gaussian(ny, nx, pitch_m, waist_mm):
    """Ideal zero-phase Gaussian amplitude at the SLM plane."""
    w=float(waist_mm)*1e-3
    x=(np.arange(nx, dtype=float)-(nx-1)/2)*pitch_m
    y=(np.arange(ny, dtype=float)-(ny-1)/2)*pitch_m
    return np.exp(-(x[None,:]**2+y[:,None]**2)/w**2)


def effective_modulation(field, h, pitch_m):
    """Return phase-only target transmittance for the specified method.

    In compensated_double_phase, T=c Utarget / Uincident within the specified
    illumination support, with max(abs(T))=1; elsewhere desired T=0.
    This is a model of the amplitude to be recovered by spatial filtering,
    NOT the raw physical SLM transmission (always unit amplitude).
    """
    amplitude=np.abs(field)
    if not np.isfinite(amplitude).all() or amplitude.max()<=0:
        raise ValueError('目标复场无效')
    kind=h['encoding']
    if kind=='phase_only':
        return np.exp(1j*np.angle(field)), 1.0
    if kind=='double_phase':
        return field/float(amplitude.max()), 1.0
    ny,nx=field.shape
    incident=incident_gaussian(ny,nx,pitch_m,h['incident_waist_mm'])
    support=incident>=float(h['incident_threshold'])
    if not support.any():
        raise ValueError('给定阈值下无有效入射照明区域')
    coverage=float(np.sum(amplitude[support]**2)/np.sum(amplitude**2))
    # The weakly illuminated target region is intentionally excluded, so no
    # uncontrolled 1/incident amplification occurs at the Gaussian tails.
    raw=np.zeros_like(field)
    raw[support]=field[support]/incident[support]
    maximum=float(np.max(np.abs(raw)))
    if not np.isfinite(maximum) or maximum<=0:
        raise ValueError('目标场未落在有效入射照明区域内')
    return raw/maximum, coverage


def double_phase_checkerboard(modulation):
    """Checkerboard phase-only representation of desired complex T, |T|<=1.

    e^(i phi1/2) alternated at single pixels, with phi±acos(|T|).
    Only the low-pass component approximates T. No physical amplitude stop.
    """
    scaled=np.minimum(np.abs(modulation),1.0)
    alpha=np.arccos(scaled)
    phase=np.angle(modulation)
    ny,nx=modulation.shape
    y,x=np.ogrid[:ny,:nx]
    return phase+np.where((x+y)%2==0,alpha,-alpha)


def encode_existing_field(result, encoding):
    """Make another hologram with the SAME computed target field (fast A/B test)."""
    c=copy.deepcopy(result.config)
    c['hologram']['encoding']=encoding
    return hologram_from_field(c,result.field,coeffs=result.coeffs,power=result.integrated_power)


def hologram_from_field(config,field,coeffs=None,power=None):
    """Encode a precomputed complex target field without solving the modes again."""
    c=normalize_config(config)
    p,h=c['physical'],c['hologram']
    ny,nx=field.shape
    if (ny,nx)!=(p['ny'],p['nx']):
        raise ValueError('模式复场尺寸与SLM配置不一致')
    pitch=p['pitch_um']*1e-6
    modulation,coverage=effective_modulation(field,h,pitch)
    if h['encoding']=='phase_only':
        holo=np.angle(field)
    else:
        holo=double_phase_checkerboard(modulation)
    notes=[]
    if h['encoding']=='compensated_double_phase':
        notes.append(f'按理想高斯入射振幅预补偿；目标能量有效区域覆盖率={coverage:.5f}；' +
                     '低照明区域被设为目标零振幅，不是实体光阑')
    if h['encoding']!='phase_only':
        notes.append('双相位棋盘编码需后续物理低通滤波；单像素交错受到实际像素串扰和带宽限制')
    if h['use_carrier']:
        coord=np.arange(nx if h['carrier_axis']=='x' else ny,dtype=float)
        carrier=h['carrier_sign']*2*np.pi*coord/h['period_px']
        holo=holo+(carrier[None,:] if h['carrier_axis']=='x' else carrier[:,None])
        if h['period_px']<8:
            notes.append('小于8px的载频需谨慎检查采样与效率')
    if h['use_lens']:
        xx=(np.arange(nx)-(nx-1)/2)*pitch
        yy=(np.arange(ny)-(ny-1)/2)*pitch
        holo=holo-np.pi*(xx[None,:]**2+yy[:,None]**2)/(p['wavelength_nm']*1e-9*h['lens_f_mm']*1e-3)
    holo=np.remainder(holo+np.deg2rad(h['offset_deg']),2*np.pi)
    gray=phase_to_u8(holo)
    notes.append('BMP不包含仿真设置的厂家Tilt，除非勾选GUI内“叠加闪耀光栅”')
    notes.append('理想256级周期灰度映射，实际LUT、响应非线性与像素填充率未标定')
    if coeffs is None: coeffs=modal_specs_and_coeffs(c)[1]
    if power is None: power=float(np.sum(np.abs(field)**2)*pitch**2)
    return Result(c,field,np.abs(field)**2,holo,gray,float(power),coeffs,tuple(notes))


def phase_to_u8(phase):
    """Periodic nominal 2pi -> 256-code mapping; NOT a measured inverse LUT."""
    wrapped = np.remainder(phase, 2*np.pi)
    codes = np.floor(256.0 * (wrapped / (2*np.pi)))
    return np.asarray(np.clip(codes, 0, 255), dtype=np.uint8)


def generate(config):
    c = normalize_config(config)
    p, g, h = c["physical"], c["geometry"], c["hologram"]
    nx, ny = p["nx"], p["ny"]
    pitch = p["pitch_um"] * 1e-6
    wavelength = p["wavelength_nm"] * 1e-9
    w0 = p["w0_mm"] * 1e-3
    # Endpoint-inclusive core grid -> EXACT pitch with (N-1)*pitch/2.
    cfg = BeamConfig(Nx=nx, Ny=ny, wavelength=wavelength, w0=w0,
                     x_half_width=(nx-1)*pitch/2.0,
                     y_half_width=(ny-1)*pitch/2.0, dtype=torch.float64)
    phy = GeneralizedPhysicsLayer(cfg, device="cpu")
    specs, coeffs = modal_specs_and_coeffs(c)
    geo = {"z": g["z_mm"]*1e-3, "rotation": math.radians(g["rotation_deg"]),
           "dx": g["dx_px"], "dy": g["dy_px"],
           "scale_x": g["scale_x"], "scale_y": g["scale_y"]}
    with torch.no_grad():
        field_t, _ = phy.forward_mixed(
            torch.tensor(coeffs[None, :], dtype=torch.complex128), specs,
            geo_params=geo, normalize_intensity=False)
    field = field_t[0].detach().cpu().numpy()
    if not np.all(np.isfinite(field)) or np.max(np.abs(field)) <= 1e-30:
        raise ValueError("复场无效或近似为0；请检查模式叠加系数和参数")
    return hologram_from_field(c,field,coeffs=coeffs,
                               power=float(np.sum(np.abs(field)**2)*phy.dx_grid*phy.dy_grid))


def export_result(result, folder, base_name="hologram", save_npy=True):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    safe = "".join(ch if (ch.isalnum() or ch in "_-.") else "_" for ch in str(base_name)).strip("._") or "hologram"
    bmp = folder/(safe + "_SLM_8bit.bmp")
    png = folder/(safe + "_SLM_8bit.png")
    Image.fromarray(result.hologram_gray, mode="L").save(str(bmp))
    Image.fromarray(result.hologram_gray, mode="L").save(str(png))
    config_path = folder/(safe + "_params.json")
    extra = {"field_grid_power": result.integrated_power,
             "notes": list(result.notes),
             "coefficients": [{"real": float(z.real), "imag": float(z.imag)} for z in result.coeffs]}
    with config_path.open("w", encoding="utf-8") as f:
        json.dump({"configuration": result.config, "diagnostics": extra},
                  f, ensure_ascii=False, indent=2)
    paths = [bmp, png, config_path]
    if save_npy:
        cp = folder/(safe + "_complex_field.npy")
        hp = folder/(safe + "_hologram_phase_rad.npy")
        np.save(str(cp), result.field)
        np.save(str(hp), result.hologram_phase)
        paths.extend((cp, hp))
    return paths
