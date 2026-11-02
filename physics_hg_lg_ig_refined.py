# -*- coding: utf-8 -*-
"""Scientific HG/LG/IG complex-field generator (revised from user-supplied model).

No SLM SDK and no hologram encoding. Python 3.8+, NumPy, PyTorch.

Definitions:
  HG index (n,m); LG index (p,l), exp(+i*l*theta);
  IG index (p,m,epsilon,parity), parity in {'even','odd'}.
  epsilon=2*f(z)^2/w(z)^2, fixed along analytic Gaussian-mode propagation.
  E(x,y,z) uses exp(+i*k*z + i*k*r^2/(2R) - i*(N+1)*Gouy).
  Coordinate units: SI meters, wavelength / waist in meters, z in meters.

Quality safeguards:
  - IG odd modes: -i factor makes their waist field real and matches epsilon->0
    standing-wave LG limit, under the stated modal phase convention.
  - Normalized analytic HG/LG modes; IG normalized ONCE on base z=0 grid;
    no artificial re-normalization with propagation. Finite windows can clip power.
  - Geometric stretch is explicitly non-propagative; 1/sqrt(sx*sy) maintains
    continuous-space power under positive coordinate stretches. It is NOT
    a physical Gaussian-mode propagation operation.
  - Intensity = abs(field)**2 by default. Optional peak-normalization rescales
    BOTH returned field and intensity, giving dimensionless display values.
  - Machine-learning dataset fixes global phase by a permanently nonzero real
    reference coefficient, and masks inactive mode phases. This REMOVES only
    global-phase gauge freedom; single-plane intensity inversions can remain
    non-injective (e.g. opposite OAM signs and other degeneracies). Optional
    four-step interferometric images use a known coherent LO for phase diversity.
  - Finite-epsilon IG currently computes on NumPy/CPU, DETACHING geometric
    variables; it is NOT differentiable w.r.t. coordinates/z/epsilon.
    Complex modal coefficients remain differentiable in the returned torch field.

The Ince tridiagonal recurrences were preserved from the supplied code, and
representative results should still be checked against an independent published
reference for demanding high-order or large-epsilon applications.
"""
from __future__ import annotations

import math
import random
from functools import lru_cache
from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import IterableDataset, get_worker_info


def _nonnegative_int(name, value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer, got {value!r}")
    return int(value)


class BeamConfig:
    """Physics grid, centered in meters. N is the backward-compatible square size.

    Nx/Ny may be set separately. Half-widths default to range_factor*w0.
    This grid uses endpoint-including linspace, hence pixel step is 2*half/(N-1).
    Increase half-widths for high-order modes or large z to avoid truncation.
    """

    def __init__(self, N=256, wavelength=532e-9, w0=0.8e-3,
                 range_factor=4.0, dtype=torch.float64,
                 Nx=None, Ny=None, x_half_width=None, y_half_width=None):
        if not np.isfinite(wavelength) or wavelength <= 0:
            raise ValueError("wavelength must be positive and finite")
        if not np.isfinite(w0) or w0 <= 0:
            raise ValueError("w0 must be positive and finite")
        if not np.isfinite(range_factor) or range_factor <= 0:
            raise ValueError("range_factor must be positive and finite")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("dtype must be torch.float32 or torch.float64")
        self.Nx = int(N if Nx is None else Nx)
        self.Ny = int(N if Ny is None else Ny)
        if self.Nx < 2 or self.Ny < 2:
            raise ValueError("Nx and Ny must be at least 2")
        self.N = int(N)  # compatibility alias; Nx/Ny are authoritative
        self.wavelength = float(wavelength)
        self.w0 = float(w0)
        self.range_factor = float(range_factor)
        self.x_half_width = float(x_half_width if x_half_width is not None else range_factor*w0)
        self.y_half_width = float(y_half_width if y_half_width is not None else range_factor*w0)
        if not all(np.isfinite(v) and v > 0 for v in (self.x_half_width, self.y_half_width)):
            raise ValueError("Half-widths must be finite and positive")
        self.L = 2.0 * self.x_half_width   # backward-compatible x extent
        self.XY_LIMIT = self.x_half_width  # backward-compatible x limit
        self.dtype = dtype
        self.mode_type = "HG"  # legacy attribute, does not drive selection

    @property
    def k(self):
        return 2.0*math.pi/self.wavelength

    @property
    def zR(self):
        return math.pi*self.w0**2/self.wavelength


class IncePolynomial:
    """Finite-dimensional Ince eigenproblem from the supplied source code.

    Finite-epsilon coefficients are real and trig-series-normalized. Matrix
    eigensolver residual checks reject gross failures; they do not establish
    accuracy for arbitrarily large p or epsilon.
    """

    @staticmethod
    def _check_indices(p, m, parity):
        p = _nonnegative_int('IG p', p)
        m = _nonnegative_int('IG m', m)
        if m > p or (p-m) % 2:
            raise ValueError("IG requires 0 <= m <= p, with p-m even")
        if parity not in ('even','odd'):
            raise ValueError("IG parity must be 'even' or 'odd'")
        if parity == 'odd' and m == 0:
            raise ValueError("Odd IG requires m >= 1")

    @staticmethod
    @lru_cache(maxsize=2048)
    def coefficients(p, m, epsilon_rounded, parity):
        p, m = int(p), int(m)
        q = float(epsilon_rounded)
        IncePolynomial._check_indices(p,m,parity)
        if not np.isfinite(q) or q < 0:
            raise ValueError("IG epsilon must be finite and >= 0")
        if q > 1e3:
            raise ValueError("epsilon > 1000 is outside the tested numeric range")
        if p == 0:
            return np.array([1.0],dtype=np.float64),0.0

        if parity == 'even':
            if p % 2 == 0:
                j = p//2; size=j+1; idx=m//2
                d = np.concatenate(([0.0],4.0*np.arange(1,size,dtype=float)**2))
                su = q*(j+np.arange(1,size,dtype=float))
                lo = (np.concatenate((np.array([2.0*q*j]),q*(j-np.arange(1,size-1,dtype=float))))
                      if size>1 else np.empty(0))
            else:
                j=(p-1)//2; size=j+1; idx=(m-1)//2
                r=np.arange(0,size-1,dtype=float)
                su=0.5*q*(p+2.0*r+3.0)
                r2=np.arange(1,size,dtype=float)
                lo=0.5*q*(p-(2.0*r2-1.0))
                d=np.concatenate((np.array([0.5*q+0.5*p*q+1.0]),
                                  (2.0*np.arange(1,size,dtype=float)+1.0)**2))
        else:
            if p % 2 == 0:
                j=p//2;size=j;idx=m//2-1
                r=np.arange(1,size+1,dtype=float)
                d=4.0*r**2
                if size>1:
                    su=q*(j+np.arange(2,size+1,dtype=float))
                    lo=q*(j-np.arange(1,size,dtype=float))
                else:
                    su=np.empty(0);lo=np.empty(0)
            else:
                j=(p-1)//2;size=j+1;idx=(m-1)//2
                r=np.arange(0,size-1,dtype=float)
                su=0.5*q*(p+2.0*r+3.0)
                r2=np.arange(1,size,dtype=float)
                lo=0.5*q*(p-(2.0*r2-1.0))
                d=np.concatenate((np.array([-0.5*q-0.5*p*q+1.0]),
                                  (2.0*np.arange(1,size,dtype=float)+1.0)**2))

        M=np.diag(d.astype(float))
        if len(su):M+=np.diag(su.astype(float),1)
        if len(lo):M+=np.diag(lo.astype(float),-1)
        vals, vecs=np.linalg.eig(M)
        if np.max(np.abs(np.imag(vals))) > 1e-8*max(1.0,np.max(np.abs(vals))):
            raise ArithmeticError("Ince eigenspectrum has unexpected imaginary values")
        order=np.argsort(np.real(vals))
        vals=np.real(vals[order]);vecs=np.real_if_close(vecs[:,order],tol=1000)
        if np.iscomplexobj(vecs):
            raise ArithmeticError("Ince eigenvectors unexpectedly complex")
        c=np.asarray(vecs[:,idx],dtype=np.float64)
        eta=float(vals[idx])
        residual=np.linalg.norm(M@c-eta*c)/max(1.0,np.linalg.norm(M)*np.linalg.norm(c))
        if not np.isfinite(residual) or residual>1e-8:
            raise ArithmeticError(f"Ince eigenpair residual too large: {residual:.3g}")

        # Retain original reproducible Fourier-coefficient phase convention.
        # For q>0, the leading (lowest-frequency) trig coefficient remains
        # positive in the tested range; the sign convention is a global gauge.
        if parity=='even' and p%2==0:
            norm2=2*c[0]**2+np.sum(c[1:]**2)
            sign_ref=np.sign(np.sum(c))
        elif parity=='even' and p%2==1:
            norm2=np.sum(c**2);sign_ref=np.sign(np.sum(c))
        elif parity=='odd' and p%2==0:
            rr=np.arange(1,len(c)+1,dtype=float)
            norm2=np.sum(c**2);sign_ref=np.sign(np.sum(rr*c))
        else:
            rr=2.0*np.arange(len(c),dtype=float)+1.0
            norm2=np.sum(c**2);sign_ref=np.sign(np.sum(rr*c))
        if sign_ref==0:sign_ref=1.0
        if norm2<=0 or not np.isfinite(norm2):
            raise ArithmeticError("Invalid Ince trigonometric norm")
        c=c/math.sqrt(float(norm2))*float(sign_ref)
        c.setflags(write=False)  # safeguard cached result
        return c,eta

    @staticmethod
    def evaluate(p,m,epsilon,z,parity):
        IncePolynomial._check_indices(p,m,parity)
        eps_key=round(float(epsilon),12)
        c,_=IncePolynomial.coefficients(int(p),int(m),eps_key,parity)
        z=np.asarray(z,dtype=np.complex128)
        if p==0:
            return np.ones_like(z,dtype=np.complex128)
        if parity=='even':
            rr=(2.0*np.arange(len(c)) if p%2==0
                else 2.0*np.arange(len(c))+1.0)
            basis=np.cos(z[...,None]*rr)
        else:
            rr=(2.0*np.arange(1,len(c)+1) if p%2==0
                else 2.0*np.arange(len(c))+1.0)
            basis=np.sin(z[...,None]*rr)
        return np.tensordot(basis,c,axes=([-1],[0]))


class GeneralizedPhysicsLayer(nn.Module):
    """Analytic complex HG/LG modes and numeric Ince-polynomial IG modes.

    forward returns (field,intensity), both consistent: intensity=abs(field)**2.
    normalize_intensity=True is a DISPLAY normalization (peak intensity=1).
    For quantitatively normalized physical amplitudes use False (default).
    """
    def __init__(self,config:BeamConfig,device:Union[str,torch.device]='cpu'):
        super().__init__()
        self.cfg=config
        self.device_name=str(device)
        x=torch.linspace(-config.x_half_width,config.x_half_width,config.Nx,dtype=config.dtype)
        y=torch.linspace(-config.y_half_width,config.y_half_width,config.Ny,dtype=config.dtype)
        Y,X=torch.meshgrid(y,x,indexing='ij')
        self.register_buffer('X_base',X.to(device))
        self.register_buffer('Y_base',Y.to(device))
        self.dx_grid=float(2.0*config.x_half_width/(config.Nx-1))
        self.dy_grid=float(2.0*config.y_half_width/(config.Ny-1))
        self._ig_norm_cache={}

    def gaussian_parameters(self,z):
        zR=self.cfg.zR
        w=self.cfg.w0*torch.sqrt(1.0+(z/zR)**2)
        gouy=torch.atan(z/zR)
        inv_R=z/(z*z+zR*zR)
        return w,gouy,inv_R

    def _parse_geo(self,B,geo_params):
        if geo_params is None:geo_params={}
        if not isinstance(geo_params,dict):
            raise TypeError('geo_params must be a dictionary')
        dev=self.X_base.device;dt=self.X_base.dtype
        def as_param(key,default):
            val=geo_params.get(key,default)
            if val is None:val=default
            v=torch.as_tensor(val,dtype=dt,device=dev)
            if v.numel()==1:v=v.reshape(1).expand(B)
            elif v.numel()!=B:
                raise ValueError(f"geo_params['{key}'] must have size 1 or batch size {B}")
            return v.reshape(B,1,1)
        gp={name:as_param(name,val) for name,val in
            [('scale_x',1.0),('scale_y',1.0),('rotation',0.0),
             ('dx',0.0),('dy',0.0),('z',0.0)]}
        for key,val in gp.items():
            if not bool(torch.isfinite(val).all()):
                raise ValueError(f"Invalid nonfinite geometric parameter: {key}")
        if bool((gp['scale_x']<=0).any()) or bool((gp['scale_y']<=0).any()):
            raise ValueError('scale_x and scale_y must be strictly positive')
        return gp

    def get_coordinate_grid(self,scale_x,scale_y,rotation,dx,dy):
        X=self.X_base.unsqueeze(0)-dx*self.dx_grid
        Y=self.Y_base.unsqueeze(0)-dy*self.dy_grid
        Xr=X*torch.cos(rotation)-Y*torch.sin(rotation)
        Yr=X*torch.sin(rotation)+Y*torch.cos(rotation)
        return Xr/scale_x,Yr/scale_y

    @staticmethod
    def cart2pol(x,y):
        return torch.sqrt(x*x+y*y),torch.atan2(y,x)

    @staticmethod
    def hermite_poly(n,x):
        n=_nonnegative_int('HG order',n)
        if n==0:return torch.ones_like(x)
        if n==1:return 2*x
        hm2=torch.ones_like(x);hm1=2*x
        for k in range(2,n+1):
            h=2*x*hm1-2*(k-1)*hm2
            hm2,hm1=hm1,h
        return hm1

    @staticmethod
    def associated_laguerre_poly(p,l_abs,x):
        p=_nonnegative_int('LG radial order',p)
        l_abs=_nonnegative_int('LG azimuthal absolute order',l_abs)
        if p==0:return torch.ones_like(x)
        lm2=torch.ones_like(x);lm1=(1.0+l_abs)-x
        if p==1:return lm1
        for n in range(1,p):
            ln=((2*n+1.0+l_abs-x)*lm1-(n+l_abs)*lm2)/(n+1.0)
            lm2,lm1=lm1,ln
        return lm1

    def _hg_mode(self,n,m,X,Y,z,w,gouy,inv_R):
        n=_nonnegative_int('HG n',n);m=_nonnegative_int('HG m',m)
        r2=X*X+Y*Y
        Hn=self.hermite_poly(n,math.sqrt(2)*X/w)
        Hm=self.hermite_poly(m,math.sqrt(2)*Y/w)
        norm=math.sqrt(2/(math.pi*2**(n+m)*math.factorial(n)*math.factorial(m)))/w
        ph=self.cfg.k*z+0.5*self.cfg.k*r2*inv_R-(n+m+1)*gouy
        return norm*Hn*Hm*torch.exp(-r2/w**2)*torch.exp(1j*ph)

    def _lg_mode(self,p,l,X,Y,z,w,gouy,inv_R):
        p=_nonnegative_int('LG p',p)
        if isinstance(l,(bool,np.bool_)) or not isinstance(l,(int,np.integer)):
            raise ValueError('LG l must be an integer')
        l=int(l)
        r,phi=self.cart2pol(X,Y)
        r2=r*r;la=abs(l)
        Lpl=self.associated_laguerre_poly(p,la,2*r2/w**2)
        norm=math.sqrt(2*math.factorial(p)/(math.pi*math.factorial(p+la)))/w
        ph=self.cfg.k*z+0.5*self.cfg.k*r2*inv_R-(2*p+la+1)*gouy+l*phi
        return norm*(math.sqrt(2)*r/w)**la*Lpl*torch.exp(-r2/w**2)*torch.exp(1j*ph)

    def _lg_parity_limit(self,p,m,parity,X,Y,z,w,gouy,inv_R):
        n=(p-m)//2;l=m
        if l==0:
            return self._lg_mode(n,0,X,Y,z,w,gouy,inv_R)
        plus=self._lg_mode(n,+l,X,Y,z,w,gouy,inv_R)
        minus=self._lg_mode(n,-l,X,Y,z,w,gouy,inv_R)
        if parity=='even':return (plus+minus)/math.sqrt(2)
        return (plus-minus)/(1j*math.sqrt(2))

    @staticmethod
    def _elliptical_coordinates_np(x,y,f):
        if f<=0:raise ValueError('IG semifocal distance must be positive')
        rp=np.sqrt((x+f)**2+y**2)
        rm=np.sqrt((x-f)**2+y**2)
        xi=np.arccosh(np.maximum((rp+rm)/(2*f),1.0))
        eta=np.sign(np.where(y==0,1.0,y))*np.arccos(np.clip((rp-rm)/(2*f),-1,1))
        return xi,eta

    def _ig_raw_np(self,p,m,epsilon,parity,x,y,z_scalar):
        if epsilon<=0:raise ValueError('finite-IG raw routine needs epsilon>0')
        zR=self.cfg.zR
        w=self.cfg.w0*math.sqrt(1+(z_scalar/zR)**2)
        gouy=math.atan(z_scalar/zR)
        inv_R=z_scalar/(z_scalar*z_scalar+zR*zR)
        r2=x*x+y*y
        f=math.sqrt(epsilon/2)*w
        xi,eta=self._elliptical_coordinates_np(x,y,f)
        radial=IncePolynomial.evaluate(p,m,epsilon,1j*xi,parity)
        angular=IncePolynomial.evaluate(p,m,epsilon,eta,parity)
        ph=self.cfg.k*z_scalar+0.5*self.cfg.k*r2*inv_R-(p+1)*gouy
        # S(i*xi) is imaginary. The -i gauge makes finite-epsilon odd IG
        # real at waist and consistent with epsilon=0 standing-wave LG.
        phase_gauge=(-1j if parity=='odd' else 1.0)
        return phase_gauge*(radial*angular*np.exp(-r2/w**2)*np.exp(1j*ph))/w

    def _ig_norm_constant(self,p,m,epsilon,parity):
        key=(p,m,round(float(epsilon),10),parity)
        if key in self._ig_norm_cache:return self._ig_norm_cache[key]
        x=self.X_base.detach().cpu().numpy()
        y=self.Y_base.detach().cpu().numpy()
        raw=self._ig_raw_np(p,m,epsilon,parity,x,y,0.0)
        power=float(np.sum(np.abs(raw)**2)*self.dx_grid*self.dy_grid)
        if not np.isfinite(power) or power<=0:
            raise ArithmeticError(f'IG source grid normalization failed: {key}, P={power}')
        norm=1.0/math.sqrt(power)
        self._ig_norm_cache[key]=norm
        return norm

    def _ig_mode(self,p,m,epsilon,parity,X,Y,z):
        IncePolynomial._check_indices(p,m,parity)
        epsilon=float(epsilon)
        if not np.isfinite(epsilon) or epsilon<0 or epsilon>1e3:
            raise ValueError('IG epsilon must lie in [0,1000]')
        if epsilon==0.0:
            w,gouy,inv_R=self.gaussian_parameters(z)
            return self._lg_parity_limit(p,m,parity,X,Y,z,w,gouy,inv_R)
        norm=self._ig_norm_constant(p,m,epsilon,parity)
        out=[]
        for b in range(X.shape[0]):
            xx=X[b].detach().cpu().numpy()
            yy=Y[b].detach().cpu().numpy()
            z0=float(z[b,0,0].detach().cpu())
            raw=self._ig_raw_np(p,m,epsilon,parity,xx,yy,z0)
            out.append(torch.from_numpy(norm*raw).to(device=X.device,dtype=torch.complex128))
        return torch.stack(out,dim=0)

    def _field_from_specs(self,mode_coeffs,mode_specs,geo_params,normalize_intensity):
        c=torch.as_tensor(mode_coeffs,device=self.X_base.device)
        if c.ndim==1:c=c.unsqueeze(0)
        if c.ndim!=2 or c.shape[1]!=len(mode_specs) or c.shape[1]==0:
            raise ValueError('mode_coeffs must have shape [B,K], matching nonempty mode_specs')
        if not torch.is_complex(c):c=c.to(torch.complex128)
        else:c=c.to(torch.complex128)
        if not bool(torch.isfinite(c.real).all() and torch.isfinite(c.imag).all()):
            raise ValueError('Coefficients must be finite')
        B=c.shape[0]
        gp=self._parse_geo(B,geo_params)
        X,Y=self.get_coordinate_grid(gp['scale_x'],gp['scale_y'],gp['rotation'],gp['dx'],gp['dy'])
        z=gp['z']
        w,gouy,inv_R=self.gaussian_parameters(z)
        field=torch.zeros((B,self.cfg.Ny,self.cfg.Nx),dtype=torch.complex128,device=self.X_base.device)
        for i,(family,idx) in enumerate(mode_specs):
            family=family.upper()
            if family=='HG':
                if len(idx)!=2:raise ValueError('HG index must be (n,m)')
                basis=self._hg_mode(*idx,X,Y,z,w,gouy,inv_R)
            elif family=='LG':
                if len(idx)!=2:raise ValueError('LG index must be (p,l)')
                basis=self._lg_mode(*idx,X,Y,z,w,gouy,inv_R)
            elif family=='IG':
                if len(idx)!=4:raise ValueError('IG index must be (p,m,epsilon,parity)')
                basis=self._ig_mode(*idx,X,Y,z)
            else:
                raise ValueError(f'Unsupported mode family {family!r}')
            field=field+c[:,i].reshape(B,1,1)*basis
        # Proper Jacobian power factor for arbitrary positive anisotropic stretches.
        field=field/torch.sqrt(gp['scale_x']*gp['scale_y'])
        if normalize_intensity:
            peak=(torch.abs(field)**2).flatten(1).amax(dim=1).reshape(B,1,1)
            if bool((peak<=1e-30).any()):
                raise ValueError('Cannot peak-normalize a zero or vanishing field')
            field=field/torch.sqrt(peak)
        intensity=torch.abs(field)**2
        return field,intensity

    def forward(self,mode_coeffs,mode_indices,geo_params=None,mode_type='HG',normalize_intensity=False):
        """Same-family coherent superposition. Output field & intensity agree exactly.

        input coeffs: [B,K] complex; output field,intensity: [B,Ny,Nx].
        """
        family=str(mode_type).upper()
        if family not in ('HG','LG','IG'):
            raise ValueError('mode_type must be HG, LG or IG')
        return self._field_from_specs(mode_coeffs,[(family,idx) for idx in mode_indices],
                                      geo_params,normalize_intensity)

    def forward_mixed(self,mode_coeffs,mode_specs,geo_params=None,normalize_intensity=False):
        """Cross-family coherent sum; mode_specs = [('HG',(n,m)), ('LG',(p,l)), ...].

        HG/LG/IG form alternative bases, so cross-family specs are generally
        NOT orthogonal; unit coefficient norm does NOT imply unit field power.
        """
        return self._field_from_specs(mode_coeffs,mode_specs,geo_params,normalize_intensity)

    @torch.no_grad()
    def generate_single(self,mode_type,index,geo_params=None,normalize_intensity=False):
        c=torch.ones((1,1),dtype=torch.complex128,device=self.X_base.device)
        return self.forward(c,[index],geo_params,mode_type,normalize_intensity)

    def mode_power(self,field):
        """Integral over the finite numerical grid (not infinite-space power)."""
        return torch.sum(torch.abs(field)**2,dim=(-2,-1))*self.dx_grid*self.dy_grid

    def overlap(self,field_a,field_b):
        numerator=torch.sum(torch.conj(field_a)*field_b,dim=(-2,-1))*self.dx_grid*self.dy_grid
        return numerator/torch.sqrt(self.mode_power(field_a)*self.mode_power(field_b)+1e-30)


class MultiMode_SCI_Dataset(IterableDataset):
    """Reproducible intensity dataset with a fixed complex-coefficient gauge.

    ALWAYS includes the first mode as an active real-positive reference.
    Relative phase of other modes is measured from this reference, not an
    unknowable common absolute phase. Inactive phases are 0 AND masked false.
    Single-intensity mode still does not guarantee unique coefficient recovery.
    Optional four_step_interferometry supplies four reference-phase images,
    allowing complex-field reconstruction (where the known LO is nonzero).

    Item is a dict with: intensity [Ny,Nx] or [4,Ny,Nx] float32, amplitudes [K],
    relative_phases [K], active_mask [K], phase_valid_mask [K],
    reference_index scalar int64, geometry [6] float64, intensity_scale
    scalar float64, plus optional complex_coefficients [K] if enabled.
    Metadata mode_indices are available at dataset.indices, not in each item.
    """
    def __init__(self,physics:GeneralizedPhysicsLayer,mode_type='HG',max_order=4,
                 ig_epsilon=2.0,ig_parity='even',add_image_noise=True,
                 seed=12345,max_active=5,noise_std=0.01,
                 return_coefficients=False,measurement="single_intensity",
                 reference_amplitude=1.0):
        super().__init__()
        self.physics=physics
        self.device=physics.X_base.device
        self.mode_type=str(mode_type).upper()
        self.max_order=_nonnegative_int('max_order',max_order)
        self.ig_epsilon=float(ig_epsilon)
        self.ig_parity=str(ig_parity).lower()
        self.add_image_noise=bool(add_image_noise)
        self.seed=_nonnegative_int('seed',seed)
        self.max_active=max(1,int(max_active))
        self.noise_std=float(noise_std)
        self.return_coefficients=bool(return_coefficients)
        self.measurement=str(measurement)
        self.reference_amplitude=float(reference_amplitude)
        if self.measurement not in ('single_intensity','four_step_interferometry'):
            raise ValueError('measurement must be single_intensity or four_step_interferometry')
        if not np.isfinite(self.reference_amplitude) or self.reference_amplitude<=0:
            raise ValueError('reference_amplitude must be finite and positive')
        if not np.isfinite(self.noise_std) or self.noise_std<0:
            raise ValueError('noise_std must be finite and nonnegative')
        if self.mode_type=='HG':
            self.indices=[(n,m) for n in range(self.max_order+1)
                          for m in range(self.max_order+1) if n+m<=self.max_order]
        elif self.mode_type=='LG':
            self.indices=[(p,l) for p in range(self.max_order//2+1)
                          for l in range(-self.max_order,self.max_order+1)
                          if 2*p+abs(l)<=self.max_order]
            self.indices.remove((0,0))
            self.indices.insert(0,(0,0))  # stable non-vortex Gaussian reference
        elif self.mode_type=='IG':
            if self.ig_parity not in ('even','odd'):
                raise ValueError('ig_parity must be even or odd')
            if not np.isfinite(self.ig_epsilon) or not 0<=self.ig_epsilon<=1e3:
                raise ValueError('ig_epsilon must lie in [0,1000]')
            self.indices=[(p,m,self.ig_epsilon,self.ig_parity)
                          for p in range(self.max_order+1)
                          for m in range(0 if self.ig_parity=='even' else 1,p+1)
                          if (p-m)%2==0 and not(self.ig_parity=='odd' and m==0)]
        else:raise ValueError('mode_type must be HG, LG or IG')
        if not self.indices:
            raise ValueError('No valid modes, increase max_order for odd IG')
        self.reference_index=0 # fixed and nonzero in EVERY item

    def _generate_random_coeffs(self,rng):
        K=len(self.indices)
        n_active=rng.randint(1,min(self.max_active,K))
        active=[self.reference_index]
        if n_active>1:
            candidates=[k for k in range(K) if k!=self.reference_index]
            active+=rng.sample(candidates,n_active-1)
        vals=np.zeros(K,dtype=np.complex128)
        # real-positive anchor fixes the unavoidable global U(1) phase gauge
        vals[self.reference_index]=rng.uniform(0.2,1.0)
        for j in active[1:]:
            amp=rng.uniform(0.1,1.0)
            ph=rng.uniform(-math.pi,math.pi)
            vals[j]=amp*np.exp(1j*ph)
        vals/=np.linalg.norm(vals)
        return torch.as_tensor(vals,device=self.device,dtype=torch.complex128)

    def __iter__(self):
        info=get_worker_info()
        worker_id=0 if info is None else info.id
        local_seed=self.seed+100003*worker_id
        rng=random.Random(local_seed)
        tg=torch.Generator(device='cpu')
        tg.manual_seed(local_seed+17)
        while True:
            coeffs=self._generate_random_coeffs(rng)
            geo={
                'scale_x':1.0,'scale_y':1.0,
                'rotation':rng.uniform(-math.pi,math.pi),
                'dx':rng.uniform(-1.0,1.0),
                'dy':rng.uniform(-1.0,1.0),
                'z':rng.uniform(-0.5,0.5)*self.physics.cfg.zR,
            }
            with torch.no_grad():
                E,_=self.physics.forward(coeffs.unsqueeze(0),self.indices,geo,self.mode_type,
                                         normalize_intensity=False)
                E=E.squeeze(0)
                if self.measurement=='single_intensity':
                    raw_intensity=torch.abs(E)**2
                else:
                    # Four known LO phase steps (0, pi/2, pi, 3pi/2).
                    # A phase-stable external local oscillator is assumed.
                    LO=self.physics.generate_single('HG',(0,0),geo_params=geo,
                                                    normalize_intensity=False)[0].squeeze(0)
                    LO=self.reference_amplitude*LO
                    phase_steps=(1.0,1j,-1.0,-1j)
                    raw_intensity=torch.stack([torch.abs(E+s*LO)**2 for s in phase_steps])
                intensity_scale=raw_intensity.amax().double()
                if not bool(torch.isfinite(intensity_scale)) or intensity_scale<=1e-30:
                    raise ArithmeticError('Invalid simulated intensity scale')
                I=(raw_intensity/intensity_scale).float().cpu()
                if self.add_image_noise and self.noise_std>0:
                    noise=torch.randn(I.shape,generator=tg,dtype=torch.float32)*self.noise_std
                    I=torch.clamp(I+noise,0.0,1.0)
            amp=torch.abs(coeffs).float().cpu()
            mask=(amp>1e-12)
            ph=torch.angle(coeffs).float().cpu()
            ph=torch.where(mask,ph,torch.zeros_like(ph))
            # By construction the coefficient at index 0 is always positive.
            ph[self.reference_index]=0.0
            record={
                'intensity':I,
                'intensity_scale':intensity_scale.cpu(),
                'amplitudes':amp,
                'relative_phases':ph,
                'active_mask':mask,
                'phase_valid_mask':mask.clone(),
                'reference_index':torch.tensor(self.reference_index,dtype=torch.int64),
                'geometry':torch.tensor([geo[k] for k in
                                         ('scale_x','scale_y','rotation','dx','dy','z')],dtype=torch.float64),
            }
            if self.return_coefficients:
                record['complex_coefficients']=coeffs.cpu()
            yield record


if __name__=='__main__':
    cfg=BeamConfig(N=160,wavelength=532e-9,w0=0.8e-3,range_factor=5.0)
    physics=GeneralizedPhysicsLayer(cfg,device='cpu')
    for name,family,index in [
        ('HG_20','HG',(2,0)),('LG_0_2','LG',(0,2)),
        ('IG_even_5_3_eps4','IG',(5,3,4.0,'even')),
        ('IG_odd_5_3_eps4','IG',(5,3,4.0,'odd')),
    ]:
        E,I=physics.generate_single(family,index)
        print(f'{name:25s} grid_power={physics.mode_power(E).item():.7f} ' 
              f'max_I={I.max().item():.4e} max|Im(E at waist)|={E.imag.abs().max().item():.2e}')
    ds=MultiMode_SCI_Dataset(physics,mode_type='HG',max_order=3,
                             add_image_noise=False,seed=2026)
    s=next(iter(ds))
    print('Example dataset item:',list(s.keys()),'reference=',s['reference_index'].item())
