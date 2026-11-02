# -*- coding: utf-8 -*-
"""HG/LG/IG structured-light visualizer + hologram design + Fourier filtering GUI.

Python 3.8+, no SLM SDK. Run `python SLM_HG_LG_IG_GUI.py`.
All modal physics delegated to physics_hg_lg_ig_refined.py.
"""
from __future__ import annotations
import copy
import json
import math
import queue
import re
import threading
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.patches import Circle
from matplotlib import rcParams, font_manager

from slm_hologram_core import DEFAULT_CONFIG, generate, export_result, normalize_config, encode_existing_field, phase_to_u8
from diffraction_simulation import reconstruct
from research_analysis import (METHODS, run_aperture_study, pareto_rows,
                               write_study_csv, load_ccd_image, compare_ccd_roi)


def setup_fonts():
    names=('Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'WenQuanYi Micro Hei')
    present={f.name for f in font_manager.fontManager.ttflist}
    chosen=next((f for f in names if f in present),None)
    if not chosen:
        try:
            import subprocess
            path=subprocess.check_output(['fc-match','-f','%{file}',':lang=zh-cn'],
                                          text=True,stderr=subprocess.DEVNULL,timeout=2).strip()
            if Path(path).is_file():
                font_manager.fontManager.addfont(path)
                chosen=font_manager.FontProperties(fname=path).get_name()
        except Exception:
            pass
    if chosen:
        rcParams['font.sans-serif']=[chosen,'DejaVu Sans']
    rcParams['axes.unicode_minus']=False

setup_fonts()


def auto_filename(config):
    # Keep filenames tied to the actual mode list, not to an editable label.
    chunks=[]
    for m in config['modes']:
        fam=m['family']
        idx=f"{m['a']}_{m['b']}"
        if fam=='IG':
            idx+=f"_e{m['epsilon']:g}_{'e' if m['parity']=='even' else 'o'}"
        chunks.append(f"{fam}{idx}_a{m['amplitude']:g}_d{m['phase_deg']:g}")
    h=config['hologram']
    suffix=(f"{h['encoding']}_"+
            (f"G{h['carrier_axis']}{h['carrier_sign']:+d}_{h['period_px']:g}px" if h['use_carrier'] else 'NoCarrier'))
    return re.sub(r'[^A-Za-z0-9_.+\-]','_', '__'.join(chunks)+'__'+suffix)[:150]


class ScrollPanel(ttk.Frame):
    """Scrollable physical parameters to avoid bottom controls being clipped."""
    def __init__(self,parent,**kwargs):
        super().__init__(parent,**kwargs)
        can=tk.Canvas(self,highlightthickness=0)
        bar=ttk.Scrollbar(self,orient='vertical',command=can.yview)
        self.inner=ttk.Frame(can,padding=(12,12))
        self.inner.bind('<Configure>',lambda e:can.configure(scrollregion=can.bbox('all')))
        self.window=can.create_window((0,0),window=self.inner,anchor='nw')
        can.bind('<Configure>',lambda e:can.itemconfigure(self.window,width=e.width))
        can.configure(yscrollcommand=bar.set)
        can.pack(side='left',fill='both',expand=True)
        bar.pack(side='right',fill='y')


class App:
    def __init__(self,root):
        self.root=root
        root.title('HG/LG/IG 复振幅全息科研工具 V5 · 实验验证版（无需SDK）')
        root.geometry('1420x875')
        root.minsize(1020,690)
        self.config=copy.deepcopy(DEFAULT_CONFIG)
        self.result=None
        self.recon=None
        self.result_signature=None
        self.tasks=queue.Queue()
        self.busy=False
        self.editor_loading=False
        self.editor_dirty=False
        self._edit_after=None
        self.vars={}
        self.mode_vars={}
        self.status=tk.StringVar(value='就绪｜按顺序设置模式、参数，点击“计算与预览”。')
        self.mode_hint=tk.StringVar(value='')
        self.metric_target=tk.StringVar(value='目标场尚未计算')
        self.metric_recon=tk.StringVar(value='尚未计算；衍射重建使用理想相位响应和高斯照明假设')
        self.auto_name=tk.BooleanVar(value=True)
        self.filename_var=tk.StringVar(value='')
        self.zoom_var=tk.StringVar(value='自动')
        self.incident_var=tk.StringVar(value='2.5')
        self.threshold_var=tk.StringVar(value='0.03')
        self.aperture_var=tk.StringVar(value='0.70')
        self.fourier_f_var=tk.StringVar(value='300')
        self.tilt_angle_var=tk.StringVar(value='-0.680')
        self.external_tilt_var=tk.BooleanVar(value=True)
        self.compare_recs=None
        self.study_rows=None
        self.study_signature=None
        self.ccd_path=None
        self.ccd_data=None
        self.study_min_var=tk.StringVar(value='0.25')
        self.study_max_var=tk.StringVar(value='1.10')
        self.study_points_var=tk.StringVar(value='7')
        self.study_method_var=tk.StringVar(value='三种编码')
        self.study_status_var=tk.StringVar(value='尚未进行参数扫描。建议先以HG10测试，之后测试HG23/IG64。')
        self.metric_compare=tk.StringVar(value='先计算目标模式，再点击“三种编码对照”。')
        self._init_style()
        self._build_toolbar()
        self.tabs=ttk.Notebook(root)
        self.tabs.pack(fill='both',expand=True,padx=12,pady=6)
        self.mode_page=ttk.Frame(self.tabs,padding=9)
        self.physics_page=ScrollPanel(self.tabs)
        self.target_page=ttk.Frame(self.tabs,padding=7)
        self.holo_page=ttk.Frame(self.tabs,padding=7)
        self.recon_page=ttk.Frame(self.tabs,padding=7)
        self.compare_page=ttk.Frame(self.tabs,padding=7)
        self.study_page=ttk.Frame(self.tabs,padding=7)
        self.tabs.add(self.mode_page,text='① 模式与复系数')
        self.tabs.add(self.physics_page,text='② 物理 / 编码 / 重建参数')
        self.tabs.add(self.target_page,text='③ 目标场')
        self.tabs.add(self.holo_page,text='④ SLM全息图')
        self.tabs.add(self.recon_page,text='⑤ 傅里叶重建')
        self.tabs.add(self.compare_page,text='⑥ 三编码对照')
        self.tabs.add(self.study_page,text='⑦ 扫描 / 实验验证')
        self._build_modes()
        self._build_params()
        self._build_plots()
        ttk.Label(root,textvariable=self.status,anchor='w',relief='sunken',padding=(9,6)).pack(
            side='bottom',fill='x',padx=10,pady=(0,7))
        self._load_config_to_ui(self.config)
        self.root.after(100,self._poll)

    def _init_style(self):
        s=ttk.Style()
        try:s.theme_use('clam')
        except tk.TclError:pass
        s.configure('Header.TLabel',font=('Microsoft YaHei UI',12,'bold'))
        s.configure('Subhead.TLabel',font=('Microsoft YaHei UI',10,'bold'))
        s.configure('Hint.TLabel',foreground='#526275')
        s.configure('Run.TButton',padding=(11,8))
        s.configure('TNotebook.Tab',padding=(12,7))
        s.configure('Treeview',rowheight=27)

    def _build_toolbar(self):
        line=ttk.Frame(self.root,padding=(13,10,13,4))
        line.pack(fill='x')
        ttk.Label(line,text='HG / LG / IG 复振幅全息 · V5',style='Header.TLabel').pack(side='left',padx=(0,14))
        self.btn_calc=ttk.Button(line,text='计算与预览',style='Run.TButton',command=self.calculate)
        self.btn_calc.pack(side='left',padx=4)
        self.btn_export=ttk.Button(line,text='导出 8位 BMP / PNG',style='Run.TButton',command=self.export)
        self.btn_export.pack(side='left',padx=4)
        self.btn_export_single=ttk.Button(line,text='导出单张图',command=self.export_single_dialog)
        self.btn_export_single.pack(side='left',padx=4)
        self.btn_compare=ttk.Button(line,text='三种编码对照',command=self.calculate_comparison)
        self.btn_compare.pack(side='left',padx=4)
        ttk.Button(line,text='保存配置',command=self.save_json).pack(side='left',padx=4)
        ttk.Button(line,text='载入配置',command=self.load_json).pack(side='left',padx=4)
        ttk.Button(line,text='恢复默认',command=self.reset).pack(side='left',padx=4)
        ttk.Label(line,text='无需HOLOEYE SDK',style='Hint.TLabel').pack(side='right',padx=4)

    def _build_modes(self):
        parent=self.mode_page
        ttk.Label(parent,text='复振幅叠加：U = Σ aₖ exp(i δₖ) Uₖ ；模式更改后需重新计算。',
                  style='Hint.TLabel').pack(anchor='w',pady=(1,9))
        row=ttk.Panedwindow(parent,orient='horizontal')
        row.pack(fill='both',expand=True)
        left=ttk.Frame(row,padding=3)
        right=ttk.LabelFrame(row,text='选中模式 · 编辑参数',padding=14)
        row.add(left,weight=5)
        row.add(right,weight=3)
        self.tree=ttk.Treeview(left,columns=('family','idx','amp','phase'),show='headings',height=11,selectmode='browse')
        for key,text,width in [('family','类别',85),('idx','阶数与椭圆率',235),('amp','振幅系数',90),('phase','相对相位 (°)',105)]:
            self.tree.heading(key,text=text)
            self.tree.column(key,width=width,anchor='center',stretch=True)
        scroll=ttk.Scrollbar(left,orient='vertical',command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side='left',fill='both',expand=True)
        scroll.pack(side='right',fill='y')
        self.tree.bind('<<TreeviewSelect>>',self._select_mode)
        self.mode_vars={k:tk.StringVar(value=v) for k,v in
            [('family','LG'),('a','0'),('b','10'),('epsilon','4.0'),('parity','even'),('amplitude','1'),('phase_deg','0')]}
        descriptors=[('模式类型','family','combo',('HG','LG','IG')),
                     ('第一阶数 n / p','a','entry',None),('第二阶数 m / l','b','entry',None),
                     ('IG椭圆率 ε','epsilon','entry',None),('IG奇偶性','parity','combo',('even','odd')),
                     ('振幅系数 a','amplitude','entry',None),('相对相位 δ（度）','phase_deg','entry',None)]
        self.editor_widgets={}
        for rowno,(txt,key,typ,opts) in enumerate(descriptors):
            ttk.Label(right,text=txt).grid(row=rowno,column=0,sticky='w',padx=5,pady=6)
            if typ=='combo':
                widget=ttk.Combobox(right,values=opts,textvariable=self.mode_vars[key],state='readonly',width=18)
            else:
                widget=ttk.Entry(right,textvariable=self.mode_vars[key],width=20)
            widget.grid(row=rowno,column=1,sticky='ew',padx=5,pady=6)
            self.editor_widgets[key]=widget
        right.columnconfigure(1,weight=1)
        self.index_hint=ttk.Label(right,text='',style='Hint.TLabel',wraplength=350)
        self.index_hint.grid(row=7,column=0,columnspan=2,sticky='w',pady=(5,9))
        ttk.Label(right,text='振幅快捷调整 (0–2)').grid(row=8,column=0,sticky='w')
        self.amp_slider=tk.Scale(right,from_=0,to=2,resolution=.02,orient='horizontal',
                                 showvalue=False,length=160,command=lambda v:self.mode_vars['amplitude'].set(v))
        self.amp_slider.grid(row=8,column=1,sticky='ew')
        ttk.Label(right,text='相位快捷调整 (−180°～180°)').grid(row=9,column=0,sticky='w')
        self.ph_slider=tk.Scale(right,from_=-180,to=180,resolution=1,orient='horizontal',
                                showvalue=False,command=lambda v:self.mode_vars['phase_deg'].set(v))
        self.ph_slider.grid(row=9,column=1,sticky='ew')
        ttk.Label(right,text='参数可自动应用于选中行；数值未填完整时等待输入。',
                  style='Hint.TLabel',wraplength=380).grid(row=10,column=0,columnspan=2,sticky='w',pady=4)
        for var in self.mode_vars.values():
            var.trace_add('write',self._editor_changed)
        self.mode_vars['family'].trace_add('write',lambda *_:self._mode_index_hint())
        commands=ttk.Frame(parent)
        commands.pack(fill='x',pady=10)
        for text,command in [('添加模式',self.add_mode),('应用修改',self.apply_mode),('删除选中',self.delete_mode),
                             ('HG₁₀ + HG₀₁',lambda:self.demo('HG')),
                             ('LG₀⁺¹ + LG₀⁻¹',lambda:self.demo('LG')),
                             ('IG₅₃',lambda:self.demo('IG')),
                             ('LG₀¹⁰',lambda:self.demo('L10'))]:
            ttk.Button(commands,text=text,command=command).pack(side='left',padx=3)
        self.norm_var=tk.BooleanVar(value=True)
        ttk.Checkbutton(parent,text='系数向量L2归一化（跨模族/非正交叠加时不等于场功率归一化）',variable=self.norm_var).pack(anchor='w')

    def _build_params(self):
        parent=self.physics_page.inner
        parent.columnconfigure(0,weight=1)
        parent.columnconfigure(1,weight=1)
        physical=ttk.LabelFrame(parent,text='物理采样（真实SLM像元）',padding=10)
        geometry=ttk.LabelFrame(parent,text='模式观察平面 / 几何变换',padding=10)
        hologram=ttk.LabelFrame(parent,text='SLM相位编码',padding=10)
        simulation=ttk.LabelFrame(parent,text='理想傅里叶面滤波 / 重建仿真',padding=10)
        physical.grid(row=0,column=0,sticky='nsew',padx=7,pady=7)
        geometry.grid(row=0,column=1,sticky='nsew',padx=7,pady=7)
        hologram.grid(row=1,column=0,sticky='nsew',padx=7,pady=7)
        simulation.grid(row=1,column=1,sticky='nsew',padx=7,pady=7)
        def fields(frame,values):
            frame.columnconfigure(1,weight=1)
            for i,(key,label,value) in enumerate(values):
                ttk.Label(frame,text=label).grid(row=i,column=0,sticky='w',pady=5,padx=3)
                v=tk.StringVar(value=str(value))
                self.vars[key]=v
                ttk.Entry(frame,textvariable=v,width=17).grid(row=i,column=1,sticky='ew',pady=5,padx=5)
        p=DEFAULT_CONFIG['physical'];g=DEFAULT_CONFIG['geometry'];h=DEFAULT_CONFIG['hologram']
        fields(physical,[('nx','横向像素 NX',p['nx']),('ny','纵向像素 NY',p['ny']),
                         ('pitch_um','像元间距 (μm)',p['pitch_um']),
                         ('wavelength_nm','波长 (nm)',p['wavelength_nm']),
                         ('w0_mm','目标模式束腰 w₀ (mm)',p['w0_mm'])])
        fields(geometry,[('z_mm','解析传播位置 z (mm)',g['z_mm']),
                         ('rotation_deg','模式旋转 (度)',g['rotation_deg']),
                         ('dx_px','中心X偏移 (px)',g['dx_px']),('dy_px','中心Y偏移 (px)',g['dy_px']),
                         ('scale_x','横向几何伸缩',g['scale_x']),('scale_y','纵向几何伸缩',g['scale_y'])])
        form=ttk.Frame(hologram)
        form.pack(fill='x')
        form.columnconfigure(1,weight=1)
        ttk.Label(form,text='编码方式').grid(row=0,column=0,sticky='w',pady=6)
        self.encoding=tk.StringVar(value=h['encoding'])
        ttk.Combobox(form,textvariable=self.encoding,values=('phase_only','double_phase','compensated_double_phase'),
                     width=18,state='readonly').grid(row=0,column=1,sticky='ew',padx=5)
        self.carrier=tk.BooleanVar(value=h['use_carrier'])
        self.lens=tk.BooleanVar(value=h['use_lens'])
        ttk.Checkbutton(form,text='叠加闪耀光栅',variable=self.carrier).grid(row=1,column=0,columnspan=2,sticky='w')
        ttk.Checkbutton(form,text='叠加数字透镜',variable=self.lens).grid(row=2,column=0,columnspan=2,sticky='w')
        for i,(key,label,value) in enumerate([('period_px','光栅周期（原生px）',h['period_px']),
                                               ('lens_f_mm','数字透镜焦距 (mm)',h['lens_f_mm']),
                                               ('offset_deg','附加相位偏置 (度)',h['offset_deg'])],start=3):
            ttk.Label(form,text=label).grid(row=i,column=0,sticky='w',pady=5)
            self.vars[key]=tk.StringVar(value=str(value))
            ttk.Entry(form,textvariable=self.vars[key],width=16).grid(row=i,column=1,sticky='ew',padx=5,pady=5)
        ttk.Label(form,text='光栅梯度轴').grid(row=6,column=0,sticky='w',pady=4)
        self.axis=tk.StringVar(value='x')
        ttk.Combobox(form,textvariable=self.axis,values=('x','y'),width=18,state='readonly').grid(row=6,column=1,sticky='ew',padx=5)
        ttk.Label(form,text='光栅载频符号').grid(row=7,column=0,sticky='w',pady=4)
        self.sign=tk.StringVar(value='1')
        ttk.Combobox(form,textvariable=self.sign,values=('1','-1'),width=18,state='readonly').grid(row=7,column=1,sticky='ew',padx=5)
        ttk.Label(hologram,text='推荐：导出时关闭自制光栅，厂家软件再叠加Tilt。双相位需傅里叶面低通；理想256级映射未校准LUT。',
                  style='Hint.TLabel',wraplength=440).pack(anchor='w',pady=7)
        # The SAME incident waist is used for precompensation and simulation.
        self.vars['incident_waist_mm']=self.incident_var
        self.vars['incident_threshold']=self.threshold_var
        simulation.columnconfigure(1,weight=1)
        fields_sim=[('SLM面入射高斯束腰 (mm)',self.incident_var),
                    ('有效照明振幅阈值 (0~1)',self.threshold_var),
                    ('实体傅里叶透镜焦距 (mm)',self.fourier_f_var),
                    ('实体光阑半径 (mm)',self.aperture_var),
                    ('厂家Tilt X偏转角 (deg)',self.tilt_angle_var)]
        for i,(label,var) in enumerate(fields_sim):
            ttk.Label(simulation,text=label).grid(row=i,column=0,sticky='w',pady=5)
            ttk.Entry(simulation,textvariable=var,width=17).grid(row=i,column=1,sticky='ew',padx=5,pady=5)
        ttk.Checkbutton(simulation,text='仿真叠加厂家Tilt（BMP不含此Tilt）',variable=self.external_tilt_var).grid(
            row=5,column=0,columnspan=2,sticky='w',pady=(8,3))
        ttk.Label(simulation,text='频谱滤波是实体毫米圆孔，坐标从 λ·f·FFT频率换算。预测为理想4f共轭平面，不是现有单透镜CCD任意位置的直接预测。',
                  style='Hint.TLabel',wraplength=460,justify='left').grid(row=6,column=0,columnspan=2,sticky='w',pady=5)
        ttk.Label(simulation,text='入射束腰应使用SLM面实际1/e振幅半径（等于1/e²强度半径）；不要随意依赖2.5mm默认假设。',
                  style='Hint.TLabel',wraplength=460).grid(row=7,column=0,columnspan=2,sticky='w',pady=5)
        note=ttk.Label(parent,text='物理注意：z是解析模式平面，不等同于SLM后任意传播距离；几何缩放不是自由传播。' ,
                       style='Hint.TLabel')
        note.grid(row=2,column=0,columnspan=2,sticky='w',padx=10,pady=7)

    def _new_figure(self,tab):
        fig=Figure(figsize=(10.6,6.3),dpi=98,facecolor='white')
        axes=fig.subplots(2,2)
        fig.subplots_adjust(left=.075,right=.985,bottom=.095,top=.93,wspace=.19,hspace=.32)
        cvs=FigureCanvasTkAgg(fig,master=tab)
        cvs.get_tk_widget().pack(fill='both',expand=True)
        toolbar=NavigationToolbar2Tk(cvs,tab,pack_toolbar=False)
        toolbar.update(); toolbar.pack(anchor='w')
        for ax in axes.flat:
            ax.text(.5,.5,'计算后显示',ha='center',va='center',transform=ax.transAxes)
            ax.set_xticks([]);ax.set_yticks([])
        cvs.draw_idle()
        return fig,axes,cvs

    def _build_plots(self):
        target_top=ttk.Frame(self.target_page)
        target_top.pack(fill='x',pady=3)
        ttk.Label(target_top,text='目标模式复场',style='Subhead.TLabel').pack(side='left',padx=8)
        ttk.Label(target_top,text='中心局部放大倍数').pack(side='left',padx=(20,6))
        ttk.Combobox(target_top,textvariable=self.zoom_var,values=('自动','1','1.5','2','3','4','6','8'),width=5,
                     state='readonly').pack(side='left')
        ttk.Button(target_top,text='刷新显示',command=self.redraw).pack(side='left',padx=10)
        ttk.Label(self.target_page,textvariable=self.metric_target,style='Hint.TLabel',wraplength=1300).pack(anchor='w',padx=10,pady=(0,3))
        self.fig_t,self.axes_t,self.cvs_t=self._new_figure(self.target_page)
        top=ttk.Frame(self.holo_page)
        top.pack(fill='x')
        ttk.Label(top,text='全息图：左侧全幅，右下局部显示原生像素 / 线剖面',style='Subhead.TLabel').pack(side='left',padx=8)
        self.fig_h,self.axes_h,self.cvs_h=self._new_figure(self.holo_page)
        top=ttk.Frame(self.recon_page)
        top.pack(fill='x')
        ttk.Label(top,text='理想4f傅里叶面选级与共轭面重建',style='Subhead.TLabel').pack(side='left',padx=8)
        ttk.Button(top,text='重新计算滤波',command=self.calculate).pack(side='left',padx=13)
        ttk.Label(self.recon_page,textvariable=self.metric_recon,style='Hint.TLabel',wraplength=1350).pack(anchor='w',padx=10,pady=4)
        self.fig_r,self.axes_r,self.cvs_r=self._new_figure(self.recon_page)
        compare_top=ttk.Frame(self.compare_page)
        compare_top.pack(fill='x',pady=3)
        ttk.Label(compare_top,text='相同目标场、相同照明及物理滤波，对照三种编码重建',style='Subhead.TLabel').pack(side='left',padx=8)
        ttk.Button(compare_top,text='计算三种编码',command=self.calculate_comparison).pack(side='left',padx=10)
        ttk.Label(self.compare_page,textvariable=self.metric_compare,style='Hint.TLabel',wraplength=1280).pack(anchor='w',padx=10,pady=4)
        self.fig_c,self.axes_c,self.cvs_c=self._new_figure(self.compare_page)
        study_top=ttk.Frame(self.study_page)
        study_top.pack(fill='x',pady=(3,7))
        ttk.Label(study_top,text='实体傅里叶光阑半径 (mm)：').pack(side='left',padx=3)
        for label,variable,width in [('起始',self.study_min_var,7),('结束',self.study_max_var,7),('点数',self.study_points_var,5)]:
            ttk.Label(study_top,text=label).pack(side='left',padx=(8,2))
            ttk.Entry(study_top,textvariable=variable,width=width).pack(side='left')
        ttk.Label(study_top,text='编码').pack(side='left',padx=(10,2))
        ttk.Combobox(study_top,textvariable=self.study_method_var,values=('三种编码',)+METHODS,
                     state='readonly',width=23).pack(side='left')
        self.btn_study=ttk.Button(study_top,text='扫描',command=self.calculate_study)
        self.btn_study.pack(side='left',padx=5)
        self.btn_study_csv=ttk.Button(study_top,text='导出扫描CSV',command=self.export_study_csv)
        self.btn_study_csv.pack(side='left',padx=3)
        control=ttk.Frame(self.study_page)
        control.pack(fill='x',pady=(0,4))
        ttk.Button(control,text='一键导出三编码实验BMP+参数',command=self.batch_export).pack(side='left',padx=4)
        ttk.Button(control,text='导入原始CCD灰度图对照',command=self.ccd_dialog).pack(side='left',padx=4)
        ttk.Label(control,text='仅扫描/仿真厂家Tilt；导出BMP不含Tilt，不再自行添加光栅。',style='Hint.TLabel').pack(side='left',padx=12)
        ttk.Label(self.study_page,textvariable=self.study_status_var,style='Hint.TLabel',wraplength=1320).pack(anchor='w',padx=8,pady=2)
        self.fig_s,self.axes_s,self.cvs_s=self._new_figure(self.study_page)
        self.output_controls=ttk.Frame(self.holo_page)
        self.output_controls.pack(fill='x',padx=10,pady=3)
        ttk.Checkbutton(self.output_controls,text='根据实际模式自动命名',variable=self.auto_name,
                        command=self._update_auto_name).pack(side='left')
        ttk.Label(self.output_controls,text='文件前缀').pack(side='left',padx=(20,5))
        ttk.Entry(self.output_controls,textvariable=self.filename_var,width=80).pack(side='left',fill='x',expand=True)

    def _set_editor(self,row):
        self.editor_loading=True
        for k,v in row.items():
            if k in self.mode_vars:
                self.mode_vars[k].set(str(v))
        self.amp_slider.set(min(2,max(0,float(row['amplitude']))))
        self.ph_slider.set(min(180,max(-180,float(row['phase_deg']))))
        self.mode_vars['amplitude'].set(str(row['amplitude']))
        self.mode_vars['phase_deg'].set(str(row['phase_deg']))
        self.editor_loading=False
        self.editor_dirty=False
        self._mode_index_hint()

    def _mode_index_hint(self):
        fam=self.mode_vars['family'].get()
        messages={'HG':'HG：(n,m)，两个阶数均为非负整数。',
                  'LG':'LG：(p,l)，径向 p≥0；拓扑荷 l 可为负。',
                  'IG':'IG：(p,m,ε,parity)；0≤m≤p、同奇偶；奇IG m≥1。'}
        self.index_hint.configure(text=messages.get(fam,''))
        for key in ('epsilon','parity'):
            widget=self.editor_widgets.get(key) if hasattr(self,'editor_widgets') else None
            if widget is not None:
                widget.configure(state=('normal' if key=='epsilon' else 'readonly') if fam=='IG' else 'disabled')

    def _editor_changed(self,*args):
        if self.editor_loading:return
        self.editor_dirty=True
        if self._edit_after is not None:
            self.root.after_cancel(self._edit_after)
        self._edit_after=self.root.after(450,self._try_auto_apply)

    def _try_auto_apply(self):
        self._edit_after=None
        if not self.editor_dirty:return
        i=self._selected()
        if i is None:return
        try:
            new=self._read_editor()
        except Exception:return
        self.config['modes'][i]=new
        self.editor_dirty=False
        self._refresh_rows(i)
        self._mark_changed()

    def _read_editor(self):
        row={'family':self.mode_vars['family'].get(),
             'a':int(self.mode_vars['a'].get()),'b':int(self.mode_vars['b'].get()),
             'epsilon':float(self.mode_vars['epsilon'].get()),
             'parity':self.mode_vars['parity'].get(),
             'amplitude':float(self.mode_vars['amplitude'].get()),
             'phase_deg':float(self.mode_vars['phase_deg'].get())}
        c=copy.deepcopy(DEFAULT_CONFIG)
        c['modes']=[row] if row['amplitude']>0 else [row,copy.deepcopy(DEFAULT_CONFIG['modes'][0])]
        normalize_config(c)
        return row

    def _selected(self):
        selection=self.tree.selection()
        return int(selection[0]) if selection else None

    def _refresh_rows(self,focus=None):
        self.tree.delete(*self.tree.get_children())
        for i,m in enumerate(self.config['modes']):
            idx=f"({m['a']},{m['b']})" if m['family']!='IG' else f"({m['a']},{m['b']},ε={m['epsilon']},{m['parity']})"
            self.tree.insert('',tk.END,iid=str(i),values=(m['family'],idx,f"{m['amplitude']:g}",f"{m['phase_deg']:g}"))
        if focus is not None and 0<=focus<len(self.config['modes']):
            self.tree.selection_set(str(focus));self.tree.focus(str(focus))
        self._update_auto_name()

    def _select_mode(self,event=None):
        if self._edit_after is not None:
            self.root.after_cancel(self._edit_after)
            self._edit_after=None
        i=self._selected()
        if i is not None and i<len(self.config['modes']):
            self._set_editor(self.config['modes'][i])

    def apply_mode(self):
        i=self._selected()
        if i is None:
            messagebox.showinfo('提示','请先选中模式表中的一行');return
        try:
            self.config['modes'][i]=self._read_editor()
            self.editor_dirty=False
            self._refresh_rows(i);self._mark_changed()
        except Exception as e:
            messagebox.showerror('模式参数错误',str(e))

    def _flush_pending_editor(self):
        if not self.editor_dirty:return
        i=self._selected()
        if i is not None:
            self.config['modes'][i]=self._read_editor()
            self.editor_dirty=False
            self._refresh_rows(i)

    def add_mode(self):
        try:
            self._flush_pending_editor()
            self.config['modes'].append(self._read_editor())
            self._refresh_rows(len(self.config['modes'])-1);self._mark_changed()
        except Exception as e:messagebox.showerror('参数错误',str(e))

    def delete_mode(self):
        i=self._selected()
        if i is None:return
        del self.config['modes'][i]
        self._refresh_rows(min(i,len(self.config['modes'])-1) if self.config['modes'] else None)
        if self.config['modes']:
            self._set_editor(self.config['modes'][min(i,len(self.config['modes'])-1)])
        self._mark_changed()

    def demo(self,name):
        d=copy.deepcopy(DEFAULT_CONFIG['modes'][0])
        if name=='HG':
            data=[dict(d,family='HG',a=1,b=0,amplitude=1.,phase_deg=0.),
                  dict(d,family='HG',a=0,b=1,amplitude=1.,phase_deg=90.)]
        elif name=='LG':
            data=[dict(d,family='LG',a=0,b=1,phase_deg=0.),
                  dict(d,family='LG',a=0,b=-1,phase_deg=0.)]
        elif name=='IG':
            data=[dict(d,family='IG',a=5,b=3,epsilon=4.,parity='even')]
        else:
            data=[dict(d,family='LG',a=0,b=10,phase_deg=0.)]
        self.config['modes']=data
        self._refresh_rows(0);self._set_editor(data[0]);self._mark_changed()

    def _load_config_to_ui(self,c):
        self.config=copy.deepcopy(c)
        for section in ('physical','geometry','hologram'):
            for key,value in c[section].items():
                if key in self.vars:
                    self.vars[key].set(str(value))
        h=c['hologram']
        self.encoding.set(h['encoding']);self.carrier.set(h['use_carrier'])
        self.lens.set(h['use_lens']);self.axis.set(h['carrier_axis']);self.sign.set(str(h['carrier_sign']))
        self.norm_var.set(c.get('normalize_coefficients',True))
        self._refresh_rows(0)
        if c['modes']:
            self._set_editor(c['modes'][0])
        self._update_auto_name()

    def _collect(self):
        self._flush_pending_editor()
        c=copy.deepcopy(self.config)
        p,g,h=c['physical'],c['geometry'],c['hologram']
        p['nx']=int(self.vars['nx'].get());p['ny']=int(self.vars['ny'].get())
        for key in ('pitch_um','wavelength_nm','w0_mm'):
            p[key]=float(self.vars[key].get())
        for key in ('z_mm','rotation_deg','dx_px','dy_px','scale_x','scale_y'):
            g[key]=float(self.vars[key].get())
        h['encoding']=self.encoding.get();h['use_carrier']=self.carrier.get()
        h['use_lens']=self.lens.get();h['carrier_axis']=self.axis.get();h['carrier_sign']=int(self.sign.get())
        for key in ('period_px','lens_f_mm','offset_deg','incident_waist_mm','incident_threshold'):
            h[key]=float(self.vars[key].get())
        c['normalize_coefficients']=self.norm_var.get()
        return normalize_config(c)

    @staticmethod
    def _signature(c):
        return json.dumps(c,sort_keys=True,ensure_ascii=False)

    def _update_auto_name(self):
        if self.auto_name.get():
            try:
                c=copy.deepcopy(self.config)
                if self.editor_dirty and self._selected() is not None:
                    c['modes'][self._selected()]=self._read_editor()
                h=c['hologram']
                if hasattr(self,'encoding'):h['encoding']=self.encoding.get()
                if hasattr(self,'carrier'):h['use_carrier']=self.carrier.get()
                if 'period_px' in self.vars:
                    try:h['period_px']=float(self.vars['period_px'].get())
                    except ValueError:pass
                if hasattr(self,'axis'):h['carrier_axis']=self.axis.get()
                if hasattr(self,'sign'):h['carrier_sign']=int(self.sign.get())
                self.filename_var.set(auto_filename(c))
            except Exception:
                pass

    def _mark_changed(self):
        self._update_auto_name()
        self.status.set('参数已修改。点击“计算与预览”更新目标场与频域重建。')

    def _simulation_settings(self):
        f=float(self.fourier_f_var.get())
        r=float(self.aperture_var.get())
        angle=float(self.tilt_angle_var.get())
        enabled=bool(self.external_tilt_var.get())
        if not (math.isfinite(f) and f>0 and math.isfinite(r) and r>0 and math.isfinite(angle)):
            raise ValueError('傅里叶焦距、圆孔半径需为正数且Tilt角度需有限')
        return (f,r,enabled,angle)

    @staticmethod
    def _reconstruct_for(result,settings):
        f,r,enable,ang=settings
        return reconstruct(result,fourier_f_mm=f,aperture_radius_mm=r,
                           use_external_tilt=enable,tilt_x_deg=ang)

    def _results_current(self):
        c=self._collect()
        return self.result is not None and self._signature(c)==self.result_signature and self._simulation_settings()==self.last_sim_settings

    def calculate_comparison(self):
        if self.busy:return
        try:
            if not self._results_current():
                messagebox.showwarning('请先重算','模式、编码或仿真参数已修改，请先点击“计算与预览”。')
                return
        except Exception as e:
            messagebox.showerror('参数错误',str(e));return
        self.busy=True
        self.btn_compare.configure(state='disabled')
        self.btn_calc.configure(state='disabled')
        self.btn_export.configure(state='disabled')
        self.btn_export_single.configure(state='disabled')
        settings=self.last_sim_settings
        base=self.result
        self.status.set('正在以同一目标复场计算三种编码并对比重建……')
        def worker():
            try:
                recs={}
                for kind in ('phase_only','double_phase','compensated_double_phase'):
                    variant=encode_existing_field(base,kind)
                    recs[kind]=self._reconstruct_for(variant,settings)
                self.tasks.put(('compare_done',recs))
            except Exception:
                self.tasks.put(('error',traceback.format_exc()))
        threading.Thread(target=worker,daemon=True).start()

    def _draw_comparison(self):
        if not self.compare_recs:return
        ax=self.axes_c.flat
        target=np.abs(self.result.field)**2
        full=self._target_extent()
        images=[('理想目标强度',target),
                ('纯相位重建',np.abs(self.compare_recs['phase_only'].recovered)**2),
                ('棋盘双相位重建',np.abs(self.compare_recs['double_phase'].recovered)**2),
                ('入射补偿双相位重建',np.abs(self.compare_recs['compensated_double_phase'].recovered)**2)]
        for a,(name,data) in zip(ax,images):
            self._image(a,data/max(float(data.max()),1e-300),name+'（各自峰值归一）','inferno',full,0,1)
            self._zoom(a)
            a.set_xlabel('x (mm)');a.set_ylabel('y (mm)')
        recs=self.compare_recs
        lines=[]
        for key,label in [('phase_only','纯相位'),('double_phase','普通双相位'),('compensated_double_phase','补偿双相位')]:
            rec=recs[key]
            lines.append(f'{label}：复场重叠={rec.fidelity:.4f}，强度相关={rec.intensity_correlation:.4f}，通带能量比例={rec.efficiency:.4f}')
        self.metric_compare.set('  |  '.join(lines)+'\n数值重叠不是实测模式纯度；三方案使用相同滤波孔径。')
        self.cvs_c.draw_idle()

    def calculate_study(self):
        if self.busy:return
        try:
            if not self._results_current():
                messagebox.showwarning('请先重算','模式、编码或仿真参数已更改。请先计算与预览。')
                return
            start=float(self.study_min_var.get());end=float(self.study_max_var.get())
            count=int(self.study_points_var.get())
            if not (math.isfinite(start) and math.isfinite(end) and 0<start<end and 2<=count<=15):
                raise ValueError('扫描需满足 0<起始<结束（mm），点数2～15')
            methods=METHODS if self.study_method_var.get()=='三种编码' else (self.study_method_var.get(),)
            radii=np.linspace(start,end,count)
        except Exception as e:
            messagebox.showerror('扫描参数错误',str(e));return
        self.busy=True
        for btn in (self.btn_study,self.btn_calc,self.btn_compare,self.btn_export,self.btn_export_single):
            btn.configure(state='disabled')
        self.study_rows=None
        signature=(self.result_signature,self.last_sim_settings,tuple(float(x) for x in radii),tuple(methods))
        base=self.result; settings=self.last_sim_settings
        self.study_status_var.set(f'计算中：{len(methods)}种编码 × {count}个半径，完整SLM网格；请等待……')
        def worker():
            try:
                rows=run_aperture_study(base,radii,settings,methods)
                self.tasks.put(('study_done',(rows,signature)))
            except Exception:
                self.tasks.put(('error',traceback.format_exc()))
        threading.Thread(target=worker,daemon=True).start()

    def _draw_study(self):
        if not self.study_rows:return
        ax=self.axes_s.flat
        for a in ax:a.clear();a.grid(alpha=.2)
        rows=self.study_rows
        groups=list(dict.fromkeys(r['method'] for r in rows))
        colors=['#1776b5','#b56525','#169269']
        for color,name in zip(colors,groups):
            rr=[r for r in rows if r['method']==name]
            radii=[r['radius_mm'] for r in rr]
            ax[0].plot(radii,[r['fidelity_squared_ideal'] for r in rr],'-o',ms=4,label=name,color=color)
            ax[1].plot(radii,[r['filtered_power_fraction_ideal'] for r in rr],'-o',ms=4,label=name,color=color)
            ax[2].plot(radii,[r['intensity_correlation_ideal'] for r in rr],'-o',ms=4,label=name,color=color)
            ax[3].plot([r['filtered_power_fraction_ideal'] for r in rr],
                       [r['fidelity_squared_ideal'] for r in rr],'-o',ms=4,label=name,color=color)
        for a,title,ylab in zip(ax[:3],('理想复场重叠 vs 光阑','理想通带能量比例 vs 光阑','理想强度相关 vs 光阑'),
                                 ('重叠模平方','通带能量比例','相关系数')):
            a.set_title(title,fontsize=10);a.set_xlabel('光阑半径 (mm)');a.set_ylabel(ylab)
        ax[3].set_title('保真度—能量权衡（非实测）',fontsize=10)
        ax[3].set_xlabel('理想通带能量比例');ax[3].set_ylabel('重叠模平方')
        ax[0].legend(fontsize=7)
        frontier=pareto_rows(rows)
        # Highlight cross-method nondominated solutions.
        for r,good in zip(rows,frontier):
            if good:
                ax[3].scatter(r['filtered_power_fraction_ideal'],r['fidelity_squared_ideal'],
                              s=72,facecolors='none',edgecolors='black',linewidths=1.0)
        candidates=[r for r in rows if r['filtered_power_fraction_ideal']>=0.05]
        best=max(candidates,key=lambda r:r['fidelity_squared_ideal']) if candidates else None
        hint=(f"示例高保真点(理想通带能量≥5%)：{best['method']}，R={best['radius_mm']:.3f}mm，"
              f"F={best['fidelity_squared_ideal']:.4f}，E={best['filtered_power_fraction_ideal']:.4f}。") if best else '扫描范围内未找到理想通带能量≥5%的点。'
        self.study_status_var.set(hint+' 圈出帕累托非支配点；不是实际实验最优光阑。')
        self.cvs_s.draw_idle()

    def export_study_csv(self):
        if not self.study_rows:
            messagebox.showinfo('提示','请先执行扫描');return
        if self.study_signature[:2] != (self.result_signature,self.last_sim_settings) or not self._results_current():
            messagebox.showwarning('参数变化','扫描所用物理/仿真参数已改变；请先重新扫描。');return
        path=filedialog.asksaveasfilename(title='导出带宽-保真度扫描CSV',defaultextension='.csv',
                 initialfile=auto_filename(self.result.config)+'_aperture_study.csv',filetypes=[('CSV','*.csv')])
        if not path:return
        try:
            write_study_csv(self.study_rows,path)
            self.fig_s.savefig(str(Path(path).with_suffix('.png')),dpi=165)
            self.status.set('扫描数据与曲线已导出：'+path)
        except Exception as e:messagebox.showerror('CSV导出失败',str(e))

    def batch_export(self):
        if self.busy or self.result is None:
            messagebox.showinfo('提示','请先完成模式计算');return
        try:
            if not self._results_current():
                messagebox.showwarning('参数变化','请重新计算，再导出三种编码。');return
            if self.result.config['hologram']['use_carrier']:
                messagebox.showwarning('避免重复载频','当前自制光栅已勾选。请关闭并重新计算，保证输出是待厂家叠加Tilt的纯编码BMP。')
                return
            if self.result.config['hologram']['use_lens']:
                messagebox.showwarning('额外透镜','当前数字透镜已勾选。请关闭并重新计算后再导出基准对照。')
                return
        except Exception as e:
            messagebox.showerror('检查失败',str(e));return
        folder=filedialog.askdirectory(title='选择三种编码实验BMP保存目录')
        if not folder:return
        try:
            base=self.result
            f,r,on,ang=self.last_sim_settings
            manifest={'purpose':'A/B/C experimental comparison: export without internal carrier; add ONE external vendor Tilt',
                      'gui_version':'V5', 'physical':base.config['physical'],
                      'geometry':base.config['geometry'],
                      'modes':base.config['modes'],
                      'nominal_external_tilt_deg':ang if on else None,
                      'fourier_f_mm_sim':f,'aperture_radius_mm_sim':r,
                      'models':[], 'cautions':['Ideal 256-code phase, not measured LUT',
                      'Ideal camera intensity similarity is not modal purity',
                      'Factory background compensation and other overlays must remain identical']}
            path=Path(folder)
            for name in METHODS:
                variant=encode_existing_field(base,name)
                base_name=auto_filename(variant.config)
                exports=export_result(variant,path,base_name=base_name,save_npy=False)
                with Image.open(exports[0]) as im:
                    arr=np.asarray(im)
                    assert im.size == (base.config['physical']['nx'],base.config['physical']['ny'])
                    assert arr.dtype==np.uint8 and np.array_equal(arr,variant.hologram_gray)
                manifest['models'].append({'encoding':name,'file':exports[0].name,
                                           'parameters':exports[2].name})
            out=path/'实验对照_三种编码_记录.json'
            out.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
            self.status.set('三种编码BMP/PNG及实验配置已导出：'+str(path))
            messagebox.showinfo('导出完成','三种编码各一张1920×1080原生8位BMP/PNG。\n文件均不包含厂家Tilt；请仅在Pattern Generator中叠加一次Tilt。')
        except Exception as e:messagebox.showerror('批量导出失败',str(e))

    def ccd_dialog(self):
        if self.result is None:
            messagebox.showinfo('提示','请先计算理论目标场');return
        path=filedialog.askopenfilename(title='选择原始CCD单通道灰度图',
                        filetypes=[('CCD gray images','*.bmp *.png *.tif *.tiff'),('All files','*.*')])
        if not path:return
        try:data=load_ccd_image(path)
        except Exception as e:messagebox.showerror('CCD读取失败',str(e));return
        win=tk.Toplevel(self.root)
        win.title('CCD灰度图初步对照｜不能替代模式纯度测量')
        win.geometry('970x750')
        frame=ttk.Frame(win,padding=9);frame.pack(fill='both',expand=True)
        top=ttk.Frame(frame);top.pack(fill='x')
        ttk.Label(top,text=f'CCD: {Path(path).name}  |  尺寸: {data.shape[1]}×{data.shape[0]}').pack(side='left',padx=4)
        settings=ttk.Frame(frame);settings.pack(fill='x',pady=6)
        s=max(16,int(min(data.shape)*0.7))
        values={'x0':str((data.shape[1]-s)//2),'y0':str((data.shape[0]-s)//2),'size':str(s)}
        var={k:tk.StringVar(value=v) for k,v in values.items()}
        for k,label in [('x0','ROI X'),('y0','ROI Y'),('size','方形边长(px)')]:
            ttk.Label(settings,text=label).pack(side='left',padx=(8,2))
            ttk.Entry(settings,textvariable=var[k],width=8).pack(side='left')
        fx=tk.BooleanVar(value=False);fy=tk.BooleanVar(value=False)
        ttk.Checkbutton(settings,text='水平翻转',variable=fx).pack(side='left',padx=6)
        ttk.Checkbutton(settings,text='垂直翻转',variable=fy).pack(side='left',padx=6)
        info=tk.StringVar(value='请选取只包含同一目标衍射级的方形ROI；截图和过曝图不适合定量分析。')
        ttk.Label(frame,textvariable=info,style='Hint.TLabel',wraplength=920).pack(anchor='w',padx=5,pady=3)
        fig=Figure(figsize=(9,6),dpi=92);aa=fig.subplots(2,2)
        fig.subplots_adjust(left=.07,right=.98,top=.93,bottom=.08,hspace=.35,wspace=.19)
        canvas=FigureCanvasTkAgg(fig,master=frame)
        canvas.get_tk_widget().pack(fill='both',expand=True)
        results={}
        def draw():
            try:
                comp=compare_ccd_roi(data,self.result.intensity,int(var['x0'].get()),
                        int(var['y0'].get()),int(var['size'].get()),fx.get(),fy.get())
                results.clear();results.update(comp)
                display=[(comp['raw'],'原始CCD图（未缩放裁剪）','gray'),
                         (comp['measured'],'CCD选区（显示归一化）','inferno'),
                         (comp['target'],'理论目标强度（独立缩放）','inferno'),
                         (comp['residual'],'差异（仅形状对照）','coolwarm')]
                for a,(img,ttl,cmap) in zip(aa.flat,display):
                    a.clear();a.imshow(img,cmap=cmap,origin='upper');a.set_title(ttl,fontsize=10)
                    a.set_xticks([]);a.set_yticks([])
                xx,yy,side=comp['roi_box']
                from matplotlib.patches import Rectangle
                aa[0,0].add_patch(Rectangle((xx,yy),side,side,ec='cyan',lw=1.6,fill=False))
                info.set(f'探索性强度形状相关={comp["correlation_exploratory"]:.4f}；归一化RMSE={comp["rmse_exploratory"]:.4f}。'
                         '两幅图被独立裁剪和缩放；这不是定量光学配准，更不是复场模式纯度。')
                canvas.draw_idle()
            except Exception as e:
                messagebox.showerror('CCD对照失败',str(e))
        buttons=ttk.Frame(frame);buttons.pack(fill='x',pady=6)
        ttk.Button(buttons,text='更新ROI并对照',command=draw).pack(side='left',padx=4)
        def save_report():
            if not results:return
            filepath=filedialog.asksaveasfilename(defaultextension='.png',initialfile='CCD_vs_target_preliminary.png',
                                     filetypes=[('PNG','*.png')])
            if not filepath:return
            fig.savefig(filepath,dpi=165)
            payload={'ccd_source':str(path),'roi':results['roi_box'],
                     'flip_x':fx.get(),'flip_y':fy.get(),
                     'morphology_corr_exploratory':results['correlation_exploratory'],
                     'rmse_exploratory':results['rmse_exploratory'],
                     'caution':results['caution'],'model':self.result.config}
            Path(filepath).with_suffix('.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
        ttk.Button(buttons,text='保存初筛PNG及JSON',command=save_report).pack(side='left',padx=4)
        ttk.Button(buttons,text='关闭',command=win.destroy).pack(side='right',padx=5)
        draw()

    def calculate(self):
        if self.busy:return
        try:
            c=self._collect()
            settings=self._simulation_settings()
        except Exception as e:
            messagebox.showerror('输入参数错误',str(e));return
        signature=self._signature(c)
        self.busy=True
        self.btn_calc.configure(state='disabled')
        self.btn_export.configure(state='disabled')
        self.btn_export_single.configure(state='disabled')
        self.btn_compare.configure(state='disabled')
        self.btn_study.configure(state='disabled')
        self.compare_recs=None
        self.study_rows=None
        self.study_signature=None
        self.status.set('正在计算完整分辨率模式、全息图和傅里叶重建；高阶IG可能耗时较长……')
        def worker():
            try:
                res=generate(c)
                sim=self._reconstruct_for(res,settings)
                self.tasks.put(('done',(res,sim,signature,settings)))
            except Exception:
                self.tasks.put(('error',traceback.format_exc()))
        threading.Thread(target=worker,daemon=True).start()

    def _poll(self):
        try:
            while True:
                typ,payload=self.tasks.get_nowait()
                self.busy=False
                self.btn_calc.configure(state='normal')
                self.btn_export.configure(state='normal')
                self.btn_export_single.configure(state='normal')
                self.btn_compare.configure(state='normal')
                self.btn_study.configure(state='normal')
                if typ=='study_done':
                    self.study_rows,self.study_signature=payload
                    self._draw_study()
                    self.tabs.select(self.study_page)
                    self.status.set('带宽—保真度—通带能量扫描已完成。')
                elif typ=='compare_done':
                    self.compare_recs=payload
                    self._draw_comparison()
                    self.tabs.select(self.compare_page)
                    self.status.set('三种编码重建对照已完成，按相同物理孔径计算。')
                elif typ=='error':
                    self.status.set('计算失败；见错误详情')
                    messagebox.showerror('计算失败',payload[-6000:])
                else:
                    self.result,self.recon,self.result_signature,settings=payload
                    self.last_sim_settings=settings
                    self._draw_target();self._draw_hologram();self._draw_recon()
                    self._update_auto_name()
                    self.tabs.select(self.target_page)
                    c=self.result.config
                    self.status.set(f"计算完成｜{c['physical']['nx']}×{c['physical']['ny']}｜"
                                    f"有限网格功率 {self.result.integrated_power:.5g}｜"
                                    f"理想复场重叠 {self.recon.fidelity:.4f}（非实测纯度）")
        except queue.Empty:pass
        self.root.after(100,self._poll)

    def _image(self,ax,image,title,cmap,extent=None,vmin=None,vmax=None):
        ax.clear()
        n=max(image.shape)
        step=max(1,math.ceil(n/950))
        arr=image[::step,::step]
        ax.imshow(arr,cmap=cmap,origin='lower',extent=extent,vmin=vmin,vmax=vmax,
                  interpolation='nearest',aspect='equal')
        ax.set_title(title,fontsize=10)
        ax.tick_params(labelsize=8)

    def _target_extent(self):
        p=self.result.config['physical']
        s=p['pitch_um']*1e-3
        return [-p['nx']*s/2,p['nx']*s/2,-p['ny']*s/2,p['ny']*s/2]

    def _zoom(self,ax):
        rect=self._target_extent()
        maxhalf=min(abs(rect[0]),abs(rect[1]),abs(rect[2]),abs(rect[3]))
        if self.zoom_var.get()=='自动' and self.result is not None:
            # Keep the main high-intensity structure visible, including high-order
            # LG rings. A fixed 3x zoom in the rectangular SLM grid clips it.
            intensity=self.result.intensity
            peak=float(np.max(intensity))
            yy,xx=np.where(intensity > peak*0.015)
            pitch_mm=self.result.config['physical']['pitch_um']*0.001
            if xx.size:
                cx=(intensity.shape[1]-1)/2.0
                cy=(intensity.shape[0]-1)/2.0
                radial=max(float(np.max(np.abs(xx-cx))),float(np.max(np.abs(yy-cy))))*pitch_mm
                half=min(maxhalf,max(.2,1.22*radial+.08))
            else:
                half=maxhalf
        else:
            fac=max(1.,float(self.zoom_var.get()))
            half=maxhalf/fac
        ax.set_xlim(-half,half)
        ax.set_ylim(-half,half)

    def _draw_target(self):
        if self.result is None:return
        I=self.result.intensity
        A=np.abs(self.result.field)
        ph=np.angle(self.result.field)
        peak=max(float(A.max()),1e-300)
        shown=np.ma.array(ph,mask=(A/peak<0.001))
        full=self._target_extent()
        ax=self.axes_t.flat
        self._image(ax[0],I/I.max(),'目标强度 · 全幅','inferno',full,vmin=0,vmax=1)
        self._image(ax[1],I/I.max(),'目标强度 · 中心放大','inferno',full,vmin=0,vmax=1)
        self._zoom(ax[1])
        self._image(ax[2],shown,'目标复相位 · 中心放大 (rad)','twilight',full,-np.pi,np.pi)
        self._zoom(ax[2])
        self._image(ax[3],A/peak,'目标振幅 · 中心放大','viridis',full,vmin=0,vmax=1)
        self._zoom(ax[3])
        for a in ax:a.set_xlabel('x (mm)',fontsize=8);a.set_ylabel('y (mm)',fontsize=8)
        self.metric_target.set(f"有限窗积分功率 {self.result.integrated_power:.6g} | "
                               f"模式数 {len(self.result.config['modes'])} | "
                               '图中极小振幅区域的相位仅在显示时遮罩；实际全息图未遮罩。')
        self.cvs_t.draw_idle()

    def _draw_hologram(self):
        r=self.result
        ph=r.hologram_phase
        gray=r.hologram_gray
        ax=self.axes_h.flat
        e=[0,gray.shape[1],0,gray.shape[0]]
        self._image(ax[0],ph,'全幅包裹相位 (0–2π)','twilight',e,0,2*np.pi)
        self._image(ax[1],gray,'全幅8位SLM灰度','gray',e,0,255)
        ny,nx=gray.shape
        half=64
        patch=gray[max(0,ny//2-half):min(ny,ny//2+half),
                   max(0,nx//2-half):min(nx,nx//2+half)]
        self._image(ax[2],patch,'中心128×128原生像素','gray',None,0,255)
        ax[2].set_xlabel('局部像素')
        ax[3].clear()
        span=min(nx,240)
        left=max(0,nx//2-span//2)
        row=gray[ny//2,left:left+span]
        ax[3].plot(np.arange(left,left+len(row)),row,linewidth=1)
        ax[3].set_ylim(-8,263)
        ax[3].grid(alpha=.2)
        ax[3].set_title('中心水平灰度剖面 (实际像素)',fontsize=10)
        ax[3].set_xlabel('x (SLM pixel)');ax[3].set_ylabel('gray')
        self.cvs_h.draw_idle()

    def _draw_recon(self):
        r=self.recon
        if r is None:return
        ax=self.axes_r.flat
        F=r.spectrum_log
        ny,nx=F.shape
        extent=r.spectrum_extent_mm
        self._image(ax[0],F,'傅里叶面 log 幅值 · 物理坐标(mm)','magma',extent)
        ax[0].add_patch(Circle((r.spectrum_x,r.spectrum_y),r.aperture_radius,
                                ec='#40f2ba',fc='none',lw=2))
        ax[0].scatter([0],[0],marker='+',s=40,c='white')
        ax[0].plot(r.spectrum_x,r.spectrum_y,marker='+',ms=8,color='lime')
        if r.spectrum_x or r.spectrum_y:
            ax[0].set_xlim(min(0,r.spectrum_x)-2*r.aperture_radius,
                           max(0,r.spectrum_x)+2*r.aperture_radius)
            ax[0].set_ylim(min(0,r.spectrum_y)-2*r.aperture_radius,
                           max(0,r.spectrum_y)+2*r.aperture_radius)
        rec=r.recovered
        target=r.target
        i_rec=np.abs(rec)**2
        i_tar=np.abs(target)**2
        full=self._target_extent()
        self._image(ax[1],i_tar/max(i_tar.max(),1e-300),'目标强度 · 同一平面','inferno',full,0,1)
        self._zoom(ax[1])
        self._image(ax[2],i_rec/max(i_rec.max(),1e-300),'滤波后重建强度 · 共轭平面','inferno',full,0,1)
        self._zoom(ax[2])
        pmask=np.abs(rec)/max(float(np.abs(rec).max()),1e-300) < .02
        self._image(ax[3],np.ma.array(np.angle(rec),mask=pmask),'重建复场相位（去载频）','twilight',full,-np.pi,np.pi)
        self._zoom(ax[3])
        warning='；'.join(r.warnings)
        self.metric_recon.set(f"复场重叠模平方={r.fidelity:.4f}；强度相关={r.intensity_correlation:.4f}；"
                              f"拟合误差={r.relative_error:.4f}；通带能量比例={r.efficiency:.4f}；"
                              f"圆孔半径={r.aperture_radius:.3f}mm；中心=({r.spectrum_x:.3f},{r.spectrum_y:.3f})mm\n{warning}")
        self.cvs_r.draw_idle()

    def redraw(self):
        if self.result is not None:
            try:self._draw_target();self._draw_recon()
            except Exception as e:messagebox.showerror('预览刷新错误',str(e))


    def _single_export_items(self):
        if self.result is None:
            return {}
        items={}
        field=self.result.field
        amp=np.abs(field)
        intensity=self.result.intensity.astype(np.float64)
        peak_i=max(float(intensity.max()),1e-300)
        peak_a=max(float(amp.max()),1e-300)
        items['目标强度（峰值归一化）']={
            'array': np.clip(intensity/peak_i,0.0,1.0), 'kind':'norm01',
            'default_stem':'target_intensity'}
        items['目标振幅（峰值归一化）']={
            'array': np.clip(amp/peak_a,0.0,1.0), 'kind':'norm01',
            'default_stem':'target_amplitude'}
        items['目标复场相位（-π~π，可视化）']={
            'array': np.angle(field).astype(np.float64), 'kind':'phase_pm_pi',
            'default_stem':'target_phase_wrapped'}
        items['目标复场相位（0~2π，SLM映射）']={
            'array': np.remainder(np.angle(field),2*np.pi).astype(np.float64), 'kind':'phase_0_2pi',
            'default_stem':'target_phase_0_2pi'}
        items['SLM全息相位（0~2π）']={
            'array': self.result.hologram_phase.astype(np.float64), 'kind':'phase_0_2pi',
            'default_stem':'hologram_phase_0_2pi'}
        items['SLM 8位灰度图（加载推荐）']={
            'array': self.result.hologram_gray.copy(), 'kind':'u8',
            'default_stem':'SLM_8bit_gray'}
        if self.recon is not None:
            rec=self.recon.recovered
            i_rec=np.abs(rec)**2
            peak_r=max(float(i_rec.max()),1e-300)
            items['重建强度（峰值归一化）']={
                'array': np.clip(i_rec/peak_r,0.0,1.0), 'kind':'norm01',
                'default_stem':'reconstructed_intensity'}
            items['重建复场相位（-π~π，可视化）']={
                'array': np.angle(rec).astype(np.float64), 'kind':'phase_pm_pi',
                'default_stem':'reconstructed_phase_wrapped'}
        return items

    def _save_single_image(self, array, kind, path, fmt, encoding):
        path=Path(path)
        fmt=fmt.upper()
        encoding=str(encoding)
        arr=np.asarray(array)
        if kind=='u8':
            u8=arr.astype(np.uint8, copy=False)
            if fmt=='TIFF' and encoding=='float32':
                Image.fromarray(u8.astype(np.float32), mode='F').save(str(path), format='TIFF')
            elif fmt=='TIFF' and encoding=='16bit':
                Image.fromarray((u8.astype(np.uint16)*257), mode='I;16').save(str(path), format='TIFF')
            else:
                Image.fromarray(u8, mode='L').save(str(path), format=fmt)
            return
        if kind=='norm01':
            arr=np.nan_to_num(arr.astype(np.float64), nan=0.0, posinf=1.0, neginf=0.0)
            arr=np.clip(arr,0.0,1.0)
            if fmt=='TIFF' and encoding=='float32':
                Image.fromarray(arr.astype(np.float32), mode='F').save(str(path), format='TIFF')
            elif fmt=='TIFF' and encoding=='16bit':
                Image.fromarray(np.round(arr*65535.0).astype(np.uint16), mode='I;16').save(str(path), format='TIFF')
            else:
                Image.fromarray(np.round(arr*255.0).astype(np.uint8), mode='L').save(str(path), format=fmt)
            return
        if kind=='phase_pm_pi':
            arr=np.remainder(arr+np.pi,2*np.pi)/(2*np.pi)
            if fmt=='TIFF' and encoding=='float32':
                # Save wrapped phase in radians for quantitative use.
                rad=np.angle(np.exp(1j*np.asarray(array,dtype=np.float64)))
                Image.fromarray(rad.astype(np.float32), mode='F').save(str(path), format='TIFF')
            elif fmt=='TIFF' and encoding=='16bit':
                Image.fromarray(np.round(arr*65535.0).astype(np.uint16), mode='I;16').save(str(path), format='TIFF')
            else:
                Image.fromarray(phase_to_u8(arr*2*np.pi), mode='L').save(str(path), format=fmt)
            return
        if kind=='phase_0_2pi':
            cyc=np.remainder(arr,2*np.pi)/(2*np.pi)
            if fmt=='TIFF' and encoding=='float32':
                Image.fromarray(np.remainder(arr,2*np.pi).astype(np.float32), mode='F').save(str(path), format='TIFF')
            elif fmt=='TIFF' and encoding=='16bit':
                Image.fromarray(np.round(cyc*65535.0).astype(np.uint16), mode='I;16').save(str(path), format='TIFF')
            else:
                Image.fromarray(phase_to_u8(arr), mode='L').save(str(path), format=fmt)
            return
        raise ValueError(f'不支持的导出类型：{kind}')

    def export_single_dialog(self):
        if self.busy or self.result is None:
            messagebox.showinfo('提示','请先计算并等待完成')
            return
        try:
            if not self._results_current():
                messagebox.showwarning('参数未重新计算','请先重新计算，再导出单张图，避免导出旧参数对应的结果。')
                return
        except Exception as exc:
            messagebox.showerror('参数无效',str(exc));return
        items=self._single_export_items()
        if not items:
            messagebox.showinfo('提示','当前没有可导出的图像')
            return
        win=tk.Toplevel(self.root)
        win.title('导出单张图像')
        win.transient(self.root)
        win.grab_set()
        win.resizable(False, False)
        frm=ttk.Frame(win,padding=14)
        frm.pack(fill='both',expand=True)
        item_var=tk.StringVar(value=list(items.keys())[0])
        fmt_var=tk.StringVar(value='TIFF')
        enc_var=tk.StringVar(value='8bit')
        auto_var=tk.BooleanVar(value=True)
        name_var=tk.StringVar(value='')
        note_var=tk.StringVar(value='')

        def refresh_note(*_):
            item=items[item_var.get()]
            fmt=fmt_var.get().upper()
            if fmt in ('BMP','PNG'):
                opts=['8bit']
            else:
                opts=['8bit','16bit','float32'] if item['kind']!='u8' else ['8bit','16bit','float32']
            box_enc.configure(values=opts, state='readonly')
            if enc_var.get() not in opts:
                enc_var.set(opts[0])
            if auto_var.get():
                stem=(auto_filename(self.result.config)+'_'+item['default_stem'])
                if item['kind']=='u8' and fmt in ('BMP','PNG'):
                    stem=auto_filename(self.result.config)+'_SLM_8bit_gray'
                name_var.set(stem)
            meaning={
                'norm01':'这是光强/振幅数据，不是SLM可直接实现的复振幅全息图。BMP/PNG为8位灰度；TIFF可选16位/float32。',
                'phase_pm_pi':'相位图：BMP/PNG与8/16位TIFF按包裹相位映射到灰度；float32 TIFF直接保存弧度值（-π~π）。',
                'phase_0_2pi':'相位图：BMP/PNG与8/16位TIFF按0~2π周期映射到灰度；float32 TIFF直接保存弧度值（0~2π）。',
                'u8':'该图本身就是用于SLM加载的8位灰度图；BMP/PNG直接保存原始像素，TIFF可另存16位或float32。'}
            note_var.set(meaning[item['kind']])

        ttk.Label(frm,text='图像类型').grid(row=0,column=0,sticky='w',padx=4,pady=6)
        box_item=ttk.Combobox(frm,textvariable=item_var,values=list(items.keys()),state='readonly',width=34)
        box_item.grid(row=0,column=1,sticky='ew',padx=4,pady=6,columnspan=2)
        ttk.Label(frm,text='文件格式').grid(row=1,column=0,sticky='w',padx=4,pady=6)
        box_fmt=ttk.Combobox(frm,textvariable=fmt_var,values=['BMP','PNG','TIFF'],state='readonly',width=12)
        box_fmt.grid(row=1,column=1,sticky='w',padx=4,pady=6)
        ttk.Label(frm,text='数据编码').grid(row=1,column=2,sticky='w',padx=4,pady=6)
        box_enc=ttk.Combobox(frm,textvariable=enc_var,values=['8bit','16bit','float32'],state='readonly',width=14)
        box_enc.grid(row=1,column=3,sticky='w',padx=4,pady=6)
        ttk.Checkbutton(frm,text='自动文件名',variable=auto_var,command=refresh_note).grid(row=2,column=0,sticky='w',padx=4,pady=6)
        ttk.Label(frm,text='文件名（不含扩展名）').grid(row=3,column=0,sticky='w',padx=4,pady=6)
        ttk.Entry(frm,textvariable=name_var,width=34).grid(row=3,column=1,sticky='ew',padx=4,pady=6,columnspan=3)
        ttk.Label(frm,textvariable=note_var,style='Hint.TLabel',wraplength=560).grid(row=4,column=0,columnspan=4,sticky='w',padx=4,pady=(4,10))
        ttk.Label(frm,text='提示：若用于SLM加载，优先选择“SLM 8位灰度图”，并导出 BMP/PNG。',style='Hint.TLabel',wraplength=560).grid(row=5,column=0,columnspan=4,sticky='w',padx=4,pady=(0,10))
        btns=ttk.Frame(frm)
        btns.grid(row=6,column=0,columnspan=4,sticky='e')

        def do_export():
            item=items[item_var.get()]
            fmt=fmt_var.get().upper()
            ext={'BMP':'.bmp','PNG':'.png','TIFF':'.tiff'}[fmt]
            stem=name_var.get().strip() or item['default_stem']
            path=filedialog.asksaveasfilename(
                title='导出单张图像', defaultextension=ext,
                initialfile=stem+ext,
                filetypes=[(fmt, '*' + ext), ('All files','*.*')])
            if not path:
                return
            try:
                self._save_single_image(item['array'], item['kind'], path, fmt, enc_var.get())
                self.status.set('已导出单张图像：'+str(path))
                messagebox.showinfo('导出完成',f'已导出：{Path(path).name}')
                win.destroy()
            except Exception as e:
                messagebox.showerror('导出失败',str(e))

        ttk.Button(btns,text='导出',command=do_export).pack(side='right',padx=4)
        ttk.Button(btns,text='取消',command=win.destroy).pack(side='right',padx=4)
        frm.columnconfigure(1,weight=1)
        frm.columnconfigure(2,weight=0)
        frm.columnconfigure(3,weight=0)
        for v in (item_var, fmt_var):
            v.trace_add('write', refresh_note)
        refresh_note()

    def export(self):
        if self.busy or self.result is None:
            messagebox.showinfo('提示','请先计算并等待完成');return
        try:
            c=self._collect()
            if self._signature(c)!=self.result_signature:
                messagebox.showwarning('参数已修改','当前模式/参数与上次计算不一致，请重新计算再导出。')
                return
            sim_current=self._simulation_settings()
            if sim_current!=self.last_sim_settings:
                messagebox.showwarning('仿真参数已修改','请重新计算衍射重建，再导出对应的完整记录。')
                return
        except Exception as e:messagebox.showerror('参数错误',str(e));return
        folder=filedialog.askdirectory(title='请选择全息图文件夹')
        if not folder:return
        try:
            base=(auto_filename(self.result.config) if self.auto_name.get() else self.filename_var.get().strip())
            paths=export_result(self.result,folder,base_name=base,save_npy=True)
            safe=paths[0].name.replace('_SLM_8bit.bmp','')
            targetpath=Path(folder)/f'{safe}_target_preview.png'
            holopath=Path(folder)/f'{safe}_hologram_preview.png'
            reconpath=Path(folder)/f'{safe}_reconstruction_preview.png'
            self.fig_t.savefig(targetpath,dpi=125)
            self.fig_h.savefig(holopath,dpi=125)
            self.fig_r.savefig(reconpath,dpi=125)
            diag={'incident_waist_mm_assumed':self.result.config['hologram']['incident_waist_mm'],
                  'fourier_f_mm':self.last_sim_settings[0],
                  'filter_radius_mm':self.last_sim_settings[1],
                  'external_tilt_enabled_sim_only':self.last_sim_settings[2],
                  'external_tilt_deg_sim_only':self.last_sim_settings[3],
                  'fidelity_squared_ideal':self.recon.fidelity,
                  'relative_complex_error_ideal':self.recon.relative_error,
                  'filtered_fft_power_fraction_ideal':self.recon.efficiency,
                  'warnings':list(self.recon.warnings),
                  'description':'Ideal numerical Fourier-order filtering; NOT experimental mode purity or measured diffraction efficiency.'}
            json_path=Path(folder)/f'{safe}_reconstruction_diagnostics.json'
            json_path.write_text(json.dumps(diag,ensure_ascii=False,indent=2),encoding='utf-8')
            self.status.set(f'已导出 {paths[0].name} 及全部物理与重建数据')
            messagebox.showinfo('导出完成',f'8位BMP与PNG：{paths[0].name}\n已同时保存配置、复场、相位数组和3张预览图。')
        except Exception as e:messagebox.showerror('导出失败',str(e))

    def save_json(self):
        try:c=self._collect()
        except Exception as e:messagebox.showerror('参数错误',str(e));return
        path=filedialog.asksaveasfilename(defaultextension='.json',filetypes=[('JSON','*.json')])
        if path:
            c['_simulation']={'incident_waist_mm':float(self.incident_var.get()),
                              'aperture_radius_mm':float(self.aperture_var.get()),
                              'fourier_f_mm':float(self.fourier_f_var.get()),
                              'external_tilt_enabled':self.external_tilt_var.get(),
                              'external_tilt_deg':float(self.tilt_angle_var.get())}
            Path(path).write_text(json.dumps(c,ensure_ascii=False,indent=2),encoding='utf-8')
            self.status.set('配置已保存：'+str(path))

    def load_json(self):
        path=filedialog.askopenfilename(filetypes=[('JSON','*.json')])
        if not path:return
        try:
            obj=json.loads(Path(path).read_text(encoding='utf-8'))
            c=normalize_config(obj.get('configuration',obj))
            self._load_config_to_ui(c)
            sim=obj.get('_simulation',{})
            self.incident_var.set(str(c['hologram'].get('incident_waist_mm',sim.get('incident_waist_mm',2.5))))
            self.aperture_var.set(str(sim.get('aperture_radius_mm',0.7)))
            self.fourier_f_var.set(str(sim.get('fourier_f_mm',300)))
            self.external_tilt_var.set(bool(sim.get('external_tilt_enabled',True)))
            self.tilt_angle_var.set(str(sim.get('external_tilt_deg',-0.680)))
            self._mark_changed()
        except Exception as e:messagebox.showerror('载入失败',str(e))

    def reset(self):
        self._load_config_to_ui(DEFAULT_CONFIG)
        self.incident_var.set('2.5');self.threshold_var.set('0.03')
        self.aperture_var.set('0.70');self.fourier_f_var.set('300')
        self.external_tilt_var.set(True);self.tilt_angle_var.set('-0.680')
        self._mark_changed()


def main():
    root=tk.Tk();App(root);root.mainloop()

if __name__=='__main__':
    main()
