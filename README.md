# SLM HG / LG / IG GUI V5

结构光复振幅全息 GUI 的原始 V5 源码快照，由用户提供的五个 Python 文件导入。

## 启动

在支持这些依赖的 Python 环境中安装依赖：

```bash
python -m pip install -r requirements.txt
python SLM_HG_LG_IG_GUI.py
```

GUI 使用 Tkinter，需要可用的图形桌面及 Tk 支持。各 Python 文件须保持在同一目录。
当前未锁定依赖版本，尚未验证 Windows 7 兼容性。

## 文件

| 文件 | 用途 |
| --- | --- |
| SLM_HG_LG_IG_GUI.py | V5 图形界面入口 |
| physics_hg_lg_ig_refined.py | HG / LG / IG 模式与物理计算 |
| slm_hologram_core.py | 配置、全息编码和导出 |
| diffraction_simulation.py | 衍射重建仿真 |
| research_analysis.py | 光阑扫描等分析工具 |

## 导入说明

- 上传附件原名带有 `(1)`，入库时移除此下载后缀，以匹配模块导入名。
- 五个 Python 文件源码未作算法修改。
- 新增 README、依赖清单与 Git 忽略规则，方便后续开发。
- 已完成语法解析检查；尚未完成 GUI 运行、数值正确性及 SLM 实验验证。
- LG / HG / IG、double phase 和 double phase compensate 的详细审查与修订应作为后续独立变更，保留此原始基线。
