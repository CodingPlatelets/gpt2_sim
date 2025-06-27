# Trapezoid Pipeline DC Synthesis Guide

本目录包含使用Synopsys Design Compiler对Trapezoid Pipeline进行芯片面积测试的完整脚本和配置文件。

## 文件说明

### 核心脚本文件
- `synthesis_script.tcl` - 主要的DC综合脚本
- `file_list.tcl` - 设计文件列表和库配置
- `constraints.sdc` - 时序约束文件
- `run_synthesis.sh` - 自动化运行脚本

### 目录结构
```
dc_synthesis/
├── README.md              # 本说明文件
├── synthesis_script.tcl   # 主综合脚本
├── file_list.tcl          # 文件列表和库设置
├── constraints.sdc        # 时序约束
├── run_synthesis.sh       # 运行脚本
├── reports/               # 综合报告输出目录
├── outputs/               # 网表输出目录
└── logs/                  # 日志文件目录
```

## 使用前准备

### 1. 环境配置
确保您的系统已安装Synopsys Design Compiler，并且具有14nm工艺库。

### 2. 修改配置文件

#### 2.1 更新库文件路径
编辑 `file_list.tcl`，修改以下部分以匹配您的14nm库文件：

```tcl
# 将这些库文件名替换为您实际的14nm库文件
set target_library_list [list \
    "your_14nm_library.db" \
    "your_14nm_memory_library.db" \
]

set link_library [list "*" $target_library_list]
set symbol_library [list "your_14nm_library.sdb"]
```

#### 2.2 更新环境变量
编辑 `run_synthesis.sh`，修改以下路径：

```bash
# 修改为您的Synopsys安装路径
export SYNOPSYS_ROOT="/your/synopsys/installation/path"

# 修改为您的14nm库路径
export LIBRARY_PATH="/your/14nm/library/path"

# 修改为您的许可证服务器
export LM_LICENSE_FILE="port@your-license-server"
```

### 3. 设计参数配置

当前配置的设计参数：
- **M = 8** (矩阵行数)
- **K = 128** (矩阵中间维度)
- **N = 128** (矩阵列数)
- **PE_ROWS = 8** (PE阵列行数)
- **PE_COLS = 16** (PE阵列列数，总共128个PE)
- **时钟频率 = 1GHz** (1ns周期)

如需修改这些参数，请编辑相应的Verilog文件。

## 运行综合

### 方法1: 使用自动化脚本（推荐）

```bash
cd trapezoid_rtl/dc_synthesis
chmod +x run_synthesis.sh
./run_synthesis.sh
```

### 方法2: 手动运行DC

```bash
cd trapezoid_rtl/dc_synthesis
dc_shell -f synthesis_script.tcl
```

## 结果分析

综合完成后，检查以下文件获取详细结果：

### 面积报告
- `reports/area_hierarchy.rpt` - 层次化面积报告
- `reports/area_designware.rpt` - DesignWare组件面积
- `reports/cell_usage.rpt` - 单元使用统计

### 时序报告
- `reports/timing_max.rpt` - 最大延迟时序报告
- `reports/timing_min.rpt` - 最小延迟时序报告
- `reports/constraints.rpt` - 约束违反报告

### 功耗报告
- `reports/power.rpt` - 功耗分析报告

### 综合质量报告
- `reports/qor_summary.rpt` - 综合质量总结

## 关键指标解读

### 1. 芯片面积
在面积报告中查找：
```
Total cell area: XXXXX.XX
```
这是您的设计在14nm工艺下的总面积（单位通常为μm²）。

### 2. 时序性能
在时序报告中查找：
```
slack (MET): X.XX ns
```
- 正值表示满足时序要求
- 负值表示时序违反

### 3. 功耗估算
在功耗报告中查找总功耗数据。

## 常见问题

### Q: 综合失败，提示库文件未找到
**A:** 检查 `file_list.tcl` 中的库文件路径和名称是否正确。

### Q: 时序违反严重
**A:** 考虑：
1. 降低时钟频率（修改 `constraints.sdc` 中的时钟周期）
2. 优化关键路径
3. 使用更快的库单元

### Q: 面积过大
**A:** 考虑：
1. 减少PE数量
2. 优化数据位宽
3. 使用面积优化的综合选项

## 面积优化建议

1. **减少流水线深度**: 当前BF16运算单元使用5级流水线，可考虑减少到3-4级
2. **优化PE数量**: 根据实际需求调整PE阵列大小
3. **使用低功耗库**: 如果时序满足，可选择面积优化的库单元
4. **门级优化**: 启用更积极的布尔优化和结构优化

## 进阶配置

### 多目标优化
修改 `synthesis_script.tcl` 中的编译选项：

```tcl
# 面积优化模式
compile_ultra -gate_clock -retime -area_high_effort_script

# 功耗优化模式  
compile_ultra -gate_clock -retime -low_power

# 时序优化模式
compile_ultra -gate_clock -retime -timing_high_effort_script
```

### 层次化综合
对于大型设计，可启用层次化综合以提高运行时间。

---

**注意**: 本脚本基于标准的14nm工艺库配置。实际使用时请根据您的具体工艺库文件进行相应调整。 