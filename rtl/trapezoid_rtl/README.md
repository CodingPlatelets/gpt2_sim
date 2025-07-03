# Trapezoid Pipeline RTL Project

## 🎯 项目概述

本项目是梯形流水线矩阵乘法器的RTL实现，专门用于稀疏矩阵乘法的硬件加速。该设计基于Python模拟器转换而来，采用5级流水线架构，支持BF16数值格式。

## 🏗️ 架构特性

- **5级流水线结构**: 高效的数据处理流水线
- **多PE并行处理**: 支持多个处理单元并行计算
- **稀疏矩阵支持**: 原生支持CSR格式稀疏矩阵
- **BF16数值格式**: 优化的16位浮点数格式
- **树形加法结构**: 高效的并行累加
- **MFIU索引生成**: 智能索引生成单元

## 📁 项目结构

```
rtl/trapezoid_rtl/
├── src/                          # 源代码目录
│   ├── bf16_add_pipeline.v       # BF16加法流水线
│   ├── bf16_multiply_pipeline.v  # BF16乘法流水线
│   ├── compute_units.v           # 计算单元(乘法器、加法器、MAC)
│   ├── add_tree.v               # 树形加法结构
│   ├── trapezoid_pipeline.v     # 主流水线模块
│   └── trapezoid_pe_array.v     # PE阵列模块
├── testbench/                    # 测试台目录
│   ├── bf16_add_testbench.v      # BF16加法器测试台
│   ├── bf16_multiply_testbench.v # BF16乘法器测试台
│   ├── compute_units_testbench.v # 计算单元测试台
│   ├── add_tree_testbench.v      # 加法树测试台
│   └── trapezoid_pipeline_testbench.v # 完整流水线测试台
├── build/                        # 构建输出目录 (自动生成)
├── dc_synthesis/                 # DC综合脚本
├── Makefile                      # 构建管理文件
└── README.md                     # 本文件
```

## 🚀 快速开始

### 环境要求

- **Icarus Verilog**: 用于仿真和编译
- **GTKWave**: 用于波形查看
- **Make**: 用于构建管理

### 安装依赖 (Ubuntu/Debian)

```bash
sudo apt-get update
sudo apt-get install iverilog gtkwave make
```

### 基本使用

1. **查看帮助信息**
   ```bash
   make help
   ```

2. **语法检查**
   ```bash
   make syntax_check
   ```

3. **编译基础模块**
   ```bash
   make all
   ```

4. **运行基础模块测试**
   ```bash
   make test_basic
   ```

5. **运行完整测试**
   ```bash
   make test_all
   ```

## 🧪 测试指南

### 逐步测试建议

按照以下顺序进行测试，确保每个模块都能正常工作：

1. **BF16基础模块测试**
   ```bash
   make test_bf16_add      # 测试BF16加法器
   make test_bf16_multiply # 测试BF16乘法器
   ```

2. **计算单元测试**
   ```bash
   make test_compute_units # 测试乘法、加法、MAC单元
   ```

3. **加法树测试**
   ```bash
   make test_add_tree      # 测试树形并行加法
   ```

4. **完整流水线测试**
   ```bash
   make test_trapezoid     # 测试完整的梯形流水线
   ```

### 波形查看

要查看测试过程中的波形，可以使用：
```bash
make test_wave_bf16_add_test      # 查看BF16加法器波形
make test_wave_compute_units_test # 查看计算单元波形
make test_wave_trapezoid_pipeline_test # 查看完整流水线波形
```

## 📊 模块说明

### BF16模块
- **bf16_add_pipeline.v**: BF16格式的加法流水线，支持5级流水线
- **bf16_multiply_pipeline.v**: BF16格式的乘法流水线，支持5级流水线

### 计算单元
- **multiply_unit**: 乘法单元，包含索引队列管理
- **add_unit**: 加法单元，包含索引队列管理  
- **mac_unit**: 乘法累加单元，组合multiply和add单元

### 加法树
- **add_tree.v**: 树形结构的并行加法器，支持多PE输入的高效累加

### 主流水线
- **trapezoid_pipeline.v**: 完整的梯形流水线，集成所有子模块
- **trapezoid_pe_array.v**: PE阵列管理模块

## 🛠️ 开发指南

### 添加新的测试

1. 在`testbench/`目录下创建新的测试台文件
2. 在`Makefile`中添加相应的编译和测试规则
3. 运行`make syntax_check`确保语法正确
4. 运行新的测试

### 修改模块

1. 修改`src/`目录下的源文件
2. 运行`make syntax_check`检查语法
3. 运行相关测试确保功能正确
4. 如有需要，更新对应的测试台

### 调试技巧

1. 使用`$display`和`$monitor`添加调试输出
2. 使用`$dumpfile`和`$dumpvars`生成波形文件
3. 利用GTKWave查看信号波形
4. 检查流水线的各个阶段状态

## 🔧 Makefile目标详解

### 编译目标
- `all`: 编译所有模块
- `bf16_add_test`: 编译BF16加法器测试
- `bf16_multiply_test`: 编译BF16乘法器测试
- `compute_units_test`: 编译计算单元测试
- `add_tree_test`: 编译加法树测试
- `trapezoid_pipeline_test`: 编译完整流水线测试

### 测试目标
- `test_basic`: 运行基础模块测试
- `test_all`: 运行所有测试
- `test_bf16_add`: 运行BF16加法器测试
- `test_bf16_multiply`: 运行BF16乘法器测试
- `test_compute_units`: 运行计算单元测试
- `test_add_tree`: 运行加法树测试
- `test_trapezoid`: 运行完整流水线测试

### 其他目标
- `syntax_check`: 检查所有模块语法
- `clean`: 清理构建文件
- `help`: 显示帮助信息
- `info`: 显示项目详细信息

## 📝 测试结果说明

每个测试台都会输出详细的测试结果，包括：
- 测试用例名称和输入参数
- 期望结果和实际结果
- 通过/失败状态
- 测试总结统计

## 🐛 常见问题

### 编译错误
1. 检查Icarus Verilog是否正确安装
2. 确保所有源文件路径正确
3. 运行`make syntax_check`检查语法错误

### 测试失败
1. 查看测试输出的详细错误信息
2. 使用波形查看器检查信号时序
3. 检查测试台的输入数据和期望结果

### 性能问题
1. 检查流水线级数设置
2. 确认时钟频率设置合理
3. 优化关键路径时序

## 🔄 与Python模拟器的对应关系

本RTL设计基于以下Python模拟器模块转换：
- `add_tree_sim.py` → `add_tree.v`
- `advance_add_sim.py` → `compute_units.v`中的add_unit
- `compute_sim.py` → `compute_units.v`
- `mfiu_sim.py` → (需要添加MFIU模块)
- `shift_sim.py` → (需要添加Shift模块)
- `trapezoid_sim.py` → `trapezoid_pipeline.v`

## 📈 后续改进计划

1. **添加MFIU模块**: 实现索引生成功能
2. **添加Shift模块**: 实现移位操作
3. **性能优化**: 优化关键路径时序
4. **综合支持**: 完善DC综合脚本
5. **更多测试**: 添加更全面的测试用例

---

**从Python模拟器到硬件实现 - 加速计算的强大力量！** 🚀 