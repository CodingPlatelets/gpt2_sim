# GPT-2 硬件模拟器

一个用于GPT-2硬件加速的模拟器，支持Trapezoid流水线、BF16运算和内存管理。

## 项目结构

```
gpt2_sim/
├── gpt2_sim/                    # 主要源代码
│   ├── bf16_module/            # BF16运算模块
│   ├── trapezoid_module/       # Trapezoid流水线模块 
│   ├── hbm/                    # 高带宽内存模块
│   ├── utils/                  # 工具函数
│   └── *.py                    # 其他核心模块
├── main.py                     # 主入口文件
├── pyproject.toml              # 项目配置
└── README.md                   # 本文档
```

## 使用方法

### 环境设置
使用uv管理项目依赖：
```bash
uv sync
```

### 运行测试

**主程序:**
```bash
uv run main.py
```

**Trapezoid模块测试:**
```bash
uv run gpt2_sim/trapezoid_module/test_trapezoid.py
```

**计算模块测试:**
```bash  
uv run gpt2_sim/computeTest.py
```

**内存模块测试:**
```bash
uv run gpt2_sim/memTest.py
```

**MFIU模块测试:**
```bash
uv run gpt2_sim/trapezoid_module/test/test_mfiu_quick.py
```

## 主要特性

- ✅ **标准化项目结构**: 符合Python包管理规范
- ✅ **统一构建系统**: 使用uv管理依赖和运行
- ✅ **模块化设计**: 清晰的模块分离
- ✅ **直接运行**: 每个测试文件都可以独立运行
- ✅ **GPU加速**: 支持CUDA加速计算

## 依赖

- Python >= 3.13
- PyTorch >= 2.0.0
- NumPy >= 1.20.0  
- SciPy >= 1.15.3
- tqdm >= 4.65.0

所有依赖通过uv自动管理。
