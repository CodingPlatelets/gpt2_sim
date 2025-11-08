# Makefile for Trapezoid Pipeline
# 梯形流水线矩阵乘法器编译和测试管理

# 工具链
IVERILOG = iverilog
VVP = vvp
GTKWAVE = gtkwave

# 编译选项 - 启用SystemVerilog支持
VERILOG_FLAGS = -g2009

# 目录设置
SRC_DIR = src
TB_DIR = testbench
BUILD_DIR = build

# 源文件分类
BF16_SRCS = $(SRC_DIR)/bf16_add_pipeline.v \
            $(SRC_DIR)/bf16_multiply_pipeline.v

COMPUTE_SRCS = $(SRC_DIR)/compute_units.v

TREE_SRCS = $(SRC_DIR)/add_tree.v

TRAPEZOID_SRCS = $(SRC_DIR)/trapezoid_pipeline.v \
                $(SRC_DIR)/trapezoid_pe_array.v

ALL_SRCS = $(BF16_SRCS) $(COMPUTE_SRCS) $(TREE_SRCS) $(TRAPEZOID_SRCS)

# 测试台文件
BF16_TESTBENCHES = $(TB_DIR)/bf16_add_testbench.v \
                   $(TB_DIR)/bf16_multiply_testbench.v

COMPUTE_TESTBENCHES = $(TB_DIR)/compute_units_testbench.v

TREE_TESTBENCHES = $(TB_DIR)/add_tree_testbench.v

MAIN_TESTBENCHES = $(TB_DIR)/trapezoid_pipeline_testbench.v

ALL_TESTBENCHES = $(BF16_TESTBENCHES) $(COMPUTE_TESTBENCHES) $(TREE_TESTBENCHES) $(MAIN_TESTBENCHES)

# 编译目标
BASIC_TARGETS = bf16_add_test bf16_multiply_test
COMPUTE_TARGETS = compute_units_test
TREE_TARGETS = add_tree_test
MAIN_TARGETS = trapezoid_pipeline_test

ALL_TARGETS = $(BASIC_TARGETS) $(COMPUTE_TARGETS) $(TREE_TARGETS) $(MAIN_TARGETS)

# 默认目标 - 编译所有模块
.PHONY: all
all: $(ALL_TARGETS)

# 创建构建目录
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# BF16 模块测试
bf16_add_test: $(BUILD_DIR) $(BF16_SRCS) $(TB_DIR)/bf16_add_testbench.v
	@echo "编译 BF16 加法器测试..."
	$(IVERILOG) $(VERILOG_FLAGS) -o $(BUILD_DIR)/bf16_add_test \
		-I$(SRC_DIR) \
		$(TB_DIR)/bf16_add_testbench.v \
		$(SRC_DIR)/bf16_add_pipeline.v
	@echo "编译完成: $(BUILD_DIR)/bf16_add_test"

bf16_multiply_test: $(BUILD_DIR) $(BF16_SRCS) $(TB_DIR)/bf16_multiply_testbench.v
	@echo "编译 BF16 乘法器测试..."
	$(IVERILOG) $(VERILOG_FLAGS) -o $(BUILD_DIR)/bf16_multiply_test \
		-I$(SRC_DIR) \
		$(TB_DIR)/bf16_multiply_testbench.v \
		$(SRC_DIR)/bf16_multiply_pipeline.v
	@echo "编译完成: $(BUILD_DIR)/bf16_multiply_test"

# 计算单元测试
compute_units_test: $(BUILD_DIR) $(BF16_SRCS) $(COMPUTE_SRCS) $(TB_DIR)/compute_units_testbench.v
	@echo "编译计算单元测试..."
	$(IVERILOG) $(VERILOG_FLAGS) -o $(BUILD_DIR)/compute_units_test \
		-I$(SRC_DIR) \
		$(TB_DIR)/compute_units_testbench.v \
		$(COMPUTE_SRCS) $(BF16_SRCS)
	@echo "编译完成: $(BUILD_DIR)/compute_units_test"

# 加法树测试
add_tree_test: $(BUILD_DIR) $(BF16_SRCS) $(COMPUTE_SRCS) $(TREE_SRCS) $(TB_DIR)/add_tree_testbench.v
	@echo "编译加法树测试..."
	$(IVERILOG) $(VERILOG_FLAGS) -o $(BUILD_DIR)/add_tree_test \
		-I$(SRC_DIR) \
		$(TB_DIR)/add_tree_testbench.v \
		$(TREE_SRCS) $(COMPUTE_SRCS) $(BF16_SRCS)
	@echo "编译完成: $(BUILD_DIR)/add_tree_test"

# 完整流水线测试
trapezoid_pipeline_test: $(BUILD_DIR) $(ALL_SRCS) $(TB_DIR)/trapezoid_pipeline_testbench.v
	@echo "编译梯形流水线测试..."
	$(IVERILOG) $(VERILOG_FLAGS) -o $(BUILD_DIR)/trapezoid_pipeline_test \
		-I$(SRC_DIR) \
		$(TB_DIR)/trapezoid_pipeline_testbench.v \
		$(ALL_SRCS)
	@echo "编译完成: $(BUILD_DIR)/trapezoid_pipeline_test"

# 运行测试 - 基础模块
.PHONY: test_bf16_add
test_bf16_add: bf16_add_test
	@echo "运行 BF16 加法器测试..."
	cd $(BUILD_DIR) && $(VVP) bf16_add_test
	@echo "BF16 加法器测试完成"

.PHONY: test_bf16_multiply
test_bf16_multiply: bf16_multiply_test
	@echo "运行 BF16 乘法器测试..."
	cd $(BUILD_DIR) && $(VVP) bf16_multiply_test
	@echo "BF16 乘法器测试完成"

.PHONY: test_compute_units
test_compute_units: compute_units_test
	@echo "运行计算单元测试..."
	cd $(BUILD_DIR) && $(VVP) compute_units_test
	@echo "计算单元测试完成"

.PHONY: test_add_tree
test_add_tree: add_tree_test
	@echo "运行加法树测试..."
	cd $(BUILD_DIR) && $(VVP) add_tree_test
	@echo "加法树测试完成"

.PHONY: test_trapezoid
test_trapezoid: trapezoid_pipeline_test
	@echo "运行梯形流水线测试..."
	cd $(BUILD_DIR) && $(VVP) trapezoid_pipeline_test
	@echo "梯形流水线测试完成"

# 运行所有测试
.PHONY: test_all
test_all: test_bf16_add test_bf16_multiply test_compute_units test_add_tree test_trapezoid
	@echo "所有测试完成"

# 快速测试 - 只测试基础模块
.PHONY: test_basic
test_basic: test_bf16_add test_bf16_multiply test_compute_units
	@echo "基础模块测试完成"

# 语法检查
.PHONY: syntax_check
syntax_check: $(BUILD_DIR)
	@echo "检查语法..."
	@echo "检查 BF16 模块..."
	@$(IVERILOG) $(VERILOG_FLAGS) -t null $(BF16_SRCS) || exit 1
	@echo "检查计算单元..."
	@$(IVERILOG) $(VERILOG_FLAGS) -t null -I$(SRC_DIR) $(COMPUTE_SRCS) $(BF16_SRCS) || exit 1
	@echo "检查加法树..."
	@$(IVERILOG) $(VERILOG_FLAGS) -t null -I$(SRC_DIR) $(TREE_SRCS) $(COMPUTE_SRCS) $(BF16_SRCS) || exit 1
	@echo "检查完整系统..."
	@$(IVERILOG) $(VERILOG_FLAGS) -t null -I$(SRC_DIR) $(ALL_SRCS) || exit 1
	@echo "语法检查通过"

# 生成波形文件的测试
.PHONY: test_wave_%
test_wave_%: %
	@echo "运行 $< 测试（生成波形）..."
	cd $(BUILD_DIR) && $(VVP) $<
	@if [ -f $(BUILD_DIR)/$<.vcd ]; then \
		echo "打开波形查看器..."; \
		$(GTKWAVE) $(BUILD_DIR)/$<.vcd; \
	else \
		echo "未找到VCD波形文件"; \
	fi

# 清理
.PHONY: clean
clean:
	@echo "清理构建文件..."
	rm -rf $(BUILD_DIR)
	@echo "清理完成"

# 显示帮助
.PHONY: help
help:
	@echo "梯形流水线 Makefile 使用说明"
	@echo "=================================="
	@echo "编译目标："
	@echo "  all                    - 编译所有模块"
	@echo "  bf16_add_test         - 编译 BF16 加法器测试"
	@echo "  bf16_multiply_test    - 编译 BF16 乘法器测试"
	@echo "  compute_units_test    - 编译计算单元测试"
	@echo "  add_tree_test         - 编译加法树测试"
	@echo "  trapezoid_pipeline_test - 编译完整流水线测试"
	@echo ""
	@echo "测试目标："
	@echo "  test_basic            - 运行基础模块测试"
	@echo "  test_all              - 运行所有测试"
	@echo "  test_bf16_add         - 运行 BF16 加法器测试"
	@echo "  test_bf16_multiply    - 运行 BF16 乘法器测试"
	@echo "  test_compute_units    - 运行计算单元测试"
	@echo "  test_add_tree         - 运行加法树测试"
	@echo "  test_trapezoid        - 运行完整流水线测试"
	@echo ""
	@echo "波形测试："
	@echo "  test_wave_TARGET      - 运行测试并查看波形"
	@echo ""
	@echo "其他："
	@echo "  syntax_check          - 检查所有模块语法"
	@echo "  clean                 - 清理构建文件"
	@echo "  help                  - 显示此帮助信息"
	@echo ""
	@echo "示例："
	@echo "  make test_basic           # 测试基础模块"
	@echo "  make test_trapezoid       # 测试完整流水线"
	@echo "  make test_wave_bf16_add_test  # 测试BF16加法器并查看波形"
	@echo "  make syntax_check         # 检查语法"

# 详细信息
.PHONY: info
info:
	@echo "项目信息"
	@echo "========"
	@echo "项目名称: 梯形流水线矩阵乘法器 (Trapezoid Pipeline Matrix Multiplier)"
	@echo "描述: 基于BF16的稀疏矩阵乘法加速器"
	@echo ""
	@echo "架构特性:"
	@echo "- 5级流水线结构"
	@echo "- 多PE并行处理"
	@echo "- 稀疏矩阵支持 (CSR格式)"
	@echo "- BF16数值格式"
	@echo "- 树形加法结构"
	@echo "- MFIU索引生成单元"
	@echo ""
	@echo "模块结构:"
	@echo "  BF16模块: $(BF16_SRCS)"
	@echo "  计算单元: $(COMPUTE_SRCS)"
	@echo "  加法树:   $(TREE_SRCS)"
	@echo "  主流水线: $(TRAPEZOID_SRCS)"
	@echo ""
	@echo "测试文件:"
	@for file in $(ALL_TESTBENCHES); do \
		echo "  $$file"; \
	done

# 静态依赖关系
$(COMPUTE_TARGETS): $(BF16_SRCS)
$(TREE_TARGETS): $(BF16_SRCS) $(COMPUTE_SRCS)
$(MAIN_TARGETS): $(ALL_SRCS) 