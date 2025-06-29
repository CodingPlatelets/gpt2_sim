# Makefile for Vector Matrix Row Product HBM Multi-Batch
# 向量矩阵乘法器编译和测试管理

# 工具链
IVERILOG = iverilog
VVP = vvp
GTKWAVE = gtkwave

# 目录设置
SRC_DIR = src
TB_DIR = testbench
BUILD_DIR = build

# 源文件
BF16_SRCS = $(SRC_DIR)/bf16_add_pipeline.v \
            $(SRC_DIR)/bf16_multiply_pipeline.v \
            $(SRC_DIR)/fp32_to_bf16_pipeline.v

CORE_SRCS = $(SRC_DIR)/processing_element.v \
            $(SRC_DIR)/processing_element_row.v \
            $(SRC_DIR)/hbm_controller.v \
            $(SRC_DIR)/vector_matrix_row_product_hbm_multi_batch.v

ALL_SRCS = $(BF16_SRCS) $(CORE_SRCS)

# 测试台文件
TESTBENCHES = $(TB_DIR)/vector_matrix_testbench.v

# 编译目标
TARGETS = vector_matrix_test

# 默认目标
.PHONY: all
all: $(TARGETS)

# 创建构建目录
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# 编译向量矩阵测试
vector_matrix_test: $(BUILD_DIR) $(ALL_SRCS) $(TB_DIR)/vector_matrix_testbench.v
	@echo "编译向量矩阵乘法器测试..."
	@echo "包含的源文件: $(ALL_SRCS)"
	$(IVERILOG) -o $(BUILD_DIR)/vector_matrix_test \
		-I$(SRC_DIR) \
		$(TB_DIR)/vector_matrix_testbench.v \
		$(ALL_SRCS)
	@echo "编译完成: $(BUILD_DIR)/vector_matrix_test"

# 运行测试
.PHONY: test
test: vector_matrix_test
	@echo "运行向量矩阵测试..."
	cd $(BUILD_DIR) && $(VVP) vector_matrix_test
	@echo "测试完成"

# 运行测试并查看波形
.PHONY: test_wave
test_wave: vector_matrix_test
	@echo "运行向量矩阵测试（生成波形）..."
	cd $(BUILD_DIR) && $(VVP) vector_matrix_test
	@if [ -f $(BUILD_DIR)/vector_matrix_test.vcd ]; then \
		echo "打开波形查看器..."; \
		$(GTKWAVE) $(BUILD_DIR)/vector_matrix_test.vcd; \
	else \
		echo "未找到VCD波形文件"; \
	fi

# 语法检查
.PHONY: syntax_check
syntax_check: $(BUILD_DIR)
	@echo "检查语法..."
	@echo "检查基础BF16模块..."
	@$(IVERILOG) -t null $(BF16_SRCS) || exit 1
	@echo "检查完整系统..."
	@$(IVERILOG) -t null -I$(SRC_DIR) $(ALL_SRCS) || exit 1
	@echo "语法检查通过"

# 单独测试各个模块
.PHONY: test_pe
test_pe: $(BUILD_DIR) $(SRC_DIR)/processing_element.v $(BF16_SRCS)
	@echo "测试处理元素模块..."
	$(IVERILOG) -o $(BUILD_DIR)/test_pe \
		-I$(SRC_DIR) \
		$(SRC_DIR)/processing_element.v $(BF16_SRCS)
	@echo "PE模块编译成功"

.PHONY: test_perow
test_perow: $(BUILD_DIR) $(SRC_DIR)/processing_element_row.v $(SRC_DIR)/processing_element.v $(BF16_SRCS)
	@echo "测试处理元素行模块..."
	$(IVERILOG) -o $(BUILD_DIR)/test_perow \
		-I$(SRC_DIR) \
		$(SRC_DIR)/processing_element_row.v \
		$(SRC_DIR)/processing_element.v \
		$(BF16_SRCS)
	@echo "PERow模块编译成功"

.PHONY: test_hbm
test_hbm: $(BUILD_DIR) $(SRC_DIR)/hbm_controller.v
	@echo "测试HBM控制器模块..."
	$(IVERILOG) -o $(BUILD_DIR)/test_hbm \
		-I$(SRC_DIR) \
		$(SRC_DIR)/hbm_controller.v
	@echo "HBM控制器模块编译成功"

# 清理
.PHONY: clean
clean:
	@echo "清理构建文件..."
	rm -rf $(BUILD_DIR)
	@echo "清理完成"

# 显示帮助
.PHONY: help
help:
	@echo "向量矩阵乘法器 Makefile 使用说明"
	@echo "=================================="
	@echo "可用目标："
	@echo "  all           - 编译所有模块"
	@echo "  test          - 运行完整测试"
	@echo "  test_wave     - 运行测试并查看波形"
	@echo "  syntax_check  - 检查所有模块语法"
	@echo "  test_pe       - 单独测试PE模块"
	@echo "  test_perow    - 单独测试PERow模块"
	@echo "  test_hbm      - 单独测试HBM控制器"
	@echo "  clean         - 清理构建文件"
	@echo "  help          - 显示此帮助信息"
	@echo ""
	@echo "示例："
	@echo "  make test           # 运行完整测试"
	@echo "  make syntax_check   # 检查语法"
	@echo "  make clean          # 清理文件"

# 详细信息
.PHONY: info
info:
	@echo "项目信息"
	@echo "========"
	@echo "项目名称: 向量矩阵乘法器 (Vector Matrix Row Product HBM Multi-Batch)"
	@echo "描述: 基于BF16的并行向量矩阵乘法加速器"
	@echo ""
	@echo "架构特性:"
	@echo "- 支持多批次处理"
	@echo "- 稀疏矩阵支持 (CSR格式)"
	@echo "- HBM内存块化处理"
	@echo "- BF16数值格式"
	@echo "- 流水线处理单元"
	@echo ""
	@echo "源文件:"
	@for file in $(ALL_SRCS); do \
		echo "  $$file"; \
	done
	@echo ""
	@echo "测试文件:"
	@for file in $(TESTBENCHES); do \
		echo "  $$file"; \
	done 