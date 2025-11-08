# MFIU RTL Makefile - 简化版
# 编译和测试MFIU Pipeline Verilog实现

# 编译器设置
IVERILOG = iverilog
VVP = vvp

# 文件设置
TESTBENCH = mfiu_testbench.v
TOP_MODULE = mfiu_testbench
SOURCES = mfiu_pipeline.v $(TESTBENCH)
OUTPUT = mfiu_test

# 默认目标
all: compile run

# 编译
compile:
	@echo "🔧 编译MFIU Pipeline..."
	$(IVERILOG) -o $(OUTPUT) $(SOURCES)
	@echo "✅ 编译完成: $(OUTPUT)"

# 运行仿真
run: compile
	@echo "🚀 运行MFIU Pipeline测试..."
	$(VVP) $(OUTPUT)
	@echo "📊 测试完成"

# 快速测试（过滤输出）
test: compile
	@echo "⚡ 快速测试MFIU Pipeline..."
	$(VVP) $(OUTPUT) | grep -E "(✅|❌|🔧|测试|周期)"

# 语法检查
syntax:
	@echo "🔍 检查Verilog语法..."
	$(IVERILOG) -t null $(SOURCES)
	@echo "✅ 语法检查通过"

# 清理
clean:
	@echo "🧹 清理生成文件..."
	rm -f $(OUTPUT)
	@echo "✅ 清理完成"

# 帮助
help:
	@echo "MFIU RTL Makefile帮助"
	@echo "======================"
	@echo "make all     - 编译并运行测试"
	@echo "make compile - 仅编译"
	@echo "make run     - 编译并运行"
	@echo "make test    - 快速测试（过滤输出）"
	@echo "make syntax  - 语法检查"
	@echo "make clean   - 清理文件"
	@echo "make help    - 显示此帮助"

.PHONY: all compile run test syntax clean help