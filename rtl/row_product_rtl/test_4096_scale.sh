#!/bin/bash

echo "🚀 4096维GPT2向量矩阵乘法测试"
echo "================================================"
echo "配置: 32 PERows × 128 PEs = 4096 并行单元"
echo "向量规模: 4096维"
echo "矩阵规模: 4096×4096 (90%稀疏度)"
echo "预计性能: 数千MAC运算/周期"
echo "================================================"

# 创建build目录
mkdir -p build

echo "📦 编译4096维测试..."
echo "注意：编译可能需要几分钟，请耐心等待..."

iverilog -o build/test_4096_scale -I src \
  testbench/vector_matrix_testbench_4096.v \
  src/vector_matrix_row_product_hbm_multi_batch.v \
  src/fp32_to_bf16_pipeline.v \
  src/hbm_controller_fixed.v \
  src/processing_element_row.v \
  src/processing_element.v \
  src/bf16_add_pipeline.v \
  src/bf16_multiply_pipeline.v

if [ $? -eq 0 ]; then
    echo "✅ 4096维测试编译成功！"
    echo ""
    echo "🚀 开始运行4096维真实规模测试..."
    echo ""
    echo "⚠️  重要提醒："
    echo "   - 此测试将加载4096维向量和稀疏矩阵"
    echo "   - 预计运行时间：数分钟到数十分钟"
    echo "   - 内存使用：可能需要几GB内存"
    echo "   - 随时可用 Ctrl+C 中断测试"
    echo ""
    echo "📊 性能指标："
    echo "   - 理论峰值: 4096 MAC/周期"
    echo "   - 目标性能: 100+ GOPS @ 100MHz"
    echo "   - 稀疏度优化: 90%稀疏矩阵"
    echo ""
    read -p "按Enter键开始测试，或Ctrl+C取消: "
    
    echo "🏃 开始运行..."
    cd build && ./test_4096_scale
else
    echo "❌ 编译失败！"
    echo "可能原因："
    echo "  1. 内存不足 - 4096维需要大量内存"
    echo "  2. 编译器限制 - 某些工具对大规模设计有限制"
    echo "  3. 参数错误 - 检查地址位宽设置"
    echo ""
    echo "建议："
    echo "  - 先运行小规模测试验证功能"
    echo "  - 增加系统内存或使用更强大的编译工具"
    echo "  - 逐步增加规模 (64维 -> 256维 -> 1024维 -> 4096维)"
fi 