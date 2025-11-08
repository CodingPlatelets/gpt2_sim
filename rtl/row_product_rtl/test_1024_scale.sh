#!/bin/bash

echo "🔧 1024维中等规模测试 - 推荐测试"
echo "================================================"
echo "配置: 8 PERows × 128 PEs = 1024 并行单元"
echo "向量规模: 1024维"
echo "矩阵规模: 1024×1024 (90%稀疏度)"
echo "预计运行时间: 数分钟"
echo "================================================"

# 创建build目录
mkdir -p build

# 创建1024维测试台（动态修改参数）
cat > testbench/vector_matrix_testbench_1024.v << 'EOF'
// 1024维中等规模测试台
`timescale 1ns / 1ps

module vector_matrix_testbench_1024;

// 中等规模参数 - 1024维
parameter NUM_PEROWS = 8;          // 8个PERow
parameter PES_PER_ROW = 128;       // 每行128个PE  
parameter VECTOR_SIZE = 1024;      // 1024维向量
parameter MAX_BATCH_SIZE = 16;     // 最大16批次
parameter MAX_BLOCKS = 256;        // 最大256个块
parameter BLOCK_SIZE = 512;        // 每块512个元素
parameter ADDR_WIDTH = 10;         // log2(1024)
parameter PEROW_ADDR_WIDTH = 3;    // log2(8)
parameter BATCH_ADDR_WIDTH = 4;    // log2(16)
parameter BLOCK_ADDR_WIDTH = 8;    // log2(256)
parameter ELEMENT_ADDR_WIDTH = 9;  // log2(512)

// 测试控制
parameter MAX_CYCLES = 100000;     // 10万周期超时
parameter SPARSITY_PERCENT = 90;   // 90%稀疏度

reg clk, rst_n;
integer cycle_count;

// 系统信号
reg start_computation;
reg [BATCH_ADDR_WIDTH:0] batch_size;
reg [BLOCK_ADDR_WIDTH:0] num_blocks;
wire computation_done;
wire system_busy;
wire [31:0] total_tasks_completed;

// A向量配置接口
reg a_vector_config_valid;
reg [BATCH_ADDR_WIDTH-1:0] a_vector_batch_id;
reg [ADDR_WIDTH-1:0] a_vector_addr;
reg [31:0] a_vector_data;
wire a_vector_config_ready;

// B矩阵配置接口
reg b_matrix_config_valid;
reg [BLOCK_ADDR_WIDTH-1:0] b_matrix_block_id;
reg [15:0] b_matrix_value;
reg [ADDR_WIDTH-1:0] b_matrix_col_index;
reg [ADDR_WIDTH-1:0] b_matrix_row_index;
reg [ELEMENT_ADDR_WIDTH-1:0] b_matrix_element_index;
reg b_matrix_block_complete;
wire b_matrix_config_ready;

// 结果输出接口
wire [31:0] total_clock_cycles;
wire result_valid;
wire [BATCH_ADDR_WIDTH-1:0] result_batch_id;
wire [ADDR_WIDTH-1:0] result_addr;
wire [31:0] result_data;
reg result_ready;

// 实例化被测试模块
vector_matrix_row_product_hbm_multi_batch #(
    .NUM_PEROWS(NUM_PEROWS),
    .PES_PER_ROW(PES_PER_ROW),
    .VECTOR_SIZE(VECTOR_SIZE),
    .MAX_BATCH_SIZE(MAX_BATCH_SIZE),
    .MAX_BLOCKS(MAX_BLOCKS),
    .BLOCK_SIZE(BLOCK_SIZE),
    .ADDR_WIDTH(ADDR_WIDTH),
    .PEROW_ADDR_WIDTH(PEROW_ADDR_WIDTH),
    .BATCH_ADDR_WIDTH(BATCH_ADDR_WIDTH),
    .BLOCK_ADDR_WIDTH(BLOCK_ADDR_WIDTH),
    .ELEMENT_ADDR_WIDTH(ELEMENT_ADDR_WIDTH)
) dut (
    .clk(clk),
    .rst_n(rst_n),
    .a_vector_config_valid(a_vector_config_valid),
    .a_vector_batch_id(a_vector_batch_id),
    .a_vector_addr(a_vector_addr),
    .a_vector_data(a_vector_data),
    .a_vector_config_ready(a_vector_config_ready),
    .b_matrix_config_valid(b_matrix_config_valid),
    .b_matrix_block_id(b_matrix_block_id),
    .b_matrix_value(b_matrix_value),
    .b_matrix_col_index(b_matrix_col_index),
    .b_matrix_row_index(b_matrix_row_index),
    .b_matrix_element_index(b_matrix_element_index),
    .b_matrix_block_complete(b_matrix_block_complete),
    .b_matrix_config_ready(b_matrix_config_ready),
    .start_computation(start_computation),
    .batch_size(batch_size),
    .num_blocks(num_blocks),
    .computation_done(computation_done),
    .total_clock_cycles(total_clock_cycles),
    .result_valid(result_valid),
    .result_batch_id(result_batch_id),
    .result_addr(result_addr),
    .result_data(result_data),
    .result_ready(result_ready),
    .system_busy(system_busy),
    .total_tasks_completed(total_tasks_completed)
);

// 时钟生成 - 100MHz
always #5 clk = ~clk;

// 性能监控
always @(posedge clk) begin
    if (!rst_n) begin
        cycle_count <= 0;
    end else begin
        cycle_count <= cycle_count + 1;
        
        // 每500周期输出一次状态
        if (cycle_count % 500 == 0) begin
            $display("周期 %d: busy=%b, done=%b, 任务=%d, 状态=%d", 
                    cycle_count, system_busy, computation_done, 
                    total_tasks_completed, dut.main_state);
        end
        
        // 超时保护
        if (cycle_count > MAX_CYCLES) begin
            $display("❌ 1024维测试：仿真超时在 %d 周期", cycle_count);
            $display("   总任务完成: %d", total_tasks_completed);
            $display("   主状态: %d", dut.main_state);
            $finish;
        end
    end
end

// 1024维测试序列
initial begin
    $display("=== 1024维GPT2向量矩阵乘法测试 ===");
    $display("配置: %d PERows × %d PEs = %d 并行单元", NUM_PEROWS, PES_PER_ROW, NUM_PEROWS * PES_PER_ROW);
    $display("向量规模: %d维", VECTOR_SIZE);
    $display("稀疏度: %d%%", SPARSITY_PERCENT);
    
    // 初始化
    clk = 0; rst_n = 0;
    a_vector_config_valid = 0;
    b_matrix_config_valid = 0;
    start_computation = 0;
    batch_size = 1;
    num_blocks = 10;  // 使用10个块进行测试
    result_ready = 1;
    
    // 复位
    #20 rst_n = 1; #50;
    
    $display("1. 加载1024维向量A...");
    for (integer i = 0; i < VECTOR_SIZE; i = i + 1) begin
        a_vector_config_valid = 1;
        a_vector_batch_id = 0;
        a_vector_addr = i;
        a_vector_data = $realtobits(1.0);
        @(posedge clk);
        if (i % 256 == 255) begin
            $display("   已加载 %d / %d 向量元素", i+1, VECTOR_SIZE);
        end
    end
    a_vector_config_valid = 0;
    
    $display("2. 加载稀疏矩阵块...");
    for (integer block_id = 0; block_id < num_blocks; block_id = block_id + 1) begin
        // 每个块有50个非零元素（约10%稀疏度）
        for (integer elem = 0; elem < 50; elem = elem + 1) begin
            b_matrix_config_valid = 1;
            b_matrix_block_id = block_id;
            b_matrix_value = 16'h3f80;  // 1.0 in BF16
            b_matrix_col_index = (elem * 20) % VECTOR_SIZE;
            b_matrix_row_index = (elem * 13) % VECTOR_SIZE; 
            b_matrix_element_index = elem;
            b_matrix_block_complete = (elem == 49);
            @(posedge clk);
        end
        b_matrix_config_valid = 0;
        b_matrix_block_complete = 0;
        $display("   已加载块 %d / %d", block_id + 1, num_blocks);
    end
    
    $display("3. 开始1024维计算...");
    start_computation = 1;
    @(posedge clk);
    start_computation = 0;
    
    $display("4. 等待完成...");
    
    // 等待完成
    wait(computation_done || (cycle_count > MAX_CYCLES));
    
    if (computation_done) begin
        $display("🎉 1024维计算完成!");
        $display("   总周期数: %d", cycle_count);
        $display("   总任务数: %d", total_tasks_completed);
        $display("   平均性能: %.2f MAC运算/周期", $itor(total_tasks_completed) / $itor(cycle_count));
    end
    
    #1000;
    $finish;
end

endmodule
EOF

echo "📦 编译1024维中等规模测试..."

iverilog -o build/test_1024_scale -I src \
  testbench/vector_matrix_testbench_1024.v \
  src/vector_matrix_row_product_hbm_multi_batch.v \
  src/fp32_to_bf16_pipeline.v \
  src/hbm_controller_fixed.v \
  src/processing_element_row.v \
  src/processing_element.v \
  src/bf16_add_pipeline.v \
  src/bf16_multiply_pipeline.v

if [ $? -eq 0 ]; then
    echo "✅ 1024维测试编译成功！"
    echo ""
    echo "🚀 运行1024维测试..."
    echo "   预计运行时间: 1-5分钟"
    echo "   内存使用: 适中"
    echo ""
    
    cd build && ./test_1024_scale
else
    echo "❌ 编译失败"
fi 