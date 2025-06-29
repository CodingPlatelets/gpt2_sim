// 详细调试测试台 - 监控PE和任务处理状态
`timescale 1ns / 1ps

module detailed_debug_testbench;

// 测试参数 - 超小规模
parameter VECTOR_SIZE = 4;
parameter NUM_PEROWS = 1;
parameter PES_PER_ROW = 2;
parameter MAX_BLOCKS = 1;
parameter MAX_BATCH_SIZE = 1;

// 计算参数
localparam ADDR_WIDTH = $clog2(VECTOR_SIZE);
localparam BATCH_ADDR_WIDTH = $clog2(MAX_BATCH_SIZE);
localparam BLOCK_ADDR_WIDTH = $clog2(MAX_BLOCKS);
localparam PEROW_ADDR_WIDTH = $clog2(NUM_PEROWS);
localparam ELEMENT_ADDR_WIDTH = 11;

reg clk, rst_n;
integer cycle_count;

// 顶层模块接口
reg a_vector_config_valid;
reg [BATCH_ADDR_WIDTH-1:0] a_vector_batch_id;
reg [ADDR_WIDTH-1:0] a_vector_addr;
reg [31:0] a_vector_data;
wire a_vector_config_ready;

reg b_matrix_config_valid;
reg [BLOCK_ADDR_WIDTH-1:0] b_matrix_block_id;
reg [15:0] b_matrix_value;
reg [ADDR_WIDTH-1:0] b_matrix_col_index;
reg [ADDR_WIDTH-1:0] b_matrix_row_index;
reg [ELEMENT_ADDR_WIDTH-1:0] b_matrix_element_index;
reg b_matrix_block_complete;
wire b_matrix_config_ready;

reg start_computation;
reg [BATCH_ADDR_WIDTH:0] batch_size;
reg [BLOCK_ADDR_WIDTH:0] num_blocks;
wire computation_done;
wire [31:0] total_clock_cycles;

wire result_valid;
wire [BATCH_ADDR_WIDTH-1:0] result_batch_id;
wire [ADDR_WIDTH-1:0] result_addr;
wire [31:0] result_data;
reg result_ready;

wire system_busy;
wire [31:0] total_tasks_completed;

// 实例化待测试模块
vector_matrix_row_product_hbm_multi_batch #(
    .NUM_PEROWS(NUM_PEROWS),
    .PES_PER_ROW(PES_PER_ROW),
    .VECTOR_SIZE(VECTOR_SIZE),
    .MAX_BATCH_SIZE(MAX_BATCH_SIZE),
    .MAX_BLOCKS(MAX_BLOCKS)
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

// 内部状态监控（通过层次化访问）
wire [3:0] main_state = dut.main_state;
wire [BLOCK_ADDR_WIDTH:0] internal_current_block_id = dut.current_block_id;
wire hbm_block_request_valid = dut.hbm_block_request_valid;
wire hbm_block_request_ready = dut.hbm_block_request_ready;
wire hbm_block_data_valid = dut.hbm_block_data_valid;
wire task_dispatch_active = dut.task_dispatch_active;

// 时钟生成
always #5 clk = ~clk;

// 详细状态监控
always @(posedge clk) begin
    if (!rst_n) begin
        cycle_count <= 0;
    end else begin
        cycle_count <= cycle_count + 1;
        
        // 每50周期显示详细状态
        if (cycle_count % 50 == 0) begin
            $display("=== 周期 %0d ===", cycle_count);
            $display("主状态: %0d, busy=%b, done=%b", main_state, system_busy, computation_done);
            $display("任务完成数: %0d, 时钟周期: %0d", total_tasks_completed, total_clock_cycles);
            $display("HBM: 请求=%b/%b, 数据=%b, 分发=%b", 
                    hbm_block_request_valid, hbm_block_request_ready, 
                    hbm_block_data_valid, task_dispatch_active);
            $display("结果: valid=%b, 地址=%0d, 数据=%h", 
                    result_valid, result_addr, result_data);
            $display("---");
        end
        
        // 异常检测
        if (cycle_count > 100 && total_tasks_completed > 1000) begin
            $display("❌ 异常：任务数量过多 (%0d)，可能有任务泄漏", total_tasks_completed);
        end
        
        if (cycle_count > 1000) begin
            $display("❌ 超时：1000周期后仍未完成");
            $display("最终状态：");
            $display("  主状态: %0d", main_state);
            $display("  完成任务数: %0d", total_tasks_completed);
            $display("  系统忙: %b", system_busy);
            $finish;
        end

        // 成功检测
        if (computation_done) begin
            $display("✅ 计算完成在周期 %0d", cycle_count);
            $display("最终完成任务数: %0d", total_tasks_completed);
            $display("总时钟周期: %0d", total_clock_cycles);
            $finish;
        end
    end
end

// 测试序列
initial begin
    $display("=== 详细调试测试：4维向量 ===");
    
    // 初始化所有信号
    clk = 0; rst_n = 0;
    
    a_vector_config_valid = 0;
    a_vector_batch_id = 0;
    a_vector_addr = 0;
    a_vector_data = 0;
    
    b_matrix_config_valid = 0;
    b_matrix_block_id = 0;
    b_matrix_value = 0;
    b_matrix_col_index = 0;
    b_matrix_row_index = 0;
    b_matrix_element_index = 0;
    b_matrix_block_complete = 0;
    
    start_computation = 0;
    batch_size = 1;
    num_blocks = 1;
    result_ready = 1;
    
    // 复位
    #20 rst_n = 1; #20;
    
    $display("1. 配置向量A = [1,1,1,1]");
    // 配置向量A的4个元素
    for (integer i = 0; i < VECTOR_SIZE; i = i + 1) begin
        a_vector_config_valid = 1;
        a_vector_batch_id = 0;
        a_vector_addr = i;
        a_vector_data = 32'h3f800000;  // 1.0 in FP32
        wait(a_vector_config_ready);
        @(posedge clk);
        a_vector_config_valid = 0;
        @(posedge clk);
    end
    
    $display("2. 配置矩阵块：B[0,0] = 1.0");
    b_matrix_config_valid = 1;
    b_matrix_block_id = 0;
    b_matrix_value = 16'h3f80;  // 1.0 in BF16
    b_matrix_col_index = 0;
    b_matrix_row_index = 0;
    b_matrix_element_index = 0;
    b_matrix_block_complete = 0;
    wait(b_matrix_config_ready);
    @(posedge clk);
    b_matrix_block_complete = 1;
    @(posedge clk);
    b_matrix_config_valid = 0;
    b_matrix_block_complete = 0;
    @(posedge clk);
    
    $display("3. 开始计算");
    start_computation = 1;
    @(posedge clk);
    start_computation = 0;
    
    $display("4. 监控执行过程...");
    // 等待完成或超时
end

endmodule 