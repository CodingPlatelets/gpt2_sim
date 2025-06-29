// 极简调试测试台 - 定位逻辑bug
`timescale 1ns / 1ps

module debug_simple_testbench;

// 最小参数 - 便于调试
parameter NUM_PEROWS = 1;          // 只用1个PERow
parameter PES_PER_ROW = 2;         // 只用2个PE
parameter VECTOR_SIZE = 4;         // 只用4维向量
parameter MAX_BATCH_SIZE = 1;      
parameter MAX_BLOCKS = 1;          // 只用1个块
parameter BLOCK_SIZE = 4;          // 每块4个元素
parameter ADDR_WIDTH = 2;          // log2(4)
parameter PEROW_ADDR_WIDTH = 1;    
parameter BATCH_ADDR_WIDTH = 1;    
parameter BLOCK_ADDR_WIDTH = 1;    
parameter ELEMENT_ADDR_WIDTH = 2;  

// 测试控制
parameter MAX_CYCLES = 10000;      // 1万周期超时

reg clk, rst_n;
integer cycle_count;

// 简化的信号
reg start_computation;
reg [BATCH_ADDR_WIDTH:0] batch_size;
reg [BLOCK_ADDR_WIDTH:0] num_blocks;
wire computation_done;
wire system_busy;
wire [31:0] total_tasks_completed;

// 固定的测试输入
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

// 时钟生成
always #5 clk = ~clk;

// 调试监控
always @(posedge clk) begin
    if (!rst_n) begin
        cycle_count <= 0;
    end else begin
        cycle_count <= cycle_count + 1;
        
        // 详细状态输出
        if (cycle_count % 100 == 0) begin
            $display("周期 %d: busy=%b, done=%b, 任务=%d, 状态=%d", 
                    cycle_count, system_busy, computation_done, 
                    total_tasks_completed, dut.main_state);
        end
        
        // 超时保护
        if (cycle_count > MAX_CYCLES) begin
            $display("❌ 调试：仿真超时在 %d 周期", cycle_count);
            $display("   主状态: %d", dut.main_state);
            $display("   current_block_id: %d", dut.current_block_id);
            $display("   num_blocks: %d", num_blocks);
            $display("   hbm_block_request_valid: %b", dut.hbm_block_request_valid);
            $display("   hbm_block_request_ready: %b", dut.hbm_block_request_ready);
            $display("   hbm_block_data_valid: %b", dut.hbm_block_data_valid);
            $display("   task_dispatch_active: %b", dut.task_dispatch_active);
            $finish;
        end
    end
end

// 极简测试序列
initial begin
    $display("=== 调试测试：4维向量 ===");
    
    // 初始化
    clk = 0; rst_n = 0;
    a_vector_config_valid = 0;
    b_matrix_config_valid = 0;
    start_computation = 0;
    batch_size = 1;
    num_blocks = 1;
    result_ready = 1;
    
    // 复位
    #20 rst_n = 1; #20;
    
    $display("1. 加载4维向量A = [1,1,1,1]");
    for (integer i = 0; i < 4; i = i + 1) begin
        a_vector_config_valid = 1;
        a_vector_batch_id = 0;
        a_vector_addr = i;
        a_vector_data = $realtobits(1.0);
        @(posedge clk);
        a_vector_config_valid = 0;
        @(posedge clk);
    end
    
    $display("2. 加载1个矩阵块：只有B[0,0]=1.0");
    b_matrix_config_valid = 1;
    b_matrix_block_id = 0;
    b_matrix_value = 16'h3f80;  // 1.0 in BF16
    b_matrix_col_index = 0;
    b_matrix_row_index = 0;
    b_matrix_element_index = 0;
    b_matrix_block_complete = 1;  // 只有一个元素
    @(posedge clk);
    b_matrix_config_valid = 0;
    b_matrix_block_complete = 0;
    @(posedge clk);
    
    $display("3. 开始计算");
    start_computation = 1;
    @(posedge clk);
    start_computation = 0;
    
    $display("4. 等待完成或超时...");
    
    // 等待完成
    wait(computation_done || (cycle_count > MAX_CYCLES));
    
    if (computation_done) begin
        $display("✅ 计算完成! 周期: %d, 任务: %d", cycle_count, total_tasks_completed);
    end
    
    #100;
    $finish;
end

endmodule 