// MFIU Pipeline 测试台 - 简化版
// 验证从Python转换的Verilog MFIU实现

`timescale 1ns / 1ps

module mfiu_testbench;

// 测试参数
parameter WIDTH = 8;
parameter BIT_WIDTH = 16;
parameter MAX_AB_WIDTH = 64;
parameter MAX_VALUES_LEN = 256;
parameter ADDR_WIDTH = 8;
parameter MAX_CYCLES = 500;

// 时钟和复位
reg clk, rst_n;
integer cycle_count;

// DUT信号
reg input_valid;
reg [31:0] len_values_A, len_values_B;
reg [15:0] mask_A_row_0, mask_A_row_1, mask_A_row_2, mask_A_row_3;
reg [ADDR_WIDTH-1:0] offset_A_row_0, offset_A_row_1, offset_A_row_2, offset_A_row_3;
reg [3:0] mask_A_row_count;
reg [15:0] mask_B_col_0, mask_B_col_1, mask_B_col_2, mask_B_col_3;
reg [ADDR_WIDTH-1:0] offset_B_col_0, offset_B_col_1, offset_B_col_2, offset_B_col_3;
reg [3:0] mask_B_col_count;

wire output_valid;
wire [31:0] output_A_0, output_A_1, output_A_2, output_A_3;
wire [31:0] output_B_0, output_B_1, output_B_2, output_B_3;
wire [31:0] output_len_A, output_len_B;
wire pipeline_active;
wire [31:0] cycle_count_dut;

// 实例化被测试模块
mfiu_pipeline #(
    .WIDTH(WIDTH),
    .BIT_WIDTH(BIT_WIDTH),
    .MAX_AB_WIDTH(MAX_AB_WIDTH),
    .MAX_VALUES_LEN(MAX_VALUES_LEN),
    .ADDR_WIDTH(ADDR_WIDTH)
) dut (
    .clk(clk),
    .rst_n(rst_n),
    .input_valid(input_valid),
    .len_values_A(len_values_A),
    .len_values_B(len_values_B),
    .mask_A_row_0(mask_A_row_0),
    .mask_A_row_1(mask_A_row_1),
    .mask_A_row_2(mask_A_row_2),
    .mask_A_row_3(mask_A_row_3),
    .offset_A_row_0(offset_A_row_0),
    .offset_A_row_1(offset_A_row_1),
    .offset_A_row_2(offset_A_row_2),
    .offset_A_row_3(offset_A_row_3),
    .mask_A_row_count(mask_A_row_count),
    .mask_B_col_0(mask_B_col_0),
    .mask_B_col_1(mask_B_col_1),
    .mask_B_col_2(mask_B_col_2),
    .mask_B_col_3(mask_B_col_3),
    .offset_B_col_0(offset_B_col_0),
    .offset_B_col_1(offset_B_col_1),
    .offset_B_col_2(offset_B_col_2),
    .offset_B_col_3(offset_B_col_3),
    .mask_B_col_count(mask_B_col_count),
    .output_valid(output_valid),
    .output_A_0(output_A_0),
    .output_A_1(output_A_1),
    .output_A_2(output_A_2),
    .output_A_3(output_A_3),
    .output_B_0(output_B_0),
    .output_B_1(output_B_1),
    .output_B_2(output_B_2),
    .output_B_3(output_B_3),
    .output_len_A(output_len_A),
    .output_len_B(output_len_B),
    .pipeline_active(pipeline_active),
    .cycle_count(cycle_count_dut)
);

// 时钟生成 - 100MHz
always #5 clk = ~clk;

// 周期计数
always @(posedge clk) begin
    if (!rst_n) begin
        cycle_count <= 0;
    end else begin
        cycle_count <= cycle_count + 1;
        
        // 每10个周期输出状态
        if (cycle_count % 10 == 0) begin
            $display("周期 %d: input_valid=%b, output_valid=%b, pipeline_active=%b", 
                    cycle_count, input_valid, output_valid, pipeline_active);
        end
        
        // 超时保护
        if (cycle_count > MAX_CYCLES) begin
            $display("❌ 测试超时在 %d 周期", cycle_count);
            $finish;
        end
    end
end

// 输出监控
always @(posedge clk) begin
    if (output_valid) begin
        $display("✅ 周期 %d: MFIU输出有效", cycle_count);
        $display("   输出长度: A=%d, B=%d", output_len_A, output_len_B);
        $display("   输出A: %d %d %d %d", 
                output_A_0, output_A_1, output_A_2, output_A_3);
        $display("   输出B: %d %d %d %d", 
                output_B_0, output_B_1, output_B_2, output_B_3);
    end
end

// 测试序列
initial begin
    $display("=== MFIU Pipeline Verilog测试 ===");
    $display("配置: WIDTH=%d, BIT_WIDTH=%d", WIDTH, BIT_WIDTH);
    
    // 初始化
    clk = 0; rst_n = 0;
    input_valid = 0;
    len_values_A = 0;
    len_values_B = 0;
    mask_A_row_count = 0;
    mask_B_col_count = 0;
    
    // 初始化掩码和偏移
    mask_A_row_0 = 0; mask_A_row_1 = 0; mask_A_row_2 = 0; mask_A_row_3 = 0;
    mask_B_col_0 = 0; mask_B_col_1 = 0; mask_B_col_2 = 0; mask_B_col_3 = 0;
    offset_A_row_0 = 0; offset_A_row_1 = 0; offset_A_row_2 = 0; offset_A_row_3 = 0;
    offset_B_col_0 = 0; offset_B_col_1 = 0; offset_B_col_2 = 0; offset_B_col_3 = 0;
    
    // 复位
    #20 rst_n = 1; #50;
    
    $display("🔧 测试1: 基本功能验证");
    
    // 测试数据1
    len_values_A = 32;
    len_values_B = 32;
    mask_A_row_count = 2;
    mask_B_col_count = 2;
    
    // A矩阵行掩码：0x00FF, 0xFF00
    mask_A_row_0 = 16'h00FF;  // 低8位为1
    mask_A_row_1 = 16'hFF00;  // 高8位为1
    offset_A_row_0 = 0;
    offset_A_row_1 = 8;
    
    // B矩阵列掩码：0x0F0F, 0xF0F0  
    mask_B_col_0 = 16'h0F0F;  // 间隔位为1
    mask_B_col_1 = 16'hF0F0;  // 间隔位为1（相反）
    offset_B_col_0 = 0;
    offset_B_col_1 = 4;
    
    // 发送输入
    @(posedge clk);
    input_valid = 1;
    @(posedge clk);
    input_valid = 0;
    
    $display("   输入数据已发送，等待流水线处理...");
    
    // 等待流水线完成
    wait(!pipeline_active);
    #100;
    
    $display("🔧 测试2: 简单掩码测试");
    
    // 测试简单掩码
    len_values_A = 16;
    len_values_B = 16;
    mask_A_row_count = 1;
    mask_B_col_count = 1;
    
    mask_A_row_0 = 16'hFFFF;  // 全1
    mask_B_col_0 = 16'h5555;  // 间隔1
    offset_A_row_0 = 0;
    offset_B_col_0 = 0;
    
    @(posedge clk);
    input_valid = 1;
    @(posedge clk);
    input_valid = 0;
    
    #200;
    
    // 等待所有处理完成
    wait(!pipeline_active);
    #200;
    
    $display("✅ MFIU Pipeline Verilog测试完成！");
    $display("   总周期数: %d", cycle_count);
    $display("   DUT周期数: %d", cycle_count_dut);
    
    $finish;
end

endmodule