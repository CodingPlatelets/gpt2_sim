// MFIU (Multi-function Integer Unit) Pipeline - Verilog实现
// 基于Python MFIU_sim.py转换
// 5级流水线：输入预处理 → 位掩码AND → 前缀和 → EC索引 → 移位输出

`timescale 1ns / 1ps

module mfiu_pipeline #(
    parameter WIDTH = 8,              // 处理宽度
    parameter BIT_WIDTH = 16,         // 位宽度
    parameter MAX_AB_WIDTH = 64,      // 最大AB宽度
    parameter MAX_VALUES_LEN = 256,   // 最大值长度
    parameter ADDR_WIDTH = 8          // 地址位宽
)(
    input wire clk,
    input wire rst_n,
    
    // 输入接口
    input wire input_valid,
    input wire [31:0] len_values_A,
    input wire [31:0] len_values_B,
    
    // A矩阵行掩码和偏移（简化为4个）
    input wire [15:0] mask_A_row_0, mask_A_row_1, mask_A_row_2, mask_A_row_3,
    input wire [ADDR_WIDTH-1:0] offset_A_row_0, offset_A_row_1, offset_A_row_2, offset_A_row_3, 
    input wire [3:0] mask_A_row_count,
    
    // B矩阵列掩码和偏移（简化为4个）
    input wire [15:0] mask_B_col_0, mask_B_col_1, mask_B_col_2, mask_B_col_3,
    input wire [ADDR_WIDTH-1:0] offset_B_col_0, offset_B_col_1, offset_B_col_2, offset_B_col_3,
    input wire [3:0] mask_B_col_count,
    
    // 输出接口（简化为4个）
    output reg output_valid,
    output reg [31:0] output_A_0, output_A_1, output_A_2, output_A_3,
    output reg [31:0] output_B_0, output_B_1, output_B_2, output_B_3,
    output reg [31:0] output_len_A,
    output reg [31:0] output_len_B,
    
    // 状态接口
    output wire pipeline_active,
    output reg [31:0] cycle_count
);

// Stage寄存器
reg stage1_valid, stage2_valid, stage3_valid, stage4_valid, stage5_valid;

// Stage 1 - 输入预处理
reg [31:0] stage1_len_values_A, stage1_len_values_B;
reg [15:0] stage1_A_mask_0, stage1_A_mask_1, stage1_B_mask_0, stage1_B_mask_1;
reg [ADDR_WIDTH-1:0] stage1_A_offset_0, stage1_A_offset_1, stage1_B_offset_0, stage1_B_offset_1;

// Stage 2 - 位掩码AND操作
reg [31:0] stage2_len_values_A, stage2_len_values_B;
reg [15:0] stage2_A_mask_0, stage2_A_mask_1, stage2_B_mask_0, stage2_B_mask_1;
reg [ADDR_WIDTH-1:0] stage2_A_offset_0, stage2_A_offset_1, stage2_B_offset_0, stage2_B_offset_1;
reg [15:0] stage2_bit_result_0, stage2_bit_result_1;

// Stage 3 - 前缀和计算
reg [31:0] stage3_len_values_A, stage3_len_values_B;
reg [ADDR_WIDTH-1:0] stage3_A_offset_0, stage3_A_offset_1, stage3_B_offset_0, stage3_B_offset_1;
reg [15:0] stage3_prefix_sum_0, stage3_prefix_sum_1;

// Stage 4 - EC索引计算
reg [31:0] stage4_len_values_A, stage4_len_values_B;
reg [ADDR_WIDTH-1:0] stage4_A_offset_0, stage4_A_offset_1, stage4_B_offset_0, stage4_B_offset_1;
reg [31:0] stage4_ec_idx_A_0, stage4_ec_idx_A_1, stage4_ec_idx_B_0, stage4_ec_idx_B_1;

// 计算bit count的函数（兼容旧版本iverilog）
function [15:0] count_bits;
    input [15:0] value;
    integer i;
    begin
        count_bits = 0;
        for (i = 0; i < 16; i = i + 1) begin
            if (value[i]) count_bits = count_bits + 1;
        end
    end
endfunction

// 周期计数器
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        cycle_count <= 0;
    end else begin
        cycle_count <= cycle_count + 1;
    end
end

// 流水线活跃状态
assign pipeline_active = stage1_valid || stage2_valid || stage3_valid || stage4_valid || stage5_valid;

// 主流水线逻辑
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        // 重置所有阶段
        stage1_valid <= 0;
        stage2_valid <= 0;
        stage3_valid <= 0;
        stage4_valid <= 0;
        stage5_valid <= 0;
        output_valid <= 0;
        
        // 重置输出
        output_A_0 <= 0; output_A_1 <= 0; output_A_2 <= 0; output_A_3 <= 0;
        output_B_0 <= 0; output_B_1 <= 0; output_B_2 <= 0; output_B_3 <= 0;
        output_len_A <= 0;
        output_len_B <= 0;
        
    end else begin
        
        // ===== Stage 5: 移位输出 =====
        stage5_valid <= stage4_valid;
        output_valid <= stage5_valid;
        
        if (stage4_valid) begin
            output_len_A <= stage4_len_values_A;
            output_len_B <= stage4_len_values_B;
            
            // 简化输出 - 基于EC索引和偏移
            output_A_0 <= stage4_ec_idx_A_0;
            output_A_1 <= stage4_ec_idx_A_1;
            output_A_2 <= 0;
            output_A_3 <= 0;
            
            output_B_0 <= stage4_ec_idx_B_0;
            output_B_1 <= stage4_ec_idx_B_1;
            output_B_2 <= 0;
            output_B_3 <= 0;
        end
        
        // ===== Stage 4: EC索引计算 =====
        stage4_valid <= stage3_valid;
        stage4_len_values_A <= stage3_len_values_A;
        stage4_len_values_B <= stage3_len_values_B;
        stage4_A_offset_0 <= stage3_A_offset_0;
        stage4_A_offset_1 <= stage3_A_offset_1;
        stage4_B_offset_0 <= stage3_B_offset_0;
        stage4_B_offset_1 <= stage3_B_offset_1;
        
        if (stage3_valid) begin
            // 简化的EC索引计算
            stage4_ec_idx_A_0 <= stage3_prefix_sum_0 + stage3_A_offset_0;
            stage4_ec_idx_A_1 <= stage3_prefix_sum_1 + stage3_A_offset_1;
            stage4_ec_idx_B_0 <= stage3_prefix_sum_0 + stage3_B_offset_0;
            stage4_ec_idx_B_1 <= stage3_prefix_sum_1 + stage3_B_offset_1;
        end
        
        // ===== Stage 3: 前缀和计算 =====
        stage3_valid <= stage2_valid;
        stage3_len_values_A <= stage2_len_values_A;
        stage3_len_values_B <= stage2_len_values_B;
        stage3_A_offset_0 <= stage2_A_offset_0;
        stage3_A_offset_1 <= stage2_A_offset_1;
        stage3_B_offset_0 <= stage2_B_offset_0;
        stage3_B_offset_1 <= stage2_B_offset_1;
        
        if (stage2_valid) begin
            // 前缀和计算 - 计算位数
            stage3_prefix_sum_0 <= count_bits(stage2_bit_result_0);
            stage3_prefix_sum_1 <= count_bits(stage2_bit_result_1);
        end
        
        // ===== Stage 2: 位掩码AND操作 =====
        stage2_valid <= stage1_valid;
        stage2_len_values_A <= stage1_len_values_A;
        stage2_len_values_B <= stage1_len_values_B;
        stage2_A_offset_0 <= stage1_A_offset_0;
        stage2_A_offset_1 <= stage1_A_offset_1;
        stage2_B_offset_0 <= stage1_B_offset_0;
        stage2_B_offset_1 <= stage1_B_offset_1;
        
        if (stage1_valid) begin
            // 位掩码AND操作
            stage2_bit_result_0 <= stage1_A_mask_0 & stage1_B_mask_0;
            stage2_bit_result_1 <= stage1_A_mask_1 & stage1_B_mask_1;
        end
        
        // ===== Stage 1: 输入预处理 =====
        stage1_valid <= input_valid;
        
        if (input_valid) begin
            stage1_len_values_A <= len_values_A;
            stage1_len_values_B <= len_values_B;
            
            // 输入掩码和偏移（简化版本）
            stage1_A_mask_0 <= mask_A_row_0;
            stage1_A_mask_1 <= mask_A_row_1;
            stage1_B_mask_0 <= mask_B_col_0;
            stage1_B_mask_1 <= mask_B_col_1;
            
            stage1_A_offset_0 <= offset_A_row_0;
            stage1_A_offset_1 <= offset_A_row_1;
            stage1_B_offset_0 <= offset_B_col_0;
            stage1_B_offset_1 <= offset_B_col_1;
        end
    end
end

endmodule