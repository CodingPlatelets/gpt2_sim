// BF16 Multiplication Pipeline
// 5-stage pipeline for BF16 floating point multiplication
`timescale 1ns/1ps

module bf16_multiply_pipeline (
    input wire clk,
    input wire rst_n,
    
    // Input interface
    input wire [15:0] bf16_a,
    input wire [15:0] bf16_b,
    input wire valid_in,
    
    // Output interface
    output reg [15:0] result,
    output reg valid_out
);

// Constants
localparam POS_INF = 16'h7F80;
localparam NEG_INF = 16'hFF80;
localparam NAN = 16'h7FC0;

// Stage 1 registers
reg [15:0] stage1_bf16_a;
reg [15:0] stage1_bf16_b;
reg stage1_valid;
reg stage1_special_case;
reg [15:0] stage1_result;

// Stage 2 registers
reg stage2_valid;
reg stage2_special_case;
reg [15:0] stage2_result;
reg stage2_sign_result;
reg [8:0] stage2_exp_a;  // 9-bit to handle denormal adjustment
reg [8:0] stage2_exp_b;  // 9-bit to handle denormal adjustment
reg [7:0] stage2_mant_a;
reg [7:0] stage2_mant_b;

// Stage 3 registers
reg stage3_valid;
reg stage3_special_case;
reg [15:0] stage3_result;
reg stage3_sign_result;
reg [9:0] stage3_exp_result;  // 10-bit to handle overflow
reg [15:0] stage3_mant_result; // 16-bit for multiplication result

// Stage 4 registers
reg stage4_valid;
reg stage4_special_case;
reg [15:0] stage4_result;
reg stage4_sign_result;
reg [9:0] stage4_exp_result;
reg [15:0] stage4_mant_result;

// Stage 5 registers
reg stage5_valid;

// Combinational logic variables
reg special_case_detected;
reg [15:0] special_case_result;
reg sign_a_comb;
reg [7:0] exp_a_comb;
reg [6:0] mant_a_comb;
reg sign_b_comb;
reg [7:0] exp_b_comb;
reg [6:0] mant_b_comb;
reg sign_result_comb;

// Pipeline temporary variables
reg [15:0] final_mant;
reg [9:0] final_exp;
reg final_sign;
reg [7:0] round_bits;
reg [7:0] half_point;
reg [7:0] denorm_mant;
reg [4:0] shift_amount;
reg sign_a_seq;
reg [7:0] exp_a_seq;
reg [6:0] mant_a_seq;
reg sign_b_seq;
reg [7:0] exp_b_seq;
reg [6:0] mant_b_seq;

// Combinational logic for special case detection
always @(*) begin
    // Extract components
    sign_a_comb = bf16_a[15];
    exp_a_comb = bf16_a[14:7];
    mant_a_comb = bf16_a[6:0];
    sign_b_comb = bf16_b[15];
    exp_b_comb = bf16_b[14:7];
    mant_b_comb = bf16_b[6:0];
    
    sign_result_comb = sign_a_comb ^ sign_b_comb;
    
    special_case_detected = 1'b0;
    special_case_result = 16'h0;
    
    // Check for NaN
    if ((exp_a_comb == 8'hFF && mant_a_comb != 7'h0) || (exp_b_comb == 8'hFF && mant_b_comb != 7'h0)) begin
        special_case_detected = 1'b1;
        special_case_result = NAN;
    end
    // Check for infinity * 0 = NaN
    else if ((exp_a_comb == 8'hFF && exp_b_comb == 8'h0 && mant_b_comb == 7'h0) ||
             (exp_b_comb == 8'hFF && exp_a_comb == 8'h0 && mant_a_comb == 7'h0)) begin
        special_case_detected = 1'b1;
        special_case_result = NAN;
    end
    // Check for zero multiplication
    else if ((exp_a_comb == 8'h0 && mant_a_comb == 7'h0) || (exp_b_comb == 8'h0 && mant_b_comb == 7'h0)) begin
        special_case_detected = 1'b1;
        special_case_result = {sign_result_comb, 15'h0};
    end
    // Check for infinity multiplication
    else if (exp_a_comb == 8'hFF || exp_b_comb == 8'hFF) begin
        special_case_detected = 1'b1;
        special_case_result = {sign_result_comb, 8'hFF, 7'h0};
    end
end

// Combinational logic for mantissa normalization
function [7:0] normalize_mantissa;
    input [7:0] exp;
    input [6:0] mant;
    
    begin
        if (exp == 8'h0 && mant != 7'h0) begin
            // Denormal number - find leading 1
            if (mant[6]) normalize_mantissa = 8'h80 | mant;
            else if (mant[5]) normalize_mantissa = 8'h40 | (mant << 1);
            else if (mant[4]) normalize_mantissa = 8'h20 | (mant << 2);
            else if (mant[3]) normalize_mantissa = 8'h10 | (mant << 3);
            else if (mant[2]) normalize_mantissa = 8'h08 | (mant << 4);
            else if (mant[1]) normalize_mantissa = 8'h04 | (mant << 5);
            else if (mant[0]) normalize_mantissa = 8'h02 | (mant << 6);
            else normalize_mantissa = 8'h80;
        end else begin
            normalize_mantissa = 8'h80 | mant;
        end
    end
endfunction

function [8:0] adjust_exp_for_denormal;
    input [7:0] exp;
    input [6:0] mant;
    
    begin
        if (exp == 8'h0 && mant != 7'h0) begin
            // Calculate adjustment for denormal numbers
            if (mant[6]) adjust_exp_for_denormal = 9'h1;
            else if (mant[5]) adjust_exp_for_denormal = 9'h0;
            else if (mant[4]) adjust_exp_for_denormal = 9'h1FF; // -1 in 2's complement
            else if (mant[3]) adjust_exp_for_denormal = 9'h1FE; // -2 in 2's complement
            else if (mant[2]) adjust_exp_for_denormal = 9'h1FD; // -3 in 2's complement
            else if (mant[1]) adjust_exp_for_denormal = 9'h1FC; // -4 in 2's complement
            else if (mant[0]) adjust_exp_for_denormal = 9'h1FB; // -5 in 2's complement
            else adjust_exp_for_denormal = 9'h1;
        end else begin
            adjust_exp_for_denormal = {1'b0, exp};
        end
    end
endfunction

// Pipeline stages
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        // Reset all pipeline registers
        stage1_bf16_a <= 16'h0;
        stage1_bf16_b <= 16'h0;
        stage1_valid <= 1'b0;
        stage1_special_case <= 1'b0;
        stage1_result <= 16'h0;
        
        stage2_valid <= 1'b0;
        stage2_special_case <= 1'b0;
        stage2_result <= 16'h0;
        stage2_sign_result <= 1'b0;
        stage2_exp_a <= 9'h0;
        stage2_exp_b <= 9'h0;
        stage2_mant_a <= 8'h0;
        stage2_mant_b <= 8'h0;
        
        stage3_valid <= 1'b0;
        stage3_special_case <= 1'b0;
        stage3_result <= 16'h0;
        stage3_sign_result <= 1'b0;
        stage3_exp_result <= 10'h0;
        stage3_mant_result <= 16'h0;
        
        stage4_valid <= 1'b0;
        stage4_special_case <= 1'b0;
        stage4_result <= 16'h0;
        stage4_sign_result <= 1'b0;
        stage4_exp_result <= 10'h0;
        stage4_mant_result <= 16'h0;
        
        stage5_valid <= 1'b0;
        result <= 16'h0;
        valid_out <= 1'b0;
    end else begin
        // Stage 5: Output stage
        valid_out <= stage4_valid;
        if (stage4_valid) begin
            if (stage4_special_case) begin
                result <= stage4_result;
            end else begin
                // Normalization and rounding
                final_mant = stage4_mant_result;
                final_exp = stage4_exp_result;
                final_sign = stage4_sign_result;
                
                if (final_mant == 16'h0) begin
                    result <= {final_sign, 15'h0};
                end else begin
                    // Handle decimal point positioning
                    if (final_mant[15]) begin
                        // Need to shift right
                        final_exp = final_exp + 1;
                        round_bits = final_mant[7:0];
                        half_point = 8'h80;
                        final_mant = final_mant >> 8;
                    end else begin
                        // Normal case
                        round_bits = final_mant[6:0];
                        half_point = 8'h40;
                        final_mant = final_mant >> 7;
                    end
                    
                    // Round to nearest even
                    if (round_bits > half_point || 
                        (round_bits == half_point && final_mant[0])) begin
                        final_mant = final_mant + 1;
                        if (final_mant[8]) begin
                            final_mant = final_mant >> 1;
                            final_exp = final_exp + 1;
                        end
                    end
                    
                    // Remove implicit bit
                    final_mant = final_mant & 16'h007F;
                    
                    // Handle overflow/underflow
                    if (final_exp <= 10'h0) begin
                        // Underflow - return denormal or zero
                        if (final_exp < -10'd6) begin
                            result <= {final_sign, 15'h0};
                        end else begin
                            // Denormal representation
                            denorm_mant = 8'h80 | final_mant[6:0];
                            shift_amount = 1 - final_exp[4:0];
                            
                            if (shift_amount < 8) begin
                                denorm_mant = denorm_mant >> shift_amount;
                            end else begin
                                denorm_mant = 8'h0;
                            end
                            
                            result <= {final_sign, 8'h0, denorm_mant[6:0]};
                        end
                    end else if (final_exp >= 10'hFF) begin
                        // Overflow - return infinity
                        result <= {final_sign, 8'hFF, 7'h0};
                    end else begin
                        // Normal case
                        result <= {final_sign, final_exp[7:0], final_mant[6:0]};
                    end
                end
            end
        end
        stage5_valid <= stage4_valid;
        
        // Stage 4: Normalization preparation stage
        stage4_valid <= stage3_valid;
        stage4_special_case <= stage3_special_case;
        stage4_result <= stage3_result;
        stage4_sign_result <= stage3_sign_result;
        stage4_exp_result <= stage3_exp_result;
        
        if (stage3_valid && !stage3_special_case) begin
            if (stage3_mant_result == 16'h0) begin
                stage4_mant_result <= 16'h0;
            end else begin
                stage4_mant_result <= stage3_mant_result;
            end
        end
        
        // Stage 3: Mantissa multiplication stage
        stage3_valid <= stage2_valid;
        stage3_special_case <= stage2_special_case;
        stage3_result <= stage2_result;
        stage3_sign_result <= stage2_sign_result;
        
        if (stage2_valid && !stage2_special_case) begin
            // Exponent addition (subtract bias)
            stage3_exp_result <= stage2_exp_a + stage2_exp_b - 10'd127;
            
            // Mantissa multiplication
            stage3_mant_result <= stage2_mant_a * stage2_mant_b;
        end
        
        // Stage 2: Decomposition and preparation stage
        stage2_valid <= stage1_valid;
        stage2_special_case <= stage1_special_case;
        stage2_result <= stage1_result;
        
        if (stage1_valid && !stage1_special_case) begin
            // Extract fields
            sign_a_seq = stage1_bf16_a[15];
            exp_a_seq = stage1_bf16_a[14:7];
            mant_a_seq = stage1_bf16_a[6:0];
            sign_b_seq = stage1_bf16_b[15];
            exp_b_seq = stage1_bf16_b[14:7];
            mant_b_seq = stage1_bf16_b[6:0];
            
            // Calculate result sign
            stage2_sign_result <= sign_a_seq ^ sign_b_seq;
            
            // Handle denormal numbers and normalize mantissas
            stage2_exp_a <= adjust_exp_for_denormal(exp_a_seq, mant_a_seq);
            stage2_exp_b <= adjust_exp_for_denormal(exp_b_seq, mant_b_seq);
            stage2_mant_a <= normalize_mantissa(exp_a_seq, mant_a_seq);
            stage2_mant_b <= normalize_mantissa(exp_b_seq, mant_b_seq);
        end
        
        // Stage 1: Input and special case detection
        stage1_bf16_a <= bf16_a;
        stage1_bf16_b <= bf16_b;
        stage1_valid <= valid_in;
        stage1_special_case <= special_case_detected;
        stage1_result <= special_case_result;
    end
end

endmodule 