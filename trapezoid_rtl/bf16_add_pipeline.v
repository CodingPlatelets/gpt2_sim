// BF16 Addition Pipeline
// 5-stage pipeline for BF16 floating point addition
module bf16_add_pipeline (
    input wire clk,
    input wire rst_n,
    input wire valid_in,
    input wire [15:0] bf16_a,
    input wire [15:0] bf16_b,
    output reg valid_out,
    output reg [15:0] result
);

    // BF16 format: [15:sign][14:7:exponent][6:0:mantissa]
    parameter POS_INF = 16'h7F80;
    parameter NEG_INF = 16'hFF80;
    parameter NAN = 16'h7FC0;

    // Stage 1: Input and special case detection
    reg stage1_valid;
    reg [15:0] stage1_bf16_a, stage1_bf16_b;
    reg stage1_special_case;
    reg [15:0] stage1_result;

    // Stage 2: Decomposition and preparation
    reg stage2_valid;
    reg stage2_special_case;
    reg [15:0] stage2_result;
    reg stage2_sign_a, stage2_sign_b;
    reg [7:0] stage2_exp_a, stage2_exp_b;
    reg [7:0] stage2_mant_a, stage2_mant_b;

    // Stage 3: Alignment
    reg stage3_valid;
    reg stage3_special_case;
    reg [15:0] stage3_result;
    reg stage3_sign_a, stage3_sign_b;
    reg [7:0] stage3_exp_result;
    reg [8:0] stage3_mant_a, stage3_mant_b; // 9 bits for overflow

    // Stage 4: Addition/Subtraction
    reg stage4_valid;
    reg stage4_special_case;
    reg [15:0] stage4_result;
    reg stage4_sign_result;
    reg [7:0] stage4_exp_result;
    reg [9:0] stage4_mant_result; // 10 bits for overflow

    // Special case detection function
    function [16:0] check_special_cases;
        input [15:0] a, b;
        reg sign_a, sign_b;
        reg [7:0] exp_a, exp_b;
        reg [6:0] mant_a, mant_b;
        begin
            sign_a = a[15];
            exp_a = a[14:7];
            mant_a = a[6:0];
            sign_b = b[15];
            exp_b = b[14:7];
            mant_b = b[6:0];

            // Check for NaN
            if ((exp_a == 8'hFF && mant_a != 0) || (exp_b == 8'hFF && mant_b != 0))
                check_special_cases = {1'b1, NAN};
            // Check for infinity
            else if (exp_a == 8'hFF) begin
                if (exp_b == 8'hFF && sign_a != sign_b)
                    check_special_cases = {1'b1, NAN};
                else
                    check_special_cases = {1'b1, {sign_a, 8'hFF, 7'h00}};
            end
            else if (exp_b == 8'hFF)
                check_special_cases = {1'b1, {sign_b, 8'hFF, 7'h00}};
            // Check for zero
            else if ((exp_a == 0 && mant_a == 0) && (exp_b == 0 && mant_b == 0)) begin
                if (sign_a == sign_b)
                    check_special_cases = {1'b1, {sign_a, 8'h00, 7'h00}};
                else
                    check_special_cases = {1'b1, 16'h0000};
            end
            else if (exp_a == 0 && mant_a == 0)
                check_special_cases = {1'b1, b};
            else if (exp_b == 0 && mant_b == 0)
                check_special_cases = {1'b1, a};
            else
                check_special_cases = {1'b0, 16'h0000};
        end
    endfunction

    // Normalize mantissa function
    function [15:0] normalize_mantissa;
        input [6:0] mant;
        input [7:0] exp;
        reg [2:0] leading_zeros;
        begin
            if (mant == 0) begin
                normalize_mantissa = {exp, 1'b0, 8'h80};
            end else begin
                leading_zeros = 0;
                if (mant[6] == 0) begin
                    if (mant[5:0] == 0) leading_zeros = 7;
                    else if (mant[5] == 0) begin
                        if (mant[4:0] == 0) leading_zeros = 6;
                        else if (mant[4] == 0) begin
                            if (mant[3:0] == 0) leading_zeros = 5;
                            else if (mant[3] == 0) begin
                                if (mant[2:0] == 0) leading_zeros = 4;
                                else if (mant[2] == 0) begin
                                    if (mant[1:0] == 0) leading_zeros = 3;
                                    else if (mant[1] == 0) leading_zeros = 2;
                                    else leading_zeros = 1;
                                end else leading_zeros = 1;
                            end else leading_zeros = 1;
                        end else leading_zeros = 1;
                    end else leading_zeros = 1;
                end
                normalize_mantissa = {exp - leading_zeros, 1'b1, mant << leading_zeros};
            end
        end
    endfunction

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            stage1_valid <= 0;
            stage2_valid <= 0;
            stage3_valid <= 0;
            stage4_valid <= 0;
            valid_out <= 0;
            result <= 0;
        end else begin
            // Stage 5: Output
            valid_out <= stage4_valid;
            if (stage4_valid) begin
                if (stage4_special_case) begin
                    result <= stage4_result;
                end else begin
                    // Normalization and composition
                    if (stage4_mant_result == 0) begin
                        result <= 16'h0000; // Positive zero
                    end else begin
                        // Handle overflow
                        if (stage4_mant_result[9]) begin
                            // Right shift mantissa and increment exponent
                            if (stage4_exp_result == 8'hFE) begin
                                // Overflow to infinity
                                result <= {stage4_sign_result, 8'hFF, 7'h00};
                            end else begin
                                result <= {stage4_sign_result, stage4_exp_result + 1, stage4_mant_result[8:2]};
                            end
                        end else if (stage4_mant_result[8]) begin
                            // Normal case
                            result <= {stage4_sign_result, stage4_exp_result, stage4_mant_result[7:1]};
                        end else begin
                            // Need to normalize
                            // Find leading one and adjust
                            if (stage4_mant_result[7]) begin
                                result <= {stage4_sign_result, stage4_exp_result - 1, stage4_mant_result[6:0]};
                            end else if (stage4_mant_result[6]) begin
                                result <= {stage4_sign_result, stage4_exp_result - 2, stage4_mant_result[5:0], 1'b0};
                            end else if (stage4_mant_result[5]) begin
                                result <= {stage4_sign_result, stage4_exp_result - 3, stage4_mant_result[4:0], 2'b0};
                            end else if (stage4_mant_result[4]) begin
                                result <= {stage4_sign_result, stage4_exp_result - 4, stage4_mant_result[3:0], 3'b0};
                            end else if (stage4_mant_result[3]) begin
                                result <= {stage4_sign_result, stage4_exp_result - 5, stage4_mant_result[2:0], 4'b0};
                            end else if (stage4_mant_result[2]) begin
                                result <= {stage4_sign_result, stage4_exp_result - 6, stage4_mant_result[1:0], 5'b0};
                            end else if (stage4_mant_result[1]) begin
                                result <= {stage4_sign_result, stage4_exp_result - 7, stage4_mant_result[0], 6'b0};
                            end else begin
                                result <= 16'h0000; // Underflow to zero
                            end
                        end
                    end
                end
            end

            // Stage 4: Addition/Subtraction
            stage4_valid <= stage3_valid;
            stage4_special_case <= stage3_special_case;
            stage4_result <= stage3_result;
            
            if (stage3_valid && !stage3_special_case) begin
                if (stage3_sign_a == stage3_sign_b) begin
                    // Same sign, add
                    stage4_mant_result <= stage3_mant_a + stage3_mant_b;
                    stage4_sign_result <= stage3_sign_a;
                end else begin
                    // Different signs, subtract
                    if (stage3_mant_a >= stage3_mant_b) begin
                        stage4_mant_result <= stage3_mant_a - stage3_mant_b;
                        stage4_sign_result <= stage3_sign_a;
                    end else begin
                        stage4_mant_result <= stage3_mant_b - stage3_mant_a;
                        stage4_sign_result <= stage3_sign_b;
                    end
                end
                stage4_exp_result <= stage3_exp_result;
            end

            // Stage 3: Alignment
            stage3_valid <= stage2_valid;
            stage3_special_case <= stage2_special_case;
            stage3_result <= stage2_result;
            
            if (stage2_valid && !stage2_special_case) begin
                stage3_sign_a <= stage2_sign_a;
                stage3_sign_b <= stage2_sign_b;
                
                // Align exponents
                if (stage2_exp_a > stage2_exp_b) begin
                    stage3_exp_result <= stage2_exp_a;
                    stage3_mant_a <= {1'b0, stage2_mant_a};
                    if (stage2_exp_a - stage2_exp_b > 8) begin
                        stage3_mant_b <= 9'h000;
                    end else begin
                        stage3_mant_b <= {1'b0, stage2_mant_b} >> (stage2_exp_a - stage2_exp_b);
                    end
                end else if (stage2_exp_b > stage2_exp_a) begin
                    stage3_exp_result <= stage2_exp_b;
                    stage3_mant_b <= {1'b0, stage2_mant_b};
                    if (stage2_exp_b - stage2_exp_a > 8) begin
                        stage3_mant_a <= 9'h000;
                    end else begin
                        stage3_mant_a <= {1'b0, stage2_mant_a} >> (stage2_exp_b - stage2_exp_a);
                    end
                end else begin
                    stage3_exp_result <= stage2_exp_a;
                    stage3_mant_a <= {1'b0, stage2_mant_a};
                    stage3_mant_b <= {1'b0, stage2_mant_b};
                end
            end

            // Stage 2: Decomposition
            stage2_valid <= stage1_valid;
            stage2_special_case <= stage1_special_case;
            stage2_result <= stage1_result;
            
            if (stage1_valid && !stage1_special_case) begin
                // Decompose A
                stage2_sign_a <= stage1_bf16_a[15];
                stage2_exp_a <= stage1_bf16_a[14:7];
                if (stage1_bf16_a[14:7] == 0) begin
                    // Denormalized number
                    stage2_mant_a <= stage1_bf16_a[6:0];
                    // For simplicity, treat as very small normal number
                    stage2_exp_a <= 8'h01;
                end else begin
                    // Add implicit 1
                    stage2_mant_a <= {1'b1, stage1_bf16_a[6:0]};
                end
                
                // Decompose B
                stage2_sign_b <= stage1_bf16_b[15];
                stage2_exp_b <= stage1_bf16_b[14:7];
                if (stage1_bf16_b[14:7] == 0) begin
                    // Denormalized number
                    stage2_mant_b <= stage1_bf16_b[6:0];
                    stage2_exp_b <= 8'h01;
                end else begin
                    // Add implicit 1
                    stage2_mant_b <= {1'b1, stage1_bf16_b[6:0]};
                end
            end

            // Stage 1: Input and special case detection
            stage1_valid <= valid_in;
            if (valid_in) begin
                stage1_bf16_a <= bf16_a;
                stage1_bf16_b <= bf16_b;
                
                {stage1_special_case, stage1_result} <= check_special_cases(bf16_a, bf16_b);
            end
        end
    end

endmodule 