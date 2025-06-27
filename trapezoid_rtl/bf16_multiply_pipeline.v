// BF16 Multiplication Pipeline
// 5-stage pipeline for BF16 floating point multiplication
module bf16_multiply_pipeline (
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

    // Stage 2: Decomposition and sign calculation
    reg stage2_valid;
    reg stage2_special_case;
    reg [15:0] stage2_result;
    reg stage2_sign_result;
    reg [7:0] stage2_exp_a, stage2_exp_b;
    reg [7:0] stage2_mant_a, stage2_mant_b;

    // Stage 3: Exponent and mantissa multiplication
    reg stage3_valid;
    reg stage3_special_case;
    reg [15:0] stage3_result;
    reg stage3_sign_result;
    reg [8:0] stage3_exp_result; // 9 bits for overflow detection
    reg [15:0] stage3_mant_result; // 16 bits for 8x8 multiplication

    // Stage 4: Normalization preparation
    reg stage4_valid;
    reg stage4_special_case;
    reg [15:0] stage4_result;
    reg stage4_sign_result;
    reg [8:0] stage4_exp_result;
    reg [15:0] stage4_mant_result;

    // Special case detection function for multiplication
    function [16:0] check_special_cases_mul;
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
                check_special_cases_mul = {1'b1, NAN};
            // Check for infinity * zero = NaN
            else if ((exp_a == 8'hFF && (exp_b == 0 && mant_b == 0)) || 
                     (exp_b == 8'hFF && (exp_a == 0 && mant_a == 0)))
                check_special_cases_mul = {1'b1, NAN};
            // Check for infinity
            else if (exp_a == 8'hFF || exp_b == 8'hFF)
                check_special_cases_mul = {1'b1, {sign_a ^ sign_b, 8'hFF, 7'h00}};
            // Check for zero
            else if ((exp_a == 0 && mant_a == 0) || (exp_b == 0 && mant_b == 0))
                check_special_cases_mul = {1'b1, {sign_a ^ sign_b, 8'h00, 7'h00}};
            else
                check_special_cases_mul = {1'b0, 16'h0000};
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
            // Stage 5: Final normalization and output
            valid_out <= stage4_valid;
            if (stage4_valid) begin
                if (stage4_special_case) begin
                    result <= stage4_result;
                end else begin
                    // Final normalization
                    if (stage4_mant_result == 0) begin
                        result <= {stage4_sign_result, 8'h00, 7'h00}; // Zero
                    end else if (stage4_mant_result[15]) begin
                        // Mantissa overflow, right shift
                        if (stage4_exp_result >= 8'hFE) begin
                            // Overflow to infinity
                            result <= {stage4_sign_result, 8'hFF, 7'h00};
                        end else begin
                            // Round to nearest even
                            if (stage4_mant_result[8] && 
                                (stage4_mant_result[7:0] != 0 || stage4_mant_result[9])) begin
                                result <= {stage4_sign_result, stage4_exp_result[7:0] + 1, 
                                          stage4_mant_result[15:9] + 1};
                            end else begin
                                result <= {stage4_sign_result, stage4_exp_result[7:0] + 1, 
                                          stage4_mant_result[15:9]};
                            end
                        end
                    end else if (stage4_mant_result[14]) begin
                        // Normal case
                        if (stage4_exp_result == 0) begin
                            result <= {stage4_sign_result, 8'h00, stage4_mant_result[13:7]};
                        end else begin
                            // Round to nearest even
                            if (stage4_mant_result[7] && 
                                (stage4_mant_result[6:0] != 0 || stage4_mant_result[8])) begin
                                result <= {stage4_sign_result, stage4_exp_result[7:0], 
                                          stage4_mant_result[14:8] + 1};
                            end else begin
                                result <= {stage4_sign_result, stage4_exp_result[7:0], 
                                          stage4_mant_result[14:8]};
                            end
                        end
                    end else begin
                        // Need to find leading 1 and normalize
                        if (stage4_mant_result[13]) begin
                            result <= {stage4_sign_result, stage4_exp_result[7:0] - 1, 
                                      stage4_mant_result[12:6]};
                        end else if (stage4_mant_result[12]) begin
                            result <= {stage4_sign_result, stage4_exp_result[7:0] - 2, 
                                      stage4_mant_result[11:5]};
                        end else if (stage4_mant_result[11]) begin
                            result <= {stage4_sign_result, stage4_exp_result[7:0] - 3, 
                                      stage4_mant_result[10:4]};
                        end else if (stage4_mant_result[10]) begin
                            result <= {stage4_sign_result, stage4_exp_result[7:0] - 4, 
                                      stage4_mant_result[9:3]};
                        end else if (stage4_mant_result[9]) begin
                            result <= {stage4_sign_result, stage4_exp_result[7:0] - 5, 
                                      stage4_mant_result[8:2]};
                        end else if (stage4_mant_result[8]) begin
                            result <= {stage4_sign_result, stage4_exp_result[7:0] - 6, 
                                      stage4_mant_result[7:1]};
                        end else if (stage4_mant_result[7]) begin
                            result <= {stage4_sign_result, stage4_exp_result[7:0] - 7, 
                                      stage4_mant_result[6:0]};
                        end else begin
                            result <= {stage4_sign_result, 8'h00, 7'h00}; // Underflow to zero
                        end
                    end
                end
            end

            // Stage 4: Copy from stage 3
            stage4_valid <= stage3_valid;
            stage4_special_case <= stage3_special_case;
            stage4_result <= stage3_result;
            stage4_sign_result <= stage3_sign_result;
            stage4_exp_result <= stage3_exp_result;
            stage4_mant_result <= stage3_mant_result;

            // Stage 3: Multiplication
            stage3_valid <= stage2_valid;
            stage3_special_case <= stage2_special_case;
            stage3_result <= stage2_result;
            stage3_sign_result <= stage2_sign_result;
            
            if (stage2_valid && !stage2_special_case) begin
                // Calculate exponent (bias = 127)
                stage3_exp_result <= stage2_exp_a + stage2_exp_b - 127;
                
                // Multiply mantissas (8-bit x 8-bit = 16-bit)
                stage3_mant_result <= stage2_mant_a * stage2_mant_b;
            end

            // Stage 2: Decomposition
            stage2_valid <= stage1_valid;
            stage2_special_case <= stage1_special_case;
            stage2_result <= stage1_result;
            
            if (stage1_valid && !stage1_special_case) begin
                // Calculate result sign
                stage2_sign_result <= stage1_bf16_a[15] ^ stage1_bf16_b[15];
                
                // Handle denormalized numbers for A
                if (stage1_bf16_a[14:7] == 0) begin
                    stage2_exp_a <= 8'h01; // Minimum exponent
                    stage2_mant_a <= stage1_bf16_a[6:0]; // No implicit 1
                end else begin
                    stage2_exp_a <= stage1_bf16_a[14:7];
                    stage2_mant_a <= {1'b1, stage1_bf16_a[6:0]}; // Add implicit 1
                end
                
                // Handle denormalized numbers for B
                if (stage1_bf16_b[14:7] == 0) begin
                    stage2_exp_b <= 8'h01; // Minimum exponent
                    stage2_mant_b <= stage1_bf16_b[6:0]; // No implicit 1
                end else begin
                    stage2_exp_b <= stage1_bf16_b[14:7];
                    stage2_mant_b <= {1'b1, stage1_bf16_b[6:0]}; // Add implicit 1
                end
            end

            // Stage 1: Input and special case detection
            stage1_valid <= valid_in;
            if (valid_in) begin
                stage1_bf16_a <= bf16_a;
                stage1_bf16_b <= bf16_b;
                
                {stage1_special_case, stage1_result} <= check_special_cases_mul(bf16_a, bf16_b);
            end
        end
    end

endmodule 