// FP32 to BF16 Pipeline Converter
// 3-stage pipeline for converting 32-bit floating point to BF16 format

module fp32_to_bf16_pipeline (
    input wire clk,
    input wire rst_n,
    
    // Input interface
    input wire [31:0] fp32_in,
    input wire valid_in,
    
    // Output interface
    output reg [15:0] bf16_out,
    output reg valid_out
);

// Pipeline stage registers
reg [31:0] stage1_fp32;
reg stage1_valid;

reg [31:0] stage2_fp32;
reg [15:0] stage2_high_bits;
reg [15:0] stage2_low_bits;
reg stage2_fp32_sign;
reg [7:0] stage2_fp32_exponent;
reg [22:0] stage2_fp32_mantissa;
reg stage2_valid;

reg [15:0] stage3_bf16;
reg stage3_valid;

// Always block for pipeline advancement
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        // Reset all pipeline registers
        stage1_fp32 <= 32'h0;
        stage1_valid <= 1'b0;
        
        stage2_fp32 <= 32'h0;
        stage2_high_bits <= 16'h0;
        stage2_low_bits <= 16'h0;
        stage2_fp32_sign <= 1'b0;
        stage2_fp32_exponent <= 8'h0;
        stage2_fp32_mantissa <= 23'h0;
        stage2_valid <= 1'b0;
        
        stage3_bf16 <= 16'h0;
        stage3_valid <= 1'b0;
        
        bf16_out <= 16'h0;
        valid_out <= 1'b0;
    end else begin
        // Stage 3: Rounding stage
        if (stage2_valid) begin
            if (stage2_fp32_exponent == 8'hFF) begin
                // Handle infinity and NaN
                if (stage2_fp32_mantissa == 23'h0) begin
                    // Infinity
                    stage3_bf16 <= stage2_high_bits;
                end else begin
                    // NaN
                    stage3_bf16 <= {stage2_fp32_sign, 8'hFF, stage2_fp32_mantissa[22:16]} | 16'h1;
                end
            end else begin
                // Round to nearest even
                if (stage2_low_bits > 16'h8000 || 
                    (stage2_low_bits == 16'h8000 && stage2_high_bits[0])) begin
                    stage3_bf16 <= stage2_high_bits + 16'h1;
                end else begin
                    stage3_bf16 <= stage2_high_bits;
                end
            end
        end
        stage3_valid <= stage2_valid;
        
        // Stage 2: Decomposition stage
        stage2_fp32 <= stage1_fp32;
        stage2_high_bits <= stage1_fp32[31:16];
        stage2_low_bits <= stage1_fp32[15:0];
        stage2_fp32_sign <= stage1_fp32[31];
        stage2_fp32_exponent <= stage1_fp32[30:23];
        stage2_fp32_mantissa <= stage1_fp32[22:0];
        stage2_valid <= stage1_valid;
        
        // Stage 1: Input stage
        stage1_fp32 <= fp32_in;
        stage1_valid <= valid_in;
        
        // Output assignment
        bf16_out <= stage3_bf16;
        valid_out <= stage3_valid;
    end
end

endmodule 