// Trapezoid PE Array Top Module
// 8x128 PE Array for Sparse Matrix Multiplication with BF16 precision
module trapezoid_pe_array #(
    parameter M = 8,
    parameter K = 128, 
    parameter N = 128,
    parameter PE_ROWS = 8,
    parameter PE_COLS = 16,  // Total 128 PEs = 8*16
    parameter PE_NUM_PER_TRAPEZOID = 4
)(
    input wire clk,
    input wire rst_n,
    
    // Control signals
    input wire enable,
    input wire [1:0] mode,  // 00: normal, 01: HBM, 10: batch, 11: weight_sharing
    
    // Input data interface
    input wire [PE_ROWS-1:0] valid_in,
    input wire [PE_ROWS*K*16-1:0] A_matrices,     // Dense input matrices A
    input wire [PE_ROWS*16*32-1:0] B_values,      // Sparse matrix B values
    input wire [PE_ROWS*16*16-1:0] B_col_indices, // Column indices for sparse B
    input wire [PE_ROWS*17*16-1:0] B_row_ptr,     // Row pointers for sparse B
    input wire [PE_ROWS*16-1:0] start_indices,
    
    // Output interface
    output wire [PE_ROWS-1:0] valid_out,
    output wire [PE_ROWS*M*N*16-1:0] C_matrices,  // Output result matrices
    output wire [PE_ROWS*32-1:0] cycle_counts,
    
    // Performance monitoring
    output wire [31:0] total_cycles,
    output wire [31:0] active_pe_count,
    output wire [15:0] utilization_percent
);

    // Internal PE array signals
    genvar i, j;
    
    // Trapezoid pipeline instances (one per PE row)
    wire [PE_ROWS-1:0] trapezoid_valid_out;
    wire [PE_ROWS*M*N*16-1:0] trapezoid_C_values;
    wire [PE_ROWS*32-1:0] trapezoid_cycle_counts;
    
    // Performance monitoring registers
    reg [31:0] global_cycle_counter;
    reg [31:0] active_pe_counter;
    reg [PE_ROWS-1:0] pe_active_status;
    
    // Generate PE array
    generate
        for (i = 0; i < PE_ROWS; i = i + 1) begin : pe_row_gen
            trapezoid_pipeline #(
                .M(M),
                .K(K),
                .N(N),
                .PE_NUM(PE_NUM_PER_TRAPEZOID)
            ) trapezoid_inst (
                .clk(clk),
                .rst_n(rst_n && enable),
                
                // Input interface
                .valid_in(valid_in[i]),
                .A_values(A_matrices[i*K*16 +: K*16]),
                .B_values(B_values[i*16*32 +: 16*32]),
                .B_col_indices(B_col_indices[i*16*16 +: 16*16]),
                .B_row_ptr(B_row_ptr[i*17*16 +: 17*16]),
                .start_index(start_indices[i*16 +: 16]),
                .is_hbm(mode == 2'b01),
                
                // Output interface
                .valid_out(trapezoid_valid_out[i]),
                .C_values(trapezoid_C_values[i*M*N*16 +: M*N*16]),
                .cycle_count(trapezoid_cycle_counts[i*32 +: 32])
            );
        end
    endgenerate
    
    // Output assignments
    assign valid_out = trapezoid_valid_out;
    assign C_matrices = trapezoid_C_values;
    assign cycle_counts = trapezoid_cycle_counts;
    assign total_cycles = global_cycle_counter;
    assign active_pe_count = active_pe_counter;
    
    // Calculate utilization percentage
    assign utilization_percent = (active_pe_counter * 100) / PE_ROWS;
    
    // Performance monitoring logic
    integer k;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            global_cycle_counter <= 0;
            active_pe_counter <= 0;
            pe_active_status <= 0;
        end else if (enable) begin
            global_cycle_counter <= global_cycle_counter + 1;
            
            // Update PE active status
            pe_active_status <= valid_in | trapezoid_valid_out;
            
            // Count active PEs
            active_pe_counter <= 0;
            for (k = 0; k < PE_ROWS; k = k + 1) begin
                if (pe_active_status[k]) begin
                    active_pe_counter <= active_pe_counter + 1;
                end
            end
        end
    end
    
    // Synthesis directives for optimal area/timing
    // synthesis translate_off
    initial begin
        $display("Trapezoid PE Array initialized with:");
        $display("  PE Array Size: %d x %d", PE_ROWS, PE_COLS);
        $display("  Matrix Dimensions: M=%d, K=%d, N=%d", M, K, N);
        $display("  Data Format: BF16 (16-bit)");
        $display("  Total PEs: %d", PE_ROWS * PE_COLS);
    end
    // synthesis translate_on

endmodule 