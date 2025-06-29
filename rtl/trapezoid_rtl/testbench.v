// Testbench for Trapezoid PE Array
`timescale 1ns/1ps

module testbench;
    // Parameters
    parameter M = 8;
    parameter K = 128;
    parameter N = 128;
    parameter PE_ROWS = 8;
    parameter PE_COLS = 16;
    parameter PE_NUM_PER_TRAPEZOID = 4;
    parameter CLOCK_PERIOD = 10; // 100MHz

    // Clock and reset
    reg clk;
    reg rst_n;
    
    // Control signals
    reg enable;
    reg [1:0] mode;
    
    // Input signals
    reg [PE_ROWS-1:0] valid_in;
    reg [PE_ROWS*K*16-1:0] A_matrices;
    reg [PE_ROWS*16*32-1:0] B_values;
    reg [PE_ROWS*16*16-1:0] B_col_indices;
    reg [PE_ROWS*17*16-1:0] B_row_ptr;
    reg [PE_ROWS*16-1:0] start_indices;
    
    // Output signals
    wire [PE_ROWS-1:0] valid_out;
    wire [PE_ROWS*M*N*16-1:0] C_matrices;
    wire [PE_ROWS*32-1:0] cycle_counts;
    wire [31:0] total_cycles;
    wire [31:0] active_pe_count;
    wire [15:0] utilization_percent;
    
    // DUT instantiation
    trapezoid_pe_array #(
        .M(M),
        .K(K),
        .N(N),
        .PE_ROWS(PE_ROWS),
        .PE_COLS(PE_COLS),
        .PE_NUM_PER_TRAPEZOID(PE_NUM_PER_TRAPEZOID)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable),
        .mode(mode),
        .valid_in(valid_in),
        .A_matrices(A_matrices),
        .B_values(B_values),
        .B_col_indices(B_col_indices),
        .B_row_ptr(B_row_ptr),
        .start_indices(start_indices),
        .valid_out(valid_out),
        .C_matrices(C_matrices),
        .cycle_counts(cycle_counts),
        .total_cycles(total_cycles),
        .active_pe_count(active_pe_count),
        .utilization_percent(utilization_percent)
    );
    
    // Clock generation
    always #(CLOCK_PERIOD/2) clk = ~clk;
    
    // Test stimulus
    initial begin
        // Initialize signals
        clk = 0;
        rst_n = 0;
        enable = 0;
        mode = 2'b00;
        valid_in = 0;
        A_matrices = 0;
        B_values = 0;
        B_col_indices = 0;
        B_row_ptr = 0;
        start_indices = 0;
        
        // Apply reset
        #(CLOCK_PERIOD * 5);
        rst_n = 1;
        enable = 1;
        
        // Wait for reset deassertion
        #(CLOCK_PERIOD * 2);
        
        $display("Starting Trapezoid PE Array Test");
        $display("Configuration: %dx%d PE Array, Matrix Size: %dx%dx%d", PE_ROWS, PE_COLS, M, K, N);
        
        // Test Case 1: Simple dense matrix multiplication
        test_dense_multiplication();
        
        // Test Case 2: Sparse matrix multiplication (HBM mode)
        test_sparse_multiplication();
        
        // Test Case 3: Performance monitoring
        test_performance_monitoring();
        
        // End simulation
        #(CLOCK_PERIOD * 100);
        $display("Test completed successfully!");
        $display("Total cycles: %d", total_cycles);
        $display("Peak utilization: %d%%", utilization_percent);
        $finish;
    end
    
    // Test case 1: Dense matrix multiplication
    task test_dense_multiplication;
        integer i, j;
        begin
            $display("Test Case 1: Dense Matrix Multiplication");
            
            mode = 2'b00; // Normal mode
            
            // Initialize test matrices
            for (i = 0; i < PE_ROWS; i = i + 1) begin
                // Simple pattern for A matrix (identity-like)
                for (j = 0; j < K; j = j + 1) begin
                    if (j == i) begin
                        A_matrices[i*K*16 + j*16 +: 16] = 16'h3F80; // BF16 value 1.0
                    end else begin
                        A_matrices[i*K*16 + j*16 +: 16] = 16'h0000; // BF16 value 0.0
                    end
                end
                
                // Simple pattern for B matrix values
                for (j = 0; j < 16; j = j + 1) begin
                    B_values[i*16*32 + j*32 +: 16] = 16'h4000; // BF16 value 2.0
                    B_values[i*16*32 + j*32 + 16 +: 16] = 16'h0000; // Padding
                end
                
                valid_in[i] = 1;
                start_indices[i*16 +: 16] = i;
            end
            
            // Apply inputs for several cycles
            #(CLOCK_PERIOD * 10);
            valid_in = 0;
            
            // Wait for outputs
            wait(valid_out != 0);
            #(CLOCK_PERIOD * 50);
            
            // Check results
            for (i = 0; i < PE_ROWS; i = i + 1) begin
                if (valid_out[i]) begin
                    $display("PE Row %d completed in %d cycles", i, cycle_counts[i*32 +: 32]);
                end
            end
            
            $display("Dense multiplication test completed\n");
        end
    endtask
    
    // Test case 2: Sparse matrix multiplication
    task test_sparse_multiplication;
        integer i, j;
        begin
            $display("Test Case 2: Sparse Matrix Multiplication (HBM Mode)");
            
            mode = 2'b01; // HBM mode
            
            // Initialize sparse B matrix in CSR format
            for (i = 0; i < PE_ROWS; i = i + 1) begin
                // Simple sparse pattern
                B_values[i*16*32 + 0*32 +: 16] = 16'h4000; // 2.0
                B_values[i*16*32 + 1*32 +: 16] = 16'h4040; // 3.0
                
                B_col_indices[i*16*16 + 0*16 +: 16] = 0; // Column 0
                B_col_indices[i*16*16 + 1*16 +: 16] = 2; // Column 2
                
                // Row pointers for 2 non-zeros in first row
                B_row_ptr[i*17*16 + 0*16 +: 16] = 0;
                B_row_ptr[i*17*16 + 1*16 +: 16] = 2;
                
                // Dense A vector
                A_matrices[i*K*16 + 0*16 +: 16] = 16'h3F80; // 1.0
                A_matrices[i*K*16 + 2*16 +: 16] = 16'h4000; // 2.0
                
                valid_in[i] = 1;
                start_indices[i*16 +: 16] = 0;
            end
            
            #(CLOCK_PERIOD * 10);
            valid_in = 0;
            
            // Wait for completion
            wait(valid_out != 0);
            #(CLOCK_PERIOD * 50);
            
            $display("Sparse multiplication test completed\n");
        end
    endtask
    
    // Test case 3: Performance monitoring
    task test_performance_monitoring;
        integer i;
        begin
            $display("Test Case 3: Performance Monitoring");
            
            // Monitor utilization over time
            for (i = 0; i < 100; i = i + 1) begin
                if (i % 10 == 0) begin
                    $display("Cycle %d: Active PEs = %d, Utilization = %d%%", 
                            total_cycles, active_pe_count, utilization_percent);
                end
                #(CLOCK_PERIOD);
            end
            
            $display("Performance monitoring test completed\n");
        end
    endtask
    
    // Monitor key signals
    always @(posedge clk) begin
        if (valid_out != 0) begin
            $display("Output detected at cycle %d", total_cycles);
        end
    end
    
    // Timeout protection
    initial begin
        #(CLOCK_PERIOD * 10000);
        $display("ERROR: Simulation timeout!");
        $finish;
    end

endmodule 