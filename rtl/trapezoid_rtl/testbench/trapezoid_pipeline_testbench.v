// Testbench for Trapezoid Pipeline
// Complete system test for the trapezoid matrix multiplication pipeline
`timescale 1ns/1ps

module trapezoid_pipeline_testbench;

    // Parameters
    parameter M = 4;
    parameter K = 16;
    parameter N = 8;
    parameter PE_NUM = 4;
    parameter CLK_PERIOD = 10; // 100MHz

    // Inputs
    reg clk;
    reg rst_n;
    reg valid_in;
    reg [K*16-1:0] A_values;              // Flat vector instead of matrix
    reg [16*32-1:0] B_values;             // Correct bit width (16*32)
    reg [16*16-1:0] B_col_indices;        // Correct bit width (16*16)
    reg [17*16-1:0] B_row_ptr;            // Correct bit width (17*16)
    reg [15:0] start_index;
    reg is_hbm;

    // Outputs
    wire valid_out;
    wire [M*N*16-1:0] C_values;           // Correct name (capital C)
    wire [31:0] cycle_count;              // Actual output from module

    // Instantiate the Unit Under Test (UUT)
    trapezoid_pipeline #(
        .M(M),
        .K(K),
        .N(N),
        .PE_NUM(PE_NUM)
    ) uut (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_in),
        .A_values(A_values),              // Correct signal name
        .B_values(B_values),
        .B_col_indices(B_col_indices),
        .B_row_ptr(B_row_ptr),
        .start_index(start_index),
        .is_hbm(is_hbm),
        .valid_out(valid_out),
        .C_values(C_values),              // Correct signal name
        .cycle_count(cycle_count)         // Actual output
    );

    // Clock generation
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    // Test vectors
    integer i, j, test_cycle_count;

    // Test sequence
    initial begin
        $dumpfile("trapezoid_pipeline_test.vcd");
        $dumpvars(0, trapezoid_pipeline_testbench);

        // Initialize inputs
        rst_n = 0;
        valid_in = 0;
        is_hbm = 0;
        start_index = 0;
        
        // Initialize matrices
        A_values = 0;
        B_values = 0;
        B_col_indices = 0;
        B_row_ptr = 0;

        // Reset sequence
        #(CLK_PERIOD*5);
        rst_n = 1;
        #(CLK_PERIOD*2);

        $display("==== Trapezoid Pipeline Testbench ====");
        $display("Parameters: M=%d, K=%d, N=%d, PE_NUM=%d", M, K, N, PE_NUM);

        // Test case 1: Simple dense matrix multiplication
        $display("\n--- Test 1: Dense Matrix Multiplication ---");
        
        // Set up A matrix (1x16, dense vector)
        A_values[15:0] = 16'h3F80; // 1.0 in BF16
        for (i = 1; i < K; i = i + 1) begin
            A_values[i*16 +: 16] = (i % 2) ? 16'h3F80 : 16'h3F00; // Alternating 1.0 and 0.5
        end

        // Set up B matrix (sparse, CSR format)
        // Simple identity-like pattern
        B_values[15:0] = 16'h3F80;   // B[0,0] = 1.0
        B_values[31:16] = 16'h3F00;  // B[0,1] = 0.5
        B_col_indices[15:0] = 16'd0;   // Column 0
        B_col_indices[31:16] = 16'd1;  // Column 1
        B_row_ptr[15:0] = 16'd0;       // Row 0 starts at index 0
        B_row_ptr[31:16] = 16'd2;      // Row 1 starts at index 2

        // Start computation
        @(posedge clk);
        valid_in = 1;
        #CLK_PERIOD;
        valid_in = 0;

        // Wait for completion
        test_cycle_count = 0;
        while (!valid_out && test_cycle_count < 100) begin
            @(posedge clk);
            test_cycle_count = test_cycle_count + 1;
        end

        if (valid_out) begin
            $display("Test 1 completed in %d cycles", test_cycle_count);
            $display("Module cycle count: %d", cycle_count);
            $display("Output C matrix values:");
            for (i = 0; i < M*N; i = i + 1) begin
                $display("  C[%d] = %h", i, C_values[i*16 +: 16]);
            end
        end else begin
            $display("Test 1 TIMEOUT - no valid output after %d cycles", test_cycle_count);
        end

        // Wait for pipeline to settle
        repeat(10) @(posedge clk);

        // Test case 2: HBM mode test
        $display("\n--- Test 2: HBM Mode ---");
        
        is_hbm = 1;
        start_index = 16'd100;
        
        // Simpler B data for HBM mode
        B_values[15:0] = 16'h4000;   // 2.0 in BF16
        B_col_indices[15:0] = 16'd0;
        B_row_ptr[15:0] = 16'd0;
        B_row_ptr[31:16] = 16'd1;

        @(posedge clk);
        valid_in = 1;
        #CLK_PERIOD;
        valid_in = 0;

        // Wait for completion
        test_cycle_count = 0;
        while (!valid_out && test_cycle_count < 100) begin
            @(posedge clk);
            test_cycle_count = test_cycle_count + 1;
        end

        if (valid_out) begin
            $display("Test 2 completed in %d cycles", test_cycle_count);
            $display("HBM mode output:");
            for (i = 0; i < 4; i = i + 1) begin
                $display("  C[%d] = %h", i, C_values[i*16 +: 16]);
            end
        end else begin
            $display("Test 2 TIMEOUT");
        end

        // Test case 3: Pipeline stress test
        $display("\n--- Test 3: Pipeline Stress Test ---");
        
        is_hbm = 0;
        
        // Multiple rapid inputs
        for (j = 0; j < 3; j = j + 1) begin
            $display("  Stress iteration %d", j);
            
            // Vary the input slightly
            A_values[15:0] = 16'h3F80 + j;
            B_values[15:0] = 16'h3F80 + j;
            
            @(posedge clk);
            valid_in = 1;
            #CLK_PERIOD;
            valid_in = 0;
            
            // Short wait between inputs
            repeat(5) @(posedge clk);
        end

        // Wait for all outputs
        test_cycle_count = 0;
        while (test_cycle_count < 200) begin
            @(posedge clk);
            test_cycle_count = test_cycle_count + 1;
            if (valid_out) begin
                $display("  Stress output at cycle %d", test_cycle_count);
            end
        end

        $display("\n--- All Tests Completed ---");
        repeat(10) @(posedge clk);
        $finish;
    end

    // Monitor key signals
    initial begin
        $monitor("Time=%t, clk=%b, rst_n=%b, valid_in=%b, valid_out=%b, cycle_count=%d",
                 $time, clk, rst_n, valid_in, valid_out, cycle_count);
    end

    // Timeout protection
    initial begin
        #(CLK_PERIOD * 1000);
        $display("GLOBAL TIMEOUT - Test did not complete");
        $finish;
    end

endmodule 