# Design Compiler Synthesis Script for Trapezoid Pipeline

# Setup library paths (adjust based on your PDK)
set_app_var search_path [list . /path/to/library/]
set_app_var target_library [list your_target_lib.db]
set_app_var link_library [list * your_target_lib.db]

# Read design
read_verilog trapezoid_pipeline.v

# Set current design
current_design trapezoid_pipeline

# Create clock with 10ns period (100MHz)
create_clock -name "clk" -period 10.0 [get_ports clk]
set_clock_uncertainty 0.5 [get_clocks clk]

# Set input/output delays
set_input_delay 2.0 -clock clk [all_inputs]
set_output_delay 2.0 -clock clk [all_outputs]

# Set drive strengths
set_driving_cell -lib_cell BUFX1 [all_inputs]
set_load 0.1 [all_outputs]

# Compile the design
compile_ultra -gate_clock

# Generate reports
report_area > reports/area_report.txt
report_timing > reports/timing_report.txt
report_power > reports/power_report.txt
report_qor > reports/qor_report.txt

# Write out netlist
write -format verilog -hierarchy -output netlists/trapezoid_pipeline_syn.v

# Display summary
puts "================================================"
puts "Synthesis Complete!"
puts "Check reports/ directory for detailed analysis"
puts "Total area: [get_attribute [current_design] area] um^2"
puts "Critical path delay: [get_attribute [get_timing_paths] slack] ns"
puts "================================================"