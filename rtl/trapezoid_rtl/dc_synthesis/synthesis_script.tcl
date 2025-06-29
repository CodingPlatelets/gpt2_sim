# Trapezoid Pipeline Synthesis Script for Synopsys Design Compiler
# Target: 14nm process technology
# Author: AI Assistant
# Date: 2024

#===============================================================================
# Setup and Configuration
#===============================================================================

# Remove existing designs
remove_design -all

# Set up libraries and search paths
source file_list.tcl

# Add search paths
set search_path [list . .. /tools/synopsys/libraries/14nm $search_path]

# Set target and link libraries
set target_library $target_library_list
set link_library $link_library
set symbol_library $symbol_library

# Create and set work library
define_design_lib $work_lib -path ./WORK
set current_design ""

#===============================================================================
# Read Design Files
#===============================================================================

echo "Reading Verilog files..."
foreach file $file_list {
    echo "Reading file: $file"
    read_verilog $file
}

# Set current design
current_design $top_module
link

#===============================================================================
# Design Setup and Constraints
#===============================================================================

echo "Setting up design constraints..."

# Source the constraints file
source constraints.sdc

# Check design
echo "Checking design..."
check_design > reports/check_design.rpt

# Report design statistics before synthesis
echo "Reporting pre-synthesis statistics..."
report_hierarchy > reports/hierarchy_pre.rpt
report_area -hierarchy > reports/area_pre.rpt

#===============================================================================
# Synthesis
#===============================================================================

echo "Starting synthesis..."

# Set synthesis options
set compile_seqmap_propagate_constants false
set compile_delete_unloaded_sequential_cells false
set compile_register_replication false

# Ultra mode compilation for best QoR
compile_ultra -gate_clock -retime

#===============================================================================
# Post-Synthesis Analysis and Reports
#===============================================================================

echo "Generating post-synthesis reports..."

# Create reports directory
file mkdir reports

# Timing reports
report_timing -max_paths 10 -nworst 2 -delay_type max > reports/timing_max.rpt
report_timing -max_paths 10 -nworst 2 -delay_type min > reports/timing_min.rpt
report_constraint -all_violators > reports/constraints.rpt

# Area reports
report_area -hierarchy > reports/area_hierarchy.rpt
report_area -designware > reports/area_designware.rpt
report_cell > reports/cell_usage.rpt

# Power reports
report_power -hierarchy > reports/power.rpt

# QoR summary
report_qor > reports/qor_summary.rpt

# Design statistics
report_design > reports/design_stats.rpt
report_compile_options > reports/compile_options.rpt

#===============================================================================
# Write Out Synthesized Design
#===============================================================================

echo "Writing out synthesized netlist..."

# Create output directory
file mkdir outputs

# Write netlist
write -format verilog -hierarchy -output outputs/${top_module}_syn.v
write -format ddc -hierarchy -output outputs/${top_module}_syn.ddc

# Write constraints
write_sdc outputs/${top_module}_syn.sdc

# Write delay format
write_sdf outputs/${top_module}_syn.sdf

#===============================================================================
# Summary and Final Reports
#===============================================================================

echo "==================================================================="
echo "Synthesis Summary for $top_module"
echo "==================================================================="

# Print area summary
echo "AREA SUMMARY:"
report_area

echo ""
echo "TIMING SUMMARY:"
report_timing -delay_type max -max_paths 1

echo ""
echo "POWER SUMMARY:"
report_power

echo ""
echo "QOR SUMMARY:"
report_qor

echo "==================================================================="
echo "Synthesis completed successfully!"
echo "Check reports/ directory for detailed analysis"
echo "Netlist available in outputs/ directory"
echo "==================================================================="

# Save final design
save_design outputs/${top_module}_final.ddc

exit 