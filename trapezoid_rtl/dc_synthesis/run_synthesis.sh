#!/bin/bash

# Trapezoid Pipeline DC Synthesis Run Script
# This script sets up the environment and runs Synopsys Design Compiler

#===============================================================================
# Environment Setup
#===============================================================================

# Set Synopsys tools path (modify according to your installation)
export SYNOPSYS_ROOT="/tools/synopsys"
export DC_SHELL="$SYNOPSYS_ROOT/syn/bin/dc_shell"

# Add Synopsys tools to PATH
export PATH="$SYNOPSYS_ROOT/syn/bin:$PATH"

# Set library paths (modify according to your 14nm library location)
export LIBRARY_PATH="/tools/synopsys/libraries/14nm"

# Set up licensing (modify according to your license server)
export LM_LICENSE_FILE="27020@license-server:$LM_LICENSE_FILE"

#===============================================================================
# Pre-synthesis Setup
#===============================================================================

echo "============================================================"
echo "Trapezoid Pipeline DC Synthesis Setup"
echo "============================================================"

# Create necessary directories
mkdir -p reports
mkdir -p outputs
mkdir -p logs

# Clean previous runs
echo "Cleaning previous synthesis results..."
rm -rf reports/*
rm -rf outputs/*
rm -rf WORK
rm -rf logs/*

#===============================================================================
# Check Prerequisites
#===============================================================================

echo "Checking prerequisites..."

# Check if DC is available
if ! command -v dc_shell &> /dev/null; then
    echo "ERROR: dc_shell not found in PATH"
    echo "Please check your Synopsys installation and update SYNOPSYS_ROOT"
    exit 1
fi

# Check if 14nm libraries exist
if [ ! -d "$LIBRARY_PATH" ]; then
    echo "WARNING: 14nm library path not found: $LIBRARY_PATH"
    echo "Please update LIBRARY_PATH in this script"
    echo "Continuing with default library settings..."
fi

# Check if all Verilog files exist
echo "Checking Verilog source files..."
for file in "../bf16_add_pipeline.v" "../bf16_multiply_pipeline.v" "../compute_units.v" "../add_tree.v" "../trapezoid_pipeline.v" "../trapezoid_pe_array.v"; do
    if [ ! -f "$file" ]; then
        echo "ERROR: Source file not found: $file"
        exit 1
    fi
    echo "  ✓ Found: $file"
done

#===============================================================================
# Run Synthesis
#===============================================================================

echo ""
echo "============================================================"
echo "Starting DC Synthesis..."
echo "============================================================"

# Record start time
start_time=$(date +%s)

# Run DC synthesis
echo "Launching Design Compiler..."
$DC_SHELL -f synthesis_script.tcl 2>&1 | tee logs/synthesis.log

# Check synthesis result
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "Synthesis completed successfully!"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "ERROR: Synthesis failed!"
    echo "Check logs/synthesis.log for details"
    echo "============================================================"
    exit 1
fi

#===============================================================================
# Post-synthesis Summary
#===============================================================================

# Record end time
end_time=$(date +%s)
duration=$((end_time - start_time))

echo ""
echo "============================================================"
echo "Synthesis Summary"
echo "============================================================"
echo "Runtime: ${duration} seconds"
echo ""

# Display key results if available
if [ -f "reports/area_hierarchy.rpt" ]; then
    echo "Area Summary:"
    echo "-------------"
    grep -A 10 "Total cell area" reports/area_hierarchy.rpt 2>/dev/null || echo "Area report not available"
    echo ""
fi

if [ -f "reports/timing_max.rpt" ]; then
    echo "Timing Summary:"
    echo "---------------"
    grep -A 5 "slack" reports/timing_max.rpt 2>/dev/null || echo "Timing report not available"
    echo ""
fi

echo "Generated Files:"
echo "----------------"
echo "Reports: reports/"
ls -la reports/ 2>/dev/null || echo "No reports generated"
echo ""
echo "Outputs: outputs/"
ls -la outputs/ 2>/dev/null || echo "No outputs generated"
echo ""

echo "============================================================"
echo "Synthesis flow completed!"
echo "Review reports/ directory for detailed analysis"
echo "============================================================" 