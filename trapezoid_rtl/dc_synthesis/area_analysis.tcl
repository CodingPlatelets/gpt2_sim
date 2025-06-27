# Area Analysis Script for Trapezoid Pipeline
# Detailed area breakdown and analysis

proc analyze_area_breakdown {} {
    echo "==================================================================="
    echo "Detailed Area Analysis for Trapezoid Pipeline"
    echo "==================================================================="
    
    # Overall area summary
    echo "\n1. OVERALL AREA SUMMARY:"
    echo "========================="
    report_area
    
    # Hierarchical area breakdown
    echo "\n2. HIERARCHICAL AREA BREAKDOWN:"
    echo "==============================="
    report_area -hierarchy -all
    
    # Module-specific analysis
    echo "\n3. MODULE-SPECIFIC ANALYSIS:"
    echo "============================"
    
    # BF16 computation units analysis
    echo "\n3.1 BF16 Computation Units:"
    echo "----------------------------"
    set bf16_add_instances [get_cells -hier -filter "ref_name =~ *bf16_add*"]
    set bf16_mul_instances [get_cells -hier -filter "ref_name =~ *bf16_mul*"]
    
    if {[sizeof $bf16_add_instances] > 0} {
        echo "BF16 Add Pipeline instances:"
        report_area [get_cells $bf16_add_instances]
    }
    
    if {[sizeof $bf16_mul_instances] > 0} {
        echo "BF16 Multiply Pipeline instances:"
        report_area [get_cells $bf16_mul_instances]
    }
    
    # PE array analysis
    echo "\n3.2 Processing Element Array:"
    echo "-----------------------------"
    set pe_instances [get_cells -hier -filter "ref_name =~ *trapezoid_pipeline*"]
    if {[sizeof $pe_instances] > 0} {
        report_area [get_cells $pe_instances]
    }
    
    # Memory elements analysis
    echo "\n3.3 Memory Elements:"
    echo "-------------------"
    set memory_instances [get_cells -hier -filter "is_sequential == true"]
    if {[sizeof $memory_instances] > 0} {
        echo "Total sequential elements:"
        report_area [get_cells $memory_instances] -all
    }
    
    # Combinational logic analysis
    echo "\n3.4 Combinational Logic:"
    echo "------------------------"
    set combo_instances [get_cells -hier -filter "is_combinational == true"]
    if {[sizeof $combo_instances] > 0} {
        echo "Total combinational logic:"
        report_area [get_cells $combo_instances] -all
    }
    
    # Critical path analysis
    echo "\n4. CRITICAL PATH ANALYSIS:"
    echo "=========================="
    report_timing -max_paths 5 -delay_type max -path_type summary
    
    # Area efficiency metrics
    echo "\n5. AREA EFFICIENCY METRICS:"
    echo "==========================="
    
    # Calculate area per PE
    set total_area [get_attribute [current_design] area]
    set pe_count 128  # 8x16 PE array
    set area_per_pe [expr $total_area / $pe_count]
    
    echo "Total design area: $total_area μm²"
    echo "Number of PEs: $pe_count"
    echo "Area per PE: $area_per_pe μm²"
    
    # Area distribution by cell type
    echo "\n6. AREA DISTRIBUTION BY CELL TYPE:"
    echo "=================================="
    report_cell -area -nosplit
    
    # Write detailed reports
    echo "\n7. GENERATING DETAILED REPORTS:"
    echo "==============================="
    
    file mkdir area_analysis_reports
    
    report_area -hierarchy -all > area_analysis_reports/detailed_hierarchy.rpt
    report_cell -area > area_analysis_reports/cell_area_breakdown.rpt
    report_timing -max_paths 10 > area_analysis_reports/critical_paths.rpt
    
    # Generate area summary CSV
    set csv_file [open "area_analysis_reports/area_summary.csv" w]
    puts $csv_file "Module,Area(μm²),Percentage"
    puts $csv_file "Total Design,$total_area,100.0"
    puts $csv_file "Area per PE,$area_per_pe,N/A"
    close $csv_file
    
    echo "Detailed reports saved to area_analysis_reports/"
    echo "==================================================================="
}

# Area optimization recommendations
proc generate_optimization_recommendations {} {
    echo "\n==================================================================="
    echo "AREA OPTIMIZATION RECOMMENDATIONS"
    echo "==================================================================="
    
    set total_area [get_attribute [current_design] area]
    
    # Check if area is within reasonable bounds
    if {$total_area > 1000000} {
        echo "WARNING: Design area is quite large (>1mm²)"
        echo "Consider the following optimizations:"
        echo "1. Reduce PE array size (currently 8x16=128 PEs)"
        echo "2. Optimize BF16 pipeline depth (currently 5 stages)"
        echo "3. Use area-optimized library cells"
        echo "4. Enable aggressive area optimization in synthesis"
    } elseif {$total_area > 500000} {
        echo "NOTICE: Design area is moderate (0.5-1mm²)"
        echo "Potential optimizations:"
        echo "1. Review pipeline depth vs area trade-off"
        echo "2. Consider resource sharing between PEs"
        echo "3. Optimize memory usage"
    } else {
        echo "GOOD: Design area is reasonable (<0.5mm²)"
        echo "The current implementation is area-efficient"
    }
    
    # Generate specific recommendations based on analysis
    echo "\nSPECIFIC RECOMMENDATIONS:"
    echo "========================"
    echo "1. Pipeline Optimization:"
    echo "   - Current BF16 pipelines use 5 stages"
    echo "   - Consider reducing to 3-4 stages for area savings"
    echo "   - Trade-off: Reduced frequency capability"
    
    echo "\n2. PE Array Scaling:"
    echo "   - Current: 8x16 = 128 PEs"
    echo "   - Alternative configurations: 8x8, 4x16, 4x8"
    echo "   - Linear area scaling with PE count"
    
    echo "\n3. Memory Optimization:"
    echo "   - Review queue depths in each module"
    echo "   - Consider using single-port vs dual-port memories"
    echo "   - Optimize data path widths"
    
    echo "\n4. Library Selection:"
    echo "   - Use High-Vt cells for non-critical paths"
    echo "   - Apply low-power techniques where applicable"
    echo "   - Consider different drive strength options"
    
    echo "==================================================================="
}

# Main execution
if {[current_design] != ""} {
    analyze_area_breakdown
    generate_optimization_recommendations
} else {
    echo "ERROR: No design loaded. Please run synthesis first."
} 