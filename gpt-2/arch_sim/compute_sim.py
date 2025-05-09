from memory_sim import SRAMOverflowError, DDROverflowError, _HardwareSimulator
from torch import nn
import torch
import math
import numpy as np


class ComputeSimulator(_HardwareSimulator):
    def __init__(self):
        super().__init__()
        self._compute_initialized = False
        
    def initialize(self, read_delay=5, write_delay=10, sram_capacity=1024*1024,
                  ddr_capacity=8*1024*1024, ddr_read_delay=100, ddr_write_delay=80,
                  ddr_to_sram_delay=50, mac_units=256, add_units=512,
                  mac_delay=1, add_delay=1):
        """
        Initialize both memory and compute simulation parameters
        
        Args:
            mac_units: Number of multiply-accumulate units available
            add_units: Number of adder units available
            mac_delay: Cycles needed for one MAC operation
            add_delay: Cycles needed for one addition operation
        """
        # Initialize memory subsystem
        super().initialize(read_delay, write_delay, sram_capacity, 
                          ddr_capacity, ddr_read_delay, ddr_write_delay,
                          ddr_to_sram_delay)
        
        # Initialize compute subsystem
        self.mac_units = mac_units
        self.add_units = add_units
        self.mac_delay = mac_delay
        self.add_delay = add_delay
        self._compute_initialized = True
        
    def _check_initialization(self):
        if not self._initialized or not self._compute_initialized:
            raise RuntimeError("ComputeSimulator not initialized. Call initialize() first.")
    
    def _calculate_max_tile_size(self, operation='matmul'):
        """Calculate maximum tile size that fits in SRAM based on operation type"""
        # For matmul, we need to fit A tile, B tile, and C tile (result)
        # For simplicity, assume square tiles and float32 (4 bytes)
        bytes_per_element = 4  # float32
        
        if operation == 'matmul':
            # For A*B, need space for A, B, and partial result
            # Assuming we need 3 equally sized tiles
            max_elements = self.sram_capacity // (3 * bytes_per_element)
            # For square tiles, each dimension is sqrt(max_elements)
            tile_dim = int(math.sqrt(max_elements))
            return tile_dim
        elif operation == 'add':
            # For A+B, need space for A, B, and result
            max_elements = self.sram_capacity // (3 * bytes_per_element)
            tile_dim = int(math.sqrt(max_elements))
            return tile_dim
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    
    def matrix_multiply(self, A_key, B_key, result_key, store_in_ddr=True):
        """
        Perform matrix multiplication C = A * B using tiling if necessary
        
        Args:
            A_key: Key for matrix A in memory
            B_key: Key for matrix B in memory
            result_key: Key to store resulting matrix
            store_in_ddr: If True, store result directly in DDR
            
        Returns:
            Result matrix
        """
        self._check_initialization()
        
        # Try to load matrices A and B from DDR if not in SRAM
        try:
            A = self.sram_read(A_key)
        except KeyError:
            # If not in memory at all, raise an error
            raise KeyError(f"Matrix {A_key} not found in memory")
            
        try:
            B = self.sram_read(B_key)
        except KeyError:
            # If not in memory at all, raise an error
            raise KeyError(f"Matrix {B_key} not found in memory")
        
        # Check matrix dimensions
        if A.shape[1] != B.shape[0]:
            raise ValueError(f"Matrix dimensions mismatch: {A.shape} and {B.shape}")
            
        # Get matrix dimensions
        M, K = A.shape
        K, N = B.shape
        
        # Check if we can fit both matrices entirely in SRAM
        total_size = self._calc_tensor_size(A) + self._calc_tensor_size(B)
        result_size = M * N * 4  # Assuming float32
        
        if total_size + result_size <= self.sram_capacity:
            # Both matrices fit in SRAM, perform direct multiplication
            result = torch.matmul(A, B)
            
            # Calculate computation cycles
            # Each element in the result requires K MAC operations
            total_mac_ops = M * N * K
            parallel_ops = min(total_mac_ops, self.mac_units)
            compute_cycles = (total_mac_ops // parallel_ops) * self.mac_delay
            if total_mac_ops % parallel_ops > 0:
                compute_cycles += self.mac_delay
                
            self.cycles += compute_cycles
            self.access_log.append(f"COMPUTE matmul {A_key}*{B_key} -> {result_key} ({compute_cycles} cycles)")
            
            # Store result
            if store_in_ddr:
                self.ddr_load(result_key, result)
            else:
                self.sram_load(result_key, result)
                
        else:
            # Matrices don't fit, use tiling
            tile_size = self._calculate_max_tile_size('matmul')
            result = torch.zeros((M, N), dtype=A.dtype, device=A.device)
            
            # Compute using tiles
            total_compute_cycles = 0
            
            for i in range(0, M, tile_size):
                i_end = min(i + tile_size, M)
                for j in range(0, N, tile_size):
                    j_end = min(j + tile_size, N)
                    
                    # Initialize tile result
                    tile_result = torch.zeros((i_end - i, j_end - j), dtype=A.dtype, device=A.device)
                    
                    for k in range(0, K, tile_size):
                        k_end = min(k + tile_size, K)
                        
                        # Load tiles from A and B
                        A_tile_key = f"{A_key}_tile_{i}_{k}"
                        B_tile_key = f"{B_key}_tile_{k}_{j}"
                        
                        A_tile = A[i:i_end, k:k_end]
                        B_tile = B[k:k_end, j:j_end]
                        
                        # Store tiles in SRAM
                        self.sram_load(A_tile_key, A_tile)
                        self.sram_load(B_tile_key, B_tile)
                        
                        # Perform tile multiplication
                        sub_result = torch.matmul(A_tile, B_tile)
                        
                        # Accumulate to tile result
                        tile_result += sub_result
                        
                        # Calculate compute cycles for this tile operation
                        tile_mac_ops = (i_end - i) * (j_end - j) * (k_end - k)
                        parallel_ops = min(tile_mac_ops, self.mac_units)
                        tile_cycles = (tile_mac_ops // parallel_ops) * self.mac_delay
                        if tile_mac_ops % parallel_ops > 0:
                            tile_cycles += self.mac_delay
                        
                        total_compute_cycles += tile_cycles
                        
                        # Clean up SRAM
                        del self.sram[A_tile_key]
                        self.sram_used -= self._calc_tensor_size(A_tile)
                        del self.sram[B_tile_key]
                        self.sram_used -= self._calc_tensor_size(B_tile)
                    
                    # Update the result matrix with the tile result
                    result[i:i_end, j:j_end] = tile_result
            
            self.cycles += total_compute_cycles
            self.access_log.append(f"COMPUTE tiled matmul {A_key}*{B_key} -> {result_key} ({total_compute_cycles} cycles)")
            
            # Store the final result
            if store_in_ddr:
                self.ddr_load(result_key, result)
            else:
                self.sram_load(result_key, result)
        
        return result
    
    def matrix_add(self, A_key, B_key, result_key, store_in_ddr=True):
        """
        Perform matrix addition C = A + B using tiling if necessary
        
        Args:
            A_key: Key for matrix A in memory
            B_key: Key for matrix B in memory
            result_key: Key to store resulting matrix
            store_in_ddr: If True, store result directly in DDR
            
        Returns:
            Result matrix
        """
        self._check_initialization()
        
        # Try to load matrices A and B from DDR if not in SRAM
        try:
            A = self.sram_read(A_key)
        except KeyError:
            # If not in memory at all, raise an error
            raise KeyError(f"Matrix {A_key} not found in memory")
            
        try:
            B = self.sram_read(B_key)
        except KeyError:
            # If not in memory at all, raise an error
            raise KeyError(f"Matrix {B_key} not found in memory")
        
        # Check matrix dimensions
        if A.shape != B.shape:
            raise ValueError(f"Matrix dimensions mismatch: {A.shape} and {B.shape}")
            
        # Get matrix dimensions
        M, N = A.shape
        
        # Check if we can fit both matrices entirely in SRAM
        total_size = self._calc_tensor_size(A) + self._calc_tensor_size(B)
        result_size = M * N * 4  # Assuming float32
        
        if total_size + result_size <= self.sram_capacity:
            # Both matrices fit in SRAM, perform direct addition
            result = A + B
            
            # Calculate computation cycles
            total_add_ops = M * N
            parallel_ops = min(total_add_ops, self.add_units)
            compute_cycles = (total_add_ops // parallel_ops) * self.add_delay
            if total_add_ops % parallel_ops > 0:
                compute_cycles += self.add_delay
                
            self.cycles += compute_cycles
            self.access_log.append(f"COMPUTE add {A_key}+{B_key} -> {result_key} ({compute_cycles} cycles)")
            
            # Store result
            if store_in_ddr:
                self.ddr_load(result_key, result)
            else:
                self.sram_load(result_key, result)
                
        else:
            # Matrices don't fit, use tiling
            tile_size = self._calculate_max_tile_size('add')
            result = torch.zeros((M, N), dtype=A.dtype, device=A.device)
            
            # Compute using tiles
            total_compute_cycles = 0
            
            for i in range(0, M, tile_size):
                i_end = min(i + tile_size, M)
                for j in range(0, N, tile_size):
                    j_end = min(j + tile_size, N)
                    
                    # Load tiles from A and B
                    A_tile_key = f"{A_key}_tile_{i}_{j}"
                    B_tile_key = f"{B_key}_tile_{i}_{j}"
                    
                    A_tile = A[i:i_end, j:j_end]
                    B_tile = B[i:i_end, j:j_end]
                    
                    # Store tiles in SRAM
                    self.sram_load(A_tile_key, A_tile)
                    self.sram_load(B_tile_key, B_tile)
                    
                    # Perform tile addition
                    tile_result = A_tile + B_tile
                    
                    # Calculate compute cycles for this tile operation
                    tile_add_ops = (i_end - i) * (j_end - j)
                    parallel_ops = min(tile_add_ops, self.add_units)
                    tile_cycles = (tile_add_ops // parallel_ops) * self.add_delay
                    if tile_add_ops % parallel_ops > 0:
                        tile_cycles += self.add_delay
                    
                    total_compute_cycles += tile_cycles
                    
                    # Update the result matrix with the tile result
                    result[i:i_end, j:j_end] = tile_result
                    
                    # Clean up SRAM
                    del self.sram[A_tile_key]
                    self.sram_used -= self._calc_tensor_size(A_tile)
                    del self.sram[B_tile_key]
                    self.sram_used -= self._calc_tensor_size(B_tile)
            
            self.cycles += total_compute_cycles
            self.access_log.append(f"COMPUTE tiled add {A_key}+{B_key} -> {result_key} ({total_compute_cycles} cycles)")
            
            # Store the final result
            if store_in_ddr:
                self.ddr_load(result_key, result)
            else:
                self.sram_load(result_key, result)
        
        return result
