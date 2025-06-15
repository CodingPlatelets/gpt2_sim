import numpy as np
from collections import deque
from scipy.sparse import csr_matrix
from .compute_sim import MACUnit

class SparsePE:
    """
    具有N个乘加单元的PE，用于处理稀疏矩阵乘法
    每行PE处理一行B的数据，A的元素被广播到所有乘法器
    """
    
    def __init__(self, pe_id, num_mac_units=8, max_queue_size=64):
        """
        初始化稀疏PE
        
        Args:
            pe_id: PE的ID
            num_mac_units: 乘加单元数量
            max_queue_size: 队列最大长度
        """
        self.pe_id = pe_id
        self.num_mac_units = num_mac_units
        self.max_queue_size = max_queue_size
        
        # 乘加单元阵列
        self.mac_units = [MACUnit() for _ in range(num_mac_units)]
        
        # 输入队列 - 存储待处理的B矩阵行数据
        self.input_queue = deque(maxlen=max_queue_size)
        
        # 当前处理的A元素（广播值）
        self.current_a_value = 0
        self.current_a_valid = False
        
        # 当前处理的B行数据
        self.current_b_row = None
        self.current_b_valid = False
        
        # 输出缓冲区
        self.output_buffer = []
        
        # 状态标志
        self.busy = False
        self.cycle_count = 0
        
    def is_queue_full(self):
        """检查输入队列是否已满"""
        return len(self.input_queue) >= self.max_queue_size
    
    def is_queue_empty(self):
        """检查输入队列是否为空"""
        return len(self.input_queue) == 0
    
    def can_accept_input(self):
        """检查是否可以接受新的输入"""
        return not self.is_queue_full()
    
    def enqueue_b_row(self, b_row_data):
        """
        将B矩阵的一行数据加入队列
        
        Args:
            b_row_data: 字典格式 {
                'row_id': 行ID,
                'values': CSR格式的values数组,
                'col_indices': CSR格式的列索引数组,
                'start_col': 起始列索引,
                'end_col': 结束列索引
            }
        """
        if not self.can_accept_input():
            return False
            
        self.input_queue.append(b_row_data)
        return True
    
    def _pad_or_split_b_row(self, values, col_indices):
        """
        对B行数据进行填充或分割处理
        
        Args:
            values: 非零元素值数组
            col_indices: 对应的列索引数组
            
        Returns:
            处理后的数据块列表，每个块最多包含num_mac_units个元素
        """
        chunks = []
        
        # 如果元素数量超过MAC单元数量，需要分割
        if len(values) > self.num_mac_units:
            for i in range(0, len(values), self.num_mac_units):
                chunk_values = values[i:i + self.num_mac_units]
                chunk_indices = col_indices[i:i + self.num_mac_units]
                chunks.append((chunk_values, chunk_indices))
        else:
            # 如果元素数量不足，需要填充
            padded_values = list(values)
            padded_indices = list(col_indices)
            
            # 用0填充到MAC单元数量
            while len(padded_values) < self.num_mac_units:
                padded_values.append(0)
                padded_indices.append(-1)  # -1表示无效索引
                
            chunks.append((padded_values, padded_indices))
            
        return chunks
    
    def set_broadcast_a(self, a_value, valid=True):
        """
        设置广播的A值
        
        Args:
            a_value: A矩阵的当前元素值
            valid: 是否有效
        """
        self.current_a_value = a_value
        self.current_a_valid = valid
    
    def _process_current_chunk(self, values, col_indices):
        """
        处理当前数据块
        
        Args:
            values: 当前块的值数组
            col_indices: 当前块的列索引数组
        """
        # 将数据分配给各个MAC单元
        for i, mac in enumerate(self.mac_units):
            if i < len(values):
                b_value = values[i]
                col_idx = col_indices[i]
                
                # 只有当列索引有效时才进行计算
                if col_idx >= 0:
                    mac.get_input(
                        input_valid=self.current_a_valid and self.current_b_valid,
                        input1=self.current_a_value,
                        input2=b_value,
                        sft_index=col_idx
                    )
                else:
                    # 无效数据，传入0
                    mac.get_input(
                        input_valid=False,
                        input1=0,
                        input2=0,
                        sft_index=-1
                    )
            else:
                # 超出范围的MAC单元设为无效
                mac.get_input(
                    input_valid=False,
                    input1=0,
                    input2=0,
                    sft_index=-1
                )
    
    def clock_cycle(self):
        """
        执行一个时钟周期
        
        Returns:
            输出结果列表，每个元素为 (列索引, 计算结果)
        """
        self.cycle_count += 1
        outputs = []
        
        # 收集MAC单元的输出
        for mac in self.mac_units:
            result = mac.clock_cycle()
            if mac.valid and len(mac.index_queue) > 0:
                col_idx = mac.index_queue.pop(0)
                if col_idx >= 0:  # 只有有效索引才输出
                    outputs.append((col_idx, result))
        
        # 如果没有当前处理的B行数据，尝试从队列获取
        if not self.current_b_valid and not self.is_queue_empty():
            self.current_b_row = self.input_queue.popleft()
            self.current_b_valid = True
            
            # 处理新的B行数据
            if self.current_b_row:
                values = self.current_b_row['values']
                col_indices = self.current_b_row['col_indices']
                
                # 分割或填充数据
                chunks = self._pad_or_split_b_row(values, col_indices)
                
                # 将分割后的块重新加入队列（除了第一个块）
                for chunk in chunks[1:]:
                    chunk_data = {
                        'row_id': self.current_b_row['row_id'],
                        'values': chunk[0],
                        'col_indices': chunk[1],
                        'start_col': self.current_b_row.get('start_col', 0),
                        'end_col': self.current_b_row.get('end_col', len(chunk[1]))
                    }
                    if len(self.input_queue) < self.max_queue_size:
                        self.input_queue.append(chunk_data)
                
                # 处理第一个块
                if chunks:
                    self._process_current_chunk(chunks[0][0], chunks[0][1])
        
        # 更新忙碌状态
        self.busy = (not self.is_queue_empty() or 
                    self.current_b_valid or 
                    any(mac.is_active() for mac in self.mac_units))
        
        # 如果当前B行处理完成，标记为无效
        if self.current_b_valid and not any(mac.is_active() for mac in self.mac_units):
            self.current_b_valid = False
            self.current_b_row = None
        
        return outputs
    
    def is_active(self):
        """检查PE是否仍在活跃处理数据"""
        return (self.busy or 
                not self.is_queue_empty() or 
                any(mac.is_active() for mac in self.mac_units))
    
    def reset(self):
        """重置PE状态"""
        self.input_queue.clear()
        self.current_a_value = 0
        self.current_a_valid = False
        self.current_b_row = None
        self.current_b_valid = False
        self.output_buffer.clear()
        self.busy = False
        self.cycle_count = 0
        
        # 重置所有MAC单元
        for mac in self.mac_units:
            mac.multiply_pipeline.reset()
            mac.input1 = 0
            mac.input2 = 0
            mac.index_queue = []
            mac.valid = False
            mac.input_valid = False
    
    def get_status(self):
        """获取PE状态信息"""
        active_macs = sum(1 for mac in self.mac_units if mac.is_active())
        
        return {
            'pe_id': self.pe_id,
            'cycle_count': self.cycle_count,
            'busy': self.busy,
            'queue_size': len(self.input_queue),
            'queue_capacity': self.max_queue_size,
            'current_a_value': self.current_a_value,
            'current_a_valid': self.current_a_valid,
            'current_b_valid': self.current_b_valid,
            'active_mac_units': active_macs,
            'total_mac_units': self.num_mac_units,
            'utilization': active_macs / self.num_mac_units if self.num_mac_units > 0 else 0
        }
    
    def print_status(self):
        """打印PE状态"""
        status = self.get_status()
        print(f"\n=== PE #{status['pe_id']} 状态 (周期 {status['cycle_count']}) ===")
        print(f"忙碌状态: {'是' if status['busy'] else '否'}")
        print(f"输入队列: {status['queue_size']}/{status['queue_capacity']}")
        print(f"当前A值: {status['current_a_value']} (有效: {status['current_a_valid']})")
        print(f"当前B行: {'有效' if status['current_b_valid'] else '无效'}")
        print(f"活跃MAC单元: {status['active_mac_units']}/{status['total_mac_units']}")
        print(f"利用率: {status['utilization']:.2%}")


class SparsePEArray:
    """
    稀疏PE阵列，管理多个PE
    """
    
    def __init__(self, num_pes=4, mac_units_per_pe=8, max_queue_size=64):
        """
        初始化PE阵列
        
        Args:
            num_pes: PE数量
            mac_units_per_pe: 每个PE的MAC单元数量
            max_queue_size: 每个PE的队列最大长度
        """
        self.num_pes = num_pes
        self.mac_units_per_pe = mac_units_per_pe
        
        # 创建PE阵列
        self.pes = [SparsePE(i, mac_units_per_pe, max_queue_size) 
                   for i in range(num_pes)]
        
        self.cycle_count = 0
    
    def distribute_b_matrix(self, b_csr_matrix):
        """
        将B矩阵的行分配给各个PE
        
        Args:
            b_csr_matrix: B矩阵的CSR格式
        """
        num_rows = b_csr_matrix.shape[0]
        
        for row_id in range(num_rows):
            # 计算该行应该分配给哪个PE
            pe_id = row_id % self.num_pes
            
            # 提取该行的数据
            start_idx = b_csr_matrix.indptr[row_id]
            end_idx = b_csr_matrix.indptr[row_id + 1]
            
            if start_idx < end_idx:  # 该行有非零元素
                values = b_csr_matrix.data[start_idx:end_idx]
                col_indices = b_csr_matrix.indices[start_idx:end_idx]
                
                row_data = {
                    'row_id': row_id,
                    'values': values.tolist(),
                    'col_indices': col_indices.tolist(),
                    'start_col': int(col_indices[0]) if len(col_indices) > 0 else 0,
                    'end_col': int(col_indices[-1]) if len(col_indices) > 0 else 0
                }
                
                # 尝试将数据加入对应PE的队列
                if not self.pes[pe_id].enqueue_b_row(row_data):
                    print(f"警告: PE {pe_id} 队列已满，无法加入行 {row_id}")
    
    def broadcast_a_value(self, a_value, valid=True):
        """
        向所有PE广播A值
        
        Args:
            a_value: A矩阵的当前元素值
            valid: 是否有效
        """
        for pe in self.pes:
            pe.set_broadcast_a(a_value, valid)
    
    def clock_cycle(self):
        """
        执行一个时钟周期
        
        Returns:
            所有PE的输出结果字典 {pe_id: [(列索引, 计算结果), ...]}
        """
        self.cycle_count += 1
        all_outputs = {}
        
        for pe in self.pes:
            outputs = pe.clock_cycle()
            if outputs:
                all_outputs[pe.pe_id] = outputs
        
        return all_outputs
    
    def is_active(self):
        """检查是否有PE仍在活跃处理数据"""
        return any(pe.is_active() for pe in self.pes)
    
    def reset(self):
        """重置所有PE"""
        self.cycle_count = 0
        for pe in self.pes:
            pe.reset()
    
    def get_overall_status(self):
        """获取整体状态"""
        total_queue_size = sum(len(pe.input_queue) for pe in self.pes)
        total_active_macs = sum(
            sum(1 for mac in pe.mac_units if mac.is_active()) 
            for pe in self.pes
        )
        total_macs = self.num_pes * self.mac_units_per_pe
        
        return {
            'cycle_count': self.cycle_count,
            'num_pes': self.num_pes,
            'total_queue_size': total_queue_size,
            'total_active_macs': total_active_macs,
            'total_macs': total_macs,
            'overall_utilization': total_active_macs / total_macs if total_macs > 0 else 0,
            'active_pes': sum(1 for pe in self.pes if pe.is_active())
        }
    
    def print_overall_status(self):
        """打印整体状态"""
        status = self.get_overall_status()
        print(f"\n=== PE阵列整体状态 (周期 {status['cycle_count']}) ===")
        print(f"PE数量: {status['num_pes']}")
        print(f"活跃PE: {status['active_pes']}")
        print(f"总队列大小: {status['total_queue_size']}")
        print(f"活跃MAC单元: {status['total_active_macs']}/{status['total_macs']}")
        print(f"整体利用率: {status['overall_utilization']:.2%}")
        
        # 打印各个PE的详细状态
        for pe in self.pes:
            if pe.is_active():
                pe.print_status()


def test_sparse_pe():
    """测试稀疏PE功能"""
    print("=== 测试稀疏PE ===")
    
    # 创建测试矩阵
    A = np.array([[1, 2, 0, 3]])  # 1x4 稠密向量
    B = np.array([
        [1, 0, 2, 0],
        [0, 3, 0, 4], 
        [5, 0, 0, 6],
        [0, 7, 8, 0]
    ])  # 4x4 稀疏矩阵
    
    print("A矩阵:")
    print(A)
    print("B矩阵:")
    print(B)
    
    # 转换B为CSR格式
    B_csr = csr_matrix(B)
    print(f"\nB矩阵CSR格式:")
    print(f"Values: {B_csr.data}")
    print(f"Indices: {B_csr.indices}")
    print(f"Indptr: {B_csr.indptr}")
    
    # 创建PE阵列
    pe_array = SparsePEArray(num_pes=2, mac_units_per_pe=4)
    
    # 分配B矩阵
    pe_array.distribute_b_matrix(B_csr)
    
    # 模拟计算过程
    print(f"\n开始计算...")
    
    # 对A的每个元素进行广播计算
    for a_idx, a_val in enumerate(A[0]):
        if a_val != 0:  # 只处理非零元素
            print(f"\n处理A[{a_idx}] = {a_val}")
            pe_array.broadcast_a_value(a_val, True)
            
            # 运行几个周期让计算完成
            for cycle in range(10):
                outputs = pe_array.clock_cycle()
                if outputs:
                    print(f"  周期 {cycle}: {outputs}")
                
                if not pe_array.is_active():
                    break
    
    # 打印最终状态
    pe_array.print_overall_status()


if __name__ == "__main__":
    test_sparse_pe()