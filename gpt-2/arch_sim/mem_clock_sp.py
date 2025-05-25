from enum import Enum, auto


class MemoryOperationType(Enum):
    IDLE = auto()
    READ_SINGLE = auto()
    WRITE_SINGLE = auto()
    READ_SEQUENCE = auto()
    WRITE_SEQUENCE = auto()


class MemoryRequestStatus(Enum):
    ACCEPTED = auto()  # Request accepted, operation initiated
    BUSY = auto()      # Memory is busy with another operation
    ERROR = auto()     # Error in request (e.g., out of bounds, though handled before starting)


class MemoryTickStatus(Enum):
    IDLE = auto()               # Memory is idle, no operation in progress
    BUSY_PROCESSING = auto()    # Memory is busy processing an ongoing operation
    READ_COMPLETED = auto()     # A read operation has just completed, data is ready
    WRITE_COMPLETED = auto()    # A write operation has just completed
    # ERROR (could be a status if tick itself causes an error, less common for simple mem)


class SimpleCycleAccurateMemory:
    """
    一个简易的周期精确（多周期操作）内存硬件模拟器。
    (A simple cycle-accurate (multi-cycle operation) memory hardware simulator.)
    操作被初始化，然后花费定义的时钟周期数，通过外部 'tick()' 调用来推进。
    (Operations are initiated and then take a defined number of clock cycles,
    advanced by an external 'tick()' call.)
    """

    # 默认操作延迟 (Default operation latencies in clock cycles)
    DEFAULT_SINGLE_BYTE_READ_LATENCY = 3
    DEFAULT_SINGLE_BYTE_WRITE_LATENCY = 2
    DEFAULT_SEQUENCE_READ_BASE_LATENCY = 2  # Base cost before per-byte
    DEFAULT_SEQUENCE_READ_PER_BYTE_LATENCY = 1
    DEFAULT_SEQUENCE_WRITE_BASE_LATENCY = 1  # Base cost before per-byte
    DEFAULT_SEQUENCE_WRITE_PER_BYTE_LATENCY = 1

    def __init__(self, size_in_bytes: int,
                 single_byte_read_latency: int = DEFAULT_SINGLE_BYTE_READ_LATENCY,
                 single_byte_write_latency: int = DEFAULT_SINGLE_BYTE_WRITE_LATENCY,
                 sequence_read_base_latency: int = DEFAULT_SEQUENCE_READ_BASE_LATENCY,
                 sequence_read_per_byte_latency: int = DEFAULT_SEQUENCE_READ_PER_BYTE_LATENCY,
                 sequence_write_base_latency: int = DEFAULT_SEQUENCE_WRITE_BASE_LATENCY,
                 sequence_write_per_byte_latency: int = DEFAULT_SEQUENCE_WRITE_PER_BYTE_LATENCY
                 ):
        if not isinstance(size_in_bytes, int) or size_in_bytes <= 0:
            raise ValueError(
                "内存大小必须是一个正整数。(Memory size must be a positive integer.)")

        self.memory_size = size_in_bytes
        self.memory_storage = bytearray(size_in_bytes)

        # 模拟寄存器 (Simulated Registers)
        # 内存地址寄存器 (Memory Address Register - stores address of current/last op)
        self.mar = 0
        # 内存数据寄存器 (Memory Data Register - for single byte reads/writes when ready/initiated)
        self.mdr = 0

        # 延迟设置 (Latency settings)
        self.latencies = {
            MemoryOperationType.READ_SINGLE: single_byte_read_latency,
            MemoryOperationType.WRITE_SINGLE: single_byte_write_latency,
            MemoryOperationType.READ_SEQUENCE: (sequence_read_base_latency, sequence_read_per_byte_latency),
            MemoryOperationType.WRITE_SEQUENCE: (sequence_write_base_latency, sequence_write_per_byte_latency),
        }

        # 进行中操作的内部状态 (Internal state for ongoing operations)
        self._busy = False
        self._current_operation_type = MemoryOperationType.IDLE
        self._cycles_remaining = 0

        self._op_address = 0
        # 用于写操作 (byte or bytes) (For write operations (byte or bytes))
        self._op_data_to_write = None
        # 用于读操作 (byte or bytes) (For read operations (byte or bytes))
        self._op_read_data_buffer = None
        self._op_sequence_length = 0  # 用于序列读 (For sequence reads)

        # 当读取数据在MDR或缓冲区中可用时为True (True when read data is available in mdr or buffer)
        self._data_ready_flag = False

        print(
            f"周期精确内存初始化完毕。大小: {self.memory_size} 字节。(Cycle-Accurate Memory initialized. Size: {self.memory_size} bytes.)")
        print(f"  延迟 (时钟周期) (Latencies (cycles)):")
        print(
            f"    单字节读 (Single Read): {self.latencies[MemoryOperationType.READ_SINGLE]}")
        print(
            f"    单字节写 (Single Write): {self.latencies[MemoryOperationType.WRITE_SINGLE]}")
        print(
            f"    序列读 (Seq Read): 基准 (Base) {self.latencies[MemoryOperationType.READ_SEQUENCE][0]} + 每字节 (PerByte) {self.latencies[MemoryOperationType.READ_SEQUENCE][1]}")
        print(
            f"    序列写 (Seq Write): 基准 (Base) {self.latencies[MemoryOperationType.WRITE_SEQUENCE][0]} + 每字节 (PerByte) {self.latencies[MemoryOperationType.WRITE_SEQUENCE][1]}")

    def _is_valid_address(self, address: int, num_bytes: int = 1) -> bool:
        """检查地址和长度是否在有效范围内。(Checks if address and length are within valid range.)"""
        if not (0 <= address < self.memory_size):
            return False
        if address + num_bytes > self.memory_size:  # 结束地址 address + num_bytes - 1 < memory_size
            return False
        return True

    def is_busy(self) -> bool:
        """检查内存当前是否正在处理操作。(Checks if the memory is currently processing an operation.)"""
        return self._busy

    def is_data_ready(self) -> bool:
        """检查已完成的读操作的数据是否准备就绪。(Checks if data from a completed read operation is ready.)"""
        return self._data_ready_flag

    def request_read_byte(self, address: int) -> MemoryRequestStatus:
        """初始化单字节读操作。(Initiates a single byte read operation.)"""
        if self._busy:
            return MemoryRequestStatus.BUSY
        if not self._is_valid_address(address):
            print(
                f"错误: 读地址 0x{address:X} 越界。(Error: Read address 0x{address:X} out of bounds.)")
            return MemoryRequestStatus.ERROR

        self._busy = True
        self._current_operation_type = MemoryOperationType.READ_SINGLE
        self._cycles_remaining = self.latencies[MemoryOperationType.READ_SINGLE]
        self.mar = address  # 设置 MAR (Set MAR)
        self._op_address = address
        self._data_ready_flag = False
        # 清除之前的读取数据 (Clear previous read data)
        self._op_read_data_buffer = None
        self.mdr = 0  # 操作完成前MDR无效 (MDR not valid until operation completes)
        # print(f"MemDebug: 从 0x{address:X} 读取字节请求已发出, 需要 {self._cycles_remaining} 周期。")
        return MemoryRequestStatus.ACCEPTED

    def request_write_byte(self, address: int, data: int) -> MemoryRequestStatus:
        """初始化单字节写操作。(Initiates a single byte write operation.)"""
        if self._busy:
            return MemoryRequestStatus.BUSY
        if not (0 <= data <= 255):
            # 通常由请求者确保数据有效，但这里也检查一下
            # (Usually requester ensures data is valid, but check here too)
            print(
                f"错误: 写入数据 {data} 不是有效字节 (0-255)。(Error: Write data {data} is not a valid byte (0-255).)")
            return MemoryRequestStatus.ERROR  # 或者抛出 ValueError
        if not self._is_valid_address(address):
            print(
                f"错误: 写地址 0x{address:X} 越界。(Error: Write address 0x{address:X} out of bounds.)")
            return MemoryRequestStatus.ERROR

        self._busy = True
        self._current_operation_type = MemoryOperationType.WRITE_SINGLE
        self._cycles_remaining = self.latencies[MemoryOperationType.WRITE_SINGLE]
        self.mar = address  # 设置 MAR (Set MAR)
        # self.mdr = data # CPU将数据放入MDR，内存稍后取用 (CPU places data in MDR, memory takes it later)
        self._op_address = address
        # 内存内部的待写入数据缓冲区 (Memory internal buffer for data to write)
        self._op_data_to_write = data
        # print(f"MemDebug: 向 0x{address:X} 写入字节请求已发出, 需要 {self._cycles_remaining} 周期。")
        return MemoryRequestStatus.ACCEPTED

    def request_read_sequence(self, address: int, num_bytes: int) -> MemoryRequestStatus:
        """初始化序列读操作。(Initiates a sequence read operation.)"""
        if self._busy:
            return MemoryRequestStatus.BUSY
        if num_bytes < 0:
            # print(f"错误: 读取序列的字节数 {num_bytes} 不能为负。")
            return MemoryRequestStatus.ERROR  # 或者抛出 ValueError
        if num_bytes == 0:
            self._op_read_data_buffer = b""
            self._data_ready_flag = True
            # print(f"MemDebug: 从 0x{address:X} 读取0字节序列请求，立即完成。")
            return MemoryRequestStatus.ACCEPTED
        if not self._is_valid_address(address, num_bytes):
            print(
                f"错误: 从 0x{address:X} 读取 {num_bytes} 字节序列越界。(Error: Read sequence from 0x{address:X} for {num_bytes} bytes out of bounds.)")
            return MemoryRequestStatus.ERROR

        self._busy = True
        self._current_operation_type = MemoryOperationType.READ_SEQUENCE
        base_lat, pb_lat = self.latencies[MemoryOperationType.READ_SEQUENCE]
        self._cycles_remaining = base_lat + num_bytes * pb_lat
        self.mar = address
        self._op_address = address
        self._op_sequence_length = num_bytes
        self._data_ready_flag = False
        self._op_read_data_buffer = None
        # print(f"MemDebug: 从 0x{address:X} ({num_bytes} 字节) 读取序列请求已发出, 需要 {self._cycles_remaining} 周期。")
        return MemoryRequestStatus.ACCEPTED

    def request_write_sequence(self, address: int, data_sequence: bytes) -> MemoryRequestStatus:
        """初始化序列写操作。(Initiates a sequence write operation.)"""
        if not isinstance(data_sequence, (bytes, bytearray)):
            # print(f"错误: 序列写入的数据类型必须是 bytes 或 bytearray。")
            return MemoryRequestStatus.ERROR  # 或者抛出 TypeError
        if self._busy:
            return MemoryRequestStatus.BUSY

        num_bytes = len(data_sequence)
        if num_bytes == 0:
            # print(f"MemDebug: 向 0x{address:X} 写入0字节序列请求，无操作。")
            return MemoryRequestStatus.ACCEPTED

        if not self._is_valid_address(address, num_bytes):
            print(
                f"错误: 向 0x{address:X} 写入 {num_bytes} 字节序列越界。(Error: Write sequence to 0x{address:X} for {num_bytes} bytes out of bounds.)")
            return MemoryRequestStatus.ERROR

        self._busy = True
        self._current_operation_type = MemoryOperationType.WRITE_SEQUENCE
        base_lat, pb_lat = self.latencies[MemoryOperationType.WRITE_SEQUENCE]
        self._cycles_remaining = base_lat + num_bytes * pb_lat
        self.mar = address
        self._op_address = address
        self._op_data_to_write = data_sequence
        # print(f"MemDebug: 向 0x{address:X} ({num_bytes} 字节) 写入序列请求已发出, 需要 {self._cycles_remaining} 周期。")
        return MemoryRequestStatus.ACCEPTED

    def get_read_data_byte(self) -> int:
        """返回已完成的单字节读操作的数据。清除就绪标志。(Returns the data from a completed single byte read. Clears ready flag.)"""
        if self._data_ready_flag and self._op_read_data_buffer is not None and isinstance(self._op_read_data_buffer, int):
            # data = self.mdr # mdr 应该也包含此值
            data = self._op_read_data_buffer
            self._data_ready_flag = False
            # self._op_read_data_buffer = None # 保留给下一次非序列读操作前清除
            return data
        print(
            f"警告/错误: 调用 get_read_data_byte 时数据未就绪或类型错误。(Warning/Error: get_read_data_byte called when data not ready or wrong type. Ready: {self._data_ready_flag}, Buffer: {type(self._op_read_data_buffer)})")
        return -1  # 表示错误或无效状态 (Indicate error or invalid state)

    def get_read_data_sequence(self) -> bytes:
        """返回已完成的序列读操作的数据。清除就绪标志。(Returns the data from a completed sequence read. Clears ready flag.)"""
        if self._data_ready_flag and self._op_read_data_buffer is not None and isinstance(self._op_read_data_buffer, bytes):
            data = self._op_read_data_buffer
            self._data_ready_flag = False
            # self._op_read_data_buffer = None
            return data
        print(
            f"警告/错误: 调用 get_read_data_sequence 时数据未就绪或类型错误。(Warning/Error: get_read_data_sequence called when data not ready or wrong type. Ready: {self._data_ready_flag}, Buffer: {type(self._op_read_data_buffer)})")
        return b""  # 表示错误或无效状态 (Indicate error or invalid state)

    def tick(self) -> MemoryTickStatus:
        """
        将内存状态推进一个时钟周期。由主模拟器循环调用。
        (Advances memory state by one clock cycle. Called by the main simulator loop.)
        """
        if not self._busy:
            return MemoryTickStatus.IDLE

        if self._cycles_remaining > 0:
            self._cycles_remaining -= 1

        if self._cycles_remaining == 0:
            # 操作在本周期完成 (Operation completes in this tick)
            op_type_completed = self._current_operation_type

            if op_type_completed == MemoryOperationType.WRITE_SINGLE:
                self.memory_storage[self._op_address] = self._op_data_to_write
                # print(f"MemDebug (tick): 向 0x{self._op_address:X} 写入值 0x{self._op_data_to_write:02X} 已完成。")
            elif op_type_completed == MemoryOperationType.READ_SINGLE:
                data_val = self.memory_storage[self._op_address]
                self.mdr = data_val  # 数据现在在MDR中 (Data is now in MDR)
                # 也在内部缓冲区 (And also in our internal buffer)
                self._op_read_data_buffer = data_val
                self._data_ready_flag = True
                # print(f"MemDebug (tick): 从 0x{self._op_address:X} 读取值 0x{data_val:02X} 已完成。数据在MDR中就绪。")
            elif op_type_completed == MemoryOperationType.WRITE_SEQUENCE:
                num_bytes_seq = len(self._op_data_to_write)
                self.memory_storage[self._op_address: self._op_address +
                                    num_bytes_seq] = self._op_data_to_write
                # print(f"MemDebug (tick): 向 0x{self._op_address:X} ({num_bytes_seq} 字节) 的序列写入已完成。")
            elif op_type_completed == MemoryOperationType.READ_SEQUENCE:
                self._op_read_data_buffer = bytes(
                    self.memory_storage[self._op_address: self._op_address + self._op_sequence_length])
                self._data_ready_flag = True
                # print(f"MemDebug (tick): 从 0x{self._op_address:X} ({self._op_sequence_length} 字节) 的序列读取已完成。数据在缓冲区中就绪。")

            # 为下一个操作重置状态 (Reset state for next operation)
            self._busy = False
            self._current_operation_type = MemoryOperationType.IDLE  # 重要：在返回前重置
            # self._op_data_to_write = None # 可以在请求时清除
            # _op_read_data_buffer 由 get_read_data_* 清除或在下次读请求时覆盖

            if op_type_completed in [MemoryOperationType.READ_SINGLE, MemoryOperationType.READ_SEQUENCE]:
                return MemoryTickStatus.READ_COMPLETED
            else:  # WRITE_SINGLE, WRITE_SEQUENCE
                return MemoryTickStatus.WRITE_COMPLETED

        return MemoryTickStatus.BUSY_PROCESSING  # 仍然繁忙 (Still busy)

    def dump(self, start_address: int, num_bytes: int):
        # Dump 操作是瞬时的，不影响内存状态/时序
        # (Dump operation is instantaneous and doesn't affect memory state/timing)
        if num_bytes < 0:
            print(
                f"错误 (dump): 字节数 {num_bytes} 不能为负。(Error (dump): Number of bytes {num_bytes} cannot be negative.)")
            return

        if num_bytes == 0:
            # 允许 start_address == memory_size 来 dump 0 字节
            if not (0 <= start_address <= self.memory_size):
                print(
                    f"错误 (dump): dump 0 字节的起始地址 0x{start_address:X} 越界。有效范围 0 到 0x{self.memory_size:X}。(Error (dump): Dump 0 bytes start address 0x{start_address:X} is out of bounds. Valid range 0 to 0x{self.memory_size:X}.)")
                return
            print(
                f"信息 (dump): 请求从地址 0x{start_address:04X} dump 0 字节。(Info (dump): Requested to dump 0 bytes from address 0x{start_address:04X}.)")
            print(f"\n内存内容 Dump (从地址 0x{start_address:04X} 开始，共 0 字节):")
            print("(空)\n")
            return

        # 如果 num_bytes > 0, start_address 必须是严格有效的内存地址 (< memory_size)
        # self._is_valid_address 检查 address < memory_size
        if not self._is_valid_address(start_address):
            print(
                f"错误 (dump): dump 的起始地址 0x{start_address:X} 越界。有效范围 0 到 0x{self.memory_size - 1:X}。(Error (dump): Dump start address 0x{start_address:X} is out of bounds. Valid range 0 to 0x{self.memory_size - 1:X}.)")
            return

        actual_num_bytes = num_bytes
        if start_address + num_bytes > self.memory_size:
            actual_num_bytes = self.memory_size - start_address
            # 此处 actual_num_bytes 不可能为负，因为 start_address < memory_size
            print(f"警告 (dump): 从 0x{start_address:X} dump {num_bytes} 字节的请求超出内存范围，将只 dump {actual_num_bytes} 字节到内存末尾。(Warning (dump): Dump request for {num_bytes} bytes from 0x{start_address:X} exceeds memory range. Will dump only {actual_num_bytes} bytes to the end of memory.)")

        print(
            f"\n内存内容 Dump (从地址 0x{start_address:04X} 开始，共 {actual_num_bytes} 字节):")
        for i in range(actual_num_bytes):
            current_address = start_address + i
            if i % 16 == 0:
                print(f"\n0x{current_address:04X}: ", end="")
            print(f"{self.memory_storage[current_address]:02X} ", end="")
        print("\n")


# --- 周期精确内存使用示例 (Example Usage for Cycle-Accurate Memory) ---
if __name__ == "__main__":
    print("--- 周期精确内存模拟示例 (Cycle-Accurate Memory Simulation Example) ---")
    memory = SimpleCycleAccurateMemory(256)  # 使用默认延迟 (Using default latencies)

    global_clock = 0

    def tick_memory_and_print(mem_instance: SimpleCycleAccurateMemory, scenario_name: str = "") -> MemoryTickStatus:
        """辅助函数：执行一次内存tick并打印状态。(Helper: Ticks memory once and prints status.)"""
        global global_clock
        global_clock += 1
        status = mem_instance.tick()
        print(
            f"时钟: {global_clock:03d} | {scenario_name} | 内存Tick状态: {status.name}")
        return status

    # --- 场景 1: 单字节读 (地址 0x0A) ---
    print("\n--- 场景 1: 单字节读 (地址 0x0A) (Scenario 1: Single Byte Read (Address 0x0A)) ---")
    SCENARIO = "S1 Read"
    memory.memory_storage[0x0A] = 0xDD  # 预加载值 (Pre-load value)
    global_clock = 0

    request_status = memory.request_read_byte(0x0A)
    print(
        f"时钟: {global_clock:03d} | {SCENARIO} | 请求读字节 (0x0A): {request_status.name}")

    if request_status == MemoryRequestStatus.ACCEPTED:
        # 循环直到操作完成 (Loop until operation completes)
        # 单字节读延迟默认为 3 (Single byte read latency is 3 by default)
        for _ in range(memory.latencies[MemoryOperationType.READ_SINGLE]):
            status = tick_memory_and_print(memory, SCENARIO)
            if status == MemoryTickStatus.READ_COMPLETED:
                if memory.is_data_ready():
                    data = memory.get_read_data_byte()
                    print(
                        f"    {SCENARIO} CPU: 从 MAR 0x{memory.mar:X} 读取字节 0x{data:02X} (MDR: 0x{memory.mdr:02X})")
                    if data != 0xDD:
                        print(
                            f"    {SCENARIO} 错误: 读取值不匹配! (Error: Read value mismatch!)")
                break  # 操作完成，退出循环 (Operation complete, exit loop)
        # 再tick几次看看空闲状态 (Tick a few more times to see idle state)
        for _ in range(2):
            tick_memory_and_print(memory, f"{SCENARIO} Idle")

    # --- 场景 2: 单字节写 (0x0B, 值 0xCC) 然后读 ---
    print("\n--- 场景 2: 单字节写 (0x0B, 值 0xCC) 然后读 (Scenario 2: Single Byte Write (0x0B, value 0xCC) then Read) ---")
    SCENARIO = "S2 Write/Read"
    global_clock = 0
    memory = SimpleCycleAccurateMemory(256)  # 新实例 (New instance)

    # 写操作 (Write operation)
    request_status_write = memory.request_write_byte(0x0B, 0xCC)
    print(
        f"时钟: {global_clock:03d} | {SCENARIO} | 请求写字节 (0x0B, 0xCC): {request_status_write.name}")
    if request_status_write == MemoryRequestStatus.ACCEPTED:
        # 单字节写延迟默认为 2 (Single byte write latency is 2 by default)
        for _ in range(memory.latencies[MemoryOperationType.WRITE_SINGLE]):
            status = tick_memory_and_print(memory, f"{SCENARIO} Write")
            if status == MemoryTickStatus.WRITE_COMPLETED:
                print(f"    {SCENARIO} CPU: 通知写操作到 MAR 0x{memory.mar:X} 已完成。")
                break

    # 读操作 (Read operation)
    request_status_read = memory.request_read_byte(0x0B)
    print(
        f"时钟: {global_clock:03d} | {SCENARIO} | 请求读字节 (0x0B): {request_status_read.name}")
    if request_status_read == MemoryRequestStatus.ACCEPTED:
        for _ in range(memory.latencies[MemoryOperationType.READ_SINGLE]):
            status = tick_memory_and_print(memory, f"{SCENARIO} Read")
            if status == MemoryTickStatus.READ_COMPLETED:
                if memory.is_data_ready():
                    data = memory.get_read_data_byte()
                    print(
                        f"    {SCENARIO} CPU: 从 MAR 0x{memory.mar:X} 读取字节 0x{data:02X}")
                    if data != 0xCC:
                        print(
                            f"    {SCENARIO} 错误: 读取值不匹配! (Error: Read value mismatch!)")
                break
    memory.dump(0x0A, 4)

    # --- 场景 3: 序列写和读 ---
    print("\n--- 场景 3: 序列写和读 (Scenario 3: Sequence Write and Read) ---")
    SCENARIO = "S3 SeqWrite/Read"
    global_clock = 0
    memory = SimpleCycleAccurateMemory(256)

    my_sequence = b'\x11\x22\x33\x44'
    seq_addr = 0x20
    seq_len = len(my_sequence)

    # 序列写 (Sequence Write)
    # 默认延迟: 基准 1 + 每字节 1 * 4 = 5 周期 (Default latency: Base 1 + PerByte 1 * 4 = 5 cycles)
    expected_write_seq_latency = memory.latencies[MemoryOperationType.WRITE_SEQUENCE][0] + \
        seq_len * memory.latencies[MemoryOperationType.WRITE_SEQUENCE][1]
    req_stat_w_seq = memory.request_write_sequence(seq_addr, my_sequence)
    print(f"时钟: {global_clock:03d} | {SCENARIO} | 请求写序列 (0x{seq_addr:X}, 长 {seq_len}): {req_stat_w_seq.name}, 预计 {expected_write_seq_latency} 周期。")
    if req_stat_w_seq == MemoryRequestStatus.ACCEPTED:
        for _ in range(expected_write_seq_latency):
            status = tick_memory_and_print(memory, f"{SCENARIO} SeqWrite")
            if status == MemoryTickStatus.WRITE_COMPLETED:
                print(f"    {SCENARIO} CPU: 通知序列写到 MAR 0x{memory.mar:X} 已完成。")
                break
    memory.dump(seq_addr, seq_len + 2)

    # 序列读 (Sequence Read)
    # 默认延迟: 基准 2 + 每字节 1 * 4 = 6 周期 (Default latency: Base 2 + PerByte 1 * 4 = 6 cycles)
    expected_read_seq_latency = memory.latencies[MemoryOperationType.READ_SEQUENCE][0] + \
        seq_len * memory.latencies[MemoryOperationType.READ_SEQUENCE][1]
    req_stat_r_seq = memory.request_read_sequence(seq_addr, seq_len)
    print(f"时钟: {global_clock:03d} | {SCENARIO} | 请求读序列 (0x{seq_addr:X}, 长 {seq_len}): {req_stat_r_seq.name}, 预计 {expected_read_seq_latency} 周期。")
    if req_stat_r_seq == MemoryRequestStatus.ACCEPTED:
        for _ in range(expected_read_seq_latency):
            status = tick_memory_and_print(memory, f"{SCENARIO} SeqRead")
            if status == MemoryTickStatus.READ_COMPLETED:
                if memory.is_data_ready():
                    retrieved_data = memory.get_read_data_sequence()
                    print(
                        f"    {SCENARIO} CPU: 从 MAR 0x{memory.mar:X} 读取序列 {retrieved_data.hex()}")
                    if retrieved_data != my_sequence:
                        print(
                            f"    {SCENARIO} 错误: 序列不匹配! (Error: Sequence mismatch!)")
                break

    # --- 场景 4: 内存忙碌 ---
    print("\n--- 场景 4: 内存忙碌 (Scenario 4: Memory Busy) ---")
    SCENARIO = "S4 Busy"
    global_clock = 0
    # 使用自定义的长读取延迟 (Using custom long read latency)
    memory = SimpleCycleAccurateMemory(256, single_byte_read_latency=5)

    req1 = memory.request_read_byte(0x50)
    print(f"时钟: {global_clock:03d} | {SCENARIO} | 请求 1 (读 0x50): {req1.name}")

    # 读操作的第1个周期 (Cycle 1 of read op)
    tick_memory_and_print(memory, f"{SCENARIO} Op1 Tick 1")

    # 尝试在忙碌时请求 (Try to request while busy)
    req2 = memory.request_write_byte(0x51, 0xFF)
    # 应该是 BUSY (Should be BUSY)
    print(f"时钟: {global_clock:03d} | {SCENARIO} | 请求 2 (写 0x51): {req2.name}")
    if req2 != MemoryRequestStatus.BUSY:
        print(f"    {SCENARIO} 错误: 内存应该忙碌! (Error: Memory should be busy!)")

    # 继续完成第一个读操作 (Continue to complete the first read operation)
    # -1因为已tick一次, +3看空闲 ( -1 because ticked once, +3 to see idle)
    for i in range(memory.latencies[MemoryOperationType.READ_SINGLE] - 1 + 3):
        status = tick_memory_and_print(memory, f"{SCENARIO} Op1 Tick {i+2}")
        if status == MemoryTickStatus.READ_COMPLETED and req1 == MemoryRequestStatus.ACCEPTED:
            if memory.is_data_ready():
                data = memory.get_read_data_byte()  # 消耗数据 (Consume data)
                print(
                    f"    {SCENARIO} CPU: 从 MAR 0x{memory.mar:X} 读取字节 0x{data:02X} (请求1完成)。")
            # break # 可以不break，继续看idle tick

    print("\n--- 周期精确内存模拟示例结束 (End of Cycle-Accurate Memory Simulation Example) ---")
