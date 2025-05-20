from bf16_module import BF16AddPipeline, BF16MultiplyPipeline

class MACUnit:
    def __init__(self):
        self.multiply_pipeline = BF16MultiplyPipeline()
        self.input1 = 0
        self.input2 = 0
        self.index_queue = []
        self.valid = False
        self.input_valid = False

    def get_input(self, input_valid, input1, input2, sft_index):
        self.input1 = input1
        self.input2 = input2
        self.input_valid = input_valid
        if input_valid:
            self.index_queue.append(sft_index)

    def clock_cycle(self):
        result = self.multiply_pipeline.clock_cycle(
            self.input1, self.input2, self.input_valid
        )
        self.valid = result["valid_output"]
        if self.valid:
            return self.multiply_pipeline.outputs.pop(0)
        return None

    def is_active(self):
        return self.multiply_pipeline.is_active()


class AddUnit:
    def __init__(self):
        self.add_pipeline = BF16AddPipeline()
        self.input1 = 0
        self.input2 = 0
        self.index_queue = []  # 只有相同的index才能被送入加法单元
        self.valid = False
        self.input_valid = False

    def get_input(self, input_valid, input1, input2, sft_index):
        self.input1 = input1
        self.input2 = input2
        self.input_valid = input_valid
        if input_valid:  # 防止污染数据
            self.index_queue.append(sft_index)  # 存放至sft_index队列中

    def clock_cycle(self):
        result = self.add_pipeline.clock_cycle(
            self.input1, self.input2, self.input_valid
        )
        self.valid = result["valid_output"]
        if self.valid:
            return self.add_pipeline.outputs.pop(0)
        return None

    def is_active(self):
        return self.add_pipeline.is_active()
