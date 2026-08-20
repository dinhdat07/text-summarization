import torch
from transformers import LogitsProcessor
import math

class GTFPenaltyLogitsProcessor(LogitsProcessor):
    """
    GTF-Penalty (Global Term Frequency Penalty) Logits Processor.
    Trừng phạt các từ lặp lại dựa trên độ hiếm của từ đó trong tập huấn luyện.
    Không yêu cầu huấn luyện lại mô hình, chỉ can thiệp vào quá trình giải mã (generation).
    """
    def __init__(self, vocab_frequencies: dict, alpha: float = 1.0, base_penalty: float = 0.6):
        """
        Args:
            vocab_frequencies: Dictionary ánh xạ từ token_id sang số lần xuất hiện trong tập train (C_v^{train}).
            alpha: Hệ số kiểm soát biên độ phạt (mặc định: 1.0).
            base_penalty: Mức phạt cơ bản (b) áp dụng cho mọi từ bị lặp (mặc định: 0.6).
        """
        self.vocab_frequencies = vocab_frequencies
        self.alpha = alpha
        self.base_penalty = base_penalty
        
        # Tìm C_max
        if vocab_frequencies:
            self.c_max = max(vocab_frequencies.values())
        else:
            self.c_max = 1
            
        self.log_c_max = math.log(self.c_max) if self.c_max > 1 else 1.0

        # Tính toán trước (Precompute) P_v cho toàn bộ vocab để tối ưu tốc độ O(1)
        # Giả định token_id lớn nhất trong từ vựng
        max_token_id = max(vocab_frequencies.keys()) if vocab_frequencies else 0
        self.P_v = torch.zeros(max_token_id + 1)
        
        for token_id, count in vocab_frequencies.items():
            if count > 0:
                p_v = self.alpha * (1.0 - (math.log(count) / self.log_c_max))
            else:
                p_v = self.alpha # Phạt tối đa nếu từ không có trong tập train
            self.P_v[token_id] = p_v
            
        # Đưa tensor P_v lên cùng device khi chạy
        self.P_v_device = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        input_ids: (batch_size, sequence_length) chứa các tokens đã sinh
        scores: (batch_size, vocab_size) chứa logits của token tiếp theo
        """
        batch_size, vocab_size = scores.shape
        
        if self.P_v_device is None or self.P_v_device.device != scores.device:
            # Resize self.P_v nếu vocab_size của model lớn hơn max_token_id trong vocab_frequencies
            if vocab_size > len(self.P_v):
                extended_P_v = torch.full((vocab_size,), self.alpha)
                extended_P_v[:len(self.P_v)] = self.P_v
                self.P_v = extended_P_v
                
            self.P_v_device = self.P_v.to(scores.device)

        # Tính tần suất lặp của mỗi từ trong chuỗi hiện tại (C_v^{curr})
        for i in range(batch_size):
            # Lấy chuỗi của batch i
            seq = input_ids[i]
            
            # Đếm số lần xuất hiện của các token (bỏ qua padding/special tokens nếu cần)
            unique_tokens, counts = torch.unique(seq, return_counts=True)
            
            # Chỉ phạt những từ nằm trong phạm vi vocab_size
            valid_mask = unique_tokens < vocab_size
            unique_tokens = unique_tokens[valid_mask]
            counts = counts[valid_mask].float()
            
            if len(unique_tokens) > 0:
                # Áp dụng công thức GTF-Penalty
                # logits'_v = logits_v - [ P_v * C_v^{curr} + b ]
                penalty = (self.P_v_device[unique_tokens] * counts) + self.base_penalty
                
                # Trừ điểm phạt vào logits
                scores[i, unique_tokens] -= penalty

        return scores

# ==========================================
# HƯỚNG DẪN SỬ DỤNG CHO QWEN2.5 HOẶC MÔ HÌNH BẤT KỲ
# ==========================================
if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

    print("Ví dụ tích hợp GTF-Penalty cho Qwen2.5...")
    
    # 1. Khởi tạo tokenizer và model
    # tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    
    # 2. Tạo vocab_frequencies giả lập (Trong thực tế, bạn cần load từ corpus VietNews)
    # Ví dụ: token 'và' (count: 100000) -> P_v thấp, token 'Nguyễn_Văn_A' (count: 10) -> P_v cao
    # vocab_freq = { 1024: 100000, 5032: 10, ... }
    dummy_vocab_freq = {1: 100000, 2: 50, 3: 5} 
    
    # 3. Khởi tạo GTF-Penalty Logits Processor
    gtf_processor = GTFPenaltyLogitsProcessor(
        vocab_frequencies=dummy_vocab_freq,
        alpha=1.0, 
        base_penalty=0.6
    )
    
    # 4. Thêm vào LogitsProcessorList
    logits_processors = LogitsProcessorList([gtf_processor])
    
    # 5. Truyền vào hàm generate
    # input_ids = tokenizer("Ví dụ đoạn văn bản:", return_tensors="pt").input_ids
    # outputs = model.generate(
    #     input_ids,
    #     max_new_tokens=100,
    #     logits_processor=logits_processors  # <--- Tích hợp tại đây
    # )
    print("Hoàn tất cài đặt mô phỏng. Xem code để biết chi tiết tích hợp.")
