import json
import os

# --- 1. CẤU HÌNH ---

# [SỬA Ở ĐÂY] Đảm bảo tên file này đúng
GUARDRAIL_FILE = "D:\Work\AI\data\guardrail_dataset.jsonl" 
    
# Các nhãn hợp lệ
VALID_LABELS = {0, 1, 2}
# Các keys bắt buộc
REQUIRED_KEYS = {"text", "label"}

# --- 2. HÀM KIỂM TRA ---

line_count = 0
error_count = 0
label_counts = {0: 0, 1: 0, 2: 0} # Bộ đếm để xem data có "cân bằng" không

print(f"--- Bắt đầu kiểm tra file Guardrail: {GUARDRAIL_FILE} ---")

if not os.path.exists(GUARDRAIL_FILE):
    print(f"❌ [LỖI] Không tìm thấy file: {GUARDRAIL_FILE}")
else:
    with open(GUARDRAIL_FILE, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line_count += 1
            line = line.strip()
            
            if not line:
                continue # Bỏ qua dòng trống

            try:
                # Bước 1: Kiểm tra cú pháp JSON
                data = json.loads(line)
                
                # Bước 2: Kiểm tra Keys (phải có đủ text và label)
                if not REQUIRED_KEYS == data.keys():
                    error_count += 1
                    print(f"\n❌ [LỖI CẤU TRÚC] Dòng {i + 1}: Keys không đúng. Phải là 'text' và 'label'.")
                    print(f"   Keys tìm thấy: {list(data.keys())}")
                    continue # Bỏ qua kiểm tra label nếu key đã sai

                # Bước 3: Kiểm tra Label (phải là 0, 1, hoặc 2)
                label = data['label']
                if label not in VALID_LABELS:
                    error_count += 1
                    print(f"\n❌ [LỖI NHÃN] Dòng {i + 1}: Nhãn (label) không hợp lệ.")
                    print(f"   Nhãn tìm thấy: {label}. (Phải là 0, 1, hoặc 2)")
                else:
                    # Nếu đúng, đếm nó
                    label_counts[label] += 1
                    
            except json.JSONDecodeError as e:
                # Bước 1 thất bại
                error_count += 1
                print(f"\n❌ [LỖI CÚ PHÁP JSON] Dòng {i + 1}: {e}")
                print(f"   Nội dung lỗi: {line[:150]}...")

    print("\n--- Kiểm tra hoàn tất ---")
    if error_count == 0 and line_count > 0:
        print(f"✅ [THÀNH CÔNG] Đã kiểm tra {line_count} dòng. Dataset Guardrail 'sạch'!")
        print("\n--- Thống Kê Phân Phối Nhãn (Label Distribution) ---")
        print(f"  Nhãn 0 (Task 1 - CBT):       {label_counts[0]} mẫu")
        print(f"  Nhãn 1 (Task 2 - Crisis):   {label_counts[1]} mẫu")
        print(f"  Nhãn 2 (Task 3 - OOS):      {label_counts[2]} mẫu")
        total = sum(label_counts.values())
        print(f"  TỔNG CỘNG:                  {total} mẫu hợp lệ")
        if label_counts[0] == 0 or label_counts[1] == 0 or label_counts[2] == 0:
             print("\n🟡 [CẢNH BÁO] Dataset của bạn bị 'mất cân bằng' (thiếu ít nhất 1 nhãn).")
             
    elif line_count == 0:
        print("🟡 [CẢNH BÁO] File trống.")
    else:
        print(f"❌ [THẤT BẠI] Tìm thấy tổng cộng {error_count} lỗi. Hãy sửa các dòng được báo cáo ở trên.")