import json
import os

# --- 1. CẤU HÌNH ---

# [SỬA Ở ĐÂY] Đảm bảo tên file này đúng
DPO_FILE_PATH = r"D:\Work\AI\dataset_dpo.jsonl"

# Đây là 3 keys bắt buộc của DPO
REQUIRED_KEYS = {"prompt", "chosen", "rejected"}

# --- 2. HÀM KIỂM TRA ---

line_count = 0
error_count = 0

print(f"--- Bắt đầu kiểm tra file DPO: {DPO_FILE_PATH} ---")

if not os.path.exists(DPO_FILE_PATH):
    print(f"❌ [LỖI] Không tìm thấy file: {DPO_FILE_PATH}")
    print("Vui lòng kiểm tra lại tên file!")
else:
    with open(DPO_FILE_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line_count += 1
            
            # Bỏ qua các dòng trống (nếu có)
            if not line.strip():
                continue

            try:
                # Bước 1: Kiểm tra xem có phải JSON hợp lệ không
                data = json.loads(line)
                
                # Bước 2: Kiểm tra xem có đủ 3 keys DPO không
                # (Dùng set.issubset() để kiểm tra nhanh)
                if not REQUIRED_KEYS.issubset(data.keys()):
                    error_count += 1
                    print(f"\n❌ [LỖI CẤU TRÚC] Dòng {i + 1}: Thiếu 1 trong 3 key (prompt, chosen, rejected).")
                    print(f"   Nội dung: {line[:150]}...")

            except json.JSONDecodeError as e:
                # Bước 1 thất bại (JSON bị lỗi cú pháp)
                error_count += 1
                print(f"\n❌ [LỖI CÚ PHÁP JSON] Dòng {i + 1}: {e}")
                print(f"   Nội dung lỗi: {line[:150]}...")

    print("\n--- Kiểm tra hoàn tất ---")
    if error_count == 0 and line_count > 0:
        print(f"✅ [THÀNH CÔNG] Đã kiểm tra {line_count} dòng. Dataset DPO của bạn 'sạch'!")
    elif line_count == 0:
        print("🟡 [CẢNH BÁO] File trống, không có dữ liệu để kiểm tra.")
    else:
        print(f"❌ [THẤT BẠI] Tìm thấy tổng cộng {error_count} lỗi. Hãy sửa các dòng được báo cáo ở trên.")