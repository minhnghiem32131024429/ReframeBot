import json
import os

file_path = "D:\\Work\\AI\\dataset.jsonl"
line_count = 0
error_count = 0

print(f"--- Bắt đầu kiểm tra file: {file_path} ---")

if not os.path.exists(file_path):
    print(f"[LỖI] Không tìm thấy file: {file_path}")
else:
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line_count += 1
            try:
                # Thử đọc từng dòng như một đối tượng JSON
                json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[LỖI CÚ PHÁP] Dòng {i + 1}: {e}")
                print(f"   Nội dung lỗi: {line[:100]}...") # In 100 ký tự đầu của dòng lỗi
                error_count += 1

    print("--- Kiểm tra hoàn tất ---")
    if error_count == 0:
        print(f"✅ [THÀNH CÔNG] Đã kiểm tra {line_count} dòng. Không tìm thấy lỗi cú pháp!")
    else:
        print(f"❌ [THẤT BẠI] Tìm thấy {error_count} lỗi cú pháp. Hãy sửa các dòng trên.")