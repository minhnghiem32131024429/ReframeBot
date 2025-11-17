from huggingface_hub import HfApi
import os

HF_REPO_ID = "Nhatminh1234/ReframeBot-Llama-3.1-8B-Adapter" 
LOCAL_CHECKPOINT_PATH = "D:/Work/AI/results_reframebot_llama3/checkpoint-700"

MY_WRITE_TOKEN = os.getenv("HF_TOKEN")  # Dùng environment variable thay vì hardcode

api = HfApi(token=MY_WRITE_TOKEN)

print(f"Đang tải các file từ '{LOCAL_CHECKPOINT_PATH}' lên '{HF_REPO_ID}' (với quyền 'write')...")

api.upload_folder(
    folder_path=LOCAL_CHECKPOINT_PATH,
    repo_id=HF_REPO_ID,
    repo_type="model"
)

print("--- UPLOAD HOÀN TẤT! ---")