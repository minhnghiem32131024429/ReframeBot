import os
from huggingface_hub import HfApi

# Lấy token từ environment variable
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    print("❌ Chưa set HF_TOKEN environment variable!")
    print("Chạy: $env:HF_TOKEN='your_token_here'")
    exit(1)

api = HfApi(token=HF_TOKEN)

# Định nghĩa 3 models cần push
models = [
    {
        "name": "SFT Adapter (Prototype 5)",
        "local_path": "D:/Work/AI/Prototypes/Prototype_5/checkpoint-423",
        "repo_id": "Nhatminh1234/ReframeBot-SFT-Llama3.1-8B",
    },
    {
        "name": "DPO Adapter",
        "local_path": "D:/Work/AI/results_reframebot_DPO/checkpoint-90",
        "repo_id": "Nhatminh1234/ReframeBot-DPO-Llama3.1-8B",
    },
    {
        "name": "Guardrail Classifier",
        "local_path": "D:/Work/AI/guardrail_model/checkpoint-950",
        "repo_id": "Nhatminh1234/ReframeBot-Guardrail-DistilBERT",
    }
]

print("=" * 60)
print("🚀 PUSH MODELS LÊN HUGGING FACE")
print("=" * 60)

for idx, model in enumerate(models, 1):
    print(f"\n[{idx}/3] Đang upload: {model['name']}")
    print(f"   📁 Local: {model['local_path']}")
    print(f"   🌐 Repo: {model['repo_id']}")
    
    if not os.path.exists(model['local_path']):
        print(f"   ⚠️  SKIP: Folder không tồn tại!")
        continue
    
    try:
        api.upload_folder(
            folder_path=model['local_path'],
            repo_id=model['repo_id'],
            repo_type="model",
            commit_message=f"Upload {model['name']}"
        )
        print(f"   ✅ Thành công!")
    except Exception as e:
        print(f"   ❌ Lỗi: {e}")

print("\n" + "=" * 60)
print("🎉 HOÀN TẤT!")
print("=" * 60)
