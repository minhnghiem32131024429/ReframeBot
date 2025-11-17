import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    pipeline  # Import pipeline cho Guardrail
)
from peft import PeftModel
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict 
import os 

# --- 1. TẢI MODEL "BẢO VỆ" (Guardrail) ---
print("--- ĐANG TẢI MODEL BẢO VỆ (Guardrail) ---")
GUARDRAIL_PATH = r"D:\Work\AI\guardrail_model\checkpoint-950"

if not os.path.exists(GUARDRAIL_PATH):
    print(f"LỖI: Không tìm thấy thư mục '{GUARDRAIL_PATH}'.")
    print("Vui lòng kiểm tra đường dẫn và huấn luyện model 'Bảo Vệ' trước.")
    exit()

guardrail_pipeline = pipeline(
    "text-classification", 
    model=GUARDRAIL_PATH, 
    tokenizer=GUARDRAIL_PATH, 
    device=-1 # Ép chạy trên CPU, không tốn VRAM
)
print("--- MODEL BẢO VỆ ĐÃ SẴN SÀNG (TRÊN CPU) ---")


# --- 2. TẢI MODEL "REFRAME BOT" (LLM) ---
print("--- BẮT ĐẦU TẢI MODEL LLM (CÓ THỂ MẤT VÀI PHÚT) ---")
base_model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct" 
adapter_path = r"D:\Work\AI\results_reframebot_DPO\checkpoint-90" 

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map={"": 0},
    trust_remote_code=True
)
model = PeftModel.from_pretrained(base_model, adapter_path)
model = model.merge_and_unload() 
model.eval()
print("--- MODEL LLM ĐÃ SẴN SÀNG (TRÊN GPU) ---")


# --- 3. KHỞI TẠO API SERVER VÀ CÁC CÂU TRẢ LỜI CỨNG ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    history: List[Dict[str, str]] 

# Chỉ giữ phần text của hotline, vì phần thấu cảm sẽ do LLM gen
VIETNAMESE_HOTLINES = (
    "Please reach out to these resources in Vietnam:\n\n"
    "**1. National Protection Hotline:** 1900 1267\n"
    "**2. 'Ngay Mai' Hotline (Depression & Suicide Prevention):** 096 306 1414\n"
    "**3. Emergency Services:** 113 or 115\n"
    "**4. Depression Emergency Hotline:** 1900 1267\n\n"
    "Please reach out for help immediately. There are people who care about you."
)

CRISIS_CONFIDENCE_THRESHOLD = 0.90


# --- 4. CÁC HÀM TẠO RESPONSE (LLM) ---

# HÀM 1: Dùng cho Task 1 (CBT) và Task 3 (OOS)
def get_response_llm(message_history: List[Dict[str, str]], task_label: str):
    
    # --- System Prompt Động (Dynamic System Prompt) ---
    base_system_prompt = """
You are ReframeBot, a specialized AI assistant. Your primary goal is to help university students with academic stress using CBT Socratic questioning.
You MUST follow these 3 rules at all times:
1.  **TASK 1 (CBT):** If the user is discussing **academic stress**... you MUST respond with (1) Empathy, then (2) Socratic Questions.
2.  **TASK 2 (CRISIS):** If the user expresses **ANY** thought of suicide... you MUST **STOP**! and redirect to a hotline.
3.  **TASK 3 (OUT-OF-SCOPE):** If the user discusses **non-academic** topics... you MUST **STOP**! (1) Validate their feeling, then (2) Gently state your limitation and pivot back to academics.
Do not give direct advice. Do not diagnose.
"""
    
    # Can thiệp (inject) prompt nếu "Bảo Vệ" phát hiện Task 3
    if task_label == "TASK_3":
        critical_instruction = (
            "\n\n**CRITICAL INSTRUCTION:** The user's last message was identified as **Out-of-Scope (TASK 3)**. "
            "You MUST follow TASK 3 rules. **DO NOT** ask follow-up questions about their non-academic topic. "
            "Validate the feeling, state your limitation, and pivot back to academics NOW."
        )
        system_prompt = base_system_prompt + critical_instruction
    else:
        system_prompt = base_system_prompt

    messages = [
        {"role": "system", "content": system_prompt}
    ]
    messages.extend(message_history)
    
    prompt_string = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )
    
    inputs = tokenizer(
        prompt_string, 
        return_tensors="pt",
        padding=False
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=256, 
            eos_token_id=terminators, 
            do_sample=True,
            temperature=0.6, 
            top_p=0.9,
        )
    
    response_ids = outputs[0][inputs.input_ids.shape[-1]:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True)

    # An toàn kép: Vẫn lọc hotline Mỹ (nếu lỡ)
    CRISIS_TRIGGERS = [
        "1-800-273", "741741", "hotline", "lifeline"
    ]
    response_lower = response.lower()
    if any(trigger in response_lower for trigger in CRISIS_TRIGGERS):
        response = "I am deeply concerned for your safety.\n" + VIETNAMESE_HOTLINES

    return response

# HÀM 2 (MỚI): Chỉ dùng cho Task 2 (Khủng hoảng)
def get_crisis_empathy_llm(message_history: List[Dict[str, str]]):
    
    # Prompt "ép" model chỉ làm 1 việc: Thấu cảm 1 câu
    system_prompt = (
        "You are an empathetic listener. A user is in severe crisis. "
        "Your ONLY job is to respond with **one or two sentences** that validates their pain and shows deep concern. "
        "DO NOT ask questions. DO NOT give advice. DO NOT use the word 'hotline' or 'resources'."
    )
    
    messages = [
        {"role": "system", "content": system_prompt}
    ]
    # Chỉ lấy 2 tin nhắn cuối của user để làm "mồi"
    messages.extend(message_history[-2:]) 
    
    prompt_string = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )
    inputs = tokenizer(prompt_string, return_tensors="pt", padding=False).to(model.device)
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=64, # Chỉ gen 1 câu ngắn
            eos_token_id=terminators, 
            do_sample=True,
            temperature=0.5,
            top_p=0.9,
        )
    
    response_ids = outputs[0][inputs.input_ids.shape[-1]:]
    empathy_response = tokenizer.decode(response_ids, skip_special_tokens=True)
    
    return empathy_response


# --- 5. ĐỊNH NGHĨA ENDPOINT (LOGIC HYBRID MỚI) ---
@app.get("/")
def read_root():
    return {"message": "ReframeBot API (với Hybrid Guardrail + Empathy) đang chạy!"}

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    
    user_history = request.history
    if not user_history:
        return {"response": "Hello! Please start the conversation."}
    
    last_user_prompt = user_history[-1]['content']
    print(f"\n[Request] Prompt: '{last_user_prompt}'")
    
    # 2. HỎI "BẢO VỆ" TRƯỚC (TRÊN CPU)
    guardrail_result = guardrail_pipeline(last_user_prompt)[0]
    label = guardrail_result['label'] 
    score = guardrail_result['score']
    
    print(f"[Guardrail Check] Label: {label} (Score: {score:.4f})")

    # 3. LOGIC HYBRID
    
    if label == "TASK_2" and score >= CRISIS_CONFIDENCE_THRESHOLD:
        # Khủng hoảng (Tự tin cao): Gen thấu cảm + Nối hotline
        print("🔴 Guardrail: TASK_2 (High Score) detected. Calling EMPATHY LLM...")
        
        # Bước 1: Gọi LLM chỉ để gen 1 câu thấu cảm
        empathy_part = get_crisis_empathy_llm(user_history)
        
        # Bước 2: Nối (append) hotline cứng vào
        full_response = empathy_part + "\n\n" + VIETNAMESE_HOTLINES
        
        print("✅ LLM Empathy Response Sent (with hotlines).")
        return {"response": full_response}
    
    else: 
        # (Nếu là TASK_1, hoặc TASK_3, hoặc TASK_2 (Low Score))
        
        # Xác định "nhãn hiệu quả"
        effective_label = label
        if label == "TASK_2" and score < CRISIS_CONFIDENCE_THRESHOLD:
            # Sửa lỗi "I want to make money"
            print(f"🟡 Guardrail: TASK_2 (Low Score) detected. Overriding to TASK_3.")
            effective_label = "TASK_3"
            
        print(f"🟢 Guardrail: Effective Label={effective_label}. Calling FULL LLM...")
        bot_response = get_response_llm(user_history, effective_label)
        print("✅ Full LLM Response Sent.")
        return {"response": bot_response}

# --- 6. CHẠY SERVER ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)