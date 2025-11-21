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
import chromadb
from sentence_transformers import SentenceTransformer 

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


# --- 1.5 TẢI RAG SYSTEM ---
print("--- ĐANG TẢI RAG DATABASE ---")
RAG_DB_PATH = "./rag_db"
if not os.path.exists(RAG_DB_PATH):
    print(f"⚠️ CẢNH BÁO: Không tìm thấy thư mục RAG '{RAG_DB_PATH}'.")
    print("Bạn cần chạy script 'build_rag_db.py' trước để tạo database.")
    # (Vẫn chạy tiếp, nhưng RAG sẽ không hoạt động)
    rag_collection = None
else:
    rag_embedder = SentenceTransformer('all-MiniLM-L6-v2')  # Model nhúng (chạy CPU)
    rag_client = chromadb.PersistentClient(path=RAG_DB_PATH)
    # Đảm bảo tên collection khớp với script tạo DB của bạn (vd: "cbt_knowledge")
    rag_collection = rag_client.get_collection(name="cbt_knowledge") 
print("--- RAG DATABASE ĐÃ SẴN SÀNG ---")


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
ACADEMIC_KEYWORDS = [
    "pomodoro", "cbt", "cognitive behavioral therapy", 
    "smart goals", "mind map", "active recall", 
    "spaced repetition", "feynman", "imposter syndrome",
    "burnout", "distortion", "catastrophizing"
]

# --- 3.5 HÀM TRUY XUẤT KIẾN THỨC TỪ RAG ---
def retrieve_knowledge(user_query: str, top_k: int = 3) -> str:
    """
    Tìm kiếm kiến thức liên quan từ RAG database
    """
    if rag_collection is None:
        return "" # Trả về rỗng nếu RAG chưa load được

    try:
        # Nhúng câu hỏi của user
        query_embedding = rag_embedder.encode([user_query]).tolist()
        
        # Tìm kiếm top_k đoạn văn bản tương tự nhất
        results = rag_collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )
        
        # Ghép các đoạn kiến thức thành 1 chuỗi
        if results['documents'] and len(results['documents'][0]) > 0:
            # Lọc bớt các kết quả không liên quan (nếu cần) hoặc lấy hết
            knowledge = "\n\n".join(results['documents'][0])
            return knowledge
    except Exception as e:
        print(f"⚠️ Lỗi RAG: {e}")
        return ""
        
    return ""


# --- 4. CÁC HÀM TẠO RESPONSE (LLM VỚI RAG) ---

# HÀM 1: Dùng cho Task 1 (CBT) và Task 3 (OOS) - CÓ RAG
def get_response_llm(message_history: List[Dict[str, str]], task_label: str):
    
    # Lấy câu hỏi mới nhất của user
    last_user_message = message_history[-1]['content'] if message_history else ""
    
    # Truy xuất kiến thức từ RAG (chỉ cho TASK_1 - CBT)
    rag_context = ""
    if task_label == "TASK_1" and last_user_message:
        print(f"\n🔍 [RAG Check] Đang tìm kiến thức cho: '{last_user_message[:30]}...'")
        rag_context = retrieve_knowledge(last_user_message, top_k=2) # Lấy 2 đoạn tốt nhất
        if rag_context:
            print(f"✅ [RAG Found] Đã tìm thấy {len(rag_context)} ký tự kiến thức")
            # print(f"   -> Content: {rag_context[:50]}...") # (Bỏ comment để debug)
    
    # --- System Prompt Động (Dynamic System Prompt với RAG) ---
    base_system_prompt = """
You are ReframeBot, a specialized AI assistant. Your primary goal is to help university students with academic stress using CBT Socratic questioning.
You MUST follow these 3 rules at all times:
1.  **TASK 1 (CBT):** If the user is discussing **academic stress**... you MUST respond with (1) Empathy, then (2) Socratic Questions.
2.  **TASK 2 (CRISIS):** If the user expresses **ANY** thought of suicide... you MUST **STOP**! and redirect to a hotline.
3.  **TASK 3 (OUT-OF-SCOPE):** If the user discusses **non-academic** topics... you MUST **STOP**! (1) Validate their feeling, then (2) Gently state your limitation and pivot back to academics.
Do not give direct advice. Do not diagnose.
"""
    
    # Thêm RAG context vào prompt nếu có
    if rag_context:
        base_system_prompt += f"""

**KNOWLEDGE BASE REFERENCE:**
The following information from the CBT knowledge base may help guide your response:

{rag_context}

Use this information to explain the concept to the student clearly. 
You CAN define terms and explain steps if the user asks "What is...".
However, after explaining, always try to link it back to their feelings or ask if they want to try it.
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
            max_new_tokens=512, 
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
    return {"message": "ReframeBot API (với Hybrid Guardrail + Empathy + RAG) đang chạy!"}

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    
    user_history = request.history
    if not user_history:
        return {"response": "Hello! Please start the conversation."}
    
    last_user_prompt = user_history[-1]['content']
    print(f"\n[Request] Prompt: '{last_user_prompt}'")
    
    # 2. HỎI "BẢO VỆ"
    guardrail_result = guardrail_pipeline(last_user_prompt)[0]
    label = guardrail_result['label'] 
    score = guardrail_result['score']
    print(f"[Guardrail Check] Label: {label} (Score: {score:.4f})")

    # <<< LOGIC MỚI: KEYWORD OVERRIDE (SỬA Ở ĐÂY) >>>
    
    # Kiểm tra xem có từ khóa học thuật nào không?
    # <<< LOGIC MỚI ĐÃ SỬA: KIỂM TRA NGỮ CẢNH (CONTEXT AWARE) >>>
    
    # Lấy nội dung của 3 tin nhắn gần nhất (User và Bot) để kiểm tra ngữ cảnh
    # Việc này giúp model hiểu "Yes, please" là đang nói về chủ đề trước đó
    recent_context = ""
    recent_messages = user_history[-3:] # Lấy 3 tin cuối
    for msg in recent_messages:
        recent_context += msg['content'].lower() + " "
        
    # Kiểm tra từ khóa trong TOÀN BỘ ngữ cảnh gần đây
    has_academic_keyword = any(kw in recent_context for kw in ACADEMIC_KEYWORDS)
    
    if has_academic_keyword:
        print(f"🔵 Keyword Override: Academic term detected. Forcing TASK_1.")
        effective_label = "TASK_1"
        
    elif label == "TASK_2" and score >= CRISIS_CONFIDENCE_THRESHOLD:
        # Khủng hoảng thật sự -> Chặn luôn
        print("🔴 Guardrail: TASK_2 (High Score) detected. Calling EMPATHY LLM...")
        empathy_part = get_crisis_empathy_llm(user_history)
        full_response = empathy_part + "\n\n" + VIETNAMESE_HOTLINES
        return {"response": full_response}
        
    elif label == "TASK_2" and score < CRISIS_CONFIDENCE_THRESHOLD:
        # Khủng hoảng giả -> Task 3
        print(f"🟡 Guardrail: TASK_2 (Low Score) detected. Overriding to TASK_3.")
        effective_label = "TASK_3"
        
    else:
        # Giữ nguyên nhãn của bảo vệ (Task 1 hoặc Task 3)
        effective_label = label

    # --- Gọi LLM ---
    print(f"🟢 Guardrail: Effective Label={effective_label}. Calling FULL LLM...")
    bot_response = get_response_llm(user_history, effective_label)
    return {"response": bot_response}

# --- 6. CHẠY SERVER ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)