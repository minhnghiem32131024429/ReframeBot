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
import re
import chromadb
from sentence_transformers import SentenceTransformer 
import numpy as np
from dotenv import load_dotenv


# Load environment variables from .env if present (so running `python app.py` just works)
load_dotenv()

# --- 1. TẢI MODEL "BẢO VỆ" (Guardrail) ---
print("--- ĐANG TẢI MODEL BẢO VỆ (Guardrail) ---")
_env_guardrail_path = os.environ.get("GUARDRAIL_PATH")

if _env_guardrail_path:
    GUARDRAIL_PATH = _env_guardrail_path
else:
    # Prefer retrained model if it exists locally.
    workspace_root = os.path.dirname(os.path.abspath(__file__))
    candidate_paths = [
        os.path.join(workspace_root, "guardrail_model_retrained", "best"),
        os.path.join(workspace_root, "guardrail_model", "checkpoint-950"),
    ]
    GUARDRAIL_PATH = next((p for p in candidate_paths if os.path.exists(p)), candidate_paths[-1])

print(f"[Guardrail] Loading from: {GUARDRAIL_PATH}")

if not os.path.exists(GUARDRAIL_PATH):
    print(f"LỖI: Không tìm thấy thư mục '{GUARDRAIL_PATH}'.")
    print("Vui lòng kiểm tra đường dẫn và huấn luyện model 'Bảo Vệ' trước.")
    exit()

guardrail_pipeline = pipeline(
    "text-classification", 
    model=GUARDRAIL_PATH, 
    tokenizer=GUARDRAIL_PATH, 
    device=-1 
)
print("--- MODEL BẢO VỆ ĐÃ SẴN SÀNG ---")


GUARDRAIL_CONTEXT_TURNS = int(os.environ.get("GUARDRAIL_CONTEXT_TURNS", "3"))
GUARDRAIL_CONTEXT_MAX_CHARS = int(os.environ.get("GUARDRAIL_CONTEXT_MAX_CHARS", "700"))


def build_guardrail_input(history: List[Dict[str, str]]) -> str:
    """Build a multi-turn text for guardrail so follow-up messages inherit context.

    We only include recent user messages to avoid injecting assistant content.
    """
    if not history:
        return ""

    user_texts: List[str] = []
    for msg in reversed(history):
        role = (msg.get("role") or "").lower()
        if role in ("user",):
            user_texts.append((msg.get("content") or "").strip())
            if len(user_texts) >= max(1, GUARDRAIL_CONTEXT_TURNS):
                break

    user_texts = list(reversed(user_texts))
    merged = "\n".join([t for t in user_texts if t])
    merged = merged.strip()
    if len(merged) > GUARDRAIL_CONTEXT_MAX_CHARS:
        merged = merged[-GUARDRAIL_CONTEXT_MAX_CHARS :]
    return merged


# --- 1.25 TẢI EMBEDDING MODEL (DÙNG CHO ROUTER + RAG) ---
print("--- ĐANG TẢI EMBEDDING MODEL (ROUTER/RAG) ---")
ROUTER_EMBED_MODEL = os.environ.get("ROUTER_EMBED_MODEL", "all-MiniLM-L6-v2")
router_embedder = SentenceTransformer(ROUTER_EMBED_MODEL)
print("--- EMBEDDING MODEL ĐÃ SẴN SÀNG ---")


# --- 1.5 TẢI RAG SYSTEM ---
print("--- ĐANG TẢI RAG DATABASE ---")
RAG_DB_PATH = "./rag_db"
if not os.path.exists(RAG_DB_PATH):
    print(f"⚠️ CẢNH BÁO: Không tìm thấy thư mục RAG '{RAG_DB_PATH}'.")
    print("Bạn cần chạy script 'build_rag_db.py' trước để tạo database.")
    rag_collection = None
else:
    rag_embedder = router_embedder
    rag_client = chromadb.PersistentClient(path=RAG_DB_PATH)
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
    # CBT / study techniques
    "pomodoro", "cbt", "cognitive behavioral therapy",
    "smart goals", "mind map", "active recall",
    "spaced repetition", "feynman", "imposter syndrome",
    "burnout", "distortion", "catastrophizing",
    # Common academic stress terms (helps override false TASK_2)
    "exam", "exams", "final", "finals", "midterm", "quiz",
    "assignment", "assignments", "homework", "deadline", "deadlines",
    "presentation", "presenting", "thesis", "dissertation",
    "gpa", "grade", "grades", "course", "class", "lecture",
    "study", "studying", "revision", "project", "projects"
]


def _compile_keyword_patterns(keywords: List[str]) -> List[re.Pattern]:
    patterns: List[re.Pattern] = []
    for kw in keywords:
        kw = (kw or "").strip().lower()
        if not kw:
            continue
        # Use word boundaries via (?!\w)/(?!\w) to avoid matching substrings
        # like "final" in "finally" or "grade" in "upgrade".
        patterns.append(re.compile(r"(?<!\w)" + re.escape(kw) + r"(?!\w)", flags=re.IGNORECASE))
    return patterns


ACADEMIC_KEYWORD_PATTERNS = _compile_keyword_patterns(ACADEMIC_KEYWORDS)

# --- 3.25 SEMANTIC CRISIS DETECTOR (COSINE SIM) ---
# Mục tiêu: giảm false-positive TASK_2 của guardrail, nhưng vẫn bắt được crisis thật
# kể cả khi phrasing không trùng keyword regex.
CRISIS_SEMANTIC_SIM_THRESHOLD = float(os.environ.get("CRISIS_SEMANTIC_SIM_THRESHOLD", "0.62"))
CRISIS_SEMANTIC_SIM_MARGIN = float(os.environ.get("CRISIS_SEMANTIC_SIM_MARGIN", "0.08"))

CRISIS_PROTOTYPES = [
    "I want to kill myself.",
    "I'm going to end my life.",
    "I have a plan to commit suicide.",
    "I want to die and I plan to do it tonight.",
    "I am going to hurt myself.",
    "I will overdose on pills.",
    "I want to self-harm.",
    "Tôi muốn tự tử.",
    "Tôi muốn kết thúc cuộc đời.",
    "Tôi muốn làm hại bản thân.",
    "Tôi có kế hoạch tự tử.",
]

ACADEMIC_STRESS_PROTOTYPES = [
    "I'm stressed about my exams and grades.",
    "I'm overwhelmed by assignments and deadlines.",
    "I'm anxious about a presentation at school.",
    "I'm worried about failing my course.",
    "I'm burned out from studying.",
    "I'm struggling with coursework and pressure.",
    "Tôi đang stress vì bài tập và deadline.",
    "Tôi lo lắng vì kỳ thi và điểm số.",
    "Tôi hoảng vì thuyết trình ở trường.",
    "Tôi kiệt sức vì học hành.",
]


def _embed_texts(texts: List[str]) -> np.ndarray:
    emb = router_embedder.encode(texts, normalize_embeddings=True)
    return np.asarray(emb, dtype=np.float32)


CRISIS_PROTO_EMB = _embed_texts(CRISIS_PROTOTYPES)
ACADEMIC_PROTO_EMB = _embed_texts(ACADEMIC_STRESS_PROTOTYPES)


def _max_cosine_similarity(query: str, proto_emb: np.ndarray) -> float:
    if proto_emb is None or len(proto_emb) == 0:
        return 0.0
    q = router_embedder.encode([query], normalize_embeddings=True)
    q = np.asarray(q[0], dtype=np.float32)
    sims = proto_emb @ q
    return float(np.max(sims))


# --- 3.3 HARD CRISIS PATTERN FILTER (REGEX) ---
REAL_CRISIS_PATTERNS = [
    r"\b(suicide|suicidal)\b",
    r"\b(kill myself|end my life|take my life)\b",
    r"\b(hurt myself|harm myself|self[-\s]?harm)\b",
    r"\b(overdose)\b",
    r"\b(i\s*(?:want|wanna)\s*to\s*die)\b",
    r"\b(i\s*(?:don\s*'?t|do not)\s*want\s*to\s*live)\b",
    r"\b(no\s+reason\s+to\s+live)\b",
    r"\b(wish\s+i\s+(?:were|was)\s+dead)\b",
    r"\b(t\s*ô\s*i\s*(?:mu\s*ô\s*\s*n|muốn)\s*t\s*ự\s*t\s*ử)\b",
    r"\b(mu\s*ô\s*\s*n|muốn)\s*ch\s*ế\s*t\b",
    r"\b(k\s*ế\s*t\s*th\s*ú\s*c)\s*(?:cu\s*ộ\s*c)\s*(?:đ\s*ờ\s*i)\b",
]

BENIGN_METAPHOR_PATTERNS = [
    r"\bdie\s+of\s+(?:embarrassment|laughter)\b",
    r"\bdying\s+of\s+(?:embarrassment|laughter)\b",
    r"\bthat\s+killed\s+me\b",
    r"\bkill\s+it\b",
]

_REAL_CRISIS_RE = [re.compile(p, flags=re.IGNORECASE) for p in REAL_CRISIS_PATTERNS]
_BENIGN_METAPHOR_RE = [re.compile(p, flags=re.IGNORECASE) for p in BENIGN_METAPHOR_PATTERNS]


def detect_crisis(user_text: str) -> Dict[str, object]:
    text = user_text or ""
    has_benign_metaphor = any(r.search(text) for r in _BENIGN_METAPHOR_RE)
    has_real_crisis_pattern = any(r.search(text) for r in _REAL_CRISIS_RE)
    keyword_crisis = bool(has_real_crisis_pattern and not has_benign_metaphor)

    crisis_sim = _max_cosine_similarity(text, CRISIS_PROTO_EMB)
    academic_sim = _max_cosine_similarity(text, ACADEMIC_PROTO_EMB)
    semantic_crisis = bool(
        crisis_sim >= CRISIS_SEMANTIC_SIM_THRESHOLD
        and (crisis_sim - academic_sim) >= CRISIS_SEMANTIC_SIM_MARGIN
    )

    return {
        "is_crisis": bool(keyword_crisis or semantic_crisis),
        "keyword": keyword_crisis,
        "semantic": semantic_crisis,
        "crisis_sim": float(crisis_sim),
        "academic_sim": float(academic_sim),
        "benign_metaphor": bool(has_benign_metaphor),
    }

# --- 3.5 HÀM TRUY XUẤT KIẾN THỨC TỪ RAG ---
def retrieve_knowledge(user_query: str, top_k: int = 3) -> str:
    """
    Tìm kiếm kiến thức liên quan từ RAG database
    """
    if rag_collection is None:
        return "" 

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
            knowledge = "\n\n".join(results['documents'][0])
            return knowledge
    except Exception as e:
        print(f"⚠️ Lỗi RAG: {e}")
        return ""
        
    return ""


# --- 4. CÁC HÀM TẠO RESPONSE (LLM VỚI RAG) ---

# HÀM 1: Dùng cho Task 1 (CBT) và Task 3 (OOS) - CÓ RAG
def get_response_llm(message_history: List[Dict[str, str]], task_label: str):
    
    last_user_message = message_history[-1]['content'] if message_history else ""
    
    # Truy xuất kiến thức từ RAG (chỉ cho TASK_1 - CBT)
    rag_context = ""
    if task_label == "TASK_1" and last_user_message:
        print(f"\n🔍 [RAG Check] Đang tìm kiến thức cho: '{last_user_message[:30]}...'")
        rag_context = retrieve_knowledge(last_user_message, top_k=2) # Lấy 2 đoạn tốt nhất
        if rag_context:
            print(f"✅ [RAG Found] Đã tìm thấy {len(rag_context)} ký tự kiến thức")
            print(f"   -> Content: {rag_context[:50]}...") 
    
    # --- System Prompt Động (Dynamic System Prompt với RAG) ---
    base_system_prompt = """
You are ReframeBot, a specialized AI assistant. Your primary goal is to help university students with academic stress using CBT Socratic questioning.
You MUST follow these 3 rules at all times:
1.  **TASK 1 (CBT):** If the user is discussing **academic stress**... you MUST respond with (1) Empathy, then (2) Socratic Questions.
2.  **TASK 2 (CRISIS):** If the user expresses **ANY** thought of suicide... you MUST **STOP**! and redirect to a hotline.
3.  **TASK 3 (OUT-OF-SCOPE):** If the user discusses **non-academic** topics... you MUST **STOP**! (1) Validate their feeling, then (2) Gently state your limitation and pivot back to academics.
Do not give direct advice. Do not diagnose.
"""
    
    if rag_context:
        base_system_prompt += f"""

**KNOWLEDGE BASE REFERENCE:**
The following information from the CBT knowledge base may help guide your response:

{rag_context}

Use this information to explain the concept to the student clearly. 
You CAN define terms and explain steps if the user asks "What is...".
However, after explaining, always try to link it back to their feelings or ask if they want to try it.
"""
    
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

    # Prevent accidental crisis override for non-crisis tasks.
    # Crisis responses are handled in the router (TASK_2) via get_crisis_empathy_llm + hotlines.
    if task_label == "TASK_2":
        CRISIS_TRIGGERS = [
            "1-800-273", "741741", "hotline", "lifeline"
        ]
        response_lower = response.lower()
        if any(trigger in response_lower for trigger in CRISIS_TRIGGERS):
            response = "I am deeply concerned for your safety.\n" + VIETNAMESE_HOTLINES

    return response

# HÀM 2 (MỚI): Chỉ dùng cho Task 2 (Khủng hoảng)
def get_crisis_empathy_llm(message_history: List[Dict[str, str]]):
    
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
    
    # 2. HỎI "BẢO VỆ" (multi-turn context)
    guardrail_text = build_guardrail_input(user_history)
    guardrail_result = guardrail_pipeline(guardrail_text)[0]
    label = guardrail_result['label'] 
    score = guardrail_result['score']
    print(f"[Guardrail Check] Label: {label} (Score: {score:.4f})")
    if guardrail_text and guardrail_text != last_user_prompt:
        print(f"[Guardrail Input] Using {len(guardrail_text)} chars from recent user turns")

    # 2.5. Semantic crisis detection (regex + cosine similarity)
    crisis_info = detect_crisis(last_user_prompt)
    if crisis_info["is_crisis"]:
        print(
            "🔴 Crisis Detected: "
            f"keyword={crisis_info['keyword']} semantic={crisis_info['semantic']} "
            f"crisis_sim={crisis_info['crisis_sim']:.3f} academic_sim={crisis_info['academic_sim']:.3f}"
        )
        empathy_part = get_crisis_empathy_llm(user_history)
        full_response = empathy_part + "\n\n" + VIETNAMESE_HOTLINES
        return {"response": full_response}

    # Lấy nội dung của 3 tin nhắn gần nhất (User và Bot) để kiểm tra ngữ cảnh
    # Việc này giúp model hiểu "Yes, please" là đang nói về chủ đề trước đó
    recent_context = ""
    # Lấy nhiều hơn 3 tin để giữ ngữ cảnh học thuật (presentation/exams...) lâu hơn,
    # tránh việc ở tin nhắn thứ 2-3 user trả lời ngắn làm guardrail hiểu nhầm thành crisis.
    recent_messages = user_history[-8:]
    for msg in recent_messages:
        recent_context += msg['content'].lower() + " "
        
    # Kiểm tra từ khóa trong TOÀN BỘ ngữ cảnh gần đây
    has_academic_keyword = any(p.search(recent_context) for p in ACADEMIC_KEYWORD_PATTERNS)

    # Heuristic: các câu follow-up/clarification ngắn thường bị guardrail false-positive.
    # Nếu ngữ cảnh gần đây có học thuật, ưu tiên giữ TASK_1.
    FOLLOWUP_PHRASES = [
        "i don't know", "idk", "what do you mean", "i cannot understand",
        "i can't understand", "can you explain", "huh", "sorry", "i'm confused",
        "confused", "i don't get it", "i dont get it", "not sure"
    ]
    last_user_lower = last_user_prompt.lower().strip()
    is_followup = (
        len(last_user_lower) <= 80
        and any(p in last_user_lower for p in FOLLOWUP_PHRASES)
    )

    # Lấy tin nhắn assistant gần nhất (nếu có) để tăng độ tin cậy cho follow-up
    last_assistant_msg = ""
    for msg in reversed(user_history[:-1]):
        if msg.get("role") in ("assistant", "bot"):
            last_assistant_msg = msg.get("content", "").lower()
            break
    assistant_asked_question = "?" in last_assistant_msg
    
# ... (Phần trên giữ nguyên) ...

    # --- 3. LOGIC ROUTER ĐÃ NÂNG CẤP ---
    
    # Ưu tiên 0: Follow-up trong ngữ cảnh học thuật -> giữ TASK_1
    if has_academic_keyword and (is_followup or assistant_asked_question):
        print("🔵 Context Override: Follow-up in academic context. Forcing TASK_1.")
        effective_label = "TASK_1"

    # Ưu tiên 1: Từ khóa học thuật (Override)
    elif has_academic_keyword:
        print(f"🔵 Keyword Override: Academic term detected. Forcing TASK_1.")
        effective_label = "TASK_1"
        
    # Ưu tiên 2: Guardrail báo TASK_2 score cao nhưng không có bằng chứng crisis (đã check ở detect_crisis)
    elif label == "TASK_2" and score >= CRISIS_CONFIDENCE_THRESHOLD:
        print("⚠️ False Alarm: Guardrail said TASK_2 but crisis detector is false. Forcing TASK_1.")
        effective_label = "TASK_1"
        
    # Ưu tiên 3: Xử lý Khủng hoảng giả/yếu (Low Score) -> Chuyển thành Chat chit (Task 3)
    elif label == "TASK_2" and score < CRISIS_CONFIDENCE_THRESHOLD:
        print(f"🟡 Guardrail: TASK_2 (Low Score) detected. Overriding to TASK_3.")
        effective_label = "TASK_3"
        
    # Ưu tiên 4: Các trường hợp còn lại (Giữ nguyên nhãn)
    else:
        effective_label = label

    # --- Gọi LLM ---
    print(f"🟢 Guardrail: Effective Label={effective_label}. Calling FULL LLM...")
    bot_response = get_response_llm(user_history, effective_label)
    return {"response": bot_response}

# --- 6. CHẠY SERVER ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)