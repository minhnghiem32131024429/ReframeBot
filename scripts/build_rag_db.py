import chromadb
from sentence_transformers import SentenceTransformer
import os

# 1. Cấu hình
TEXT_FILE = "data\\knowledge.txt" # File chứa kiến thức chuẩn của bạn
DB_PATH = "./rag_db" # Nơi lưu database

# 2. Tải model Embedding (Chạy CPU, siêu nhẹ)
print("Đang tải model embedding...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# 3. Khởi tạo ChromaDB
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection(name="cbt_knowledge")

# 4. Đọc và Cắt nhỏ văn bản (Chunking)
print("Đang đọc và cắt nhỏ tài liệu...")
with open(TEXT_FILE, 'r', encoding='utf-8') as f:
    text = f.read()

# Cắt thành các đoạn khoảng 300 ký tự (đơn giản hóa)
chunks = [text[i:i+500] for i in range(0, len(text), 400)] 

# 5. Nhúng (Embed) và Lưu vào DB
print(f"Đang lưu {len(chunks)} đoạn kiến thức vào DB...")
ids = [str(i) for i in range(len(chunks))]
embeddings = embedder.encode(chunks).tolist()

collection.add(
    documents=chunks,
    embeddings=embeddings,
    ids=ids
)

print("✅ Đã tạo xong RAG Database!")