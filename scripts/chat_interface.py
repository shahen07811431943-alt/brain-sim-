import sys
import os
import json
import re
import numpy as np
from difflib import SequenceMatcher

print("📬 Receiving a new message from the user...")

# 1. قراءة السؤال من الملف
with open("question.txt", "r", encoding="utf-8") as f:
    user_question = f.read().strip()
print(f"👤 User: {user_question}")

# 2. تحميل قاعدة المعرفة (الملف الذي يحفظ فيه الدماغ معلوماته)
DB_FILE = "brain_conversation.json"
if os.path.exists(DB_FILE):
    with open(DB_FILE, "r", encoding="utf-8") as f:
        knowledge_db = json.load(f)
else:
    print("❌ Error: Brain memory (brain_conversation.json) not found. Please run the training first.")
    sys.exit(1)

# 3. محرك البحث (كما في الكود السابق)
def text_to_vector(text, dim=500):
    vector = np.zeros(dim)
    words = re.sub(r'[^\w\s]', '', text.lower()).split()
    for word in words:
        vector[hash(word) % dim] += 1
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector

def retrieve(query, top_k=2):
    query_vec = text_to_vector(query)
    query_words = set(re.sub(r'[^\w\s]', '', query.lower()).split())
    
    scores = []
    for item in knowledge_db:
        item_vec = np.array(item["vector"])
        vec_score = float(np.dot(query_vec, item_vec))
        item_words = set(re.sub(r'[^\w\s]', '', item["text"].lower()).split())
        keyword_score = len(query_words & item_words) / max(len(query_words), 1)
        title_bonus = 2.0 if item["label"].replace("_", " ") in query else 0.0
        total_score = (vec_score * 0.4) + (keyword_score * 0.4) + (title_bonus * 0.2)
        scores.append((total_score, item))
    
    scores.sort(key=lambda x: x[0], reverse=True)
    return scores[:top_k]

# 4. توليد الرد
def generate_response(query, results):
    if not results or results[0][0] < 0.05:
        return "عذراً، هذه المعلومة ليست في ذاكرتي بعد. سأتعلمها لاحقاً."
    
    best_item = results[0][1]
    response_parts = [f"بخصوص {best_item['label'].replace('_', ' ')}:"]
    
    sentences = [s.strip() + "." for s in best_item["text"].split(".") if len(s.strip()) > 20]
    response_parts.extend(sentences[:3]) # أول 3 جمل للشرح
    
    if "insight" in best_item:
        response_parts.append(f"\n💭 تحليلي الشخصي: {best_item['insight']}")
        
    response_parts.append("\nهل تريد أن أتوسع في نقطة معينة؟")
    return " ".join(response_parts)

# 5. البحث وكتابة الجواب في الملف
print("🧠 Brain is thinking and searching memory...")
results = retrieve(user_question)
brain_answer = generate_response(user_question, results)

# كتابة الجواب في ملف answer.txt
with open("answer.txt", "w", encoding="utf-8") as f:
    f.write(brain_answer)
    
# إضافة المحادثة إلى السجل العام
with open("chat_history.md", "a", encoding="utf-8") as f:
    f.write(f"**👤 أنت:** {user_question}\n\n**🧠 الدماغ:** {brain_answer}\n\n---\n")

print("✅ Brain answered! Check answer.txt")
