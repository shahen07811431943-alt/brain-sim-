import sys
import os
import json
import re
import numpy as np
import wikipediaapi

print("🧠 Initializing Conversation Cortex (100% Internal Brain)...")

DB_FILE = "brain_conversation.json"
DIMENSION = 500

def text_to_vector(text):
    vector = np.zeros(DIMENSION)
    words = re.sub(r'[^\w\s]', '', text.lower()).split()
    for word in words:
        idx = hash(word) % DIMENSION
        vector[idx] += 1
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector

def build_knowledge():
    print("🌐 Fetching knowledge from Wikipedia...")
    wiki = wikipediaapi.Wikipedia(user_agent='ConvBrain/1.0 (test@example.com)', language='ar')
    
    topics = [
        "الذكاء_الاصطناعي", "الدماغ_البشري", "القرآن", "الطب", 
        "الفيزياء", "الفلسفة", "الاقتصاد", "الفضاء", "الرياضيات",
        "الكيمياء", "التاريخ_الإسلامي", "الروبوتات", "علم_النفس",
        "علم_الأحياء", "الفلك", "الشمس", "القمر", "الماء"
    ]
    
    knowledge = []
    for topic in topics:
        page = wiki.page(topic)
        if page.exists():
            full_text = page.summary
            vector = text_to_vector(full_text)
            words = full_text.split()
            key_insight = " ".join(words[3:15]) if len(words) > 15 else full_text
            
            knowledge.append({
                "label": topic,
                "text": full_text,
                "vector": vector.tolist(),
                "insight": f"المفهوم الأساسي هو: {key_insight}...",
                "count": 1
            })
            print(f"   ✅ Learned: {topic}")
    
    with open(DB_FILE, 'w', encoding='utf-8') as f:
        json.dump(knowledge, f, ensure_ascii=False)
    return knowledge

if os.path.exists(DB_FILE):
    with open(DB_FILE, 'r', encoding='utf-8') as f:
        knowledge_db = json.load(f)
    print(f"📂 Loaded existing knowledge: {len(knowledge_db)} concepts")
else:
    knowledge_db = build_knowledge()

print(f"🏆 Brain Ready with {len(knowledge_db)} concepts stored in {DB_FILE}")
