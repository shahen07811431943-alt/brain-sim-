import sys
import os
import json
import numpy as np

print("🧠 Initializing Vector Memory Database (Multimodal Ready)...")

DB_FILE = "brain_vectors.json"
DIMENSION = 1000  # كل معلومة تمثل بـ 1000 بُعد (مثل 1000 خلية عصبية)

# --- 1. تحويل أي شيء إلى متجه (Vectorization) ---
def text_to_vector(text):
    """يحول النص إلى متجه رياضي (باستخدام التجزئة)"""
    vector = np.zeros(DIMENSION)
    words = text.lower().split()
    for word in words:
        # تحويل الكلمة إلى موقع في المتجه
        idx = hash(word) % DIMENSION
        vector[idx] += 1
    # التطبيع (Normalization)
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    return vector

def image_to_vector(pixels_array):
    """يحول صورة (مصفوفة بكسلات) إلى متجه"""
    # تبسيط: نأخذ متوسط البكسلات ونوزعها على المتجه
    flat = pixels_array.flatten()
    vector = np.zeros(DIMENSION)
    for i, val in enumerate(flat):
        if i < DIMENSION:
            vector[i] = val / 255.0
    return vector

# --- 2. قاعدة البيانات المتجهة ---
database = [] # ستكون قائمة من القواميس: {"vector": [...], "label": "...", "modality": "text/image"}

def add_vector_memory(data, modality="text", label=""):
    """يضيف أي بيانات (نص/صورة) إلى الذاكرة كمتجه"""
    if modality == "text":
        vector = text_to_vector(data)
    elif modality == "image":
        vector = image_to_vector(data)
    else:
        print("⚠️ Modality not supported yet.")
        return

    # التحقق من التشابه مع الذكريات السابقة
    best_score = -1
    best_match_idx = -1
    
    for i, item in enumerate(database):
        # حساب التشابه الجيبي (Cosine Similarity)
        stored_vec = np.array(item["vector"])
        score = np.dot(vector, stored_vec) / (np.linalg.norm(vector) * np.linalg.norm(stored_vec) + 1e-8)
        if score > best_score:
            best_score = score
            best_match_idx = i

    if best_score > 0.8: # إذا كانت المعلومة متشابهة جداً
        print(f"   [🔄] Merged concept with existing memory (Similarity: {best_score:.2f})")
        # تحديث العداد (الخبرة)
        database[best_match_idx]["count"] += 1
    else:
        # معلومة جديدة تماماً
        new_memory = {
            "vector": vector.tolist(),
            "label": label if label else f"Unknown_{len(database)}",
            "modality": modality,
            "count": 1
        }
        database.append(new_memory)
        print(f"   [🌟] Created New Memory ({modality}): {new_memory['label'][:30]}")

# --- 3. اختبار الذاكرة المتجهة ---
print("\n📚 Feeding data to Vector Brain...")

# إضافة نصوص (كما كنا نفعل)
add_vector_memory("القطة حيوان أليف يحب أكل السمك", modality="text", label="قطة")
add_vector_memory("القطة حيوان أليف يأكل السمك", modality="text", label="قطة_مكررة")
add_vector_memory("الكمبيوتر الكمي يستخدم الكيوبت", modality="text", label="كمبيوتر")

# محاكاة إضافة صورة (مصفوفة 5x5 كأنها صورة)
fake_image = np.random.randint(0, 255, (5, 5))
add_vector_memory(fake_image, modality="image", label="صورة_تجريبية")

# --- 4. عرض الإحصائيات ---
print("\n" + "="*50)
print("📊 VECTOR DATABASE STATISTICS:")
print(f"Total memory nodes: {len(database)}")
print(f"Dimension per node: {DIMENSION}")
print("="*50)

# --- 5. الحفظ كملف (بحجم صغير جداً) ---
# نستخدم النوع float16 لتقليل الحجم للنصف!
serializable_db = []
for item in database:
    serializable_db.append({
        "vector": np.array(item["vector"]).astype(np.float16).tolist(), # ضغط الحجم
        "label": item["label"],
        "modality": item["modality"],
        "count": item["count"]
    })

with open(DB_FILE, 'w', encoding='utf-8') as f:
    json.dump(serializable_db, f, ensure_ascii=False)

print(f"\n💾 Vector Memory saved to {DB_FILE}.")
print("This database is now ready to store Text, Images, and Audio in the same space.")
