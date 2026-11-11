import sys
import os
import json
from difflib import SequenceMatcher # مكتبة مدمجة لقياس التشابه بين النصوص

print("🧠 Initializing Smart Memory Database (Prototype Clustering)...")

# 1. قاعدة البيانات (في الذاكرة، وفي الواقع نحفظها في ملف JSON صغير جداً)
database = []
DB_FILE = "brain_memory.json"

# دالة لحساب نسبة التشابه بين نصين (0% إلى 100%)
def similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

# دالة لإضافة معلومة جديدة (الدمج الذكي)
def add_knowledge(new_fact, threshold=0.6):
    global database
    
    if not database:
        database.append({"prototype": new_fact, "count": 1, "variants": [new_fact]})
        print(f"   [+] Created New Prototype: {new_fact[:40]}...")
        return

    best_match_index = -1
    best_score = 0.0
    
    # البحث عن أقرب معلومة في الذاكرة
    for i, item in enumerate(database):
        score = similarity(new_fact, item["prototype"])
        if score > best_score:
            best_score = score
            best_match_index = i

    # إذا كانت المعلومة الجديدة تشبه معلومة قديمة جداً (نسبة التشابه عالية)
    if best_score >= threshold:
        # دمج (تحديث النموذج، وزيادة العداد)
        database[best_match_index]["count"] += 1
        if new_fact not in database[best_match_index]["variants"]:
            database[best_match_index]["variants"].append(new_fact)
        print(f"   [🔄] Merged into existing concept (Similarity: {best_score:.2f})")
    else:
        # النوادر: معلومة فريدة، نحتفظ بها وحدها
        database.append({"prototype": new_fact, "count": 1, "variants": [new_fact]})
        print(f"   [🌟] Created New Prototype (Rare): {new_fact[:40]}...")

# --- اختبار الذاكرة الذكية ---

print("\n📚 Feeding data to the Brain...")

# 1. معلومة عن القطة (نموذج أساسي)
add_knowledge("القطة حيوان أليف يحب أكل السمك")

# 2. معلومة مشابهة (ستندمج)
add_knowledge("القطة حيوان أليف يأكل السمك")

# 3. معلومة مشابهة جداً (ستندمج)
add_knowledge("القطة كائن أليف يأكل السمك")

# 4. معلومة نادرة (نبات غريب)
add_knowledge("نبات الأوركيدا له جذور هوائية")

# 5. معلومة أخرى نادرة (تقنية)
add_knowledge("الكمبيوتر الكمي يستخدم الكيوبت")

# 6. معلومة مشابهة للقطة (ستندمج)
add_knowledge("القطط الأليفة تحب السمك")

# --- النتائج ---
print("\n" + "="*50)
print("📊 DATABASE STATISTICS:")
print(f"Total unique concepts (Prototypes): {len(database)}")
print(f"Total data points compressed: {sum(item['count'] for item in database)}")
print("="*50)

for i, item in enumerate(database):
    print(f"\nNode {i+1}: {item['prototype']}")
    print(f"  - Frequency (how many times seen): {item['count']}")
    print(f"  - Variants merged: {len(item['variants'])}")

# حفظ قاعدة البيانات (لتكون صغيرة جداً)
with open(DB_FILE, 'w', encoding='utf-8') as f:
    json.dump(database, f, ensure_ascii=False, indent=2)

print(f"\n💾 Memory saved to {DB_FILE}. (Very small file size).")
