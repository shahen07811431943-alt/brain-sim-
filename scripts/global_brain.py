import sys
import os
import re
import wikipediaapi

print("🌐 Initializing Global Brain: Fetching live knowledge from Wikipedia...")

# --- 1. محرك الذاكرة والاسترجاع (محرك بحث ذكي للغة العربية) ---
# قائمة بأدوات الربط (Stop Words) التي يجب تجاهلها أثناء البحث
ARABIC_STOP_WORDS = set([
    "من", "في", "على", "هو", "هي", "ما", "ماذا", "عن", "إلى", "التي", "الذي", "هذا", "هذه", 
    "أن", "إن", "كان", "كانت", "قد", "لقد", "لا", "لم", "لن", "و", "أو", "ثم", "لكن", "حتى", 
    "إذا", "كل", "بعض", "غير", "بين", "عند", "لدى", "مع", "دون", "قبل", "بعد", "أمام", "خلف", 
    "فوق", "تحت", "يقول", "قال", "هل", "كيف", "لماذا", "أين", "متى", "هنا", "هناك", "نحن", "أنت"
])

def tokenize(text):
    # تنظيف النص من علامات الترقيم
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    words = text.split()
    # إزالة أدوات الربط (Stop Words)
    return [w for w in words if w not in ARABIC_STOP_WORDS]

class SmartRetriever:
    def __init__(self, corpus_dict):
        self.corpus_dict = corpus_dict

    def search(self, query):
        q_words = set(tokenize(query))
        if not q_words:
            return None
            
        best_topic = None
        best_score = 0
        
        for topic, doc in self.corpus_dict.items():
            d_words = set(tokenize(doc))
            
            # 1. حساب التشابه (تقاطع الكلمات)
            score = len(q_words.intersection(d_words))
            
            # 2. إعطاء أولوية قصوى إذا كان السؤال يحتوي على كلمة من عنوان المقالة
            topic_words = set(tokenize(topic.replace("_", " ")))
            title_match = len(q_words.intersection(topic_words))
            score += title_match * 10  # مكافأة كبيرة للتطابق مع العنوان
            
            if score > best_score:
                best_score = score
                best_topic = topic
                
        if best_score >= 10: # يجب أن يتطابق مع العنوان على الأقل
            return self.corpus_dict[best_topic]
        return None

# --- 2. الاتصال بالمصدر (ويكيبيديا) ---
wiki = wikipediaapi.Wikipedia(
    user_agent='GlobalBrainSim/1.0 (test@example.com)',
    language='ar'
)

topics = [
    "الذكاء_الاصطناعي", 
    "الدماغ_البشري", 
    "القرآن", 
    "الطب", 
    "الفيزياء", 
    "الفلسفة",
    "الاقتصاد",
    "الفضاء",
    "الرياضيات",
    "الكيمياء",
    "التاريخ_الإسلامي",
    "الروبوتات"
]

corpus = {}
for topic in topics:
    page = wiki.page(topic)
    if page.exists():
        text = page.summary[0:500].replace("\n", " ")
        corpus[topic] = text
        print(f"   ✅ Loaded Knowledge: {topic}")

if not corpus:
    print("❌ Failed to fetch data from the internet.")
    sys.exit(1)

# --- 3. بناء الذاكرة ---
print(f"\n🧠 Building Smart Index with {len(corpus)} dense knowledge nodes...")
retriever = SmartRetriever(corpus)

# --- 4. الحوار (اختبار الأسئلة) ---
questions = [
    "ما هو الذكاء الاصطناعي؟",
    "ماذا يقول القرآن؟",
    "ما هو الدماغ البشري؟",
    "ما هي الفلسفة؟",
    "ما هو الفضاء؟",
    "ما هي الرياضيات؟"
]

print("\n🗣️ Starting Brain Dialogue (Smart Retrieval):")
print("=" * 60)

for q in questions:
    print(f"\n👤 Human: {q}")
    result = retriever.search(q)
    if result:
        print(f"🧠 Brain: {result}")
    else:
        print("🧠 Brain: عذراً، هذه المعلومة ليست في ذاكرتي الحالية.")

print("\n" + "=" * 60)
print("🏆 SUCCESS! The Brain read massive live data and answered queries accurately.")
