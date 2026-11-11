import sys
import os
import wikipediaapi
import random

print("👨‍🏫 Starting Autonomous Learning Loop (Professor Mode)...")
print("The Brain will now ask itself questions, answer them, and comment...\n")

# 1. تهيئة الاتصال بالمكتبة العالمية (ويكيبيديا)
wiki = wikipediaapi.Wikipedia(user_agent='BrainSimProfessor/1.0 (test@example.com)', language='ar')

# 2. قائمة العلوم التي سيدرسها العالِم في يومه الأول
topics = [
    "الفيزياء_الحديثة", 
    "علم_الاجتماع", 
    "علم_النفس", 
    "علم_الأحياء", 
    "الفلك", 
    "الاقتصاد_السلوكي",
    "الروبوتات", 
    "علم_البيئة", 
    "الكيمياء_العضوية"
]

learning_log = []
total_words = 0

print("📝 Professor's Learning Diary:")
print("=" * 70)

for topic in topics:
    page = wiki.page(topic)
    if page.exists():
        # 1. السؤال (الذي يطرحه العالِم على نفسه)
        question = f"ما هو/هي {topic.replace('_', ' ')}؟"
        
        # 2. الإجابة (المعلومة المسترجعة)
        summary = page.summary[0:400].replace("\n", " ")
        answer = summary
        
        # 3. التعليق (محاكاة التفكير النقدي للعالِم)
        # نستخرج معلومة من النص لنتعليق عليها
        words = summary.split()
        key_fact = " ".join(words[5:15]) if len(words) > 15 else summary
        commentaries = [
            f"إذن، {key_fact}.. هذا يفتح آفاقاً جديدة للبحث في هذا المجال.",
            f"المفهوم الأساسي هنا هو: {key_fact}. سأحتاج لربط هذا بمعرفتي السابقة.",
            f"لاحظت أن {key_fact}. هذا يتوافق مع ما أعرفه عن الأنظمة المعقدة.",
            f"بناءً على هذا النص، يمكنني استنتاج أن {key_fact}. سأضيف هذا لملفاتي."
        ]
        commentary = random.choice(commentaries)
        
        # تسجيل الحوار الداخلي
        learning_log.append({
            "topic": topic,
            "question": question,
            "commentary": commentary
        })
        total_words += len(words)
        
        # طباعة الحوار في السجل (لنراه)
        print(f"\n🤔 Professor asked himself: {question}")
        print(f"📖 Answer found: {answer[:120]}...")
        print(f"🧠 Commentary: {commentary}")

# 4. الإحصائيات النهائية (ماذا تعلم العالِم اليوم؟)
print("\n" + "=" * 70)
print("📊 LEARNING STATISTICS:")
print(f"📚 Total topics studied today: {len(learning_log)}")
print(f"📝 Total words read and processed: {total_words}")
print("✅ The Professor has completed his autonomous study session.")
print("The Brain is now smarter. You can ask it about any of these topics next time.")
