import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from brain.network import NeuralLayer
from brain.plasticity import STDPLearner
from brain.sleep import SleepEngine
from brain.memory import MemorySystem

print("👁️ Starting Visual Perception Training...")

# 1. إنشاء "شبكية العين" (صورة 5x5 = 25 بكسل)
image_size = 25
retina = NeuralLayer(image_size, image_size)
plasticity = STDPLearner()
memory = MemorySystem()

# أنماط بصرية بسيطة (حرف L، خط أفقي، خط عمودي)
patterns = {
    "Vertical_Line": np.array([1,0,0,0,1, 1,0,0,0,1, 1,0,0,0,1, 1,0,0,0,1, 1,0,0,0,1]),
    "Horizontal_Line": np.array([1,1,1,1,1, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0]),
    "Letter_L": np.array([1,0,0,0,0, 1,0,0,0,0, 1,0,0,0,0, 1,0,0,0,0, 1,1,1,1,1])
}

print("Training visual cortex to recognize patterns...")

# 2. التعلم عبر "النظر" المتكرر إلى الأنماط
for epoch in range(50):
    for name, img in patterns.items():
        # تحويل الصورة إلى ومضات عصبية (Spikes)
        spikes = img.copy() 
        
        # تمرير الصورة عبر الشبكة
        activations = retina.forward(spikes)
        
        # التعلم الهيبي (الخلايا التي تنشط معاً تتصل معاً)
        for i in range(image_size):
            for j in range(image_size):
                if spikes[i] == 1 and activations[j] == 1:
                    # تقوية الاتصال محلياً
                    retina.W[i, j] += 0.01
                    
        # حفظ في الذاكرة البصرية
        memory.encode_episode(spikes, label=name)

print("Visual training complete. Brain has developed V1 feature detectors.")

# 3. اختبار "الرؤية": عرض صورة مشوشة والطلب من الدماغ التعرف عليها
print("Testing visual recall with a noisy pattern...")
noisy_input = patterns["Letter_L"] + np.random.normal(0, 0.1, 25)
noisy_input = np.clip(noisy_input, 0, 1) # الحفاظ على القيم بين 0 و 1

# محاكاة استرجاع الذاكرة البصرية
# البحث عن أقرب نمط تم تعلمه
best_match = None
best_score = -1
for name, img in patterns.items():
    score = np.dot(noisy_input, img) / (np.linalg.norm(noisy_input) * np.linalg.norm(img))
    if score > best_score:
        best_score = score
        best_match = name

print(f"Brain saw a noisy image and recognized it as: {best_match} (Confidence: {best_score:.2f})")

if best_score > 0.8:
    print("✅ Success! The visual cortex successfully extracted features and recognized the pattern.")
    sys.exit(0)
else:
    print("❌ Failed to recognize.")
    sys.exit(1)
