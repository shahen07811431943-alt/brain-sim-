import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from brain.brain import Brain

print("🧠 Starting Multimodal Binding (Visual-Motor Association)...")

# 1. الدماغ يدخل إليه 25 بكسل (صورة 5x5)
brain = Brain(input_dim=25)

# 2. تعريف إشارات بصرية بسيطة تعني (تحرك لأعلى، تحرك لليمين)
# سهم للأعلى
img_up = np.array([0,0,1,0,0, 0,1,1,1,0, 0,0,1,0,0, 0,0,1,0,0, 0,0,1,0,0])
# سهم لليمين
img_right = np.array([0,0,1,0,0, 0,0,1,0,0, 0,1,1,1,1, 0,0,1,0,0, 0,0,1,0,0])

# القواعد: إذا رأى سهم الأعلى، يتحرك للأعلى (Action 0). إذا رأى سهم اليمين، يتحرك لليمين (Action 3).
visual_rules = [
    {"image": img_up, "correct_action": 0},
    {"image": img_right, "correct_action": 3}
]

print("Training brain to associate visual cues with motor actions...")

# 3. التدريب: الدماغ يرى الصورة، يتخذ قراراً، ويتعلم من المكافأة
for epoch in range(100):
    for rule in visual_rules:
        state = rule["image"]
        correct_action = rule["correct_action"]
        
        # الدماغ يقرر الحركة بناءً على الصورة التي يراها (بدون استكشاف عشوائي في هذه المرحلة)
        action = brain.act(state, epsilon=0.0)
        
        # هل القرار صحيح؟
        if action == correct_action:
            reward = 1.0 # مكافأة دوبامين
        else:
            reward = -1.0 # عقاب
            
        # الدماغ يتعلم من الخطأ (تحديث أوزان Q)
        # نفترض أن الحالة التالية هي نفسها (لأننا نركز على الربط الحسي الحركي)
        brain.learn(state, action, reward, state)

print("Training complete. Brain has learned Visual-Motor associations.")

# 4. الاختبار: نعرض عليه صورة "سهم الأعلى" المشوشة قليلاً
print("Testing: Presenting a noisy 'UP' arrow to the brain...")
noisy_up = np.clip(img_up + np.random.normal(0, 0.1, 25), 0, 1)

# الدماغ يتخذ قراراً
chosen_action = brain.act(noisy_up, epsilon=0.0)

action_names = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
print(f"Brain saw the image and chose motor action: {action_names[chosen_action]}")

if chosen_action == 0:
    print("✅ Success! Brain correctly linked Visual Perception to Motor Action.")
    sys.exit(0)
else:
    print("❌ Failed to bind vision to action.")
    sys.exit(1)
