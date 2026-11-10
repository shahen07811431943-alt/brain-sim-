import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from brain.brain import Brain
from brain.sleep import SleepEngine

print("Starting Sleep & Memory Consolidation Test...")

brain = Brain(input_dim=5)
sleep_engine = SleepEngine(brain.memory)

# 1. تعلم المهمة A (نرسل الاسم داخل process)
task_A = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
for _ in range(20):
    brain.process(task_A, reward=1.0, label="Task_A")

# 2. تعلم المهمة B
task_B = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
for _ in range(20):
    brain.process(task_B, reward=1.0, label="Task_B")

# 3. اختبار التذكر قبل النوم
label_before, _ = brain.memory.retrieve_from_cortex(task_A)
print(f"Before Sleep, recall for Task A is: {label_before}")

# 4. النوم (إعادة تشغيل الذكريات)
print("Sleeping and consolidating memories...")
sleep_engine.run_sleep_cycle(brain.network, replay_epochs=100)

# 5. اختبار التذكر بعد النوم
label_after, score_after = brain.memory.retrieve_from_cortex(task_A)
print(f"After Sleep, recall for Task A is: {label_after} (Score: {score_after:.2f})")

if label_after == "Task_A":
    print("✅ Success! Catastrophic forgetting prevented by Sleep.")
    sys.exit(0)
else:
    print("❌ Failed to remember Task A.")
    sys.exit(1)
