import sys
import os

# هذا السطر يخبر بايثون بالبحث في المجلد الرئيسي للمشروع
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from brain.brain import Brain
from brain.sleep import SleepEngine

print("Starting Sleep Phase Consolidation...")
brain = Brain(input_dim=5)
sleep = SleepEngine(brain.memory)
# إعادة تشغيل الخبرات
weights, count = sleep.run_sleep_cycle(brain.network.W)
print(f"Consolidated {count} episodes into cortex.")
