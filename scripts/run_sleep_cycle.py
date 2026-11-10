from brain.brain import Brain
from brain.sleep import SleepEngine
import sys

print("Starting Sleep Phase Consolidation...")
brain = Brain(input_dim=5)
sleep = SleepEngine(brain.memory)
# إعادة تشغيل الخبرات
weights, count = sleep.run_sleep_cycle(brain.network.W)
print(f"Consolidated {count} episodes into cortex.")
sys.exit(0)
