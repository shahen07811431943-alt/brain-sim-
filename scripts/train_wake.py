import sys
import os

# هذا السطر يخبر بايثون بالبحث في المجلد الرئيسي للمشروع
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from brain.brain import Brain

print("Starting Wake Phase Training...")
brain = Brain(input_dim=5)
# محاكاة تدريب
for i in range(10):
    stimulus = np.random.rand(5)
    brain.process(stimulus, reward=1.0)
print("Training complete. Saved to hippocampus.")
