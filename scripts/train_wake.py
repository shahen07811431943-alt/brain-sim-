import numpy as np
from brain.brain import Brain
import sys

print("Starting Wake Phase Training...")
brain = Brain(input_dim=5)
# محاكاة تدريب
for i in range(10):
    stimulus = np.random.rand(5)
    brain.process(stimulus, reward=1.0)
print("Training complete. Saved to hippocampus.")
sys.exit(0)
