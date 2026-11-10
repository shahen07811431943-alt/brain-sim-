import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from brain.network import NeuralLayer
from brain.plasticity import STDPLearner

print("🔮 Starting Predictive Coding Training (Sequence Learning)...")

# 1. شبكة صغيرة لتوقع الأنماط
input_dim = 4
predictor_net = NeuralLayer(input_dim, input_dim)
plasticity = STDPLearner()

# 2. تعريف تسلسل بسيط (مثل: 1 -> 2 -> 3 -> 4)
sequence = [
    np.array([1, 0, 0, 0]),
    np.array([0, 1, 0, 0]),
    np.array([0, 0, 1, 0]),
    np.array([0, 0, 0, 1])
]

print("Training brain to predict the NEXT element in a sequence...")

# 3. التدريب على التسلسل
for epoch in range(100):
    for i in range(len(sequence)):
        current_pattern = sequence[i]
        next_pattern = sequence[(i + 1) % len(sequence)] 
        
        # الدماغ يرى النمط الحالي
        activations = predictor_net.forward(current_pattern)
        
        # التعلم: نعدل الأوزان لتقترب من النمط التالي
        error = next_pattern - activations
        predictor_net.W += 0.1 * np.outer(error, current_pattern)

print("Training complete. Brain has learned the temporal sequence.")

# 4. الاختبار
print("Testing prediction...")
test_pattern = sequence[0] 
activations = predictor_net.forward(test_pattern)

print(f"Input Pattern: {test_pattern}")
print(f"Brain Activations (Prediction): {activations}")

predicted_index = np.argmax(activations)
actual_next_index = np.argmax(sequence[1])

if predicted_index == actual_next_index:
    print(f"✅ Success! Brain correctly predicted the next pattern (Index: {predicted_index}).")
    sys.exit(0)
else:
    print(f"❌ Failed. Brain predicted {predicted_index}, but actual was {actual_next_index}.")
    sys.exit(1)
