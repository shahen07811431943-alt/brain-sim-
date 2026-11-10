import numpy as np
from .network import NeuralLayer
from .memory import MemorySystem
from .plasticity import STDPLearner
from .novelty import NoveltyDetector

class Brain:
    def __init__(self, input_dim=5):
        self.network = NeuralLayer(input_dim, input_dim)
        self.memory = MemorySystem()
        self.plasticity = STDPLearner()
        self.novelty = NoveltyDetector()
        self.last_prediction = np.zeros(input_dim)
        
    def process(self, stimulus, reward=0.0):
        # 1. الإدراك
        activations = self.network.forward(stimulus)
        
        # 2. التوقع (مقارنة المحفز الحالي بالتوقع السابق)
        prediction_error = np.mean((stimulus - self.last_prediction) ** 2)
        self.last_prediction = stimulus.copy() # تحديث التوقع
        
        # 3. اكتشاف الجدة
        is_novel = self.novelty.update_and_detect(activations)
        
        # 4. التعلم (يحدث فقط إذا كان هناك خطأ كبير أو مكافأة)
        if is_novel or prediction_error > 0.1 or reward > 0:
            weight_change = self.plasticity.update(0.0, 1.0, reward)
            self.network.W += weight_change * np.outer(activations, stimulus) * 0.1
            
            # 5. التخزين في الحصين
            self.memory.encode_episode(stimulus, label=f"ep_{len(self.memory.hippocampus)}")
            
        return {
            "activations": activations,
            "novelty": is_novel,
            "error": prediction_error
        }
