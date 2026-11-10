import numpy as np
from .network import NeuralLayer
from .memory import MemorySystem
from .plasticity import STDPLearner
from .novelty import NoveltyDetector

class Brain:
    def __init__(self, input_dim=10):
        self.network = NeuralLayer(input_dim, input_dim)
        self.memory = MemorySystem()
        self.plasticity = STDPLearner()
        self.novelty = NoveltyDetector()
        
    def process(self, stimulus, reward=0.0):
        # 1. الإدراك
        activations = self.network.forward(stimulus)
        
        # 2. اكتشاف الجدة
        is_novel = self.novelty.update_and_detect(activations)
        
        # 3. التعلم المحلي (اللدونة)
        weight_change = self.plasticity.update(0.0, 1.0, reward)
        self.network.W += weight_change * np.outer(activations, stimulus)
        
        # 4. الحصين (الذاكرة السريعة) إذا كان هناك شيء جديد
        if is_novel or reward > 0:
            self.memory.encode_episode(stimulus, label="novel_pattern")
            
        return is_novel, activations
