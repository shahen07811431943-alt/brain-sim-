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
        self.last_prediction = np.zeros(input_dim)
        
    def process(self, stimulus, reward=0.0, label=None):
        activations = self.network.forward(stimulus)
        prediction_error = np.mean((stimulus - self.last_prediction) ** 2)
        self.last_prediction = stimulus.copy()
        is_novel = self.novelty.update_and_detect(activations)
        
        if is_novel or prediction_error > 0.1 or reward > 0:
            weight_change = self.plasticity.update(0.0, 1.0, reward)
            self.network.W += weight_change * np.outer(activations, stimulus) * 0.1
            if label is None:
                label = f"ep_{len(self.memory.hippocampus)}"
            self.memory.encode_episode(stimulus, label=label)
            
        return {"activations": activations, "novelty": is_novel, "error": prediction_error}

    def act(self, state, epsilon=0.2):
        # استكشاف: بنسبة epsilon نختار حركة عشوائية
        if np.random.rand() < epsilon:
            return np.random.randint(4)
        
        # استغلال: نختار الحركة بناءً على ما تعلمناه
        activations = self.network.forward(state)
        action = int(np.argmax(activations)) % 4
        return action
