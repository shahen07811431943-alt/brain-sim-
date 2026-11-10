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
        
        # شبكة Q الجديدة (تحاكي الدوبامين): 4 أفعال × عدد المدخلات
        self.q_weights = np.random.randn(4, input_dim) * 0.01
        
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
        if np.random.rand() < epsilon:
            return np.random.randint(4) # استكشاف
        
        # استغلال: نختار الحركة صاحبة أعلى قيمة Q
        q_values = self.q_weights @ state
        return int(np.argmax(q_values))

    def learn(self, state, action, reward, next_state, gamma=0.9, alpha=0.1):
        """تحديث أوزان Q باستخدام فارق التوقع (TD Error) - يحاكي الدوبامين"""
        q_current = self.q_weights[action] @ state
        q_next = np.max(self.q_weights @ next_state)
        
        # خطأ التوقع (الدوبامين): المكافأة + قيمة المستقبل - قيمة الحاضر
        td_error = reward + (gamma * q_next) - q_current
        
        # تحديث أوزان الفعل بناءً على الخطأ
        self.q_weights[action] += alpha * td_error * state
