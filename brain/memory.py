import numpy as np
from collections import deque

class MemorySystem:
    def __init__(self, hippocampal_capacity=100):
        # الحصين: ذاكرة مؤقتة سريعة
        self.hippocampus = deque(maxlen=hippocampal_capacity)
        # القشرة: ذاكرة طويلة المدى
        self.cortex = {} 

    def encode_episode(self, pattern, label):
        """الحصين يخزن الخبرة فوراً"""
        self.hippocampus.append((pattern.copy(), label))

    def consolidate_to_cortex(self, pattern, label):
        """القشرة تدمج الأنماط ببطء (أثناء النوم)"""
        if label not in self.cortex:
            self.cortex[label] = pattern.copy()
        else:
            # متوسط متحرك بطيء (يمنع النسيان الكارثي)
            self.cortex[label] = 0.9 * self.cortex[label] + 0.1 * pattern

    def retrieve_from_cortex(self, cue):
        """استرجاع من الذاكرة المستقرة (القشرة)"""
        best_label, best_score = None, -1.0
        for label, pattern in self.cortex.items():
            denom = (np.linalg.norm(cue) * np.linalg.norm(pattern)) or 1.0
            score = np.dot(cue, pattern) / denom
            if score > best_score:
                best_label, best_score = label, score
        return best_label, best_score
