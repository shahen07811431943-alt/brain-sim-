from collections import deque
import numpy as np

class MemorySystem:
    """يحاكي الحصين (سريع) والقشرة (بطيئة ومستقرة)."""
    def __init__(self, hippocampal_capacity=1000):
        self.hippocampus = deque(maxlen=hippocampal_capacity) # ذاكرة عرضية سريعة
        self.cortex_patterns = {} # أنماط قشرية مستقرة

    def encode_episode(self, pattern, label):
        """الحصين يخزن الخبرة فورًا (تعلم من مثال واحد)."""
        self.hippocampus.append((pattern.copy(), label))

    def consolidate_to_cortex(self, pattern, label):
        """القشرة تدمج الأنماط ببطء (أثناء النوم)."""
        if label not in self.cortex_patterns:
            self.cortex_patterns[label] = [pattern.copy()]
        else:
            # متوسط متحرك يحاكي التثبيت البطيء
            self.cortex_patterns[label].append(pattern.copy())
            self.cortex_patterns[label] = [np.mean(self.cortex_patterns[label], axis=0)]

    def retrieve_from_cortex(self, cue):
        """استرجاع من الذاكرة المستقرة."""
        best_label, best_score = None, -1.0
        for label, patterns in self.cortex_patterns.items():
            score = np.dot(cue, patterns[0]) / (np.linalg.norm(cue) * np.linalg.norm(patterns[0]) + 1e-8)
            if score > best_score:
                best_label, best_score = label, score
        return best_label, best_score
