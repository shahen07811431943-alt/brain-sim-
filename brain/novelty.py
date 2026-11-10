import numpy as np

class NoveltyDetector:
    def __init__(self, threshold_percentile=95):
        self.threshold_percentile = threshold_percentile
        self.firing_history = []

    def update_and_detect(self, firing_rates):
        self.firing_history.append(np.array(firing_rates))
        if len(self.firing_history) < 5:
            return True # في البداية كل شيء جديد
            
        all_rates = np.concatenate(self.firing_history)
        dynamic_threshold = np.percentile(all_rates, self.threshold_percentile)
        
        # إذا كان النشاط الحالي أعلى من المعتاد، فهذا يعني جدة
        return np.mean(firing_rates) > dynamic_threshold * 1.2
