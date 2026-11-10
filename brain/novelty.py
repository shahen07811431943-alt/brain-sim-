import numpy as np

class NoveltyDetector:
    """يكتشف المدخلات الجديدة عبر عتبة تفاضلية على معدلات الإطلاق."""
    def __init__(self, threshold_percentile=99):
        self.threshold_percentile = threshold_percentile
        self.firing_history = []

    def update_and_detect(self, firing_rates):
        self.firing_history.append(np.array(firing_rates))
        if len(self.firing_history) < 10:
            return False # لا يمكننا الحكم بعد
        
        # حساب العتبة الديناميكية (DTF)
        all_rates = np.concatenate(self.firing_history)
        dynamic_threshold = np.percentile(all_rates, self.threshold_percentile)
        
        # إذا كان معدل الإطلاق الحالي أعلى بكثير من المعتاد -> جدة
        if np.mean(firing_rates) > dynamic_threshold * 1.5:
            return True
        return False
