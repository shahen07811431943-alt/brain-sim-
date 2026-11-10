import numpy as np

class SleepEngine:
    """ينفذ إعادة تشغيل الخبرات من الحصين إلى القشرة أثناء 'النوم'."""
    def __init__(self, memory_system):
        self.memory = memory_system

    def run_sleep_cycle(self, network_weights, replay_epochs=50):
        """دورة النوم: إعادة تشغيل + تحديث الأوزان القشرية."""
        consolidated_count = 0
        for _ in range(replay_epochs):
            # إعادة تشغيل عشوائية للخبرات المخزنة في الحصين
            if not self.memory.hippocampus:
                break
            pattern, label = self.memory.hippocampus[
                np.random.randint(0, len(self.memory.hippocampus))
            ]
            # تثبيت النمط في القشرة
            self.memory.consolidate_to_cortex(pattern, label)
            # هنا يتم تحديث أوزان الشبكة بناءً على النمط المعاد تشغيله
            # (محاكاة: زيادة طفيفة في الأوزان لتعزيز المسار)
            network_weights += 0.001 * np.outer(pattern, pattern)
            consolidated_count += 1
        return network_weights, consolidated_count
