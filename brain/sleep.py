import numpy as np
import random

class SleepEngine:
    def __init__(self, memory_system):
        self.memory = memory_system

    def run_sleep_cycle(self, network, replay_epochs=50):
        """دورة النوم: إعادة تشغيل الخبرات لتثبيتها"""
        episodes = list(self.memory.hippocampus)
        if not episodes:
            return 0

        for _ in range(replay_epochs):
            # اختيار خبرة عشوائية من الحصين
            pattern, label = random.choice(episodes)
            
            # 1. تثبيت في القشرة
            self.memory.consolidate_to_cortex(pattern, label)
            
            # 2. إعادة تشغيل على الشبكة لتقوية الأوزان
            activations = network.forward(pattern)
            network.W += 0.01 * np.outer(activations, pattern)
            
        return len(episodes)
