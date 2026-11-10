import numpy as np

class STDPLearner:
    def __init__(self, a_plus=0.01, a_minus=0.012, tau_plus=20.0, tau_minus=20.0):
        self.a_plus, self.a_minus = a_plus, a_minus
        self.tau_plus, self.tau_minus = tau_plus, tau_minus
        self.eligibility = 0.0  # أثر الأهلية

    def update(self, pre_spike_t, post_spike_t, reward=0.0):
        if post_spike_t > pre_spike_t:
            delta = self.a_plus * np.exp(-(post_spike_t - pre_spike_t) / self.tau_plus)
        else:
            delta = -self.a_minus * np.exp(-(pre_spike_t - post_spike_t) / self.tau_minus)
        
        # تراكم الأثر مع الوقت
        self.eligibility = 0.9 * self.eligibility + delta 
        
        # التعديل النهائي يعتمد على المكافأة (Three-Factor)
        return self.eligibility * (1.0 + reward)
