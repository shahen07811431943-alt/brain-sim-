from brain.plasticity import STDPLearner

def test_stdp_learning():
    learner = STDPLearner()
    change = learner.update(pre_spike_t=0.0, post_spike_t=10.0, reward=1.0)
    assert change > 0 # يجب أن يقوى المشبك
