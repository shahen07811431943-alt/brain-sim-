import numpy as np

class GridWorld:
    def __init__(self, size=5):
        self.size = size
        self.agent_pos = np.array([0, 0])
        self.goal_pos = np.array([size-1, size-1])

    def reset(self):
        self.agent_pos = np.array([0, 0])
        return self._get_state()

    def _get_state(self):
        # تحويل الموقع إلى مصفوفة يفهمها الدماغ (10 خلايا)
        state = np.zeros(self.size * 2)
        state[self.agent_pos[0]] = 1.0
        state[self.size + self.agent_pos[1]] = 1.0
        return state

    def step(self, action):
        # 0=أعلى, 1=أسفل, 2=يسار, 3=يمين
        if action == 0: self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == 1: self.agent_pos[0] = min(self.size - 1, self.agent_pos[0] + 1)
        elif action == 2: self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == 3: self.agent_pos[1] = min(self.size - 1, self.agent_pos[1] + 1)

        # هل وصل للهدف؟
        done = np.array_equal(self.agent_pos, self.goal_pos)
        reward = 10.0 if done else -0.1 # مكافأة كبيرة للهدف، وعقاب بسيط لكل خطوة
        return self._get_state(), reward, done
