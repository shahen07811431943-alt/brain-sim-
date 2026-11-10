import numpy as np

class GridWorld:
    def __init__(self, size=5):
        self.size = size
        self.agent_pos = np.array([0, 0])
        self.goal_pos = np.array([size-1, size-1])
        self.last_distance = 0

    def reset(self):
        self.agent_pos = np.array([0, 0])
        self.last_distance = abs(self.agent_pos[0] - self.goal_pos[0]) + abs(self.agent_pos[1] - self.goal_pos[1])
        return self._get_state()

    def _get_state(self):
        state = np.zeros(self.size * 2)
        state[self.agent_pos[0]] = 1.0
        state[self.size + self.agent_pos[1]] = 1.0
        return state

    def step(self, action):
        if action == 0: self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == 1: self.agent_pos[0] = min(self.size - 1, self.agent_pos[0] + 1)
        elif action == 2: self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == 3: self.agent_pos[1] = min(self.size - 1, self.agent_pos[1] + 1)

        done = np.array_equal(self.agent_pos, self.goal_pos)
        
        current_distance = abs(self.agent_pos[0] - self.goal_pos[0]) + abs(self.agent_pos[1] - self.goal_pos[1])
        
        # نظام المكافآت الجديد:
        if done:
            reward = 10.0 # مكافأة كبيرة عند الوصول
        elif current_distance < self.last_distance:
            reward = 0.1  # مكافأة صغيرة عند الاقتراب
        else:
            reward = -0.1 # عقاب عند الابتعاد أو الاصطدام
            
        self.last_distance = current_distance
        return self._get_state(), reward, done
