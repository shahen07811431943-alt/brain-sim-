import numpy as np

class GridWorld:
    def __init__(self, size=5):
        self.size = size
        self.start_pos = np.array([0, 0])
        
        # الحوائط الثابتة
        self.walls = [
            np.array([1, 1]), np.array([1, 2]), np.array([1, 3]),
            np.array([3, 1]), np.array([3, 2]), np.array([3, 3])
        ]
        
        self.agent_pos = self.start_pos.copy()
        self.goal_pos = np.array([size-1, size-1])
        self.last_distance = 0

    def reset(self):
        self.agent_pos = self.start_pos.copy()
        
        # اختيار هدف عشوائي لا يتداخل مع الحوائط أو البداية
        while True:
            new_goal = np.array([np.random.randint(0, self.size), np.random.randint(0, self.size)])
            if not any(np.array_equal(new_goal, wall) for wall in self.walls) and not np.array_equal(new_goal, self.start_pos):
                self.goal_pos = new_goal
                break

        self.last_distance = abs(self.agent_pos[0] - self.goal_pos[0]) + abs(self.agent_pos[1] - self.goal_pos[1])
        return self._get_state()

    def _get_state(self):
        state = np.zeros(self.size * 2)
        state[self.agent_pos[0]] = 1.0
        state[self.size + self.agent_pos[1]] = 1.0
        return state

    def step(self, action):
        new_pos = self.agent_pos.copy()
        
        if action == 0: new_pos[0] = max(0, new_pos[0] - 1)
        elif action == 1: new_pos[0] = min(self.size - 1, new_pos[0] + 1)
        elif action == 2: new_pos[1] = max(0, new_pos[1] - 1)
        elif action == 3: new_pos[1] = min(self.size - 1, new_pos[1] + 1)

        hit_wall = any(np.array_equal(new_pos, wall) for wall in self.walls)
        
        if hit_wall:
            reward = -1.0
            done = False
        else:
            self.agent_pos = new_pos
            done = np.array_equal(self.agent_pos, self.goal_pos)
            current_distance = abs(self.agent_pos[0] - self.goal_pos[0]) + abs(self.agent_pos[1] - self.goal_pos[1])
            
            if done:
                reward = 10.0
            elif current_distance < self.last_distance:
                reward = 0.1
            else:
                reward = -0.1
                
            self.last_distance = current_distance

        return self._get_state(), reward, done
