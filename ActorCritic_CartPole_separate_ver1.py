class Actor(nn.Module):
    def __init__(self, state_size, action_size, gamma, device):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, self.action_size)

        self.log_probs = []
        self.rewards = []

        self.gamma = gamma
    
    def forward(self, state):
        output1 = F.relu(self.linear1(state))
        output2 = F.relu(self.linear2(output1))
        output3 = F.relu(self.linear3(output2))
        distribution = Categorical(F.softmax(output3, dim=-1))
        action = distribution.sample()
        self.log_probs.append(distribution.log_prob(action).unsqueeze(0))
        return action
    
    def calculateLoss(self, v_t, v_t_plus_1):
        # actor_loss ver1
        log_probs = torch.cat(self.log_probs)
        returns = self.calc_returns()
        ad = (returns - v_t)
        actor_loss = -(log_probs*ad.detach()).sum()
        return actor_loss

        # actor_loss ver2
        # returns = self.calc_returns()[0:-1]
        # log_probs = torch.cat(self.log_probs[0:-1])
        # advantages = returns + self.gamma*v_t_plus_1 - v_t
        # actor_loss = - (log_probs * advantages.detach()).sum()
        # return actor_loss

    def calc_returns(self):
        R = 0
        returns = []
        for reward in self.rewards[::-1]:
            R = reward + self.gamma * R
            returns.insert(0, R)
        return torch.tensor(returns)

    def clearMemory(self):
        del self.log_probs[:]
        del self.rewards[:]

class Critic(nn.Module):
    def __init__(self, state_size, action_size, gamma, device):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, 1)

        self.predicted_state_values = []
        self.rewards = []

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        value = self.linear3(output)
        self.predicted_state_values.append(value) 
    
    def calculateLoss(self):
        v_t = torch.cat(self.predicted_state_values)
        returns = self.calc_returns()
        v_t_plus_1 = (1/2) * torch.cat(self.predicted_state_values[1::]) #1/2かけてもどちらでも...
        value_loss = ((v_t-returns).pow(2)).sum()
        return value_loss, v_t, v_t_plus_1

    def calc_returns(self):
        R = 0 # todo
        returns = []
        for reward in self.rewards[::-1]:
            R = reward + self.gamma * R
            returns.insert(0, R)
        return torch.tensor(returns)
    
    def clearMemory(self):
        del self.predicted_state_values[:]
        del self.rewards[:]

# PuyoPuyoEndlessSmall-v2 settings
env = gym.make('PuyoPuyoEndlessSmall-v2')
OBSERVATION_SPACE = 84
ACTION_SPACE = env.action_space.n
LEARNING_RATE = 0.0001
GAMMA = 0.90

state_size = OBSERVATION_SPACE
action_size = ACTION_SPACE
lr = LEARNING_RATE
gamma = GAMMA
n_episodes = 20000
n_steps = 15

# 学習
def train_actor_critic_separate_networks_puyo(actor, critic, actor_optimizer, critic_optimizer, n_episodes, n_steps):
    total_rewards = []
    total_max_chain = []
    total_chain_cnt = []
    for iter in range(n_episodes):
        env.reset()
        state = to_edited_observation(env.reset())
        episode_display_rewards = 0
        episode_max_chain_length = 0
        episode_chain_cnt = 0

        for step in range(n_steps):
            state = torch.FloatTensor(state).to(device)
            # 現在のvalueを推定
            critic(state)
            next_state, reward, done, _ = env.step(actor(state))

            if reward == -1: # 終了条件の時, この時done=True(ペナルティ)
                reward = -1
            if reward == 0: # 1ターンのコスト
                reward = 0
            if reward > 0: # 連鎖したときの条件
                chain_length = calc_chain_length_small_and_wide_env(reward)
                if episode_max_chain_length < chain_length:
                    episode_max_chain_length = chain_length
                # reward = chain_length
                episode_chain_cnt += 1

            # 両ネットワークにrewardを保存
            actor.rewards.append(reward)
            critic.rewards.append(reward)
            # stateの更新
            state = to_edited_observation(next_state)

            if done: # reward=-10と上書きしている
                break
            if step == n_steps-1: # 1-episodeの終端条件
                break
        

        # 記録
        episode_returns = sum(actor.rewards)
        total_rewards.append(episode_returns)
        total_max_chain.append(episode_max_chain_length)
        total_chain_cnt.append(episode_chain_cnt)
        
        actor_optimizer.zero_grad()
        critic_optimizer.zero_grad()

        critic_loss, v_t, v_t_plus_1 = critic.calculateLoss()

        critic_loss.backward()
        critic_optimizer.step()
        actor_loss = actor.calculateLoss(v_t, v_t_plus_1)
        actor_loss.backward()
        actor_optimizer.step()

        print(f'episode:{iter}, loss:{actor_loss+critic_loss}, max_chain:{episode_max_chain_length}\
        total_rewards:{episode_returns}, episode_chain_cnt:{episode_chain_cnt}')
        critic.clearMemory()
        actor.clearMemory()
    
    return total_rewards, total_max_chain, total_chain_cnt

actor = Actor(state_size, action_size, gamma, device)
critic = Critic(state_size, action_size, gamma, device)
actor_optimizer = optim.Adam(actor.parameters(), lr=lr)
critic_optimizer = optim.Adam(critic.parameters(), lr=lr)
total_rewards, total_max_chain, total_chain_cnt = train_actor_critic_separate_networks_puyo(actor, critic, actor_optimizer, critic_optimizer, n_episodes, n_steps)
