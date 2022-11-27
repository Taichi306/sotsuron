# Actor loss = - (log_probs*advantage).sum()
# Critic loss = ((v_t-returns).pow(2)).sum()
# advantage = (returns - v_t)

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
        returns = self.calc_returns()[0:-1]
        log_probs = torch.cat(self.log_probs[0:-1])
        advantages = returns + self.gamma*v_t_plus_1 - v_t
        actor_loss = - (log_probs * advantages.detach()).sum()
        return actor_loss

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
        v_t_plus_1 = torch.cat(self.predicted_state_values[1::]) #1/2かけてもどちらでも...
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


env = gym.make("CartPole-v0")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
lr = 0.0001
gamma = 0.99
n_episodes = 300


def calc_returns(rewards):
    R = 0
    returns = []
    for reward in rewards[::-1]:
        R = reward + 0.99 * R
        returns.insert(0, R)
    return returns

def train_separate_networks(actor, critic, n_episodes):
    actor_optim = optim.Adam(actor.parameters())
    critic_optim = optim.Adam(critic.parameters())

    for iter in range(n_episodes):
        display_reward = 0
        env.reset()
        state = env.reset()

        for step in count():
            state = torch.FloatTensor(state).to(device)
            critic(state)
            next_state, reward, done, _ = env.step(actor(state).cpu().numpy())
            display_reward += reward
            actor.rewards.append(torch.tensor([reward], dtype=torch.float, device=device))
            critic.rewards.append(torch.tensor([reward], dtype=torch.float, device=device))

            state = next_state

            if done:
                break
        
        actor_optim.zero_grad()
        critic_optim.zero_grad()

        critic_loss, v_t, v_t_plus_1  = critic.calculateLoss()
        actor_loss = actor.calculateLoss(v_t, v_t_plus_1)
        critic_loss.backward()
        actor_loss.backward()

        actor_optim.step()
        critic_optim.step()

        print(f'episode:{iter}, loss:{actor_loss}, sum_reward:{display_reward}')
        actor.clearMemory()
        critic.clearMemory()