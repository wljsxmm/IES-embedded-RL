from torch.utils.tensorboard import SummaryWriter
from replay_buffer import *
from rainbow_dqn import DQN
import argparse
from mp_mdp.environment import *


class Runner:
    def __init__(self, args, number, seed, deterministic, draw):
        self.args = args
        self.number = number
        self.seed = seed
        self.deterministic = deterministic
        self.draw = draw

        env = env_make(mode=1, day_cycle=2)
        env_evaluate = env_make(mode=0, day_cycle=2)

        self.env = env
        self.env_evaluate = env_evaluate  # When evaluating the policy, we need to rebuild an environment

        np.random.seed(seed)
        torch.manual_seed(seed)

        self.args.state_dim = 48  # TODO daily的特征是不是该浓缩一下
        self.args.action_dim = 50
        self.args.episode_limit = 9  # Maximum number of steps per episode

        if args.use_per and args.use_n_steps:
            self.replay_buffer = N_Steps_Prioritized_ReplayBuffer(args)
        elif args.use_per:
            self.replay_buffer = Prioritized_ReplayBuffer(args)
        elif args.use_n_steps:
            self.replay_buffer = N_Steps_ReplayBuffer(args)
        else:
            self.replay_buffer = ReplayBuffer(args)
        self.agent = DQN(args)

        self.algorithm = 'DQN'
        if args.use_double and args.use_dueling and args.use_noisy and args.use_per and args.use_n_steps:
            self.algorithm = 'Rainbow_' + self.algorithm
        else:
            if args.use_double:
                self.algorithm += '_Double'
            if args.use_dueling:
                self.algorithm += '_Dueling'
            if args.use_noisy:
                self.algorithm += '_Noisy'
            if args.use_per:
                self.algorithm += '_PER'
            if args.use_n_steps:
                self.algorithm += "_N_steps"

        self.writer = SummaryWriter(log_dir='runs/DQN/{}_number_{}_seed_{}'.format(self.algorithm, number, seed))

        self.evaluate_num = 0  # Record the number of evaluations
        self.evaluate_rewards = []  # Record the rewards during the evaluating
        self.total_steps = 0  # Record the total steps during the training
        if args.use_noisy:  # 如果使用Noisy net，就不需要epsilon贪心策略了
            self.epsilon = 0
        else:
            self.epsilon = self.args.epsilon_init
            self.epsilon_min = self.args.epsilon_min
            self.epsilon_decay = (self.args.epsilon_init - self.args.epsilon_min) / self.args.epsilon_decay_steps

    def run(self, ):
        self.evaluate_policy()
        while self.total_steps < self.args.max_train_steps:
            episode_steps = 0
            # s = self.env.reset(episode_steps)
            s = self.env.reset_uncertainty(episode_steps, fixed=self.deterministic)
            done = False
            episode_reward = 0
            while not done:
                episode_steps += 1

                self.total_steps += 1
                a = self.agent.choose_action(s, epsilon=self.epsilon)
                action = hst_status[a]
                # s_, r, done = self.env.step(action, episode_steps, self.args.episode_limit, self.draw)
                s_, r, done = self.env.step_uncertainty(action, episode_steps, self.args.episode_limit)

                if not self.args.use_noisy:  # Decay epsilon
                    self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon - self.epsilon_decay > self.epsilon_min else self.epsilon_min

                # When dead or win or reaching the max_episode_steps, done will be Ture, we need to distinguish them;
                # terminal means dead or win,there is no next state s';
                # but when reaching the max_episode_steps,there is a next state s' actually.
                if done and episode_steps != self.args.episode_limit:
                    terminal = True
                else:
                    terminal = False

                if episode_steps != self.args.episode_limit:
                    self.replay_buffer.store_transition(s, a, r, s_, terminal, done)  # Store the transition
                    s = s_

                if self.replay_buffer.current_size >= self.args.batch_size:
                    self.agent.learn(self.replay_buffer, self.total_steps)

                if self.total_steps % self.args.evaluate_freq == 0:
                    self.evaluate_policy()
                    self.agent.save_model(self.algorithm, self.number, self.seed)
                episode_reward += r
                # print('choose action: ', action, 'episode_steps: ', episode_steps, 'current_reward: ', r, 'done: ', done)
            self.writer.add_scalar('train_rewards', episode_reward, global_step=self.total_steps)
        # Save reward
        np.save('./data_train/{}_number_{}_seed_{}.npy'.format(self.algorithm, self.number, self.seed),
                np.array(self.evaluate_rewards))

    def evaluate_policy(self, ):
        evaluate_reward = 0
        self.agent.net.eval()  # Set the model to evaluation mode
        for _ in range(self.args.evaluate_times):
            episode_reward = 0
            episode_steps = 0
            # s = self.env_evaluate.reset(episode_steps)
            s = self.env_evaluate.reset_uncertainty(episode_steps, fixed=self.deterministic)
            done = False
            while not done:
                episode_steps += 1
                if episode_steps == 10:
                    pass
                a = self.agent.choose_action(s, epsilon=0)
                action = hst_status[a]
                # s_, r, done = self.env_evaluate.step(action, episode_steps, self.args.episode_limit, self.draw)
                s_, r, done = self.env_evaluate.step_uncertainty(action, episode_steps, self.args.episode_limit)
                episode_reward += r
                # print('choose action: ', action, 'episode_steps: ', episode_steps, 'episode_reward: ', episode_reward, 'done: ', done)
                s = s_
            evaluate_reward += episode_reward
        self.agent.net.train()  # Set the model to training mode
        evaluate_reward /= self.args.evaluate_times
        self.evaluate_rewards.append(evaluate_reward)
        print("total_steps:{} \t evaluate_reward:{} \t epsilon：{}".format(self.total_steps, evaluate_reward, self.epsilon))
        self.writer.add_scalar('evaluate_rewards', evaluate_reward, global_step=self.total_steps)

    def evaluate(self):
        self.agent.load_model(self.algorithm, self.number, self.seed)
        evaluate_reward = 0
        self.agent.net.eval()  # Set the model to evaluation mode
        for _ in range(self.args.evaluate_times):
            episode_reward = 0
            episode_steps = 0
            # s = self.env_evaluate.reset(episode_steps)
            s = self.env.reset_uncertainty(episode_steps, fixed=self.deterministic)
            done = False
            while not done:
                episode_steps += 1
                a = self.agent.choose_action(s, epsilon=0)
                action = hst_status[a]
                # action = [600, 300]
                # s_, r, done = self.env_evaluate.step(action, episode_steps, self.args.episode_limit, self.draw)
                s_, r, done = self.env.step_uncertainty(action, episode_steps, self.args.episode_limit, self.draw)
                episode_reward += r
                print('choose action: ', action, 'episode_steps: ', episode_steps, 'episode_reward: ', episode_reward,
                      'done: ', done)
                s = s_
            evaluate_reward += episode_reward
        evaluate_reward /= self.args.evaluate_times
        self.evaluate_rewards.append(evaluate_reward)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameter Setting for DQN")
    parser.add_argument("--max_train_steps", type=int, default=int(2e4), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=5e2,
                        help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--evaluate_times", type=float, default=1, help="Evaluate times")

    parser.add_argument("--buffer_capacity", type=int, default=int(1e5), help="The maximum replay-buffer capacity ")
    parser.add_argument("--batch_size", type=int, default=256, help="batch size")
    parser.add_argument("--hidden_dim", type=int, default=512,
                        help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate of actor")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--epsilon_init", type=float, default=0.8, help="Initial epsilon")
    parser.add_argument("--epsilon_min", type=float, default=0.1, help="Minimum epsilon")
    parser.add_argument("--epsilon_decay_steps", type=int, default=int(1.5e4),
                        help="How many steps before the epsilon decays to the minimum")
    parser.add_argument("--tau", type=float, default=0.005, help="soft update the target network")
    parser.add_argument("--use_soft_update", type=bool, default=True, help="Whether to use soft update")
    parser.add_argument("--target_update_freq", type=int, default=200,
                        help="Update frequency of the target network(hard update)")
    parser.add_argument("--n_steps", type=int, default=5, help="n_steps")
    parser.add_argument("--alpha", type=float, default=0.6, help="PER parameter")
    parser.add_argument("--beta_init", type=float, default=0.4, help="Important sampling parameter in PER")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Learning rate Decay")
    parser.add_argument("--grad_clip", type=float, default=10.0, help="Gradient clip")

    parser.add_argument("--use_double", type=bool, default=True, help="Whether to use double Q-learning")
    parser.add_argument("--use_dueling", type=bool, default=True, help="Whether to use dueling network")
    parser.add_argument("--use_noisy", type=bool, default=False, help="Whether to use noisy network")
    parser.add_argument("--use_per", type=bool, default=True, help="Whether to use PER")
    parser.add_argument("--use_n_steps", type=bool, default=True, help="Whether to use n_steps Q-learning")

    args = parser.parse_args()

    seed = 1
    runner = Runner(args=args, number=3, seed=seed, deterministic=False, draw=False)
    runner.run()
    # runner.evaluate()

'''
1: episode_limit = 7  


10: episode_limit = 6  random choose 10 days


20: episode_limit = 17  fixed choose scenario 20 days
'''
