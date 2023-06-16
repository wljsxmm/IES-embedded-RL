import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import argparse
from normalization import Normalization, RewardScaling
from replaybuffer import ReplayBuffer
from ppo_continuous import PPO_continuous
from mp_mdp.environment import *


def evaluate(args, env_evaluate, agent, state_norm, number, seed):
    # agent.load('./model/PPO_continuous_number_{}_seed_{}.pth'.format(number, seed))
    episode_step, episode_reward, done = 0, 0, False
    # s = env_evaluate.reset(episode_step)
    s = env_evaluate.reset_uncertainty(episode_step, fixed=False)

    if args.use_state_norm:
        s = state_norm(s, update=False)  # During the evaluating,update=False
    while not done:
        episode_step += 1
        a = agent.evaluate(s)  # We use the deterministic policy during the evaluating
        if args.policy_dist == "Beta":
            action = a * args.max_action  # [0,1]->[-max,max]
        else:
            action = a
        # action = [0, 0]
        # s_, r, done = env_evaluate.step(action, episode_step, args.max_episode_steps)
        s_, r, done = env_evaluate.step_uncertainty(action, episode_step, args.max_episode_steps)
        if args.use_state_norm:
            s_ = state_norm(s_, update=False)
        episode_reward += r
        print('choose action: ', action, 'episode_steps: ', episode_step, 'current_reward: ', r, 'done: ', done)
        s = s_
    return episode_reward


def evaluate_policy(args, env, agent, state_norm):
    times = 3
    evaluate_reward = 0
    for _ in range(times):
        episode_step, episode_reward, done = 0, 0, False
        # s = env.reset(episode_step)
        s = env.reset_uncertainty(episode_step, fixed=False)
        if args.use_state_norm:
            s = state_norm(s, update=False)  # During the evaluating,update=False
        while not done:
            episode_step += 1
            a = agent.evaluate(s)  # We use the deterministic policy during the evaluating
            if args.policy_dist == "Beta":
                action = a * args.max_action  # [0,1]->[-max,max]
            else:
                action = a
            # s_, r, done = env.step(action, episode_step, args.max_episode_steps)
            s_, r, done = env.step_uncertainty(action, episode_step, args.max_episode_steps)
            if args.use_state_norm:
                s_ = state_norm(s_, update=False)
            episode_reward += r
            s = s_
        evaluate_reward += episode_reward

    return evaluate_reward / times


def main(args, number, seed, fixed, draw):
    env = env_make(mode=1, day_cycle=3)
    env_evaluate = env_make(mode=0, day_cycle=3)

    np.random.seed(seed)
    torch.manual_seed(seed)

    args.state_dim = 72
    args.action_dim = 2
    args.max_action = 1000
    args.max_episode_steps = 7

    evaluate_num = 0  # Record the number of evaluations
    evaluate_rewards = []  # Record the rewards during the evaluating
    total_steps = 0  # Record the total steps during the training

    replay_buffer = ReplayBuffer(args)
    agent = PPO_continuous(args)

    # Build a tensorboard
    writer = SummaryWriter(log_dir='runs/PPO_continuous/env_number_{}_seed_{}'.format(number, seed))

    state_norm = Normalization(shape=args.state_dim)  # Trick 2:state normalization
    if args.use_reward_norm:  # Trick 3:reward normalization
        reward_norm = Normalization(shape=1)
    elif args.use_reward_scaling:  # Trick 4:reward scaling
        reward_scaling = RewardScaling(shape=1, gamma=args.gamma)

    while total_steps < args.max_train_steps:

        # evaluate_reward = evaluate(args, env_evaluate, agent, state_norm, number, seed)
        # print(1)

        # s = env.reset(0)
        s = env.reset_uncertainty(day=0, fixed=fixed)
        if args.use_state_norm:
            s = state_norm(s)
        if args.use_reward_scaling:
            reward_scaling.reset()
        episode_step = 0
        done = False
        episode_reward = 0
        while not done:
            episode_step += 1
            a, a_logprob = agent.choose_action(s)  # Action and the corresponding log probability
            if args.policy_dist == "Beta":
                action = a * args.max_action  # [0,1]->[-max,max]
            else:
                action = a
            # s_, r, done = env.step(action, episode_step, args.max_episode_steps)
            s_, r, done = env.step_uncertainty(action, episode_step, args.max_episode_steps)

            if args.use_state_norm:
                s_ = state_norm(s_)
            if args.use_reward_norm:
                r = reward_norm(r)
            elif args.use_reward_scaling:
                r = reward_scaling(r)

            # When dead or win or reaching the max_episode_steps, done will be Ture, we need to distinguish them;
            # dw means dead or win,there is no next state s';
            # but when reaching the max_episode_steps,there is a next state s' actually.
            if done and episode_step != args.max_episode_steps:
                dw = True
            else:
                dw = False

            # Take the 'action'，but store the original 'a'（especially for Beta）
            replay_buffer.store(s, a, a_logprob, r, s_, dw, done)
            s = s_
            total_steps += 1
            episode_reward += r

            # When the number of transitions in buffer reaches batch_size,then update
            if replay_buffer.count == args.batch_size:
                agent.update(replay_buffer, total_steps)
                replay_buffer.count = 0

            # if total_steps == 10:
            #     agent.load('./model/PPO_continuous_number_{}_seed_{}.pth'.format(number, seed))
            #     evaluate_reward = evaluate(args, env_evaluate, agent, state_norm, number, seed)
            #     return evaluate_reward

            # Evaluate the policy every 'evaluate_freq' steps
            if total_steps % args.evaluate_freq == 0:
                evaluate_num += 1
                evaluate_reward = evaluate_policy(args, env_evaluate, agent, state_norm)
                evaluate_rewards.append(evaluate_reward)
                print("==========================================================================================")
                print("evaluate_num:{} \t evaluate_reward:{} \t".format(evaluate_num, evaluate_reward))
                print("==========================================================================================")
                writer.add_scalar('step_rewards', evaluate_rewards[-1], global_step=total_steps)
                # Save the rewards
                if evaluate_num % args.save_freq == 0:
                    np.save('./data_train/PPO_continuous_number_{}_seed_{}.npy'.format(number, seed),
                            np.array(evaluate_rewards))
                    agent.save('./model/PPO_continuous_number_{}_seed_{}.pth'.format(number, seed))
        writer.add_scalar('train_rewards', episode_reward, global_step=total_steps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for PPO-continuous")
    parser.add_argument("--max_train_steps", type=int, default=int(4e4), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=5e2,
                        help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument("--policy_dist", type=str, default="Beta", help="Beta or Gaussian")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--hidden_width", type=int, default=64,
                        help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=3e-4, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=True, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=True, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")

    args = parser.parse_args()

    main(args, number=11, seed=10, fixed=False, draw=False)
