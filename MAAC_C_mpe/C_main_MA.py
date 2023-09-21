import time
import numpy as np
from tensorboardX import SummaryWriter
from env.make_env import make_env
from common_MA import (ReplayBuffer, save_1d_data, explore_action_2dim, explore_action_4dim, explore_action_6dim)
from common_MA import reward_from_state
import wandb
import datetime
import C_settings_MA


def exploration(action_all, args, agent, cur_episode, noise_stop):
    action_all_before_exploration = []
    for i in range(args.agent_count):
        action_all_before_exploration.append(action_all[i])
    # print("action_all_before_exploration ==>", action_all_before_exploration)
    action_all_after_exploration = []
    if args.env_name in ["routing6v4", "routing12v20", "routing24v128", "simple_spread"]:
        for i in range(args.agent_count):
            temp_act = None
            if args.action_dim_list[i] == 2:
                temp_act, agent.var[i] = explore_action_2dim(action_all_before_exploration[i], args.epsilon, agent.var[i], cur_episode, noise_stop)
            elif args.action_dim_list[i] == 4:
                temp_act = explore_action_4dim(action_all_before_exploration[i], args.epsilon)
            elif args.action_dim_list[i] == 6:
                temp_act = explore_action_6dim(action_all_before_exploration[i], args.epsilon)
            action_all_after_exploration.append(list(temp_act))
    else:
        raise ValueError("args.env_name is not defined! ...")
    # print("action_all_after_exploration ==>", action_all_after_exploration)
    return np.array(action_all_after_exploration)



def training(args, agent, batch, cur_episode, writer=None, training_step=0):
    observation_list, action_list, reward_list, next_observation_list = [], [], [], []
    for i in range(args.agent_count):
        observation_i_batch = np.array([e[0][i] for e in batch])
        action_i_batch = np.array([e[1][i] for e in batch])  # if action_dim==1, need add '.reshape(-1, 1)'
        reward_i_batch = np.array([e[2][i] for e in batch]).reshape(-1, 1)
        next_observation_i_batch = np.array([e[3][i] for e in batch])
        observation_list.append(observation_i_batch)
        action_list.append(action_i_batch)
        reward_list.append(reward_i_batch)
        next_observation_list.append(next_observation_i_batch)
    done_list = np.asarray([e[4] for e in batch]).astype(int).reshape(-1, 1)

    loss_c = agent.train_critic(observation_list, action_list, reward_list, next_observation_list, done_list, writer, training_step)
    loss_a = agent.train_actor(observation_list, writer, training_step)
    if cur_episode > 0 and cur_episode % 100 == 0:
        agent.train_target_network_soft()
        print("soft update executed at episode {}".format(cur_episode))
    # return loss_a, loss_c
    # print("train_actor ==> loss:", loss_a)
    # print("train_critic ==> loss:", loss_c)


def main(args, writer):
    today = datetime.date.today()
    current_date = today.strftime("%d%m%y")
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%H_%M_%S")

    wandb.login(key="efb76db851374f93228250eda60639c70a93d1ec")
    wandb.init(
        # set the wandb project where this run will be logged
        project="MADDPG_FrameWork",
        name='MADDPG_SS_test_'+str(current_date) + '_' + str(formatted_time),
        # track hyperparameters and run metadata
        config={
            "learning_rate": args.lr_actor,
            "epochs": args.episode_count,
        }
    )

    global_training_step = 0
    agent = Agent(args)
    replay_buffer = ReplayBuffer(args.buffer_size, args.batch_size)
    env = make_env("simple_spread")
    args.agent_count = env.n
    n_actions = env.world.dim_p
    n_states = env.observation_space[0].shape[0]
    episode_reward_list = []
    noise_stop = round(args.episode_count / 8)
    for episode in range(args.episode_count):
        # print("=" * 10, "episode", episode)
        args.epsilon -= args.epsilon_delta
        temp_buffer = []  # small trick: current episode experience replay
        episode_reward = 0.0
        observation_all = env.reset()
        for step in range(args.max_episode_len):
            # env.render()
            print("=" * 10, "episode", episode, "***** step", step)
            action_all = agent.generate_action(observation_all)
            # print("current step is {}, agent {}, noise is {}".format(step, 0, agent.var[0]))
            # print("current step is {}, agent {}, noise is {}".format(step, 1, agent.var[1]))
            # print("current step is {}, agent {}, noise is {}".format(step, 2, agent.var[2]))
            action_all = exploration(action_all, args, agent, episode, noise_stop)
            # rewards, next_observation_all, done, _ = env.step(action_all)
            next_observation_all, rewards, done, info = env.step(action_all)
            if True in done:
                done = True
            else:
                done = False
            # make reward consistent with other algorithm
            reward = np.array(rewards)
            rew1 = reward_from_state(next_observation_all, env.agents)
            reward_to_store = rew1 + (np.array(reward, dtype=np.float32) / 100.)

            replay_buffer.add([observation_all, action_all, reward_to_store, next_observation_all, done])
            # temp_buffer.append([observation_all, action_all, rewards, next_observation_all, done])
            observation_all = next_observation_all

            # compute the mean reward among all agents
            # single_step_mean_rw = sum(rewards) / args.agent_count
            # episode_reward += single_step_mean_rw
            # compute the accumulated sum reward
            episode_reward = episode_reward + sum(reward_to_store)

            # train the agent
            batch = replay_buffer.sample()
            # if done or step == args.max_episode_len - 1:  # small trick: always ensure the most recent experience is inside the sampled batch, at the each episode's termination
            #     batch.extend(temp_buffer)
            if len(batch) >= replay_buffer.batch_size:
                # print("executed training when length of experience replay exceed the batch size")
                training(args, agent, batch, episode, writer, global_training_step)
                global_training_step += 1
            if done:
                break
        # mean_ep_rw = episode_reward/args.max_episode_len
        # print("mean episode rewards ==>", mean_ep_rw)
        print("Episode rewards ==>", episode_reward)
        # wandb.log({'mean episode rewards': float(mean_ep_rw)})
        wandb.log({'Episode rewards': float(episode_reward)})
        # episode_reward_list.append(mean_ep_rw)
        # save the model every 250 episodes
        if episode % 250 == 0:
            agent.save_model(str(episode))
    wandb.finish()
    print("all training done")


if __name__ == "__main__":
    args = C_settings_MA.parse_arguments()

    # if args.env_name == "routing6v4":
    #     from C_env_routing6v4 import Environment
    # elif args.env_name == "routing12v20":
    #     from C_env_routing12v20 import Environment
    # elif args.env_name == "routing24v128":
    #     from C_env_routing24v128 import Environment
    # else:
    #     raise ValueError("args.env_name is not defined! ...")

    if args.agent_name in ["IND_AC", "MADDPG", "ATT_MADDPG", "NCC_AC", "MAAC", "Contrastive"]:
        from C_models_MA import Agent
    else:
        raise ValueError("args.agent_name is not defined! ...")

    for exp_id in range(1, args.exp_count + 1):
        args.epsilon = 1.0  # Critically, please always reset this value!!!
        args.exp_id = exp_id
        np.random.seed(args.seed + exp_id)

        time_begin = time.time()
        writer = None  # SummaryWriter(log_dir=f"./log/{args.exp_name}/{exp_id}")
        main(args, writer=writer)
        print("time_used ==>", time.time() - time_begin)
