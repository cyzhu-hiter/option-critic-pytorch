import logging
import os
import time
import wandb
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# class TensorboardLogger():
#     def __init__(self, logdir, run_name):
#         self.log_name = logdir + '/' + run_name
#         self.tf_writer = None
#         self.start_time = time.time()
#         self.n_eps = 0

#         if not os.path.exists(self.log_name):
#             os.makedirs(self.log_name)

#         self.writer = SummaryWriter(self.log_name)

#         logging.basicConfig(
#             level=logging.DEBUG,
#             format='%(asctime)s %(message)s',
#             handlers=[
#                 logging.StreamHandler(),
#                 logging.FileHandler(self.log_name + '/logger.log'),
#                 ],
#             datefmt='%Y/%m/%d %I:%M:%S %p'
#             )

#     def log_episode(self, steps, reward, option_lengths, ep_steps, epsilon, learning_rate):
#         self.n_eps += 1
#         logging.info(f"> ep {self.n_eps} done. total_steps={steps} | reward={reward} | episode_steps={ep_steps} "\
#             f"| hours={(time.time()-self.start_time) / 60 / 60:.3f} | epsilon={epsilon:.3f}")
#         self.writer.add_scalar(tag="episodic_rewards", scalar_value=reward, global_step=self.n_eps)
#         self.writer.add_scalar(tag='episode_lengths', scalar_value=ep_steps, global_step=self.n_eps)

#         # Keep track of options statistics
#         for option, lens in option_lengths.items():
#             # Need better statistics for this one, point average is terrible in this case
#             self.writer.add_scalar(tag=f"option_{option}_avg_length", scalar_value=np.mean(lens) if len(lens)>0 else 0, global_step=self.n_eps)
#             self.writer.add_scalar(tag=f"option_{option}_active", scalar_value=sum(lens)/ep_steps, global_step=self.n_eps)
#     def log_data(self, step, actor_loss, critic_loss, entropy, epsilon):
#         if actor_loss:
#             self.writer.add_scalar(tag="actor_loss", scalar_value=actor_loss.item(), global_step=step)
#         if critic_loss:
#             self.writer.add_scalar(tag="critic_loss", scalar_value=critic_loss.item(), global_step=step)
#         self.writer.add_scalar(tag="policy_entropy", scalar_value=entropy, global_step=step)
#         self.writer.add_scalar(tag="epsilon",scalar_value=epsilon, global_step=step)

# class WandbLogger():
#     def __init__(self, logdir, run_name):
#         self.log_name = logdir + '/' + run_name
#         self.start_time = time.time()
#         self.n_eps = 0

#         if not os.path.exists(self.log_name):
#             os.makedirs(self.log_name)

#         # W&B initialization for this run
#         wandb.init(project="option", name=run_name, config={"log_dir": logdir})

#         logging.basicConfig(
#             level=logging.DEBUG,
#             format='%(asctime)s %(message)s',
#             handlers=[
#                 logging.StreamHandler(),
#                 logging.FileHandler(self.log_name + '/logger.log'),
#             ],
#             datefmt='%Y/%m/%d %I:%M:%S %p'
#         )

#     def log_episode(self, steps, reward, option_lengths, ep_steps, epsilon):
#         self.n_eps += 1
#         logging.info(f"> ep {self.n_eps} done. total_steps={steps} | reward={reward} | episode_steps={ep_steps} "\
#                      f"| hours={(time.time() - self.start_time) / 60 / 60:.3f} | epsilon={epsilon:.3f}")
        
#         # Log episode data to W&B
#         wandb.log({"episodic_rewards": reward, "episode_lengths": ep_steps, "epsilon": epsilon, "step": steps})

#         # Log options statistics
#         for option, lens in option_lengths.items():
#             if len(lens) > 0:
#                 wandb.log({f"option_{option}_avg_length": np.mean(lens), f"option_{option}_active": sum(lens)/ep_steps, "step": steps})

#     def log_data(self, step, actor_loss, critic_loss, entropy, epsilon):
#         wandb.log({"actor_loss": actor_loss.item() if actor_loss else None,
#                    "critic_loss": critic_loss.item() if critic_loss else None,
#                    "policy_entropy": entropy,
#                    "epsilon": epsilon,
#                    "step": step})

class TensorboardLogger():
    def __init__(self, logdir, run_name):
        self.log_name = logdir + '/' + run_name
        self.tf_writer = None
        self.start_time = time.time()
        self.n_eps = 0
        if not os.path.exists(self.log_name):
            os.makedirs(self.log_name)
        self.writer = SummaryWriter(self.log_name)
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.log_name + '/logger.log'),
            ],
            datefmt='%Y/%m/%d %I:%M:%S %p'
        )

    def log_episode(self, steps, reward, option_lengths, ep_steps, epsilon, learning_rate):
        self.n_eps += 1
        logging.info(f"> ep {self.n_eps} done. total_steps={steps} | reward={reward} | episode_steps={ep_steps} "\
            f"| hours={(time.time()-self.start_time) / 60 / 60:.3f} | epsilon={epsilon:.3f} | lr={learning_rate:.6f}")
        self.writer.add_scalar("Training/Episodic Rewards", reward, self.n_eps)
        self.writer.add_scalar('Training/Episode Lengths', ep_steps, self.n_eps)
        self.writer.add_scalar("Training/Learning Rate", learning_rate, self.n_eps)

        for option, lens in option_lengths.items():
            self.writer.add_scalar(f"Options/Option {option} Average Length", np.mean(lens) if len(lens) > 0 else 0, self.n_eps)
            self.writer.add_scalar(f"Options/Option {option} Activation", sum(lens) / ep_steps, self.n_eps)

    def log_data(self, step, actor_loss, critic_loss, entropy, epsilon):
        if actor_loss:
            self.writer.add_scalar(tag="Policy/actor_loss", scalar_value=actor_loss.item(), global_step=step)
        if critic_loss:
            self.writer.add_scalar(tag="Policy/critic_loss", scalar_value=critic_loss.item(), global_step=step)
        self.writer.add_scalar(tag="Policy/policy_entropy", scalar_value=entropy, global_step=step)
        self.writer.add_scalar(tag="Policy/epsilon",scalar_value=epsilon, global_step=step)

    def log_customed_output(self, data):
        logging.info(data)

class WandbLogger():
    def __init__(self, logdir, run_name):
        self.log_name = logdir + '/' + run_name
        self.start_time = time.time()
        self.n_eps = 0

        if not os.path.exists(self.log_name):
            os.makedirs(self.log_name)
        wandb.init(project="option_critic", name=run_name, config={"log_dir": logdir})
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.log_name + '/logger.log'),
            ],
            datefmt='%Y/%m/%d %I:%M:%S %p'
        )

    def log_episode(self, steps, reward, option_lengths, ep_steps, epsilon, learning_rate):
        self.n_eps += 1  # Increment episode count at the start of logging this episode's data

        logging.info(f"> ep {self.n_eps} done. total_steps={steps} | reward={reward} | episode_steps={ep_steps} "\
                     f"| hours={(time.time() - self.start_time) / 60 / 60:.3f} | epsilon={epsilon:.3f} | lr={learning_rate:.6f}")
        
        # Log episode metrics with panel categorization
        episode_metrics = {
            "Training Parameters/Episodic Rewards": reward,
            "Training Parameters/Episode Lengths": ep_steps,
            "Training Parameters/Learning Rate": learning_rate,
            "Training Parameters/Epsilon": epsilon,
            "Training Parameters/Steps": steps,
        }

        # Log options statistics under a separate panel
        for option, lens in option_lengths.items():
            option_metrics = {
                f"Options/Option {option} Average Length": np.mean(lens) if len(lens) > 0 else 0,
                f"Options/Option {option} Activation": sum(lens) / ep_steps
            }
            episode_metrics.update(option_metrics)

        # Use the updated self.n_eps for this episode's data
        wandb.log(episode_metrics, step=self.n_eps)

    def log_data(self, steps, actor_loss, critic_loss, entropy, epsilon):
        policy_metrics = {
            "Policy/Actor Loss": actor_loss.item() if actor_loss else None,
            "Policy/Critic Loss": critic_loss.item() if critic_loss else None,
            "Policy/Policy Entropy": entropy,
            "Policy/Epsilon": epsilon
        }

        # Log policy related data using the current episode number as the step
        wandb.log(policy_metrics, step=self.n_eps)

    def log_customed_output(self, data):
        logging.info(data)


class EmptyLogger(object):
    def __init__(self, *args):
        self.n_eps = 0
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.log_name + '/logger.log'),
            ],
            datefmt='%Y/%m/%d %I:%M:%S %p'
        )

    def log_episode(self, *args):
        self.n_eps += 1

    def log_data(self, *args):
        pass

    def save(self):
        pass

    def log_customed_output(self, data):
        logging.info(data)

# if __name__=="__main__":
#     logger = TensorboardLogger(logdir='runs/', run_name='test_model-test_env')
#     steps = 200 ; reward = 5 ; option_lengths = {opt: np.random.randint(0,5,size=(5)) for opt in range(5)} ; ep_steps = 50
#     logger.log_episode(steps, reward, option_lengths, ep_steps)

