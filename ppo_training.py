from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from Energy_District_Gym_Environment import EnergyDistrictEnvironment


def make_env(rank: int = 0):
    def _init():
        env = EnergyDistrictEnvironment()
        env = Monitor(env, filename=f"./logs/monitor/monitor_{rank}")
        return env
    return _init


if __name__ == "__main__":
    N_ENVS = 12  # parallele Environments
    env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)

    eval_env = SubprocVecEnv([make_env(999)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)
    # wichtig: eval_env nicht updaten lassen, sondern nur normalize
    eval_env.training = False
    eval_env.norm_reward = False

    model = PPO(
        "MlpPolicy",
        env,
        device="auto",
        n_steps=96,
        batch_size=int(N_ENVS * 96 / 4),
        learning_rate=2e-4,
        verbose=1,
        tensorboard_log="./logs/tb/"
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./logs/best_model/",
        log_path="./logs/eval/",
        eval_freq=9_600,
        deterministic=True,
        render=False,
    )

    model.learn(total_timesteps=200000, callback=eval_callback)
    model.save("./models/ppo_energy_district")

    # Normalisierungs-Statistiken sichern
    env.save("./models/vecnormalize.pkl")

