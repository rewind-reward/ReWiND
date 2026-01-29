import gym
import numpy as np
from sympy.sets.sets import true
import torch as th
import torch.nn.functional as F
from gym import spaces
import wandb
from reward_model.base_reward_model import BaseRewardModel




# Wrapper for PCA
class PCAReducerWrapper(gym.Wrapper):
    def __init__(self, env, pca_model):
        super(PCAReducerWrapper, self).__init__(env)
        self.pca_model = pca_model
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.pca_model.n_components,),
            dtype=np.float32,
        )

    def __getstate__(self):
        """Custom method for pickling - exclude pca_model which might contain unpicklable objects"""
        state = self.__dict__.copy()
        # Remove the pca_model which might not be picklable
        if "pca_model" in state:
            del state["pca_model"]
        return state

    def __setstate__(self, state):
        """Custom method for unpickling"""
        self.__dict__.update(state)
        # Set pca_model to None - it will need to be set again after unpickling
        self.pca_model = None

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs_pca = self.pca_model.transform(obs.reshape(1, -1)).flatten()
        return obs_pca, reward, done, info

    def reset(self):
        obs = self.env.reset()
        return self.pca_model.transform(obs.reshape(1, -1)).flatten()


class RewardWrapper(gym.Wrapper):
    def __init__(self, env, sparse=True, success_bonus=0.0):
        super(RewardWrapper, self).__init__(env)
        self.sparse = sparse
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.success_bonus = success_bonus

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        # Convert dense rewards to sparse
        sparse_reward = self.success_bonus if info.get("success", False) else 0.0
        if self.sparse:
            reward = sparse_reward
        else:
            reward = reward + sparse_reward

        return obs, reward, done, info


# Wrapper for Time-based Observations
class TimeWrapper(gym.Wrapper):
    def __init__(self, env):
        super(TimeWrapper, self).__init__(env)
        self.counter = 0
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.env.observation_space.shape[0] + 1,),
            dtype=np.float32,
        )

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        t = self.counter / 500  # Assuming max steps is 500
        obs = np.concatenate([obs, [t]])
        self.counter += 1
        return obs, reward, done, info

    def reset(self):
        self.counter = 0
        obs = self.env.reset()
        return np.concatenate([obs, [0]])  # Add time as 0 at reset


# Wrapper for Language-based Observations
# All this environment does is change the observation space
# This will append a specific language feature to the observation
class LanguageWrapper(gym.Wrapper):
    def __init__(self, env, language_feature):
        super(LanguageWrapper, self).__init__(env)

        if isinstance(language_feature, th.Tensor):
            language_feature = language_feature.cpu().numpy()

        self.language_features = language_feature
        # The observation space is a dict
        # Let us add language_feature to the observation space
        current_obs_space = self.env.observation_space
        assert isinstance(current_obs_space, spaces.Dict), (
            "Observation space must be a Dict."
        )

        new_spaces = current_obs_space.spaces.copy()
        new_spaces["language_feature"] = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.language_features),),
            dtype=np.float32,
        )

        self.observation_space = spaces.Dict(new_spaces)

    def __getstate__(self):
        """Custom method for pickling"""
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        """Custom method for unpickling"""
        self.__dict__.update(state)

    def _observation(self, observation):
        observation["language_feature"] = self.language_features
        return observation

    def reset(self):
        obs = self.env.reset()
        obs = self._observation(obs)
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self._observation(obs)
        return obs, reward, done, info


class ImageEmbeddingWrapper(gym.Wrapper):
    def __init__(self, env, reward_model):
        super(ImageEmbeddingWrapper, self).__init__(env)
        self.reward_model = reward_model
        # The observation space is a dict
        # Let us add image_feature to the observation space

        current_obs_space = self.env.observation_space
        assert isinstance(current_obs_space, spaces.Dict), (
            "Observation space must be a Dict."
        )

        image_keys = self.env.image_keys

        # Define the new observation space
        new_spaces = current_obs_space.spaces.copy()
        for i, key in enumerate(image_keys):
            # Add a new key for the image feature corresponding to each image key
            new_spaces[f"image_feature_{i}"] = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(reward_model.img_output_dim,),
                dtype=np.float32,
            )

        # Set the updated observation space
        self.observation_space = spaces.Dict(new_spaces)

    def __getstate__(self):
        """Custom method for pickling - exclude reward_model which might contain unpicklable objects"""
        state = self.__dict__.copy()
        # Remove the reward_model which might not be picklable
        if "reward_model" in state:
            del state["reward_model"]
        return state

    def __setstate__(self, state):
        """Custom method for unpickling"""
        self.__dict__.update(state)
        # Set reward_model to None - it will need to be set again after unpickling
        self.reward_model = None

    def _observation(self, observation):
        for i, key in enumerate(self.image_keys):
            image = observation[key]
            image = image[None, None, :, :, :]
            image_feature = self.reward_model.encode_images(image).squeeze()
            observation[f"image_feature_{i}"] = image_feature

        return observation

    def reset(self):
        obs = self.env.reset()
        obs = self._observation(obs)
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self._observation(obs)
        return obs, reward, done, info


class LearnedRewardWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        reward_model: BaseRewardModel,
        language_features: th.Tensor,
        text_instruction: str = None,
        is_state_based: bool = False,
        dense_eval: bool = False,
    ):
        super(LearnedRewardWrapper, self).__init__(env)
        self.reward_model = reward_model
        self.is_state_based = is_state_based

        self.past_observations = {}

        for key in self.image_keys:
            self.past_observations[key] = []
        self.counter = 0

        self.dense_eval = dense_eval

        self.reward_at_every_step = self.reward_model.reward_at_every_step
        self.reward_divisor = self.reward_model.reward_divisor

        # Language features used by policy (384-dim MiniLM, for LanguageWrapper)
        self.policy_language_features = language_features
        
        # Language features used for reward calculation (needs to be re-encoded by reward model)
        if text_instruction is not None:
            # Use reward model to encode text
            with th.no_grad():
                reward_lang_feat = self.reward_model.encode_text([text_instruction]).squeeze()
            self.reward_language_features = (
                th.Tensor(reward_lang_feat)
                .float()
                .to(self.reward_model.device)
                .unsqueeze(0)
                .unsqueeze(0)
            )
        elif language_features is not None:
            # If no original text, try to use passed features (may need adaptation)
            print("Warning: Using policy language features for reward calculation. This may cause dimension mismatch for some reward models.")
            self.reward_language_features = (
                th.Tensor(language_features)
                .float()
                .to(self.reward_model.device)
                .unsqueeze(0)
                .unsqueeze(0)
            )
        else:
            print("Language features are not provided in the reward model")
            print(
                "This may be valid if the user is using sparse/dense reward in a single task"
            )
            self.reward_language_features = None

        # update the observation space to have image_feature_*
        self.observation_space = self.env.observation_space
        for i, key in enumerate(self.image_keys):
            self.observation_space.spaces[f"image_feature_{i}"] = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(reward_model.img_output_dim,),
                dtype=np.float32,
            )

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["reward_model"]
        del state["reward_language_features"]
        del state["past_observations"]
        del state["counter"]

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def _compute_reward(self):
        # print(len(self.past_observations["observation.images.main"]))
        if not hasattr(self.reward_model, "multiple_cameras"):
            stacked_sequence = np.stack(
                self.past_observations[self.image_keys[self.image_reward_idx]],
                axis=0,
            )
            stacked_sequence = (
                th.from_numpy(stacked_sequence).float().to(self.reward_model.device)
            )

            reward = self.reward_model.calculate_rewards(
                self.reward_language_features, stacked_sequence.unsqueeze(0)
            )
        else:
            stacked_sequence = {
                key: np.stack(self.past_observations[key], axis=0)
                for key in self.image_keys
            }
            stacked_sequence = {
                key: th.from_numpy(stacked_sequence[key])
                .float()
                .to(self.reward_model.device)
                for key in self.image_keys
            }
            reward_sum = 0
            rewards = []
            for key in self.image_keys:
                rewards.append(
                    self.reward_model.calculate_rewards(
                        self.reward_language_features,
                        stacked_sequence[key].unsqueeze(0),
                        key,
                    )
                )
            # print(rewards)
            reward = sum(rewards) / len(self.image_keys)

        return reward

    def step(self, action):
        self.counter += 1
        obs, original_reward, done, info = self.env.step(action)

        encoded_image = None
        # IF the model is state-based and is dense/sparse reward, we can skip this

        if f"image_feature_{self.image_reward_idx}" in obs:
            encoded_image = obs[f"image_feature_{self.image_reward_idx}"]

        encoded_images = {}
        for i, key in enumerate(self.image_keys):
            # Reward calculation can only use features encoded by reward model
            if f"reward_image_feature_{i}" in obs:
                encoded_images[key] = obs[f"reward_image_feature_{i}"]
            else:
                raise KeyError(f"reward_image_feature_{i} not found in observation. Reward calculation requires reward model encoded features.")

        if self.reward_model.name == "dense" or self.dense_eval:
            reward = original_reward / self.reward_divisor

            if info.get("success", False):
                reward += self.reward_model.success_bonus

            return obs, reward, done, info
        # Check if this is sparse/dense reward
        elif self.reward_model.name == "sparse":
            sparse_reward = (
                self.reward_model.success_bonus if info.get("success", False) else 0.0
            )
            # Note: No reward divisor for sparse reward.

            return obs, sparse_reward, done, info
        
        if encoded_images is not None:
            for i, key in enumerate(self.image_keys):
                # Reward calculation can only use features encoded by reward model
                if f"reward_image_feature_{i}" in obs:
                    self.past_observations[key].append(obs[f"reward_image_feature_{i}"])
                else:
                    raise KeyError(f"reward_image_feature_{i} not found in observation. Reward calculation requires reward model encoded features.")

        assert self.reward_language_features is not None, (
            "Language features are None in the reward model"
        )

        if self.reward_at_every_step:
            reward = self._compute_reward()
            wandb.log({"train/learned_reward_per_step": reward})
            if done:
                wandb.log({"train/learned_reward": reward})
        else:
            if done:
                reward = self._compute_reward()
                wandb.log({"train/learned_reward": reward})
            else:
                reward = 0

        reward /= self.reward_divisor

        # Success bonus
        if info.get("success", False):
            reward += self.reward_model.success_bonus
            print("adding success bonus", reward)
            wandb.log({"train/learned_reward_with_success_bonus": reward})

        if isinstance(reward, np.ndarray):
            # All wrappers act on 1 env at a time
            reward = reward[0]

        return obs, reward, done, info

    def reset(self):
        self.past_observations = {}
        for key in self.image_keys:
            self.past_observations[key] = []
        self.counter = 0

        obs = self.env.reset()

        for i, key in enumerate(self.image_keys):
            # Reward calculation can only use features encoded by reward model
            if f"reward_image_feature_{i}" in obs:
                self.past_observations[key].append(obs[f"reward_image_feature_{i}"])
            else:
                raise KeyError(f"reward_image_feature_{i} not found in observation. Reward calculation requires reward model encoded features.")

        return obs


class FlattenDictObservationWrapper(gym.Wrapper):
    def __init__(self, env, use_proprio=True):
        super().__init__(env)
        self.env = env

        # Concatenation will be done in the order of the keys
        # Proprio, text_vector, image_vectors

        obs_space = self.env.observation_space
        total_concat_size = 0

        self.orig_obs_space = obs_space

        # Policy input can only use image features encoded by policy encoder
        policy_image_keys = [
            key for key in obs_space.spaces.keys() if "policy_image_feature" in key
        ]
        
        if policy_image_keys:
            image_feature_keys = sorted(policy_image_keys)
        else:
            raise KeyError("policy_image_feature_* not found in observation space. Policy requires policy encoder encoded features.")

        # Get text keys "language_feature"
        lang_feature_key = (
            "language_feature"
            if "language_feature" in obs_space.spaces.keys()
            else None
        )

        self.use_proprio = use_proprio

        if use_proprio and "proprio" in obs_space.spaces.keys():
            proprio_key = "proprio"
        else:
            proprio_key = None

        # Get total size
        if proprio_key is not None:
            total_concat_size += obs_space[proprio_key].shape[0]

        if lang_feature_key is not None:
            total_concat_size += obs_space[lang_feature_key].shape[0]

        for key in image_feature_keys:
            total_concat_size += obs_space[key].shape[0]

        self.lang_feature_key = lang_feature_key
        self.proprio_key = proprio_key
        self.image_feature_keys = sorted(image_feature_keys)


        self.observation_space = gym.spaces.Box(
            low=-1, high=1, shape=(total_concat_size,), dtype=np.float32
        )

    def _observation(self, obs: dict):
        flattened_obs = []

        # Lang
        if "language_feature" in obs:
            lang = obs["language_feature"]
            lang = lang.reshape(-1)
            flattened_obs.append(lang)

        # Get image in order
        for key in self.image_feature_keys:
            if key in obs:
                image = obs[key]
                image = image.reshape(-1)
                flattened_obs.append(image)

        # Proprio
        if "proprio" in obs and self.use_proprio:
            proprio = obs["proprio"]
            proprio = proprio.reshape(-1)
            flattened_obs.append(proprio)

        return np.concatenate(flattened_obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._observation(obs), reward, done, info

    def reset(self):
        obs = self.env.reset()
        return self._observation(obs)


# Environment keeps an aggregate reward at each step and outputs it only when the episode ends
class RewardAtEndWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super(RewardAtEndWrapper, self).__init__(env)
        # Keep track of the total reward
        self.total_reward = 0

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.total_reward += reward
        if done:
            reward_to_return = self.total_reward
            self.total_reward = 0
            return obs, reward_to_return, done, info
        else:
            return obs, 0, done, info


class RewardScaleWrapper(gym.Wrapper):
    def __init__(self, env, divisor):
        super(RewardScaleWrapper, self).__init__(env)
        self.divisor = divisor

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward / self.divisor, done, info


class LoggingWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, logger, prefix=""):
        super(LoggingWrapper, self).__init__(env)
        self.logger = logger
        self.prefix = prefix
        self.episode_reward = 0

        self.episode_number = 0
        self.step_number = 0

    def __getstate__(self):
        """Custom method for pickling - exclude logger which might contain unpicklable objects"""
        state = self.__dict__.copy()
        # Remove the logger which might not be picklable
        if "logger" in state:
            del state["logger"]
        return state

    def __setstate__(self, state):
        """Custom method for unpickling"""
        self.__dict__.update(state)
        # Set logger to None - it will need to be set again after unpickling
        self.logger = None

    def step(self, action):
        # We want to log the reward at each step
        obs, reward, done, info = self.env.step(action)

        if self.logger is not None:
            self.logger.record(self.prefix + "/reward", reward)
        self.episode_reward += reward

        # if it is a done and a success, we want to log it
        if done:
            if self.logger is not None:
                if info.get("success", False):
                    self.logger.record(self.prefix + "/success", 1)
                else:
                    self.logger.record(self.prefix + "/success", 0)

                # also log the episode reward
                self.logger.record(self.prefix + "/episode_reward", self.episode_reward)
            self.episode_reward = 0

        return obs, reward, done, info


class ActionChunkingWrapper(gym.Wrapper):
    def __init__(self, env, chunk_size=15, n_action_steps=15):
        super(ActionChunkingWrapper, self).__init__(env)
        self.chunk_size = chunk_size
        self.n_action_steps = n_action_steps

        self.chunk = []

    def __getstate__(self):
        """Custom method for pickling"""
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        """Custom method for unpickling"""
        self.__dict__.update(state)

    def step(self, chunked_action: np.ndarray):
        # Unpack action

        if chunked_action is not None and chunked_action.ndim == 1:
            print("**" * 10)
            print()
            print("THE ACTION IS NOT CHUNKED")
            print("This may be okay if random exploration from SB3 is used")
            print()
            print("**" * 10)
            obs, reward, done, info = self.env.step(chunked_action)
            info["action"] = chunked_action[None, :]
            return obs, reward, done, info
        if self.is_chunk_empty or self.chunk is None:
            # Then let the action replace the chunk
            self.chunk = chunked_action
        # else:
        #     # If chunk is not empty, we will assert that chunked_action is None
        #     breakpoint()
        #     assert chunked_action is None
        try:
            popped_action = self.chunk[0]
            self.chunk = self.chunk[1:]
        except:
            breakpoint()
        # return self.env.step(None)

        obs, reward, done, info = self.env.step(popped_action)

        info["action"] = popped_action

        # check if we have n_action_steps used
        actions_taken = self.chunk_size - len(self.chunk)
        if actions_taken >= self.n_action_steps:
            self.chunk = []


        return obs, reward, done, info

    @property
    def is_chunk_empty(self):
        return len(self.chunk) == 0 or self.chunk is None

    def reset(self):
        # print("Resetting chunk")
        self.chunk = []
        return self.env.reset()


class ACTTemporalEnsemblerWrapper(gym.Wrapper):
    def __init__(
        self, env: gym.Env, temporal_ensemble_coeff: float, chunk_size: int
    ) -> None:
        """Temporal ensembling as described in Algorithm 2 of https://arxiv.org/abs/2304.13705.

        The weights are calculated as wᵢ = exp(-temporal_ensemble_coeff * i) where w₀ is the oldest action.
        They are then normalized to sum to 1 by dividing by Σwᵢ. Here's some intuition around how the
        coefficient works:
            - Setting it to 0 uniformly weighs all actions.
            - Setting it positive gives more weight to older actions.
            - Setting it negative gives more weight to newer actions.
        NOTE: The default value for `temporal_ensemble_coeff` used by the original ACT work is 0.01. This
        results in older actions being weighed more highly than newer actions (the experiments documented in
        https://github.com/huggingface/lerobot/pull/319 hint at why highly weighing new actions might be
        detrimental: doing so aggressively may diminish the benefits of action chunking).

        Here we use an online method for computing the average rather than caching a history of actions in
        order to compute the average offline. For a simple 1D sequence it looks something like:

        ```
        import torch

        seq = torch.linspace(8, 8.5, 100)
        print(seq)

        m = 0.01
        exp_weights = torch.exp(-m * torch.arange(len(seq)))
        print(exp_weights)

        # Calculate offline
        avg = (exp_weights * seq).sum() / exp_weights.sum()
        print("offline", avg)

        # Calculate online
        for i, item in enumerate(seq):
            if i == 0:
                avg = item
                continue
            avg *= exp_weights[:i].sum()
            avg += item * exp_weights[i]
            avg /= exp_weights[:i+1].sum()
        print("online", avg)
        ```
        """
        super().__init__(env)
        self.chunk_size = chunk_size
        self.ensemble_weights = th.exp(-temporal_ensemble_coeff * th.arange(chunk_size))
        self.ensemble_weights_cumsum = th.cumsum(self.ensemble_weights, dim=0)

        self.chunk = None
        self.chunk_count = None

    def reset(self):
        """Resets the online computation variables."""
        self.chunk = []
        # (chunk_size,) count of how many actions are in the ensemble for each time step in the sequence.
        self.chunk_count = None
        return self.env.reset()

    @property
    def is_chunk_empty(self):
        return True

    def ensure_safeish_goal_position(
        self,
        goal_pos: th.Tensor,
        present_pos: th.Tensor,
        max_relative_target: float | list[float],
    ):
        # convert everything to tensors
        goal_pos = th.tensor(goal_pos)
        present_pos = th.tensor(present_pos)
        max_relative_target = th.tensor(max_relative_target)

        # Cap relative action target magnitude for safety.
        diff = goal_pos - present_pos
        safe_diff = th.minimum(diff, max_relative_target)
        safe_diff = th.maximum(safe_diff, -max_relative_target)

        safe_goal_pos = present_pos + safe_diff

        return safe_goal_pos.numpy()

    def step(self, actions: th.Tensor) -> th.Tensor:
        """
        Takes a (batch, chunk_size, action_dim) sequence of actions, update the temporal ensemble for all
        time steps, and pop/return the next batch of actions in the sequence.
        """

        if actions is not None and actions.ndim == 1:
            print("**" * 10)
            print()
            print("THE ACTION IS NOT CHUNKED")
            print("This may be okay if random exploration from SB3 is used")
            print()
            print("**" * 10)
            obs, reward, done, info = self.env.step(actions)
            info["action"] = actions[None, :]
            return obs, reward, done, info

        # self.ensemble_weights = self.ensemble_weights.to(device=actions.device)
        # self.ensemble_weights_cumsum = self.ensemble_weights_cumsum.to(
        #     device=actions.device
        # )
        # let us also clip the actions to be between ensure_safeish_goal_position
        # current_state = self.env.current_observation["observation.state"].numpy()
        # actions = self.ensure_safeish_goal_position(
        #     goal_pos=actions, present_pos=current_state, max_relative_target=12.0
        # )
        actions = actions[None, ...]
        if self.chunk is None:
            # Initializes `self._ensembled_action` to the sequence of actions predicted during the first
            # time step of the episode.
            # actions is of shape (chunk_size, action_dim). chunks should be (batch_size, chunk_size, action_dim)
            self.chunk = actions.copy()

            # Note: The last dimension is unsqueeze to make sure we can broadcast properly for tensor
            # operations later.
            self.chunk_count = np.ones(
                (self.chunk_size, 1),
                dtype=np.int32,
            )
        else:
            # self.chunk will have shape (batch_size, chunk_size - 1, action_dim). Compute
            # the online update for those entries.
            self.chunk *= self.ensemble_weights_cumsum[self.chunk_count - 1].numpy()
            self.chunk += (
                actions[:, :-1] * self.ensemble_weights[self.chunk_count].numpy()
            )
            self.chunk /= self.ensemble_weights_cumsum[self.chunk_count].numpy()
            self.chunk_count = np.clip(
                self.chunk_count + 1, a_min=None, a_max=self.chunk_size
            )
            # The last action, which has no prior online average, needs to get concatenated onto the end.
            self.chunk = np.concatenate([self.chunk, actions[:, -1:]], axis=1)
            self.chunk_count = np.concatenate(
                [
                    self.chunk_count,
                    np.ones_like(self.chunk_count[-1:]),
                ]
            )
        # "Consume" the first action.
        action, self.chunk, self.chunk_count = (
            self.chunk[:, 0],
            self.chunk[:, 1:],
            self.chunk_count[1:],
        )
        obs, reward, done, info = self.env.step(action.squeeze())

        info["action"] = action.squeeze()

        # print(action.shape, actions.shape)
        return obs, reward, done, info

    def __getstate__(self):
        """Custom method for pickling"""
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        """Custom method for unpickling"""
        self.__dict__.update(state)



