
cd /data/minimax-dialogue/feishan/critique_rl/critique_verl

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
 data.train_files=/data/minimax-dialogue/feishan/critique_rl/verl_data/training_data/deepscaler_train_filter/train.parquet \
 data.val_files=/data/minimax-dialogue/feishan/critique_rl/verl_data/training_data/deepscaler_train_filter/test.parquet \
 data.train_batch_size=256 \
 data.max_prompt_length=1024 \
 data.max_response_length=3072 \
 actor_rollout_ref.model.path=/data/minimax-dialogue/feishan/critique_rl/verl_data/models/Qwen2.5-Math-7B \
 actor_rollout_ref.actor.optim.lr=5e-7 \
 actor_rollout_ref.actor.ppo_mini_batch_size=256 \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=20 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
 critic.ppo_micro_batch_size_per_gpu=4 \
 algorithm.kl_ctrl.kl_coef=0.001 \
 trainer.logger=['console'] \
 trainer.val_before_train=False \
 trainer.n_gpus_per_node=8 \
 trainer.nnodes=1 \
 trainer.save_freq=20 \
 trainer.test_freq=20 \
 trainer.total_epochs=15 2>&1 | tee verl_demo.log