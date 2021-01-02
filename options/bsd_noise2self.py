#### general settings
name: BSD68_noise2self
use_tb_logger: true
model: noise2self
distortion: deblur
scale: 1
gpu_ids: [0]

#### datasets
datasets:
  train:
    name: BSD68
    mode: BSD68
    interval_list: [1]
    HQ_data: ~
    LQ_data: ../BSD68/train/DCNN400_train_gaussian25.npy

    use_shuffle: true
    n_workers: 4  # per GPU
    batch_size: 32
    HQ_size: &HQ_SIZE 128
    use_flip: true
    use_rot: true
  val:
    name: BSD68
    mode: BSD68
    interval_list: [1]
    HQ_data: ../BSD68/test/bsd68_groundtruth.npy
    LQ_data: ../BSD68/test/bsd68_gaussian25.npy

    use_shuffle: true
    n_workers: 4  # per GPU
    batch_size: 1

#### network structures
network_G:
  img_channels: 1

#### path
path:
  pretrain_model_G: ../experiments/pretrained/bsd.pth
  strict_load: true
  resume_state: ~

#### training settings: learning rate scheme, loss
train:
  lr_G: !!float 4e-4
  lr_scheme: CosineAnnealingLR_Restart
  beta1: 0.9
  beta2: 0.99
  niter: 600000
  warmup_iter: -1  # -1: no warm up
  T_period: [50000, 100000, 150000, 150000, 150000]
  restarts: [50000, 150000, 300000, 450000]
  restart_weights: [1, 1, 1, 1]
  eta_min: !!float 1e-6

  pixel_criterion: l2
  pixel_weight: 1.0
  kl_weight: 0.0
  val_freq: !!float 11000

  manual_seed: 0

#### logger
logger:
  print_freq: 500
  save_checkpoint_freq: !!float 1e4
