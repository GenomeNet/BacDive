library(dplyr)
library(deepG)

# model
filters <- c(32,32,64,64,128,128,256)
kernel_size <- rep(24, length(filters))
pool_size <-  rep(4, length(filters))
maxlen <- 1000000

model <- create_model_lstm_cnn(
  maxlen = maxlen,
  layer_dense = c(256, 1),
  kernel_size = kernel_size,
  filters = filters,
  strides = NULL,
  pool_size = pool_size,
  learning_rate = 0.000025,
  bal_acc = FALSE,
  last_layer_activation = "linear",
  loss_fn = "mse",
  model_seed = 12)

# training

run_name <- "bacdive_new_pathogen_1"
model_card <- list(path_model_card = "/vol/projects/BIFO/genomenet/model_cards",
                   description = "pathogenicity prediction")
random_sampling <- TRUE
concat_seq <- ""

all_files <- list.files("/vol/projects/pmuench/bacdive_new/bacdive_references", full.names = TRUE)
all_csv <- read.csv("/vol/projects/rmreches/bacdive_labels/training_patho_labels_2024-01-30.csv")
target_from_csv <- "/vol/projects/rmreches/bacdive_labels/training_patho_2024-01-30.csv"

train_csv <- all_csv %>% filter(type == "train") %>% select(file) %>% unlist()
train_index <- basename(all_files) %in% train_csv
train_files <- as.list(all_files[train_index])

val_csv <- all_csv %>% filter(type == "validation") %>% select(file) %>% unlist()
val_index <- basename(all_files) %in% val_csv
val_files <- as.list(all_files[val_index])

train_model(
  train_type = "label_csv",
  model = model,
  path = train_files,
  path_val = val_files, 
  path_checkpoint = "/vol/projects/BIFO/genomenet/checkpoints",
  path_tensorboard = "/vol/projects/BIFO/genomenet/tensorboard",
  path_log = "/vol/projects/BIFO/genomenet/scores_log",
  model_card = model_card,
  train_val_ratio = 0.2,
  run_name = run_name,
  step = 500000,
  batch_size = 4,
  epochs = 1000,
  max_queue_size = 500,
  reduce_lr_on_plateau = TRUE,
  lr_plateau_factor = 0.95,
  patience = 10,
  cooldown = 5,
  steps_per_epoch = 20000, 
  seed = c(15, 51),
  random_sampling = random_sampling,
  sample_by_file_size = TRUE,
  max_samples = 4,
  concat_seq = concat_seq,
  target_from_csv = target_from_csv)
