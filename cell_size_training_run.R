library(dplyr)
library(deepG)

# model
filters <- c(32,32,64,64,128,128,256)
kernel_size <- rep(24, length(filters))
pool_size <-  rep(4, length(filters))
maxlen <- 1000000

model <- create_model_lstm_cnn(
  maxlen = maxlen,
  layer_dense = c(256, 4),
  kernel_size = kernel_size,
  filters = filters,
  strides = NULL,
  pool_size = pool_size,
  learning_rate = learning_rate,
  bal_acc = FALSE,
  last_layer_activation = "linear",
  loss_fn = "mse",
  model_seed = 12)

# training

run_name <- "bacdive_new_cellsize_1"
model_card <- list(path_model_card = "/vol/projects/BIFO/genomenet/model_cards",
                   description = "cell size prediction",
                   data_standardization = "m=c(2.2833333,3.5430677,0.6081660,0.7439617);   
                                           sd=c(5.4286742,8.4601631,0.3197071,0.3665785);
                                           names=c(rgStart_len,rgEnd_len,rgStart_wid,rgEnd_wid)")
random_sampling <- TRUE
learning_rate <- 1e-06

all_files <- list.files("/vol/projects/pmuench/bacdive_new/bacdive_references", full.names = TRUE)
all_csv <- read.csv("/vol/projects/rmreches/bacdive_labels/training_cell_size_labels_2024-01-26.csv")
target_from_csv <- "/vol/projects/rmreches/bacdive_labels/training_cell_size_norm_2024-01-26.csv"

train_csv <- all_csv %>% filter(type == "train") %>% select(file) %>% unlist()
train_index <- basename(all_files) %in% train_csv
train_files <- as.list(all_files[train_index])

val_csv <- all_csv %>% filter(type == "validation") %>% select(file) %>% unlist()
val_index <- basename(all_files) %in% val_csv
val_files <- as.list(all_files[val_index])  
concat_seq <- ""

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
  proportion_per_seq = NULL,
  max_samples = 4,
  concat_seq = concat_seq,
  target_from_csv = target_from_csv)

