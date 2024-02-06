library(dplyr)
library(deepG)

filters <- c(32,32,64,64,128,128,256)
kernel_size <- rep(24, length(filters))
pool_size <-  rep(4, length(filters))
maxlen <- 1000000

model <- create_model_lstm_cnn(
  maxlen = maxlen,
  layer_dense = c(128, 1),
  kernel_size = kernel_size,
  filters = filters,
  pool_size = pool_size,
  learning_rate = 0.000025,
  last_layer_activation = "sigmoid",
  loss_fn = "binary_crossentropy",
  model_seed = 12)


train_model_bacdive_spore <- function(
    run_name = "bacdive_new_spore_1",
    model_card = list(path_model_card = "/vol/projects/BIFO/genomenet/model_cards",
                      description = "spore prediction with ce loss"),
    fasta_path = "/path/to/fasta/files",
    ttv_file = "/path/to/csv/with/train/test/val/split",
    target_from_csv = "/path/to/csv/with/target/labels",
    model,
    path_checkpoint = "/checkpoint/path",
    path_tensorboard = "/tensorboard/path",
    path_log = "/log/path",
    concat_seq = "",
    random_sampling = TRUE,
    max_samples = 4,
    batch_size = 4,
    epochs = 10000,
    steps_per_epoch = 20000,
    sample_by_file_size = TRUE,
    step = 500000,
    ...) {
  
  all_files <- list.files(fasta_path, full.names = TRUE)
  all_csv <- read.csv(ttv_file)
  
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
    path_checkpoint = path_checkpoint,
    path_tensorboard = path_tensorboard,
    path_log = path_log,
    model_card = model_card,
    train_val_ratio = 0.2,
    run_name = run_name,
    step = step,
    batch_size = batch_size,
    epochs = epochs,
    steps_per_epoch = steps_per_epoch, 
    random_sampling = random_sampling,
    sample_by_file_size = sample_by_file_size,
    max_samples = max_samples,
    concat_seq = concat_seq,
    target_from_csv = target_from_csv,
    ...)
  
}

train_model_bacdive_spore(model = model)