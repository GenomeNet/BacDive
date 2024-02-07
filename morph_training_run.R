library(dplyr)
library(deepG)

run_name <- "bacdive_new_morph_1"
model_card <- list(path_model_card = "/vol/projects/BIFO/genomenet/model_cards",
                   description = "cell morph prediction")
concat_seq <- ""
learning_rate <- 0.0000001
random_sampling <- TRUE

all_files <- list.files("/vol/projects/pmuench/bacdive_new/bacdive_references", full.names = TRUE)
all_csv <- read.csv("/vol/projects/rmreches/bacdive_labels/training_cell_morph_labels_2024-01-31.csv")
target_from_csv <- "/vol/projects/rmreches/bacdive_labels/training_cell_morph_2024-01-31.csv"

train_csv <- all_csv %>% filter(type == "train") %>% select(file) %>% unlist()
train_index <- basename(all_files) %in% train_csv
train_files <- as.list(all_files[train_index])

val_csv <- all_csv %>% filter(type == "validation") %>% select(file) %>% unlist()
val_index <- basename(all_files) %in% val_csv
val_files <- as.list(all_files[val_index])

target_split <- list(
  #cellshape =
  c("is_cell_shape_rod.shaped",
    "is_cell_shape_other",
    "is_cell_shape_coccus.shaped",
    "is_cell_shape_vibrio.shaped",
    "is_cell_shape_filament.shaped",
    "is_cell_shape_sphere.shaped",
    "is_cell_shape_ovoid.shaped",
    "is_cell_shape_pleomorphic.shaped",
    "is_cell_shape_spiral.shaped",
    "is_cell_shape_curved.shaped",
    "is_cell_shape_oval.shaped"),
  #flagellum = 
  c("is_flagellum_arrangement_monotrichous",
    "is_flagellum_arrangement_monotrichous_polar",
    "is_flagellum_arrangement_polar",
    "is_flagellum_arrangement_peritrichous",
    "is_flagellum_arrangement_lophotrichous",
    "is_flagellum_arrangement_gliding"),
  #gram = 
  c("is_gram_stain_positive",
    "is_gram_stain_variable",
    "is_gram_stain_negative"),
  #motility =
  c("is_motile")
)

dense_layers <- list(c(256, length(target_split[[1]])),
                     c(256, length(target_split[[2]])),
                     c(256, length(target_split[[3]])),
                     c(256, length(target_split[[4]])))

shared_dense_layers <- 1500
last_layer_activation <- list("softmax", "softmax", "softmax", "sigmoid")
output_names <- list("cell_shape", "flagellum", "gram", "motility")
losses <- list("categorical_crossentropy", "categorical_crossentropy",
               "categorical_crossentropy", "binary_crossentropy")

# model
filters <- c(32,64,128,128,256,256,512,1024)
kernel_size <- rep(24, length(filters))
pool_size <-  rep(4, length(filters))
maxlen <- 1000000

mirrored_strategy <- tensorflow::tf$distribute$MirroredStrategy()
with(mirrored_strategy$scope(), { 
  
  base_model <- create_model_lstm_cnn(
    maxlen = maxlen,
    layer_dense = c(256),
    kernel_size = kernel_size,
    filters = filters,
    strides = NULL,
    pool_size = pool_size,
    learning_rate = learning_rate,
    bal_acc = FALSE,
    last_layer_activation = "linear",
    loss_fn = "mse",
    model_seed = 12)
  
  num_layers <- length(base_model$get_config()$layers)
  layer_name <- base_model$get_config()$layers[[num_layers-4]]$name
  
  model <- remove_add_layers(model = base_model,
                             layer_name = layer_name,
                             dense_layers = dense_layers,
                             shared_dense_layers = shared_dense_layers,
                             last_activation = last_layer_activation,
                             output_names = output_names,
                             losses = losses,
                             verbose = TRUE,
                             dropout = NULL,
                             dropout_shared = NULL,
                             freeze_base_model = FALSE,
                             compile = TRUE,
                             learning_rate = learning_rate,
                             solver = "adam",
                             flatten = FALSE,
                             global_pooling = "both_ch_last",
                             model_seed = 12)
  
  metrics_list <- list(
    cell_shape = tensorflow::tf$keras$metrics$Accuracy(name='accuracy'),
    flagellum = tensorflow::tf$keras$metrics$Accuracy(name='accuracy'),
    gram = tensorflow::tf$keras$metrics$Accuracy(name='accuracy'),
    motility = tensorflow::tf$keras$metrics$BinaryAccuracy(name='binary_accuracy')
  )
  
  model %>% keras::compile(loss = losses,
                           optimizer = model$optimizer,
                           metrics = metrics_list)
  
})


train_model(
  train_type = "label_csv",
  target_split = target_split,
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
  shuffle_file_order = TRUE,
  shuffle_input = TRUE,
  save_best_only = FALSE,
  seed = c(15, 51),
  random_sampling = random_sampling,
  sample_by_file_size = TRUE,
  proportion_per_seq = NULL,
  max_samples = 4,
  concat_seq = concat_seq,
  target_from_csv = target_from_csv)


##