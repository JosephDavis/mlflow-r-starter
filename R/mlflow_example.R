# MLflow R Example
# This script demonstrates all major MLflow features with a tracking server

# Load required libraries
suppressPackageStartupMessages({
  library(logger)
  library(mlflow)
  library(carrier)
  library(randomForest)
  library(caret)
  library(ggplot2)
})

# TRACKING SERVER SETUP
# TODO: REPLACE W/ your MLflow tracking server URL
mlflow_set_tracking_uri("http://localhost:5001")
log_info("MLflow tracking URI: {mlflow_get_tracking_uri()}")

# MLOPS FUNCTIONS

setup_experiment <- function(name) {
  experiment_id <- tryCatch(
    {
      mlflow_create_experiment(name = name)
    },
    error = function(e) {
      experiment <- mlflow_get_experiment(name = name)
      experiment$experiment_id
    }
  )

  mlflow_set_experiment(experiment_name = name)
  log_info("Using experiment: {name}, ID: {experiment_id}")

  return(experiment_id)
}

prepare_data <- function(split_ratio = 0.8, seed = 42) {
  # TODO: REPLACE W/ your dataset loading logic
  data(iris)
  set.seed(seed)

  # TODO: REPLACE W/ your target variable and dataset name
  train_indices <- createDataPartition(
    iris$Species,
    p = split_ratio,
    list = FALSE
  )
  train_data <- iris[train_indices, ]
  test_data <- iris[-train_indices, ]

  log_info("Training samples: {nrow(train_data)}")
  log_info("Test samples: {nrow(test_data)}")

  return(list(train = train_data, test = test_data))
}

train_model <- function(train_data, params) {
  start_time <- Sys.time()

  # TODO: REPLACE W/ your model algorithm and formula
  model <- randomForest(
    Species ~ .,
    data = train_data,
    ntree = params$n_trees,
    maxnodes = params$max_depth,
    nodesize = params$min_samples_split
  )

  end_time <- Sys.time()
  training_time <- as.numeric(difftime(end_time, start_time, units = "secs"))

  return(list(model = model, training_time = training_time))
}

evaluate_model <- function(model, train_data, test_data) {
  train_predictions <- predict(model, train_data)
  test_predictions <- predict(model, test_data)

  # TODO: REPLACE W/ your target variable and evaluation metrics
  train_accuracy <- sum(train_predictions == train_data$Species) /
    nrow(train_data)
  test_accuracy <- sum(test_predictions == test_data$Species) / nrow(test_data)

  # TODO: REPLACE W/ your target variable
  conf_matrix <- confusionMatrix(test_predictions, test_data$Species)

  return(list(
    train_predictions = train_predictions,
    test_predictions = test_predictions,
    train_accuracy = train_accuracy,
    test_accuracy = test_accuracy,
    conf_matrix = conf_matrix
  ))
}

log_artifacts <- function(model, test_data, predictions, temp_dir) {
  conf_matrix_plot <- ggplot(
    data = as.data.frame(predictions$conf_matrix$table),
    aes(x = Reference, y = Prediction, fill = Freq)
  ) +
    geom_tile() +
    geom_text(aes(label = Freq), vjust = 1) +
    scale_fill_gradient(low = "white", high = "steelblue") +
    labs(title = "Confusion Matrix", x = "Actual", y = "Predicted") +
    theme_minimal()

  conf_matrix_path <- file.path(temp_dir, "confusion_matrix.png")
  ggsave(conf_matrix_path, conf_matrix_plot, width = 8, height = 6)
  mlflow_log_artifact(conf_matrix_path, "plots")

  importance_data <- data.frame(
    Feature = rownames(model$importance),
    Importance = model$importance[, 1]
  )

  importance_plot <- ggplot(
    importance_data,
    aes(x = reorder(Feature, Importance), y = Importance)
  ) +
    geom_bar(stat = "identity", fill = "skyblue") +
    coord_flip() +
    labs(
      title = "Feature Importance",
      x = "Features",
      y = "Mean Decrease Gini"
    ) +
    theme_minimal()

  importance_path <- file.path(temp_dir, "feature_importance.png")
  ggsave(importance_path, importance_plot, width = 8, height = 6)
  mlflow_log_artifact(importance_path, "plots")

  model_summary_path <- file.path(temp_dir, "model_summary.txt")
  capture.output(print(model), file = model_summary_path)
  mlflow_log_artifact(model_summary_path, "model_info")

  predictions_df <- data.frame(
    actual = test_data$Species,
    predicted = predictions$test_predictions,
    correct = test_data$Species == predictions$test_predictions
  )
  predictions_path <- file.path(temp_dir, "test_predictions.csv")
  write.csv(predictions_df, predictions_path, row.names = FALSE)
  mlflow_log_artifact(predictions_path, "predictions")
}

log_model_crate <- function(model, temp_dir) {
  model_crate <- carrier::crate(
    function(new_data) {
      predict(model, new_data)
    },
    model = model
  )

  # TODO: REPLACE W/ your feature names and data types
  mlflow_log_model(
    model = model_crate,
    artifact_path = "model",
    signature = list(
      inputs = list(
        Sepal.Length = "double",
        Sepal.Width = "double",
        Petal.Length = "double",
        Petal.Width = "double"
      ),
      outputs = list("Species" = "string")
    )
  )

  model_path <- file.path(temp_dir, "random_forest_model.rds")
  saveRDS(model, model_path)
  mlflow_log_artifact(model_path, "model_files")
}

safe_execute <- function(expr, description, critical = FALSE) {
  tryCatch(
    {
      eval(expr, envir = parent.frame())
    },
    error = function(e) {
      run_success <<- FALSE
      error_msg <- paste(description, "failed:", e$message)
      error_messages <<- c(error_messages, error_msg)

      if (critical) {
        log_error(error_msg)
        stop(paste("Critical error:", error_msg))
      } else {
        log_warn(error_msg)
        return(NULL)
      }
    }
  )
}

execute_mlflow_run <- function(train_data, test_data, params, tags, run_name) {
  # Track overall success state
  run_success <- TRUE
  error_messages <- c()
  run_id <- NULL

  # Start MLflow run and capture the run object directly
  run_obj <- mlflow_start_run()

  # Extract run ID from the run object
  if (!is.null(run_obj) && !is.null(run_obj$run_uuid)) {
    run_id <- run_obj$run_uuid
  }

  # Use the run object in with block
  with(run_obj, {
    # Set custom run name using system tag
    mlflow_set_tag("mlflow.runName", run_name)

    # PARAMETER LOGGING
    log_info("Logging parameters for the run")

    for (param_name in names(params)) {
      mlflow_log_param(param_name, params[[param_name]])
    }

    # TAGS
    log_info("Setting tags for the run")

    for (tag_name in names(tags)) {
      mlflow_set_tag(tag_name, tags[[tag_name]])
    }

    # MODEL TRAINING
    model_result <- safe_execute(
      quote(train_model(train_data, params)),
      "Model training",
      critical = TRUE
    )
    model <- model_result$model
    training_time <- model_result$training_time

    # PREDICTIONS AND EVALUATION
    predictions <- safe_execute(
      quote(evaluate_model(model, train_data, test_data)),
      "Model evaluation",
      critical = TRUE
    )
    train_accuracy <- predictions$train_accuracy
    test_accuracy <- predictions$test_accuracy
    conf_matrix <- predictions$conf_matrix

    # METRICS LOGGING
    metrics <- list(
      train_accuracy = train_accuracy,
      test_accuracy = test_accuracy,
      training_time_seconds = training_time,
      n_features = ncol(train_data) - 1,
      train_samples = nrow(train_data),
      test_samples = nrow(test_data)
    )

    for (metric_name in names(metrics)) {
      mlflow_log_metric(metric_name, metrics[[metric_name]])
    }

    # Log confusion matrix metrics
    mlflow_log_metric(
      "precision_setosa",
      conf_matrix$byClass["Class: setosa", "Precision"]
    )
    mlflow_log_metric(
      "recall_setosa",
      conf_matrix$byClass["Class: setosa", "Recall"]
    )
    mlflow_log_metric("f1_setosa", conf_matrix$byClass["Class: setosa", "F1"])

    # ARTIFACT LOGGING
    # Use a temporary directory for artifacts and cleanup after run
    log_info("Logging artifacts for the run")
    temp_dir <- tempdir()
    on.exit({
      unlink(temp_dir, recursive = TRUE)
    })

    safe_execute(
      quote(log_artifacts(model, test_data, predictions, temp_dir)),
      "Artifact logging"
    )

    # MODEL LOGGING
    log_info("Logging model")
    safe_execute(
      quote(log_model_crate(model, temp_dir)),
      "Model logging"
    )

    # RUN INFORMATION
    # Display run information
    if (!is.null(run_id) && nchar(run_id) > 0) {
      log_info("Run ID: {run_id}")
    } else {
      log_warn("Run info not available")
      run_success <<- FALSE
      error_messages <<- c(error_messages, "Could not get valid run ID")
    }
  })

  # Return results for use outside the function
  return(list(
    run_success = run_success,
    error_messages = error_messages,
    run_id = run_id,
    params = params
  ))
}

register_model <- function(run_result, model_name) {
  if (run_result$run_success && !is.null(run_result$run_id)) {
    log_info("Registering model")

    # Get or create registered model
    registered_model <- safe_execute(
      quote({
        tryCatch(
          {
            model <- mlflow_get_registered_model(name = model_name)
            log_info("Found existing registered model: {model_name}")
            model
          },
          error = function(e) {
            # TODO: REPLACE W/ your model description
            model <- mlflow_create_registered_model(
              name = model_name,
              description = "Random Forest classifier for Iris species prediction"
            )
            log_info("Created new registered model: {model_name}")
            model
          }
        )
      }),
      "Model registration setup"
    )

    # Create model version from completed run
    if (!is.null(registered_model)) {
      model_uri <- paste0("runs:/", run_result$run_id, "/model")

      safe_execute(
        quote({
          model_version <- mlflow_create_model_version(
            name = model_name,
            source = model_uri,
            run_id = run_result$run_id,
            description = paste0(
              "Random Forest model with ", run_result$params$n_trees, " trees"
            )
          )
          log_info("Created model version {model_version$version} for {model_name}")
        }),
        "Creating model version"
      )
    }
  } else {
    log_warn("Skipping model registration due to run failure or missing run ID")
  }
}

# EXPERIMENT SETUP

# TODO: REPLACE W/ your experiment name
experiment_name <- "iris_classification_experiment"
experiment_id <- setup_experiment(experiment_name)

# DATA PREPARATION
log_info("Preparing data for the experiment")

# Load and prepare dataset
data_split <- prepare_data(split_ratio = 0.8)
train_data <- data_split$train
test_data <- data_split$test

# MODEL TRAINING WITH MLFLOW TRACKING

# Define parameters and tags before run execution
# TODO: REPLACE W/ your model parameters
params <- list(
  n_trees = 100,
  max_depth = 5,
  min_samples_split = 2
)

# TODO: REPLACE W/ your project-specific tags and metadata
tags <- list(
  mlflow_version = as.character(packageVersion("mlflow")),
  r_version = R.version.string,
  algorithm = "RandomForest",
  data_split_ratio = "0.8",
  random_seed = "42",
  model_type = "classification",
  dataset = "iris",
  developer = "Joey Davis",
  environment = "development",
  framework = "randomForest"
)

# TODO: REPLACE W/ your custom run naming strategy
run_name <- paste0(
  tags$algorithm, "_",
  format(Sys.time(), "%Y%m%d_%H%M%S"),
  "_trees", params$n_trees,
  "_depth", params$max_depth
)

# Execute the MLflow run
run_result <- execute_mlflow_run(train_data, test_data, params, tags, run_name)

# MODEL REGISTRATION
# TODO: REPLACE W/ your model name for registration
model_name <- "iris_classifier"
register_model(run_result, model_name)

# CLEANUP AND SUMMARY

if (run_result$run_success) {
  log_success("MLflow tracking completed successfully!")
} else {
  log_warn("MLflow tracking completed with some errors:")
  for (error in run_result$error_messages) {
    log_warn("  - {error}")
  }
}
