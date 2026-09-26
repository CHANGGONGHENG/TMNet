# Conditional inference tree (and other R-only classifiers) for the machine-learning comparison.
#
# Called from r_backend.py:
#   Rscript r_models.R <model> <train.csv> <out_dir> <predict_0.csv> <predict_1.csv> ...
#
# The training file holds the features followed by a column named `y` (0/1).
# For every prediction file one output file `pred_<i>.csv` is written, containing P(y = 1).
#
# Requires the `partykit` package (conditional inference trees).

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4) {
  stop("usage: Rscript r_models.R <model> <train.csv> <out_dir> <predict_*.csv> ...")
}

model_name <- args[1]
train_file <- args[2]
out_dir <- args[3]
pred_files <- args[4:length(args)]

dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

train <- read.csv(train_file, check.names = TRUE)
train$y <- as.factor(train$y)

fit_model <- function(name, data) {
  if (name == "ConditionalInferenceTree") {
    if (!requireNamespace("partykit", quietly = TRUE)) {
      stop("the 'partykit' package is required for ConditionalInferenceTree")
    }
    return(partykit::ctree(y ~ ., data = data))
  }
  stop(paste("unknown R model:", name))
}

fit <- fit_model(model_name, train)

for (i in seq_along(pred_files)) {
  newdata <- read.csv(pred_files[i], check.names = TRUE)
  prob <- predict(fit, newdata = newdata, type = "prob")[, "1"]
  write.csv(data.frame(p = as.numeric(prob)),
            file.path(out_dir, sprintf("pred_%d.csv", i - 1)),
            row.names = FALSE)
}

cat(sprintf("R model %s finished (%d prediction sets)\n", model_name, length(pred_files)))
