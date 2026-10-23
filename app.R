# ============================================================================
# train_all_models.R — reproduce all 6 models and save all_models.rds
# Run this once, then launch app.R
# ============================================================================
suppressPackageStartupMessages({
  library(tidyverse); library(caret); library(randomForest); library(gbm)
})

setwd("C:/Users/ADMIN/OneDrive/Desktop/caleb makini/R_poject_Dashboard")

# ---- 1. Load the exact data the Rmd used -----------------------------
stopifnot(file.exists("credit_clean.csv"))
data <- read.csv("credit_clean.csv") |>
  mutate(
    is_fraud            = factor(is_fraud,            levels = c("legit","fraud")),
    foreign_transaction = factor(foreign_transaction, levels = c("no","yes")),
    location_mismatch   = factor(location_mismatch,   levels = c("no","yes")),
    merchant_category   = factor(merchant_category)
  ) |>
  select(-transaction_id)

# ---- 2. Same split as Task 4.2 ---------------------------------------
set.seed(123)
idx   <- createDataPartition(data$is_fraud, p = 0.80, list = FALSE)
train <- data[ idx, ]
test  <- data[-idx, ]

cat("Train:", nrow(train), "| Test:", nrow(test), "\n")

# ---- 3. Same CV controls as Task 4.3 ---------------------------------
ctrl <- trainControl(
  method          = "cv", number = 5, classProbs = TRUE,
  summaryFunction = prSummary, savePredictions = "final",
  sampling        = "up"
)

# ---- 4. Train all 6 models -------------------------------------------

cat("\n[1/6] Training Logistic Regression…\n")
m_lr <- train(is_fraud ~ ., data = train, method = "glm",
              family = "binomial", trControl = ctrl, metric = "AUC")

cat("[2/6] Training Decision Tree…\n")
m_dt <- train(is_fraud ~ ., data = train, method = "rpart",
              trControl = ctrl, metric = "AUC",
              tuneGrid = expand.grid(cp = c(.0005,.001,.005,.01,.02,.05,.1)))

cat("[3/6] Training SVM… (slowest, ~3-5 min)\n")
m_svm <- train(is_fraud ~ ., data = train, method = "svmRadial",
               trControl = ctrl, metric = "AUC",
               preProcess = c("center","scale"))

cat("[4/6] Training Random Forest…\n")
m_rf <- train(is_fraud ~ ., data = train, method = "rf",
              trControl = ctrl, metric = "AUC",
              tuneGrid = expand.grid(mtry = c(2,3)), ntree = 200)

cat("[5/6] Training Gradient Boosting…\n")
m_gbm <- train(is_fraud ~ ., data = train, method = "gbm",
               trControl = ctrl, metric = "AUC", verbose = FALSE,
               tuneGrid = expand.grid(n.trees = c(100,200),
                                      interaction.depth = c(2,3),
                                      shrinkage = c(.05,.10),
                                      n.minobsinnode = 10))

cat("[6/6] Training KNN…\n")
m_knn <- train(is_fraud ~ ., data = train, method = "knn",
               trControl = ctrl, metric = "AUC",
               preProcess = c("center","scale"),
               tuneGrid = expand.grid(k = c(3,5,7,9)))

# ---- 5. Save --------------------------------------------------------
all_models <- list(
  "Logistic Regression" = m_lr,
  "Decision Tree"       = m_dt,
  "SVM"                 = m_svm,
  "Random Forest"       = m_rf,
  "Gradient Boosting"   = m_gbm,
  "KNN"                 = m_knn
)
saveRDS(all_models, "all_models.rds")

cat("\n========== DONE ==========\n")
cat("Saved", length(all_models), "models to all_models.rds\n")
cat("Models:", paste(names(all_models), collapse = ", "), "\n")

# Optional: quick verification on the test set
cat("\nTest-set AUC per model:\n")
for (nm in names(all_models)) {
  p <- predict(all_models[[nm]], test, type = "prob")[, "fraud"]
  a <- pROC::auc(test$is_fraud, p, quiet = TRUE,
                 levels = c("legit","fraud"), direction = "<")
  cat(sprintf("  %-22s AUC = %.3f\n", nm, as.numeric(a)))
}