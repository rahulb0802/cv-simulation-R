library(dplyr)
library(brms)
library(ggplot2)
library(mvtnorm)

generate_sparse_real_data <- function(
    num_people = 200,
    num_contexts = 50) {
  fixed_intercept <- 1.0
  residual_sd <- 1.0

  true_person_variance <- matrix(
    c(
      0.5, 0.15,
      0.15, 0.8
    ),
    nrow = 2
  )

  contexts <- data.frame(
    context_id = 1:num_contexts,
    context_risk = rnorm(num_contexts, mean = 0, sd = 1.5)
  )

  true_person_dna_raw <- rmvnorm(num_people, mean = c(0, 0), sigma = true_person_variance)
  true_person_dna <- data.frame(
    person_id = 1:num_people,
    true_b_i = true_person_dna_raw[, 1],
    true_s_i = true_person_dna_raw[, 2]
  )

  real_world_data <- true_person_dna %>%
    mutate(context_id = sample(contexts$context_id, size = num_people, replace = TRUE)) %>%
    left_join(contexts, by = "context_id")

  real_world_data <- real_world_data %>%
    mutate(
      outcome = fixed_intercept + true_b_i + (true_s_i * context_risk) + rnorm(num_people, 0, residual_sd)
    )

  final_real_data <- real_world_data %>% select(person_id, context_id, context_risk, outcome)

  return(list(
    real_data = final_real_data,
    true_world_params = list(
      fixed_intercept = fixed_intercept,
      person_variance_matrix = true_person_variance,
      residual_sd = residual_sd
    )
  ))
}

set.seed(42)
generated_data <- generate_sparse_real_data()
real_data <- generated_data$real_data

print("Fitting Bayesian Random Slope Model...")

bayesian_world_model <- brm(
  outcome ~ context_risk + (context_risk | person_id),
  data = real_data,
  chains = 2,
  iter = 1000,
  cores = 2,
  silent = 2,
  refresh = 0
)

print("Extracting world parameters...")

estimated_variances <- VarCorr(bayesian_world_model)

estimated_person_variance_matrix <- estimated_variances$person_id$cov
estimated_residual_sd <- estimated_variances$residual__$sd[1]

print("Comparing Estimated vs. True World Parameters")
print("True Variance-Covariance Matrix:")
print(generated_data$true_world_params$person_variance_matrix)

print("Estimated Variance-Covariance Matrix (from Bayesian Model):")
print(estimated_person_variance_matrix)

print(paste("True Residual SD:", generated_data$true_world_params$residual_sd))
print(paste("Estimated Residual SD:", round(estimated_residual_sd, 4)))
