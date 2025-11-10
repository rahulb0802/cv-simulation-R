# Robustness Check: Non-Linear Effects 
library(dplyr)
library(mvtnorm)

generate_nonlinear_data <- function(num_people = 200, num_contexts = 50) {
  contexts <- data.frame(
    context_id = 1:num_contexts,
    # Use skew dist instead of symmetric
    context_risk = rlnorm(num_contexts, meanlog=-0.5, sdlog=0.7) 
  )
  true_person_variance <- matrix(c(0.5, 0.15, 0.15, 0.8), nrow = 2)
  dna_raw <- rmvnorm(num_people, mean = c(0, 0), sigma = true_person_variance)
  dna <- data.frame(
    person_id = 1:num_people,
    true_b_i = dna_raw[, 1],
    true_s_i = dna_raw[, 2]
  )
  
  sparse_data <- dna %>%
    mutate(context_id = sample(contexts$context_id, size = num_people, replace = TRUE)) %>%
    left_join(contexts, by = "context_id") %>%
    mutate(
      # Add an extra quadratic effect
      nonlinear_effect = 0.4 * true_s_i * (context_risk^2),
      linear_predictor = -2.5 + true_b_i + (true_s_i * context_risk) + nonlinear_effect, 
      probability_overdose = plogis(linear_predictor),
      outcome = rbinom(num_people, 1, probability_overdose)
    ) %>%
    select(person_id, context_id, context_risk, outcome)
  
  return(list(
    real_data = sparse_data,
    all_contexts = contexts,
    true_person_dna = dna
  ))
}

set.seed(123)
nonlinear_phase_1_data <- generate_nonlinear_data()

saveRDS(nonlinear_phase_1_data, "nonlinear_data.rds")