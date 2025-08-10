library(brms)
library(dplyr)
library(mvtnorm)
library(lme4)
library(ggplot2)


# This function generates our initial sparse dataset (one observation per person)
generate_sparse_data <- function(num_people = 200, num_contexts = 50) {
  contexts <- data.frame(context_id = 1:num_contexts, context_risk = rnorm(num_contexts, 0, 1.5))
  true_person_variance <- matrix(c(0.5, 0.15, 0.15, 0.8), nrow = 2)
  true_person_dna_raw <- rmvnorm(num_people, mean = c(0, 0), sigma = true_person_variance)
  true_person_dna <- data.frame(person_id = 1:num_people, true_b_i = true_person_dna_raw[,1], true_s_i = true_person_dna_raw[,2])
  
  real_world_data <- true_person_dna %>%
    mutate(context_id = sample(contexts$context_id, size = num_people, replace = TRUE)) %>%
    left_join(contexts, by = "context_id") %>%
    mutate(outcome = 1.0 + true_b_i + (true_s_i * context_risk) + rnorm(num_people, 0, 1.0))
  
  final_real_data <- real_world_data %>% select(person_id, context_id, context_risk, outcome)
  
  return(list(real_data = final_real_data, all_contexts = contexts))
}

set.seed(42)
phase_1_data <- generate_sparse_data()
real_data <- phase_1_data$real_data
all_contexts <- phase_1_data$all_contexts

print("Fitting Bayesian CROSSED model to sparse data...")
bayesian_world_model <- brm(
  outcome ~ context_risk + (context_risk | person_id) + (1 | context_id), 
  data = real_data,
  chains = 2, iter = 1000, cores = 2, silent = 2, refresh = 0
)


# Extract all the learned rules from the Bayesian model
estimated_variances <- VarCorr(bayesian_world_model)
estimated_fixed_effects <- fixef(bayesian_world_model)[, "Estimate"]
estimated_residual_sd <- summary(bayesian_world_model)$spec_pars["sigma", "Estimate"]
sd_context <- estimated_variances$context_id$sd["Intercept", "Estimate"]

sds_person <- VarCorr(bayesian_world_model)$person_id$sd
sd_intercept_person <- sds_person["Intercept", "Estimate"]
sd_slope_person <- sds_person["context_risk", "Estimate"]
correlation_person <- VarCorr(bayesian_world_model)$person_id$cor["Intercept", "Estimate", "context_risk"]
var_intercept_person <- sd_intercept_person^2
var_slope_person <- sd_slope_person^2
cov_person <- correlation_person * sd_intercept_person * sd_slope_person
dna_blueprint_matrix <- matrix(c(var_intercept_person, cov_person, cov_person, var_slope_person), nrow = 2, byrow = TRUE)

num_synthetic_people <- 500
synthetic_person_dna_raw <- rmvnorm(num_synthetic_people, mean = c(0, 0), sigma = dna_blueprint_matrix)
synthetic_person_dna <- data.frame(person_id = 1:num_synthetic_people, b_i = synthetic_person_dna_raw[,1], s_i = synthetic_person_dna_raw[,2])

num_contexts_to_visit <- 20
rich_data_list <- list()

for (i in 1:nrow(synthetic_person_dna)) {
  current_person_dna <- synthetic_person_dna[i, ]
  contexts_to_visit <- all_contexts %>% sample_n(size = num_contexts_to_visit)
  
  context_shocks <- rnorm(num_contexts_to_visit, mean = 0, sd = sd_context)
  
  simulated_outcomes <- 
    estimated_fixed_effects["Intercept"] +
    (estimated_fixed_effects["context_risk"] * contexts_to_visit$context_risk) +
    current_person_dna$b_i +
    (current_person_dna$s_i * contexts_to_visit$context_risk) +
    context_shocks + # <-- Added the context effect
    rnorm(num_contexts_to_visit, mean = 0, sd = estimated_residual_sd)
  
  rich_data_list[[i]] <- data.frame(
    person_id = current_person_dna$person_id,
    context_id = contexts_to_visit$context_id,
    context_risk = contexts_to_visit$context_risk,
    simulated_outcome = simulated_outcomes
  )
}
rich_simulated_data <- dplyr::bind_rows(rich_data_list)

final_crossed_model <- lmer(
  simulated_outcome ~ context_risk + (context_risk | person_id) + (1 | context_id),
  data = rich_simulated_data
)

final_estimated_dna <- ranef(final_crossed_model)$person_id
final_indicator_df <- data.frame(
  person_id = as.numeric(rownames(final_estimated_dna)),
  FINAL_CV_INDICATOR = final_estimated_dna[, "context_risk"] # The reactivity slope
)

true_synthetic_dna <- synthetic_person_dna %>%
  rename(true_synth_s_i = s_i) %>%
  select(person_id, true_synth_s_i)


final_validation_df <- inner_join(final_indicator_df, true_synthetic_dna, by = "person_id")

# Calculate the correlation
final_correlation_test <- cor.test(
  final_validation_df$FINAL_CV_INDICATOR, 
  final_validation_df$true_synth_s_i
)

print("FINAL VALIDATION")
print(final_correlation_test)

# --- Final Visualization ---
final_validation_plot <- ggplot(final_validation_df, aes(x = true_synth_s_i, y = FINAL_CV_INDICATOR)) +
  geom_point(alpha = 0.5, color = "purple") +
  geom_smooth(method = "lm", color = "orange", linetype = "dashed") +
  labs(
    title = "Validation of the Final CV Indicator (from Crossed Model)",
    subtitle = "Comparing the estimated reactivity to the known ground truth",
    x = "True Synthetic Reactivity Slope (Ground Truth)",
    y = "Estimated Reactivity Slope (Final Indicator)"
  ) +
  theme_minimal()

print(final_validation_plot)