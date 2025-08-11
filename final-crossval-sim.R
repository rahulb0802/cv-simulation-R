library(brms)
library(dplyr)
library(mvtnorm)
library(lme4)
library(ggplot2)
library(tidyr)
library(e1071)
library(patchwork)


generate_sparse_data <- function(num_people = 800, num_contexts = 50) {
  contexts <- data.frame(context_id = 1:num_contexts, context_risk = rnorm(num_contexts, 0, 1.5))
  true_person_variance <- matrix(c(0.5, 0.15, 0.15, 0.8), nrow = 2)
  dna_raw <- rmvnorm(num_people, mean = c(0, 0), sigma = true_person_variance)
  dna <- data.frame(person_id = 1:num_people, true_b_i = dna_raw[, 1], true_s_i = dna_raw[, 2])
  sparse_data <- dna %>%
    mutate(context_id = sample(contexts$context_id, size = num_people, replace = TRUE)) %>%
    left_join(contexts, by = "context_id") %>%
    mutate(outcome = 1.0 + true_b_i + (true_s_i * context_risk) + rnorm(num_people, 0, 1.0)) %>%
    select(person_id, context_id, context_risk, outcome)
  return(list(real_data = sparse_data, all_contexts = contexts, true_person_dna = dna))
}

set.seed(42)
phase_1_data <- generate_sparse_data()
real_data <- phase_1_data$real_data
all_contexts <- phase_1_data$all_contexts

print("--- Phase 1: Fitting Bayesian model to learn world rules... ---")
world_model <- brm(
  outcome ~ context_risk + (context_risk | person_id) + (1 | context_id),
  data = real_data,
  chains = 2, iter = 1000, cores = 2, silent = 2, refresh = 0
)

# Estimate individual traits

print("--- Phase 2: Estimating the DNA for each real person... ---")
estimated_real_dna_brms <- ranef(world_model)
estimated_real_dna_df <- data.frame(
  person_id = as.numeric(rownames(estimated_real_dna_brms$person_id)),
  est_b_i = estimated_real_dna_brms$person_id[, "Estimate", "Intercept"],
  est_s_i = estimated_real_dna_brms$person_id[, "Estimate", "context_risk"]
)

# Simulate outcomes

print("--- Phase 3: Simulating transport for each person... ---")
num_contexts_to_visit <- 20
est_fx <- fixef(world_model)[, "Estimate"]
est_res_sd <- summary(world_model)$spec_pars["sigma", "Estimate"]
sd_context <- VarCorr(world_model)$context_id$sd["Intercept", "Estimate"]

rich_data_list <- list()
for (i in 1:nrow(estimated_real_dna_df)) {
  current_person_est_dna <- estimated_real_dna_df[i, ]
  contexts_to_visit <- all_contexts %>% sample_n(size = num_contexts_to_visit)
  context_shocks <- rnorm(num_contexts_to_visit, 0, sd_context)

  simulated_outcomes <-
    est_fx["Intercept"] + (est_fx["context_risk"] * contexts_to_visit$context_risk) +
    current_person_est_dna$est_b_i +
    (current_person_est_dna$est_s_i * contexts_to_visit$context_risk) +
    context_shocks + rnorm(num_contexts_to_visit, 0, est_res_sd)

  rich_data_list[[i]] <- data.frame(
    person_id = current_person_est_dna$person_id,
    context_risk = contexts_to_visit$context_risk,
    simulated_outcome = simulated_outcomes
  )
}
rich_simulated_data <- dplyr::bind_rows(rich_data_list)



print("--- Phase 3: Calculating the final, multi-part CVI profile ---")

cvi_scores <- rich_simulated_data %>%
  group_by(person_id) %>%
  group_modify(~ {
    personal_model <- lm(simulated_outcome ~ context_risk, data = .x)

    r_squared_value <- summary(personal_model)$r.squared

    reactivity_slope <- coef(personal_model)["context_risk"]

    data.frame(
      CVI_R_Squared = r_squared_value,
      CVI_Reactivity = reactivity_slope
    )
  }) %>%
  ungroup()

print("--- Final CVI Profile Calculated ---")
print(head(cvi_scores))
print("--- Starting Final Diagnostic Analysis for CVI_Reactivity ---")

true_original_dna <- phase_1_data$true_person_dna

validation_df <- inner_join(cvi_scores, true_original_dna, by = "person_id")

validation_df <- validation_df %>%
  mutate(
    CVI_Reactivity_Abs = abs(CVI_Reactivity),
    true_reactivity_magnitude = abs(true_s_i)
  )

final_validation_test <- cor.test(
  validation_df$CVI_Reactivity_Abs,
  validation_df$true_reactivity_magnitude
)

print("--- 1. Definitive Validation Result ---")
print(final_validation_test)


validation_plot <- ggplot(validation_df, aes(x = true_reactivity_magnitude, y = CVI_Reactivity_Abs)) +
  geom_point(alpha = 0.6, color = "darkred") +
  geom_smooth(method = "lm", color = "black") +
  labs(
    title = "Final Validation: CVI_Reactivity vs. Ground Truth",
    x = "True Reactivity Magnitude |s_i| (Ground Truth)",
    y = "CVI Reactivity Magnitude |slope| (Final Indicator)"
  ) +
  theme_minimal()

print(validation_plot)


# The MOST reactive person (highest absolute slope)
most_reactive_person_id <- validation_df %>%
  filter(CVI_Reactivity_Abs == max(CVI_Reactivity_Abs)) %>%
  pull(person_id)

# The LEAST reactive person (lowest absolute slope)
least_reactive_person_id <- validation_df %>%
  filter(CVI_Reactivity_Abs == min(CVI_Reactivity_Abs)) %>%
  pull(person_id)

# Get their simulated data
most_reactive_data <- rich_simulated_data %>% filter(person_id == most_reactive_person_id)
least_reactive_data <- rich_simulated_data %>% filter(person_id == least_reactive_person_id)

plot_most_reactive <- ggplot(most_reactive_data, aes(x = context_risk, y = simulated_outcome)) +
  geom_point(color = "firebrick", alpha = 0.7) +
  geom_smooth(method = "lm", color = "black", se = FALSE) +
  labs(title = paste("Persona: Most Reactive (Person", most_reactive_person_id, ")")) +
  theme_minimal()

plot_least_reactive <- ggplot(least_reactive_data, aes(x = context_risk, y = simulated_outcome)) +
  geom_point(color = "steelblue", alpha = 0.7) +
  geom_smooth(method = "lm", color = "black", se = FALSE) +
  labs(title = paste("Persona: Least Reactive (Person", least_reactive_person_id, ")")) +
  coord_cartesian(ylim = range(most_reactive_data$simulated_outcome, least_reactive_data$simulated_outcome)) +
  theme_minimal()

persona_plot <- plot_least_reactive + plot_most_reactive

print("--- 2. Vulnerability Persona Profiles ---")
print(persona_plot)


distribution_plot <- ggplot(validation_df, aes(x = CVI_Reactivity)) +
  geom_histogram(aes(y = ..density..), bins = 30, fill = "gray80", color = "black") +
  geom_density(color = "darkblue", size = 1.2) +
  labs(
    title = "Distribution of the CVI_Reactivity Indicator",
    subtitle = "Shows the landscape of predictable sensitivity in the population",
    x = "CVI Reactivity Score (Personal Slope)",
    y = "Density"
  ) +
  theme_minimal()

print("--- 3. Distribution of the Final Indicator ---")
print(distribution_plot)


print("--- Final Analysis: Does CVI_Reactivity predict extreme outcomes? ---")

consequence_metrics <- rich_simulated_data %>%
  group_by(person_id) %>%
  summarise(
    outcome_range = max(simulated_outcome) - min(simulated_outcome),
    prob_high_risk = mean(simulated_outcome > quantile(rich_simulated_data$simulated_outcome, 0.90)),
    prob_low_risk = mean(simulated_outcome < quantile(rich_simulated_data$simulated_outcome, 0.10))
  ) %>%
  ungroup()


consequence_validation_df <- inner_join(cvi_scores, consequence_metrics, by = "person_id") %>%
  mutate(CVI_Reactivity_Abs = abs(CVI_Reactivity))


# Does higher reactivity lead to a wider range of outcomes?
corr_range <- cor(consequence_validation_df$CVI_Reactivity_Abs, consequence_validation_df$outcome_range)
print(paste("Correlation between Reactivity Magnitude and Outcome Range:", round(corr_range, 4)))

plot_range <- ggplot(consequence_validation_df, aes(x = CVI_Reactivity_Abs, y = outcome_range)) +
  geom_point(alpha = 0.6, color = "darkslateblue") +
  geom_smooth(method = "lm") +
  labs(
    title = "Consequence Test 1: Reactivity vs. Outcome Range",
    x = "CVI Reactivity Magnitude |slope|",
    y = "Range of Simulated Outcomes"
  ) +
  theme_minimal()

print(plot_range)


print("--- Running Direct Validation of the Bayesian Estimate ---")

validation_df <- inner_join(estimated_real_dna_df, true_original_dna, by = "person_id")

correlation_test <- cor.test(validation_df$est_s_i, validation_df$true_s_i)

print("--- VALIDATION RESULT ---")
print("Correlation between the Direct Bayesian Estimate and the Ground Truth:")
print(correlation_test)


validation_plot <- ggplot(validation_df, aes(x = true_s_i, y = est_s_i)) +
  geom_point(alpha = 0.6, color = "darkslateblue") +
  geom_smooth(method = "lm", color = "firebrick", se = FALSE) +
  labs(
    title = "Validation of the Direct Bayesian Estimator",
    subtitle = "Comparing the model's estimate (from sparse data) to the ground truth",
    x = "True Reactivity Slope (Ground Truth)",
    y = "Estimated Reactivity Slope (Indicator)"
  ) +
  theme_minimal()

print(validation_plot)


print("--- Starting Final Diagnostic Analysis ---")

# Rank order stability (running multiple simulations)
print("--- Running Diagnostic 1: Rank-Order Stability Test ---")

median_reactivity_value <- median(cvi_scores$CVI_Reactivity) # Let's use the raw reactivity for the median

median_reactivity_person_id <- cvi_scores %>%
  mutate(dist_from_median = abs(CVI_Reactivity - median_reactivity_value)) %>%
  filter(dist_from_median == min(dist_from_median)) %>%
  pull(person_id)

stability_scores <- replicate(100, {
  person_dna <- estimated_real_dna_df %>% filter(person_id == median_reactivity_person_id)
  contexts_to_visit <- all_contexts %>% sample_n(size = num_contexts_to_visit)
  context_shocks <- rnorm(num_contexts_to_visit, 0, sd_context)
  simulated_outcomes <- est_fx["Intercept"] + (est_fx["context_risk"] * contexts_to_visit$context_risk) +
    person_dna$est_b_i + (person_dna$est_s_i * contexts_to_visit$context_risk) +
    context_shocks + rnorm(num_contexts_to_visit, 0, est_res_sd)
  personal_model <- lm(simulated_outcomes ~ contexts_to_visit$context_risk)
  coef(personal_model)["contexts_to_visit$context_risk"]
})

stability_plot <- ggplot(data.frame(scores = stability_scores), aes(x = scores)) +
  geom_histogram(aes(y = ..density..), bins = 15, fill = "gray70", color = "black") +
  geom_density(color = "darkred", size = 1.2) +
  labs(
    title = paste("Stability of CVI_Reactivity for Person", median_reactivity_person_id),
    subtitle = "A narrow peak indicates a reliable and precise indicator.",
    x = "CVI_Reactivity Score (from 100 independent simulations)",
    y = "Density"
  ) +
  theme_minimal()

print(stability_plot)


# Context contribution plot

print("--- Running Diagnostic 2: Context Contribution Plot ---")
high_reactivity_person_id <- cvi_scores %>%
  mutate(abs_reactivity = abs(CVI_Reactivity)) %>%
  filter(abs_reactivity == max(abs_reactivity)) %>%
  pull(person_id)

high_reactivity_data <- rich_simulated_data %>% filter(person_id == high_reactivity_person_id)


personal_model_high_react <- lm(simulated_outcome ~ context_risk, data = high_reactivity_data)
high_reactivity_data$leverage <- hatvalues(personal_model_high_react)


contribution_plot <- ggplot(high_reactivity_data, aes(x = context_risk, y = simulated_outcome)) +
  # The size of the point is mapped to its leverage
  geom_point(aes(size = leverage), color = "firebrick", alpha = 0.7) +
  geom_smooth(method = "lm", se = FALSE, color = "black") +
  scale_size_continuous(range = c(2, 8)) +
  labs(
    title = paste("Context Contribution for High-Reactivity Person", high_reactivity_person_id),
    subtitle = "Larger points have more influence on the slope.",
    x = "Context Risk Score",
    y = "Simulated Outcome",
    size = "Leverage"
  ) +
  theme_minimal()

print(contribution_plot)


# Conditional density plot

print("--- Running Diagnostic 3: Conditional Density Plot ---")
low_reactivity_person_id <- cvi_scores %>%
  mutate(abs_reactivity = abs(CVI_Reactivity)) %>%
  filter(abs_reactivity == min(abs_reactivity)) %>%
  pull(person_id)
high_reactivity_data <- rich_simulated_data %>% filter(person_id == high_reactivity_person_id)
low_reactivity_data <- rich_simulated_data %>% filter(person_id == low_reactivity_person_id)

high_reactivity_data <- high_reactivity_data %>%
  mutate(risk_category = cut(context_risk, breaks = 3, labels = c("Low Risk", "Medium Risk", "High Risk")))
low_reactivity_data <- low_reactivity_data %>%
  mutate(risk_category = cut(context_risk, breaks = 3, labels = c("Low Risk", "Medium Risk", "High Risk")))

# Plot for the High-Reactivity Person
density_plot_high <- ggplot(high_reactivity_data, aes(x = simulated_outcome, fill = risk_category)) +
  geom_density(alpha = 0.6) +
  labs(
    title = paste("Conditional Density for High-Reactivity Person", high_reactivity_person_id),
    subtitle = "The distribution of outcomes shifts dramatically with context risk.",
    x = "Simulated Outcome", fill = "Context Type"
  ) +
  theme_minimal()

# Plot for the Low-Reactivity Person
density_plot_low <- ggplot(low_reactivity_data, aes(x = simulated_outcome, fill = risk_category)) +
  geom_density(alpha = 0.6) +
  labs(
    title = paste("Conditional Density for Low-Reactivity Person", low_reactivity_person_id),
    subtitle = "The distribution of outcomes is stable across contexts.",
    x = "Simulated Outcome", fill = "Context Type"
  ) +
  theme_minimal()

conditional_density_comparison <- density_plot_low + density_plot_high

print(conditional_density_comparison)
