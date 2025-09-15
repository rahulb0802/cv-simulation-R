library(brms)
library(dplyr)
library(mvtnorm)
library(lme4)
library(ggplot2)
library(tidyr)
library(e1071)
library(patchwork)


generate_sparse_data <- function(num_people = 200,
                                 num_contexts = 50) {
  contexts <- data.frame(
    context_id = 1:num_contexts,
    context_risk = rnorm(num_contexts, 0, 1.5)
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
    mutate(outcome = 1.0 + true_b_i + (true_s_i * context_risk) + rnorm(num_people, 0, 1.0)) %>%
    select(person_id, context_id, context_risk, outcome)
  return(list(
    real_data = sparse_data,
    all_contexts = contexts,
    true_person_dna = dna
  ))
}

set.seed(42)
phase_1_data <- generate_sparse_data()
real_data <- phase_1_data$real_data
all_contexts <- phase_1_data$all_contexts

model_priors <- c(
  prior(
    student_t(3, 0, 2.5),
    class = "sd",
    group = "person_id",
    coef = "Intercept"
  ),
  prior(
    student_t(3, 0, 2.5),
    class = "sd",
    group = "person_id",
    coef = "context_risk"
  ),
  prior(student_t(3, 0, 2.5), class = "sd", group = "context_id"),
  prior(lkj(2), class = "cor"),
  prior(student_t(3, 0, 10), class = "Intercept"),
  prior(normal(0, 5), class = "b", coef = "context_risk")
)

print("Fitting Bayesian model to learn world rules..")
world_model <- brm(
  outcome ~ context_risk + (context_risk |
                              person_id) + (1 | context_id),
  data = real_data,
  prior = model_priors,
  chains = 2,
  iter = 1000,
  cores = 2,
  silent = 2,
  refresh = 0
)

print("Simulating draws from posterior distributions...")

posterior_draws <- as_draws_df(world_model)
print(head(posterior_draws))

num_worlds_to_simulate <- 100
worlds_to_simulate <- posterior_draws %>% sample_n(size = num_worlds_to_simulate)

master_cvi_list <- list()
print(paste(
  "Beginning simulation across",
  num_worlds_to_simulate,
  "plausible worlds..."
))

for (world_i in 1:nrow(worlds_to_simulate)) {
  current_world_rules <- worlds_to_simulate[world_i, ]
  
  # Fixed Effects
  est_fx_intercept <- current_world_rules$b_Intercept
  est_fx_context_risk <- current_world_rules$b_context_risk
  
  # Variances and Covariances
  sd_person_intercept <- current_world_rules$`sd_person_id__Intercept`
  sd_person_slope <- current_world_rules$`sd_person_id__context_risk`
  cor_person <- current_world_rules$`cor_person_id__Intercept__context_risk`
  var_int <- sd_person_intercept^2
  var_slope <- sd_person_slope^2
  cov_person <- cor_person * sd_person_intercept * sd_person_slope
  dna_blueprint_matrix <- matrix(c(var_int, cov_person, cov_person, var_slope),
                                 nrow = 2,
                                 byrow = TRUE)
  
  sd_context <- current_world_rules$`sd_context_id__Intercept`
  est_res_sd <- current_world_rules$sigma
  
  person_ranef_cols <- select(current_world_rules, starts_with("r_person_id"))
  
  estimated_real_dna_df <- person_ranef_cols %>%
    tidyr::pivot_longer(everything(), names_to = "parameter", values_to = "value") %>%
    tidyr::extract(parameter,
                   into = c("person_id", "term"),
                   regex = "r_person_id\\[(\\d+),(\\w+)\\]") %>%
    tidyr::pivot_wider(names_from = "term", values_from = "value") %>%
    rename(est_b_i = Intercept, est_s_i = context_risk) %>%
    mutate(person_id = as.numeric(person_id))
  
  num_contexts_to_visit <- 50
  rich_data_list <- list()
  for (i in 1:nrow(estimated_real_dna_df)) {
    current_person_est_dna <- estimated_real_dna_df[i, ]
    contexts_to_visit <- all_contexts %>% sample_n(size = num_contexts_to_visit)
    context_shocks <- rnorm(num_contexts_to_visit, 0, sd_context)
    simulated_outcomes <- est_fx_intercept + (est_fx_context_risk * contexts_to_visit$context_risk) +
      current_person_est_dna$est_b_i + (current_person_est_dna$est_s_i * contexts_to_visit$context_risk) +
      context_shocks + rnorm(num_contexts_to_visit, 0, est_res_sd)
    rich_data_list[[i]] <- data.frame(
      person_id = current_person_est_dna$person_id,
      context_risk = contexts_to_visit$context_risk,
      simulated_outcome = simulated_outcomes
    )
  }
  
  rich_simulated_data <- dplyr::bind_rows(rich_data_list)
  
  cvi_profile_this_world <- rich_simulated_data %>%
    group_by(person_id) %>%
    group_modify( ~ {
      personal_model <- lm(simulated_outcome ~ context_risk, data = .x)
      data.frame(CVI_Reactivity = coef(personal_model)["context_risk"])
    }) %>%
    ungroup()
  
  cvi_profile_this_world$world_id <- world_i
  master_cvi_list[[world_i]] <- cvi_profile_this_world
  
  # Print progress
  if (world_i %% 10 == 0) {
    print(
      paste(
        "...completed simulation for world",
        world_i,
        "of",
        num_worlds_to_simulate
      )
    )
  }
}

final_posterior_cvi_data <- dplyr::bind_rows(master_cvi_list)

print(head(final_posterior_cvi_data))

final_cvi_summary <- final_posterior_cvi_data %>%
  group_by(person_id) %>%
  summarise(
    # The median is a robust choice for the "best guess"
    CVI_SCORE = mean(CVI_Reactivity),
    CVI_lower_95 = quantile(CVI_Reactivity, 0.025),
    CVI_upper_95 = quantile(CVI_Reactivity, 0.975)
  ) %>%
  ungroup()

print("Head of the final summary:")
print(head(final_cvi_summary))

low_risk_person_id <- final_cvi_summary %>%
  mutate(abs_cvi = abs(CVI_SCORE)) %>%
  filter(abs_cvi == min(abs_cvi)) %>%
  pull(person_id)

high_risk_person_id <- final_cvi_summary %>%
  mutate(abs_cvi = abs(CVI_SCORE)) %>%
  filter(abs_cvi == max(abs_cvi)) %>%
  pull(person_id)


low_risk_distribution <- final_posterior_cvi_data %>% filter(person_id == low_risk_person_id)
high_risk_distribution <- final_posterior_cvi_data %>% filter(person_id == high_risk_person_id)

plot_low_risk <- ggplot(low_risk_distribution, aes(x = CVI_Reactivity)) +
  geom_density(fill = "steelblue", alpha = 0.8) +
  labs(
    title = paste("Posterior for Low-Reactivity Person", low_risk_person_id),
    subtitle = "Distribution is centered on zero.",
    x = "CV Score (Reactivity)",
    y = "Density"
  ) +
  theme_minimal()

plot_high_risk <- ggplot(high_risk_distribution, aes(x = CVI_Reactivity)) +
  geom_density(fill = "firebrick", alpha = 0.8) +
  labs(
    title = paste("Posterior for High-Reactivity Person", high_risk_person_id),
    subtitle = "Distribution is far from zero.",
    x = "CV Score (Reactivity)",
    y = "Density"
  ) +
  coord_cartesian(
    xlim = range(
      high_risk_distribution$CVI_Reactivity,
      low_risk_distribution$CVI_Reactivity
    )
  )

final_validation_plot <- plot_low_risk + plot_high_risk

print(final_validation_plot)

low_risk_distribution <- rich_simulated_data %>% filter(person_id == low_risk_person_id)
high_risk_distribution <- rich_simulated_data %>% filter(person_id == high_risk_person_id)

plot_stable <- ggplot(low_risk_distribution, aes(x = context_risk, y = simulated_outcome)) +
  geom_point(color = "darkgreen", alpha = 0.6) +
  geom_smooth(method = "lm", color = "black", se = FALSE) +
  labs(
    title = paste("Profile for Most Stable (Person", low_risk_person_id, ")"),
    subtitle = "A flat slope indicates low reactivity.",
    x = "Context Risk Score",
    y = "Simulated Outcome"
  ) +
  theme_minimal()

# Plot for the "Most Vulnerable" (high positive slope) persona
plot_vulnerable <- ggplot(high_risk_distribution, aes(x = context_risk, y = simulated_outcome)) +
  geom_point(color = "darkred", alpha = 0.6) +
  geom_smooth(method = "lm", color = "black", se = FALSE) +
  labs(
    title = paste("Profile for Most Vulnerable (Person", high_risk_person_id, ")"),
    subtitle = "A steep positive slope indicates high vulnerability.",
    x = "Context Risk Score",
    y = "Simulated Outcome"
  ) +
  theme_minimal()

final_cvi_plot <- plot_stable + plot_vulnerable

print(final_cvi_plot)

caterpillar_plot <- final_cvi_summary %>%
  mutate(person_id = reorder(as.factor(person_id), CVI_SCORE)) %>%
  ggplot(aes(x = person_id, y = CVI_SCORE)) +
  geom_errorbar(
    aes(ymin = CVI_lower_95, ymax = CVI_upper_95),
    width = 0.2,
    color = "gray50"
  ) +
  geom_point(color = "darkblue", size = 2) +
  geom_hline(yintercept = 0,
             linetype = "dashed",
             color = "red") +
  labs(
    title = "Distribution of Individual CV Scores with Uncertainty",
    subtitle = "Each point is a person's estimated reactivity, with 95% credible intervals.",
    x = "Individual Person (ordered by CV Score)",
    y = "CV Score (Reactivity)"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_blank(), axis.ticks.x = element_blank())

print(caterpillar_plot)

certainty_df <- final_cvi_summary %>%
  mutate(
    uncertainty_width = CVI_upper_95 - CVI_lower_95,
    reactivity_magnitude = abs(CVI_SCORE)
  )

certainty_plot <- ggplot(certainty_df,
                         aes(x = reactivity_magnitude, y = uncertainty_width)) +
  geom_point(alpha = 0.7, color = "darkgreen") +
  geom_smooth(method = "lm",
              color = "black",
              linetype = "dotted") +
  labs(
    title = "Certainty vs. Reactivity Magnitude",
    subtitle = "Is our uncertainty related to how reactive a person is?",
    x = "CV Score Magnitude |Reactivity|",
    y = "Width of 95% Credible Interval (Uncertainty)"
  ) +
  theme_minimal()

print(certainty_plot)

top_5_ids <- final_cvi_summary %>%
  mutate(abs_cvi = abs(CVI_SCORE)) %>%
  arrange(desc(abs_cvi)) %>%
  head(5) %>%
  pull(person_id)

top_5_data <- final_posterior_cvi_data %>%
  filter(person_id %in% top_5_ids)

top_5_plot <- ggplot(top_5_data, aes(x = CVI_Reactivity)) +
  facet_wrap( ~ person_id, ncol = 5) +
  geom_density(fill = "firebrick", alpha = 0.8) +
  labs(title = "Posterior Profiles for the Top 5 Most Vulnerable Individuals", x = "CV Score (Reactivity)", y = "Density") +
  theme_minimal() +
  theme(axis.text.y = element_blank(), axis.ticks.y = element_blank()) # Clean up y-axis

print(top_5_plot)

point_estimates <- final_posterior_cvi_data %>%
  group_by(person_id) %>%
  summarise(CVI_Reactivity = median(CVI_Reactivity)) %>%
  ungroup()

scale_asinh_robust <- function(x) {
  robust_scaled <- (x - median(x)) / IQR(x)
  compressed <- asinh(robust_scaled)
  max_abs_val <- max(abs(compressed))
  normalized <- compressed / max_abs_val
  return(normalized)
}

final_nvi_scores <- point_estimates %>%
  mutate(NVI_Score = scale_asinh_robust(CVI_Reactivity))

print("--- Final NVI Scores Calculated ---")
print("Head of the final NVI scores, sorted by score:")
print(head(final_nvi_scores %>% arrange(desc(NVI_Score))))


nvi_plot <- ggplot(final_nvi_scores, aes(x = NVI_Score)) +
  geom_histogram(
    bins = 25,
    fill = "darkblue",
    color = "white",
    alpha = 0.8
  ) +
  labs(
    title = "Distribution of the Final Normalized Vulnerability Index (NVI)",
    subtitle = "The final score is scaled to a -1 to 1 range.",
    x = "NVI Score (0 = Least Vulnerable, 1 = Most Vulnerable)",
    y = "Frequency"
  ) +
  theme_minimal()

print(nvi_plot)

high_nvi_person_id <- final_nvi_scores %>%
  filter(NVI_Score == max(NVI_Score)) %>%
  pull(person_id)

low_nvi_person_id <- final_nvi_scores %>%
  mutate(abs_nvi = abs(NVI_Score)) %>%
  filter(abs_nvi == min(abs_nvi)) %>%
  pull(person_id)

high_nvi_data <- rich_simulated_data %>% filter(person_id == high_nvi_person_id)
low_nvi_data <- rich_simulated_data %>% filter(person_id == low_nvi_person_id)

plot_high_nvi <- ggplot(high_nvi_data, aes(x = context_risk, y = simulated_outcome)) +
  geom_point(color = "darkred", alpha = 0.6) +
  geom_smooth(method = "lm",
              color = "black",
              se = FALSE) +
  labs(
    title = paste("Profile for Most Vulnerable (Person", high_nvi_person_id, ")"),
    subtitle = "A steep slope indicates high reactivity.",
    x = "Context Risk Score",
    y = "Simulated Outcome"
  ) +
  theme_minimal()

plot_low_nvi <- ggplot(low_nvi_data, aes(x = context_risk, y = simulated_outcome)) +
  geom_point(color = "darkblue", alpha = 0.6) +
  geom_smooth(method = "lm",
              color = "black",
              se = FALSE) +
  labs(
    title = paste("Profile for Least Vulnerable (Person", low_nvi_person_id, ")"),
    subtitle = "A flat slope indicates low reactivity.",
    x = "Context Risk Score",
    y = "Simulated Outcome"
  ) +
  coord_cartesian(ylim = range(
    high_nvi_data$simulated_outcome,
    low_nvi_data$simulated_outcome
  ))

final_validation_plot <- plot_low_nvi + plot_high_nvi

print(final_validation_plot)

initial_uncertainty <- final_posterior_cvi_data %>%
  group_by(person_id) %>%
  summarise(
    lower_95 = quantile(CVI_Reactivity, 0.025),
    upper_95 = quantile(CVI_Reactivity, 0.975)
  ) %>%
  ungroup()

nvi_with_uncertainty <- inner_join(final_nvi_scores, initial_uncertainty, by = "person_id")


# Caterpillar plot

nvi_caterpillar_plot <- nvi_with_uncertainty %>%
  mutate(person_id = reorder(as.factor(person_id), NVI_Score)) %>%
  ggplot(aes(x = person_id, y = CVI_Reactivity)) + # Plotting the raw reactivity
  geom_errorbar(aes(ymin = lower_95, ymax = upper_95),
                width = 0.2,
                color = "gray60") +
  geom_point(aes(color = NVI_Score), size = 2) + # Color the points by their FINAL NVI score
  scale_color_gradient2(
    low = "blue",
    mid = "grey",
    high = "red",
    midpoint = 0
  ) +
  geom_hline(yintercept = 0,
             linetype = "dashed",
             color = "black") +
  labs(
    title = "Distribution of Individual Reactivity with Uncertainty",
    subtitle = "Points colored by their final NVI Score (-1 to +1).",
    x = "Individual Person (ordered by NVI Score)",
    y = "Raw CVI_Reactivity (Unscaled)"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_blank(), axis.ticks.x = element_blank())

print(nvi_caterpillar_plot)

# Certainty plot

certainty_df <- nvi_with_uncertainty %>%
  mutate(uncertainty_width = upper_95 - lower_95,
         nvi_magnitude = abs(NVI_Score))

certainty_plot_nvi <- ggplot(certainty_df, aes(x = nvi_magnitude, y = uncertainty_width)) +
  geom_point(alpha = 0.6, color = "darkorange") +
  geom_smooth(method = "lm",
              color = "black",
              linetype = "dotted") +
  labs(
    title = "NVI Magnitude vs. Initial Uncertainty",
    subtitle = "Are we more certain about people with more extreme NVI scores?",
    x = "NVI Score Magnitude |-1 to +1|",
    y = "Width of Initial 95% Credible Interval (Uncertainty)"
  ) +
  theme_minimal()

print(certainty_plot_nvi)

high_nvi_person_id <- final_nvi_scores %>%
  filter(NVI_Score == max(NVI_Score)) %>%
  pull(person_id)
low_nvi_person_id <- final_nvi_scores %>%
  mutate(abs_nvi = abs(NVI_Score)) %>%
  filter(abs_nvi == min(abs_nvi)) %>%
  pull(person_id)

high_nvi_dist <- final_posterior_cvi_data %>% filter(person_id == high_nvi_person_id)
low_nvi_dist <- final_posterior_cvi_data %>% filter(person_id == low_nvi_person_id)

high_nvi_score <- final_nvi_scores %>%
  filter(person_id == high_nvi_person_id) %>%
  pull(NVI_Score)
low_nvi_score <- final_nvi_scores %>%
  filter(person_id == low_nvi_person_id) %>%
  pull(NVI_Score)

plot_high_nvi_dist <- ggplot(high_nvi_dist, aes(x = CVI_Reactivity)) +
  geom_density(fill = "firebrick", alpha = 0.8) +
  labs(
    title = paste(
      "Posterior Profile for Most Vulnerable (Person",
      high_nvi_person_id,
      ")"
    ),
    subtitle = paste("Final NVI Score =", round(high_nvi_score, 2)),
    x = "Raw CVI_Reactivity"
  ) +
  theme_minimal()

plot_low_nvi_dist <- ggplot(low_nvi_dist, aes(x = CVI_Reactivity)) +
  geom_density(fill = "steelblue", alpha = 0.8) +
  labs(
    title = paste(
      "Posterior Profile for Most Stable (Person",
      low_nvi_person_id,
      ")"
    ),
    subtitle = paste("Final NVI Score =", round(low_nvi_score, 2)),
    x = "Raw CVI_Reactivity"
  ) +
  coord_cartesian(xlim = range(high_nvi_dist$CVI_Reactivity, low_nvi_dist$CVI_Reactivity))

persona_density_plot <- plot_low_nvi_dist + plot_high_nvi_dist
print(persona_density_plot)
