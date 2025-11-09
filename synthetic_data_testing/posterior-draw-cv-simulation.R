library(brms)
library(dplyr)
library(mvtnorm)
library(lme4)
library(ggplot2)
library(tidyr)
library(e1071)
library(patchwork)
library(mclust)


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
  # sparse_data <- dna %>%
  #   mutate(context_id = sample(contexts$context_id, size = num_people, replace = TRUE)) %>%
  #   left_join(contexts, by = "context_id") %>%
  #   mutate(outcome = 1.0 + true_b_i + (true_s_i * context_risk) + rnorm(num_people, 0, 1.0)) %>%
  #   select(person_id, context_id, context_risk, outcome)
  sparse_data <- dna %>%
    mutate(context_id = sample(contexts$context_id, size = num_people, replace = TRUE)) %>%
    left_join(contexts, by = "context_id") %>%
    mutate(
      linear_predictor = -2.0 + true_b_i + (true_s_i * context_risk) + rnorm(num_people, 0, 1.0),
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

set.seed(42)
phase_1_data <- generate_sparse_data()
real_data <- phase_1_data$real_data
all_contexts <- phase_1_data$all_contexts

model_priors <- c(
  prior(normal(0, 0.2), class = "sd"),
  prior(lkj(2), class = "cor"),
  prior(normal(-3, 1), class = "Intercept"),
  prior(normal(0, 0.5), class = "b")
)

print("Fitting Bayesian model to learn world rules..")
world_model <- brm(
  outcome ~ context_risk + (context_risk |
                              person_id) + (1 | context_id),
  data = real_data,
  family = bernoulli(link = "logit"),
  prior = model_priors,
  # sample_prior = "only",
  chains = 2,
  iter = 2000,
  cores = 2,
  silent = 2,
  refresh = 0
)

# Plot the prior predictive distribution
# plot(pp_check(world_model, nsamples = 100))

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
  # est_res_sd <- current_world_rules$sigma
  
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
    # Removed random shock when using Bernoulli setup
    simulated_outcomes <- est_fx_intercept + (est_fx_context_risk * contexts_to_visit$context_risk) +
      current_person_est_dna$est_b_i + (current_person_est_dna$est_s_i * contexts_to_visit$context_risk) +
      context_shocks
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
  filter(CVI_SCORE == max(CVI_SCORE)) %>%
  pull(person_id)


low_risk_distribution <- final_posterior_cvi_data %>% filter(person_id == low_risk_person_id)
high_risk_distribution <- final_posterior_cvi_data %>% filter(person_id == high_risk_person_id)

plot_low_risk <- ggplot(low_risk_distribution, aes(x = CVI_Reactivity)) +
  geom_density(fill = "steelblue", alpha = 0.8) +
  labs(
    title = paste("Posterior for Low-Reactivity Person", low_risk_person_id),
    x = "CV Score (Reactivity)",
    y = "Density"
  ) +
  theme_minimal()

plot_high_risk <- ggplot(high_risk_distribution, aes(x = CVI_Reactivity)) +
  geom_density(fill = "firebrick", alpha = 0.8) +
  labs(
    title = paste("Posterior for High-Reactivity Person", high_risk_person_id),
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


top_5_ids <- final_cvi_summary %>%
  arrange(desc(CVI_SCORE)) %>%
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


print("Generating a plot to validate the CVI score's meaning")

low_risk_person_id <- final_cvi_summary %>%
  mutate(abs_cvi = abs(CVI_SCORE)) %>%
  filter(abs_cvi == min(abs_cvi)) %>%
  pull(person_id)

high_risk_person_id <- final_cvi_summary %>%
  filter(CVI_SCORE == max(CVI_SCORE)) %>%
  pull(person_id)

low_risk_posterior <- final_posterior_cvi_data %>% filter(person_id == low_risk_person_id)
high_risk_posterior <- final_posterior_cvi_data %>% filter(person_id == high_risk_person_id)

low_risk_final_score <- final_cvi_summary %>% filter(person_id == low_risk_person_id) %>% pull(CVI_SCORE)
high_risk_final_score <- final_cvi_summary %>% filter(person_id == high_risk_person_id) %>% pull(CVI_SCORE)

plot_stable_dist <- ggplot(low_risk_posterior, aes(x = CVI_Reactivity)) +
  geom_density(fill = "darkgreen", alpha = 0.7) +
  geom_vline(xintercept = low_risk_final_score, color = "black", size = 1.5, linetype = "dashed") +
  annotate("text", x = low_risk_final_score, y = 0.5, label = "Final CVI Score\n(Mean of this distribution)", hjust = -0.1) +
  labs(
    title = paste("Posterior for Most Stable Person (", low_risk_person_id, ")"),
    x = "CVI Score (Slope from a single simulated world)",
    y = "Density"
  ) +
  theme_minimal()

plot_vulnerable_dist <- ggplot(high_risk_posterior, aes(x = CVI_Reactivity)) +
  geom_density(fill = "darkred", alpha = 0.7) +
  geom_vline(xintercept = high_risk_final_score, color = "black", size = 1.5, linetype = "dashed") +
  annotate("text", x = high_risk_final_score, y = 0.5, label = "Final CVI Score\n(Mean of this distribution)", hjust = 1.1) +
  labs(
    title = paste("Posterior for Most Vulnerable Person (", high_risk_person_id, ")"),
    x = "CVI Score (Slope from a single simulated world)",
    y = "Density"
  ) +
  theme_minimal()

cvi_proof_plot <- plot_stable_dist + plot_vulnerable_dist
print(cvi_proof_plot)

vulnerable_person_id <- final_cvi_summary %>%
  filter(CVI_SCORE == max(CVI_SCORE)) %>%
  pull(person_id)

vulnerable_posterior <- final_posterior_cvi_data %>%
  filter(person_id == vulnerable_person_id)

proportion_positive <- mean(vulnerable_posterior$CVI_Reactivity > 0)

individual_proof_plot <- ggplot(vulnerable_posterior, aes(x = CVI_Reactivity)) +
  geom_density(fill = "firebrick", alpha = 0.7) +
  geom_vline(xintercept = 0, color = "black", linetype = "dashed", size = 1) +
  labs(
    title = paste("Proof of Positive Correlation for Most Vulnerable Person (", vulnerable_person_id, ")"),
    subtitle = paste(
      "Distribution of their 100 simulated CVI scores.",
      round(proportion_positive * 100, 1),
      "% of scores are positive."
    ),
    x = "CVI Score (Slope from a single simulated world)",
    y = "Density"
  ) +
  theme_minimal()

print(individual_proof_plot)
print(paste("Metric: The posterior probability that this person's CVI is > 0 is:", proportion_positive))


low_risk_person_id <- final_cvi_summary %>%
  mutate(abs_cvi = abs(CVI_SCORE)) %>%
  filter(abs_cvi == min(abs_cvi)) %>%
  pull(person_id)

high_risk_person_id <- final_cvi_summary %>%
  filter(CVI_SCORE == max(CVI_SCORE)) %>%
  pull(person_id)

prediction_grid <- data.frame(
  context_risk = seq(min(all_contexts$context_risk), max(all_contexts$context_risk), length.out = 50)
)

all_world_params <- worlds_to_simulate %>%
  mutate(world_id = 1:n()) %>%
  select(world_id, b_Intercept, b_context_risk, starts_with("r_person_id")) %>%
  pivot_longer(
    cols = starts_with("r_person_id"),
    names_to = "parameter",
    values_to = "value"
  ) %>%
  extract(
    parameter,
    into = c("person_id", "term"),
    regex = "r_person_id\\[(\\d+),(\\w+)\\]",
    convert = TRUE
  ) %>%
  pivot_wider(names_from = "term", values_from = "value") %>%
  rename(r_intercept = Intercept, r_slope = context_risk)


full_predictions <- tidyr::crossing(all_world_params, prediction_grid) %>%
  mutate(
    predicted_outcome = (b_Intercept + r_intercept) + (b_context_risk + r_slope) * context_risk
  )


prediction_summary <- full_predictions %>%
  filter(person_id %in% c(low_risk_person_id, high_risk_person_id)) %>%
  group_by(person_id, context_risk) %>%
  summarise(
    mean_outcome = mean(predicted_outcome),
    lower_95 = quantile(predicted_outcome, 0.025),
    upper_95 = quantile(predicted_outcome, 0.975),
    .groups = "drop"
  ) %>%
  mutate(
    Persona = ifelse(person_id == low_risk_person_id,
                     paste("Most Stable (Person", person_id, ")"),
                     paste("Most Vulnerable (Person", person_id, ")"))
  )


representative_plot <- ggplot(prediction_summary, aes(x = context_risk, y = mean_outcome)) +
  # The shaded ribbon represents the 95% credible interval across all 100 worlds
  geom_ribbon(aes(ymin = lower_95, ymax = upper_95, fill = Persona), alpha = 0.3) +
  geom_line(aes(color = Persona), size = 1.2) +
  facet_wrap(~ Persona, scales = "free_y") +
  labs(
    title = "Representative Profiles with Full Posterior Uncertainty",
    subtitle = "Lines are the mean prediction; ribbons are the 95% credible interval across all simulated worlds.",
    x = "Context Risk Score",
    y = "Predicted Outcome (Log-Odds)"
  ) +
  theme_minimal() +
  theme(legend.position = "none")

print(representative_plot)

# Ideas for deriving threshold

# Gaussian mixture model to separate groups
mixture_model <- Mclust(final_cvi_summary$CVI_SCORE, G = 2:3)

print(paste("Optimal Model:", mixture_model$G, mixture_model$modelName))
params <- mixture_model$parameters

group_means <- params$mean
group_order <- order(group_means)

if (mixture_model$G == 3) {
  mean_resilient <- params$mean[group_order[1]]
  mean_stable <- params$mean[group_order[2]]
  mean_vulnerable <- params$mean[group_order[3]]
  
  prop_resilient <- params$pro[group_order[1]]
  prop_stable <- params$pro[group_order[2]]
  prop_vulnerable <- params$pro[group_order[3]]
  
  # Deal with both equal var and variable var separately
  if (mixture_model$modelName == "E") {
    sd_resilient <- sd_stable <- sd_vulnerable <- sqrt(params$variance$sigmasq)
  } else { # "V"
    sd_resilient <- sqrt(params$variance$sigmasq[group_order[1]])
    sd_stable <- sqrt(params$variance$sigmasq[group_order[2]])
    sd_vulnerable <- sqrt(params$variance$sigmasq[group_order[3]])
  }
} else {
  mean_stable <- params$mean[group_order[1]]
  mean_vulnerable <- params$mean[group_order[2]]
  
  prop_stable <- params$pro[group_order[1]]
  prop_vulnerable <- params$pro[group_order[2]]
  
  # Deal with both equal var and variable var separately
  if (mixture_model$modelName == "E") {
sd_stable <- sd_vulnerable <- sqrt(params$variance$sigmasq)
  } else { # "V"
    sd_stable <- sqrt(params$variance$sigmasq[group_order[1]])
    sd_vulnerable <- sqrt(params$variance$sigmasq[group_order[2]])
  }
}

find_intersection <- function(x, p1, m1, s1, p2, m2, s2) {
  density1 <- p1 * dnorm(x, mean = m1, sd = s1)
  density2 <- p2 * dnorm(x, mean = m2, sd = s2)
  return(density1 - density2)
}

if (mixture_model$G == 3) {
  boundary_res_st <- uniroot(
    find_intersection,
    interval = c(mean_resilient, mean_stable),
    p1 = prop_resilient, m1 = mean_resilient, s1 = sd_resilient,
    p2 = prop_stable, m2 = mean_stable, s2 = sd_stable
  )$root
  print(paste("First boundary:", round(boundary_res_st, 3)))
}

boundary_st_vul <- uniroot(
  find_intersection,
  interval = c(mean_stable, mean_vulnerable),
  p1 = prop_stable, m1 = mean_stable, s1 = sd_stable,
  p2 = prop_vulnerable, m2 = mean_vulnerable, s2 = sd_vulnerable
)$root

print(paste("Second (main) boundary:", round(boundary_st_vul, 3)))

final_plot <- ggplot(final_cvi_summary, aes(x = CVI_SCORE)) +
  geom_histogram(bins = 30, fill = "grey", alpha = 0.7, aes(y = ..density..)) +
  
  # stable
  stat_function(
    fun = function(x) { prop_stable * dnorm(x, mean = mean_stable, sd = sd_stable) },
    color = "steelblue", size = 1.2
  ) +
  
  # vulnerable
  stat_function(
    fun = function(x) { prop_vulnerable * dnorm(x, mean = mean_vulnerable, sd = sd_vulnerable) },
    color = "firebrick", size = 1.2
  ) +
  
  # The precise decision boundaries
  geom_vline(xintercept = boundary_st_vul, color = "black", linetype = "dashed", size = 1.5) +
  
  labs(
    title = paste("Data-Driven Identification of Sub-populations (Optimal Model G=", mixture_model$G, ")", sep=""),
    x = "Final CVI Score",
    y = "Density"
  ) +
  theme_minimal()

if (mixture_model$G == 3) {
  final_plot <- final_plot + 
    # "resilient"
    stat_function(
      fun = function(x) { prop_resilient * dnorm(x, mean = mean_resilient, sd = sd_resilient) },
      color = "darkorange", size = 1.2
    ) +
    geom_vline(xintercept = boundary_res_st, color = "black", linetype = "dotted", size = 1) 
}

print(final_plot)
summary(mixture_model)
