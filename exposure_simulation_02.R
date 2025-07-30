library(dplyr)
library(lme4)
library(ggplot2)
library(patchwork)

generate_crossed_data <- function(
    num_people = 150,
    num_contexts = 50,
    obs_per_person = 10,
    beta = 0.5,
    G_person = 0.5,
    G_context = 0.7,
    sigma_e = 1.5) {
  person_effects <- data.frame(
    person_id = 1:num_people,
    b_i = rnorm(num_people, mean = 0, sd = sqrt(G_person))
  )

  context_effects <- data.frame(
    context_id = 1:num_contexts,
    u_j = rnorm(num_contexts, mean = 0, sd = sqrt(G_context))
  )

  records_list <- list()
  for (i in 1:num_people) {
    person_id_val <- person_effects$person_id[i]
    b_i_val <- person_effects$b_i[i]

    contexts_visited <- sample(context_effects$context_id, size = obs_per_person)

    visited_context_effects <- context_effects %>% filter(context_id %in% contexts_visited)

    x_ij <- rnorm(obs_per_person, 0, 1)
    random_shock_ij <- rnorm(obs_per_person, 0, sd = sigma_e)

    y_ij <- beta * x_ij + b_i_val + visited_context_effects$u_j + random_shock_ij

    records_list[[i]] <- data.frame(
      person_id = person_id_val,
      context_id = visited_context_effects$context_id,
      y = y_ij,
      x = x_ij
    )
  }

  final_df <- dplyr::bind_rows(records_list)

  return(list(
    data = final_df,
    person_effects = person_effects,
    context_effects = context_effects
  ))
}

set.seed(42)
crossed_data_list <- generate_crossed_data()
crossed_data <- crossed_data_list$data

crossed_model <- lmer(y ~ x + (1 | person_id) + (1 | context_id), data = crossed_data, REML = FALSE)

fixed_intercept <- fixef(crossed_model)["(Intercept)"]
fixed_beta_x <- fixef(crossed_model)["x"]

# Random effects data frames
person_effects_est <- ranef(crossed_model)$person_id
context_effects_est <- ranef(crossed_model)$context_id

sigma_e_est <- sigma(crossed_model)


person_to_simulate <- 5
print(paste("Running simulation for Person", person_to_simulate))


b_i_est <- person_effects_est[person_to_simulate, "(Intercept)"]
x_val <- 0


all_context_ids <- row.names(context_effects_est)


simulated_outcomes <- c()
for (context_j_id in all_context_ids) {
  u_j_est <- context_effects_est[context_j_id, "(Intercept)"]


  predicted_y <- fixed_intercept + (fixed_beta_x * x_val) + b_i_est + u_j_est


  random_shock <- rnorm(1, mean = 0, sd = sigma_e_est)


  final_simulated_y <- predicted_y + random_shock


  simulated_outcomes <- c(simulated_outcomes, final_simulated_y)
}



sim_results_df <- data.frame(outcomes = simulated_outcomes)

# --- Plot 1: The Risk Distribution Histogram ---
risk_distribution_plot <- ggplot(sim_results_df, aes(x = outcomes)) +
  geom_histogram(aes(y = ..density..), bins = 15, fill = "skyblue", color = "black", alpha = 0.7) +
  geom_density(color = "red", linewidth = 1.2) +
  labs(
    title = paste("Simulated Risk Exposure Distribution for Person", person_to_simulate),
    subtitle = "Distribution of likely outcomes across all contexts",
    x = "Simulated Outcome",
    y = "Density"
  ) +
  theme_minimal()

sim_results_df$context_id <- as.numeric(all_context_ids)


# Let's find the top 3 contexts that produced the highest-risk outcomes
dangerous_contexts <- sim_results_df %>%
  arrange(desc(outcomes)) %>%
  head(3)

print("--- Analysis of High-Risk Contexts for Person 5 ---")
print("The top 3 most 'dangerous' contexts for this person were:")
print(dangerous_contexts)


risk_distribution_plot_highlighted <- risk_distribution_plot +
  geom_point(
    data = dangerous_contexts, aes(x = outcomes, y = 0),
    color = "purple", size = 5, shape = 8
  ) +
  labs(caption = "Purple stars indicate the outcomes from the top 3 most dangerous contexts.")

print(risk_distribution_plot_highlighted)

# --- Analysis 2: Calculate Probabilistic Risk ---
high_risk_threshold <- 2.0


prob_exceeding_threshold <- sum(sim_results_df$outcomes > high_risk_threshold) / nrow(sim_results_df)

print(paste0("Estimated probability of outcome > ", high_risk_threshold, ": ", round(prob_exceeding_threshold * 100, 2), "%"))


percentile_95 <- quantile(sim_results_df$outcomes, 0.95)
print(paste("95th percentile outcome (reasonable worst-case):", round(percentile_95, 2)))


person_effects_true <- crossed_data_list$person_effects
low_risk_person_id <- person_effects_true %>%
  filter(b_i == min(b_i)) %>%
  pull(person_id)

print(paste("Now running comparison simulation for 'Low Risk' Person", low_risk_person_id))

# --- Re-run the simulation loop for the low-risk person ---
b_i_low_risk <- min(person_effects_true$b_i)
sim_outcomes_low_risk <- c()
for (context_j_id in all_context_ids) {
  u_j_est <- context_effects_est[context_j_id, "(Intercept)"]
  predicted_y <- fixed_intercept + (fixed_beta_x * x_val) + b_i_low_risk + u_j_est
  random_shock <- rnorm(1, mean = 0, sd = sigma_e_est)
  final_simulated_y <- predicted_y + random_shock
  sim_outcomes_low_risk <- c(sim_outcomes_low_risk, final_simulated_y)
}


sim_results_low_risk_df <- data.frame(outcomes = sim_outcomes_low_risk)
risk_dist_plot_low_risk <- ggplot(sim_results_low_risk_df, aes(x = outcomes)) +
  geom_histogram(aes(y = ..density..), bins = 15, fill = "green", color = "black", alpha = 0.7) +
  geom_density(color = "darkgreen", size = 1.2) +
  labs(
    title = paste("Simulated Risk Exposure Distribution for 'Low Risk' Person", low_risk_person_id),
    x = "Simulated Outcome", y = "Density"
  ) +
  theme_minimal() +
  coord_cartesian(xlim = range(sim_results_df$outcomes))

print(risk_dist_plot_low_risk)


print("--- Side-by-Side Comparison of Risk Profiles ---")

side_by_side_plot <- risk_distribution_plot + risk_dist_plot_low_risk
print(side_by_side_plot)
