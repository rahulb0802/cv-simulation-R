library(dplyr)
library(lme4)
library(ggplot2)

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

print(risk_distribution_plot)

# --- Analysis 2: Calculate Probabilistic Risk ---
high_risk_threshold <- 2.0


prob_exceeding_threshold <- sum(sim_results_df$outcomes > high_risk_threshold) / nrow(sim_results_df)

print(paste0("Estimated probability of outcome > ", high_risk_threshold, ": ", round(prob_exceeding_threshold * 100, 2), "%"))


percentile_95 <- quantile(sim_results_df$outcomes, 0.95)
print(paste("95th percentile outcome (reasonable worst-case):", round(percentile_95, 2)))
