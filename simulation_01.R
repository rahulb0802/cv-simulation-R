library(dplyr)
library(lme4)
library(pROC)
library(HLMdiag)
library(ggplot2)

generate_synthetic_data <- function(
    num_people=150,
    contexts_range = c(5, 15),
    beta = 0.5,
    G = 0.5,
    vulnerable_prop=0.3,
    nv_mean_var = 0.5,
    v_mean_var = 2.0,
    person_effect_dist = "normal"
) {
  nv_params <- list(shape = 2.0, scale = nv_mean_var / 2.0)
  v_params <- list(shape = 4.0, scale = v_mean_var / 4.0)
  
  person_data_list <- list()
  
  for (i in 1:num_people) {
    is_vulnerable <- runif(1) < vulnerable_prop
    
    if (is_vulnerable) {
      true_variance <- rgamma(1, shape = v_params$shape, scale = v_params$scale)
      
    } else {
      true_variance <- rgamma(1, shape = nv_params$shape, scale = nv_params$scale)
    }
    if (person_effect_dist == "t") {
      variance_of_t3 <- 3 / (3-2)
      raw_t_value <- rt(1, df = 3)
      true_u_i = (raw_t_value / sqrt(variance_of_t3)) * sqrt(G)
    } else {
      true_u_i <- rnorm(1, mean = 0, sd = sqrt(G))
    }
    
    person_data_list[[i]] <- list(
      person_id = i,
      is_truly_vulnerable = is_vulnerable,
      true_within_person_variance = true_variance,
      true_u_i = true_u_i
    )
  }
  
  true_person_info <- dplyr::bind_rows(person_data_list)
  
  records_list <- list()
  
  for (i in 1:nrow(true_person_info)) {
    person <- true_person_info[i, ]
    
    num_contexts <- sample(contexts_range[1]:contexts_range[2], 1)
    
    x_ij <- rnorm(num_contexts, mean = 0, sd = 1)
    
    person_sigma <- sqrt(person$true_within_person_variance)
    epsilon_ij <- rnorm(num_contexts, mean = 0, sd = person_sigma)
    
    y_ij <- beta * x_ij + person$true_u_i + epsilon_ij
    
    person_records <- data.frame(
      person_id = person$person_id,
      y = y_ij,
      x = x_ij
    )
    records_list[[i]] <- person_records
  }
  
  data_df <- dplyr::bind_rows(records_list)
  
  return(list(
    data_df = data_df,
    true_person_info = true_person_info
  ))
}

set.seed(42)

simulation_output <- generate_synthetic_data(person_effect_dist = 'normal')
synthetic_data <- simulation_output$data_df
true_person_info <- simulation_output$true_person_info

model <- lmer(y ~ x + (1 | person_id), data = synthetic_data, REML = FALSE)

fixed_effects_pred <- predict(model, re.form = NA)
synthetic_data$marginal_residual <- synthetic_data$y - fixed_effects_pred

print("The column names returned by hlm_resid() are:")
print(names(lcr_results))



tiv_scores <- synthetic_data %>%
  group_by(person_id) %>%
  summarise(TIV_score = var(marginal_residual)) %>%
  na.omit()

evaluation_df <- inner_join(tiv_scores, true_person_info, by = "person_id")

auc_roc <- roc(evaluation_df$is_truly_vulnerable, evaluation_df$TIV_score)
auc_value <- auc(auc_roc)

lcr_results <- hlm_resid(model, type = "LS")

lcr_results$person_id <- synthetic_data$person_id

tiv_lcr_scores <- lcr_results %>%
  group_by(person_id) %>%
  summarise(TIV_LCR_score = var(`.ls.resid`)) %>%
  na.omit()

evaluation_lcr_df <- inner_join(tiv_lcr_scores, true_person_info, by = "person_id")

auc_roc_lcr <- roc(evaluation_lcr_df$is_truly_vulnerable, evaluation_lcr_df$TIV_LCR_score)
auc_value_lcr <- auc(auc_roc_lcr)

print(paste("Validation AUC for original TIV:", round(auc_value, 4)))
print(paste("TIV-LCR AUC:", round(auc_value_lcr, 4)))

synthetic_data$conditional_residual <- residuals(model)
full_evaluation_data <- inner_join(synthetic_data, evaluation_df, by = "person_id")

person_of_interest_id <- full_evaluation_data %>%
  filter(is_truly_vulnerable == TRUE) %>%
  filter(TIV_score == max(TIV_score)) %>%
  pull(person_id) %>% # pull() extracts a single column
  unique() # Get just the one ID

print(paste("Generating Vulnerability Profile Plot for Person ID:", person_of_interest_id))

# --- Create the Vulnerability Profile Plot ---
vulnerability_profile_plot <- full_evaluation_data %>%

  filter(person_id == person_of_interest_id) %>%

  mutate(context_visit_number = 1:n()) %>%

  ggplot(aes(x = context_visit_number, y = conditional_residual)) +
  geom_point(color = "red", size = 4, alpha = 0.8) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  labs(
    title = paste("Vulnerability Profile for Person", person_of_interest_id),
    subtitle = "A 'High TIV' individual's residual pattern across contexts",
    x = "Context Visit Number",
    y = "Conditional Residual (Deviation from Personal Average)"
  ) +
  theme_minimal(base_size = 14)


print(vulnerability_profile_plot)




