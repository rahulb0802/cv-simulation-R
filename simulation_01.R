library(dplyr)
library(lme4)
library(pROC)

generate_synthetic_data <- function(
    num_people=150,
    contexts_range = c(5, 15),
    beta = 0.5,
    G = 0.5,
    vulnerable_prop=0.3,
    nv_mean_var = 0.5,
    v_mean_var = 2.0
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
    
    true_u_i <- rnorm(1, mean = 0, sd = sqrt(G))
    
    person_data_list[[i]] <- list(
      person_id = i,
      is_truly_vulnerable = is_vulnerable,
      true_within_person_variance = true_variance
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
    
    y_ij <- beta * x_ij + epsilon_ij
    
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

simulation_output <- generate_synthetic_data()
synthetic_data <- simulation_output$data_df
true_person_info <- simulation_output$true_person_info

model <- lmer(y ~ x + (1 | person_id), data = synthetic_data)

fixed_effects_pred <- predict(model, re.form = NA)
synthetic_data$marginal_residual <- synthetic_data$y - fixed_effects_pred

tiv_scores <- synthetic_data %>%
  group_by(person_id) %>%
  summarise(TIV_score = var(marginal_residual)) %>%
  na.omit()

evaluation_df <- inner_join(tiv_scores, true_person_info, by = "person_id")

auc_roc <- roc(evaluation_df$is_truly_vulnerable, evaluation_df$TIV_score)
auc_value <- auc(auc_roc)

print(paste("Validation AUC for original TIV:", round(auc_value, 4)))


