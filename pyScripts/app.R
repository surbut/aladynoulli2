library(shiny)
library(ggplot2)
library(tidyr)

# Complete Shiny app for Population Structure Analysis
ui <- fluidPage(
  tags$head(
    tags$script(src = "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_HTMLorMML"),
    tags$script("MathJax.Hub.Config({tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}});")
  ),
  
  
  titlePanel("Population Structure Discovery: EM vs Bayesian"),
  
  sidebarLayout(
    sidebarPanel(
      sliderInput("n_inds", "Number of Individuals",
                  min = 50, max = 500, value = 200, step = 50),
      sliderInput("mix_prop", "Mixing Proportion (π)",
                  min = 0.1, max = 0.9, value = 0.5, step = 0.1),
      sliderInput("true_freq1", "True Frequency Population 1",
                  min = 0.1, max = 0.9, value = 0.3, step = 0.1),
      sliderInput("true_freq2", "True Frequency Population 2",
                  min = 0.1, max = 0.9, value = 0.7, step = 0.1),
      sliderInput("alpha", "Prior α",
                  min = 1, max = 20, value = 2, step = 1),
      sliderInput("beta", "Prior β",
                  min = 1, max = 20, value = 2, step = 1),
      actionButton("update", "Run Analysis"),
      width = 3
    ),
    
    mainPanel(
      tabsetPanel(
        tabPanel("Data View", 
                 plotOutput("allele_counts"),
                 verbatimTextOutput("data_summary")),
        
        tabPanel("EM Results", 
                 plotOutput("em_progress"),
                 verbatimTextOutput("em_estimates")),
        
        tabPanel("Bayesian Results", 
                 plotOutput("bayes_progress"),
                 verbatimTextOutput("bayes_estimates")),
        
        tabPanel("Comparison",
                 plotOutput("comparison_plot"),
                 verbatimTextOutput("comparison_text")),
        
        tabPanel("Mathematical Details",
                 uiOutput("math_explanation")),
        
        tabPanel("Prior Properties",
                 fluidRow(
                   column(6, plotOutput("prior_mean_viz")),
                   column(6, plotOutput("prior_variance_viz")),
                   column(12, verbatimTextOutput("prior_properties"))
                 )),
        
        # Add this to the UI tabsetPanel:
        tabPanel("Prior Visualization",
                 fluidRow(
                   column(12,
                          plotOutput("prior_viz"),
                          verbatimTextOutput("prior_interpretation"))
                 )),
        
        # Add to UI tabsetPanel:
        tabPanel("Convergence Demo",
                 fluidRow(
                   column(3,
                          h4("Experiment Settings"),
                          selectInput("demo_scenario", "Choose Scenario:",
                                    choices = c(
                                      "Population Separation" = "separation",
                                      "Sample Size Effect" = "sample_size",
                                      "Prior Strength" = "prior_strength",
                                      "Initial Values" = "init_values",
                                      "Mixing Proportions" = "mixing"
                                    )),
                          uiOutput("scenario_controls"),
                          actionButton("run_demo", "Run Demo")
                   ),
                   column(9,
                          plotOutput("demo_plot"),
                          verbatimTextOutput("demo_summary"))
                 ))
      ),
      width = 9
    )
  )
)




server <- function(input, output, session) {
  # Generate data from two populations
  generate_data <- function(n_inds, freq1, freq2, mix_prop) {
    # Assign populations based on mixing proportion
    pop_assign <- rbinom(n_inds, 1, mix_prop) + 1
    allele_counts <- numeric(n_inds)
    for(i in 1:n_inds) {
      prob <- if(pop_assign[i] == 1) freq1 else freq2
      allele_counts[i] <- rbinom(1, size = 20, prob = prob)
    }
    data.frame(
      Individual = 1:n_inds,
      AlleleCount = allele_counts,
      TruePopulation = factor(pop_assign)
    )
  }
  
  # EM algorithm implementation
  em_algorithm <- function(data, n_iter = 10, mix_prop = 0.5) {
    # Initialize
    freq1 <- runif(1, 0.2, 0.8)
    freq2 <- runif(1, 0.2, 0.8)
    
    history <- data.frame(
      Iteration = 0,
      Pop1_freq = freq1,
      Pop2_freq = freq2,
      Mix_prop = mix_prop
    )
    
    for(iter in 1:n_iter) {
      # E-step
      resp1 <- dbinom(data$AlleleCount, size = 20, prob = freq1) * mix_prop
      resp2 <- dbinom(data$AlleleCount, size = 20, prob = freq2) * (1 - mix_prop)
      
      total_resp <- resp1 + resp2
      resp1 <- resp1 / total_resp
      resp2 <- resp2 / total_resp
      
      # M-step
      freq1 <- sum(resp1 * data$AlleleCount) / (20 * sum(resp1))
      freq2 <- sum(resp2 * data$AlleleCount) / (20 * sum(resp2))
      mix_prop <- mean(resp1)  # Update mixing proportion
      
      history <- rbind(history, 
                      data.frame(Iteration = iter,
                               Pop1_freq = freq1,
                               Pop2_freq = freq2,
                               Mix_prop = mix_prop))
    }
    
    list(history = history,
         final_resp1 = resp1,
         final_resp2 = resp2)
  }
  
  # Bayesian analysis implementation
  bayesian_analysis <- function(data, alpha, beta, mix_prop = 0.5) {
    # Initialize with prior
    freq1 <- rbeta(1, alpha, beta)
    freq2 <- rbeta(1, alpha, beta)
    
    history <- data.frame(
      Iteration = 0,
      Pop1_mean = freq1,
      Pop1_lower = qbeta(0.025, alpha, beta),
      Pop1_upper = qbeta(0.975, alpha, beta),
      Pop2_mean = freq2,
      Pop2_lower = qbeta(0.025, alpha, beta),
      Pop2_upper = qbeta(0.975, alpha, beta),
      Mix_prop = mix_prop
    )
    
    for(iter in 1:10) {
      # E-step with mixing proportions
      resp1 <- dbinom(data$AlleleCount, size = 20, prob = freq1) * mix_prop
      resp2 <- dbinom(data$AlleleCount, size = 20, prob = freq2) * (1 - mix_prop)
      total_resp <- resp1 + resp2
      resp1 <- resp1 / total_resp
      resp2 <- resp2 / total_resp
      
      # Update mixing proportion
      mix_prop <- mean(resp1)
      
      # M-step with Beta posterior
      alpha1 <- alpha + sum(resp1 * data$AlleleCount)
      beta1 <- beta + sum(resp1 * (20 - data$AlleleCount))
      alpha2 <- alpha + sum(resp2 * data$AlleleCount)
      beta2 <- beta + sum(resp2 * (20 - data$AlleleCount))
      
      freq1 <- alpha1 / (alpha1 + beta1)
      freq2 <- alpha2 / (alpha2 + beta2)
      
      history <- rbind(history,
                      data.frame(
                        Iteration = iter,
                        Pop1_mean = freq1,
                        Pop1_lower = qbeta(0.025, alpha1, beta1),
                        Pop1_upper = qbeta(0.975, alpha1, beta1),
                        Pop2_mean = freq2,
                        Pop2_lower = qbeta(0.025, alpha2, beta2),
                        Pop2_upper = qbeta(0.975, alpha2, beta2),
                        Mix_prop = mix_prop
                      ))
    }
    return(history)
  }
  
  # Reactive values for storing results
  results <- reactiveVal(NULL)
  
  # Update results when button is clicked
  observeEvent(input$update, {
    data <- generate_data(input$n_inds, input$true_freq1, input$true_freq2, input$mix_prop)
    em_results <- em_algorithm(data, mix_prop = input$mix_prop)
    bayes_results <- bayesian_analysis(data, input$alpha, input$beta, input$mix_prop)
    
    results(list(
      data = data,
      em_results = em_results,
      bayes_results = bayes_results
    ))
  })
  
  # Data view outputs
  output$allele_counts <- renderPlot({
    req(results())
    data <- results()$data
    
    # Calculate true proportions
    prop1 <- mean(data$TruePopulation == 1)
    
    ggplot(data, aes(x = AlleleCount/20, fill = TruePopulation)) +
      geom_histogram(position = "identity", alpha = 0.5, bins = 20) +
      labs(title = "Distribution of Allele Frequencies by Population",
           subtitle = sprintf("Population 1: %.1f%%, Population 2: %.1f%%", 
                            100*prop1, 100*(1-prop1)),
           x = "Observed Allele Frequency",
           y = "Count") +
      theme_minimal()
  })
  
  output$data_summary <- renderText({
    req(results())
    data <- results()$data
    prop1 <- mean(data$TruePopulation == 1)
    
    paste0("Data Summary:\n",
           "Number of individuals: ", nrow(data), "\n",
           "True frequencies: ", input$true_freq1, " and ", input$true_freq2, "\n",
           "Mixing proportions: ", round(100*prop1, 1), "% and ", 
           round(100*(1-prop1), 1), "%")
  })
  
  
  # Add to server:
  output$prior_mean_viz <- renderPlot({
    x <- seq(0, 1, length.out = 200)
    
    # Compare different means with same strength
    df <- data.frame(
      x = rep(x, 3),
      density = c(
        dbeta(x, 2, 8),   # Mean = 0.2
        dbeta(x, 5, 5),   # Mean = 0.5
        dbeta(x, 8, 2)    # Mean = 0.8
      ),
      Prior = rep(c(
        "Mean = 0.2 (α=2, β=8)",
        "Mean = 0.5 (α=5, β=5)",
        "Mean = 0.8 (α=8, β=2)"
      ), each = 200)
    )
    
    ggplot(df, aes(x = x, y = density, color = Prior)) +
      geom_line(size = 1) +
      labs(title = "Beta Distributions with Different Means",
           subtitle = "Same total strength (α+β=10), different means",
           x = "Allele Frequency",
           y = "Density") +
      theme_minimal()
  })
  
  output$prior_variance_viz <- renderPlot({
    x <- seq(0, 1, length.out = 200)
    
    # Compare different strengths with same mean
    df <- data.frame(
      x = rep(x, 3),
      density = c(
        dbeta(x, 1, 1),    # Weak prior
        dbeta(x, 5, 5),    # Medium prior
        dbeta(x, 20, 20)   # Strong prior
      ),
      Prior = rep(c(
        "Weak (α=β=1)",
        "Medium (α=β=5)",
        "Strong (α=β=20)"
      ), each = 200)
    )
    
    ggplot(df, aes(x = x, y = density, color = Prior)) +
      geom_line(size = 1) +
      labs(title = "Beta Distributions with Different Strengths",
           subtitle = "Same mean (0.5), different strengths",
           x = "Allele Frequency",
           y = "Density") +
      theme_minimal()
  })
  
  output$prior_properties <- renderText({
    # Calculate properties of current prior
    mean <- input$alpha / (input$alpha + input$beta)
    variance <- (input$alpha * input$beta) / 
      ((input$alpha + input$beta)^2 * (input$alpha + input$beta + 1))
    
    paste0(
      "Properties of Beta(", input$alpha, ", ", input$beta, "):\n",
      "\nPrior Mean = α/(α+β) = ", round(mean, 3),
      "\nPrior Variance = αβ/((α+β)²(α+β+1)) = ", round(variance, 3),
      "\nPrior Strength (α+β) = ", input$alpha + input$beta,
      "\n\nKey Relationships:",
      "\n1. Mean moves toward α/(α+β)",
      "\n2. Variance decreases as (α+β) increases",
      "\n3. Shape determined by relative sizes of α and β",
      "\n4. Uniform when α=β=1",
      "\n5. More peaked as α+β increases"
    )
  })
  # EM results outputs
  output$em_progress <- renderPlot({
    req(results())
    history <- results()$em_results$history
    
    history_long <- pivot_longer(history, 
                                 cols = c(Pop1_freq, Pop2_freq),
                                 names_to = "Population",
                                 values_to = "Frequency")
    
    ggplot(history_long, aes(x = Iteration, y = Frequency, color = Population)) +
      geom_line(size = 1) +
      geom_point(size = 3) +
      geom_hline(yintercept = input$true_freq1, linetype = "dashed", color = "red") +
      geom_hline(yintercept = input$true_freq2, linetype = "dashed", color = "blue") +
      labs(title = "EM Algorithm Progress",
           subtitle = "Dashed lines show true frequencies",
           x = "Iteration",
           y = "Estimated Allele Frequency") +
      theme_minimal()
  })
  
  output$em_estimates <- renderText({
    req(results())
    history <- results()$em_results$history
    final <- tail(history, 1)
    
    paste0("Final EM Estimates:\n",
           "Population 1: ", round(final$Pop1_freq, 3),
           " (True: ", input$true_freq1, ")\n",
           "Population 2: ", round(final$Pop2_freq, 3),
           " (True: ", input$true_freq2, ")")
  })
  
  # Bayesian results outputs
  output$bayes_progress <- renderPlot({
    req(results())
    bayes_results <- results()$bayes_results
    
    ggplot(bayes_results, aes(x = Iteration)) +
      geom_ribbon(aes(ymin = Pop1_lower, ymax = Pop1_upper, fill = "Pop1"), alpha = 0.2) +
      geom_ribbon(aes(ymin = Pop2_lower, ymax = Pop2_upper, fill = "Pop2"), alpha = 0.2) +
      geom_line(aes(y = Pop1_mean, color = "Pop1")) +
      geom_line(aes(y = Pop2_mean, color = "Pop2")) +
      geom_hline(yintercept = input$true_freq1, linetype = "dashed", color = "red") +
      geom_hline(yintercept = input$true_freq2, linetype = "dashed", color = "blue") +
      labs(title = "Bayesian Estimation Progress",
           subtitle = "Shaded areas show 95% credible intervals",
           x = "Iteration",
           y = "Allele Frequency") +
      theme_minimal()
  })
  
  
  # Add to server:
  output$iteration_details <- renderTable({
    req(results())
    data <- results()$data
    
    # Get true population assignments for comparison
    true_pop1 <- data$TruePopulation == 1
    
    # Calculate what we'd get with known assignments
    true_alpha1 <- 2 + sum(data$AlleleCount[true_pop1])
    true_beta1 <- 2 + sum(20 - data$AlleleCount[true_pop1])
    true_freq1 <- true_alpha1/(true_alpha1 + true_beta1)
    
    true_alpha2 <- 2 + sum(data$AlleleCount[!true_pop1])
    true_beta2 <- 2 + sum(20 - data$AlleleCount[!true_pop1])
    true_freq2 <- true_alpha2/(true_alpha2 + true_beta2)
    
    # Get EM iterations
    em_results <- results()$em_results
    
    # Create table of iterations
    iter_table <- data.frame(
      Iteration = 0:10,
      Pop1_alpha = NA,
      Pop1_beta = NA,
      Pop1_freq = NA,
      Pop2_alpha = NA,
      Pop2_beta = NA,
      Pop2_freq = NA,
      stringsAsFactors = FALSE
    )
    
    # Fill in values for each iteration
    for(iter in 1:10) {
      # Get responsibilities for this iteration
      resp1 <- em_results$responsibilities[[iter]]
      
      # Calculate posterior parameters
      alpha1 <- 2 + sum(resp1 * data$AlleleCount)
      beta1 <- 2 + sum(resp1 * (20 - data$AlleleCount))
      freq1 <- alpha1/(alpha1 + beta1)
      
      alpha2 <- 2 + sum((1-resp1) * data$AlleleCount)
      beta2 <- 2 + sum((1-resp1) * (20 - data$AlleleCount))
      freq2 <- alpha2/(alpha2 + beta2)
      
      iter_table[iter+1,] <- c(iter, alpha1, beta1, freq1, alpha2, beta2, freq2)
    }
    
    # Add true values as last row
    iter_table <- rbind(
      iter_table,
      c("True", true_alpha1, true_beta1, true_freq1, 
        true_alpha2, true_beta2, true_freq2)
    )
    
    iter_table
  }, digits = 3)
  
  output$bayes_estimates <- renderText({
    req(results())
    bayes_results <- results()$bayes_results
    final <- tail(bayes_results, 1)
    
    paste0("Final Bayesian Estimates:\n",
           "Population 1: ", round(final$Pop1_mean, 3),
           " (", round(final$Pop1_lower, 3), " - ", round(final$Pop1_upper, 3), ")\n",
           "Population 2: ", round(final$Pop2_mean, 3),
           " (", round(final$Pop2_lower, 3), " - ", round(final$Pop2_upper, 3), ")")
  })
  
  # Comparison outputs
  output$comparison_plot <- renderPlot({
    req(results())
    em_results <- results()$em_results$history
    bayes_results <- results()$bayes_results
    
    ggplot() +
      # EM results
      geom_line(data = em_results, 
                aes(x = Iteration, y = Pop1_freq, color = "EM Pop1")) +
      geom_line(data = em_results, 
                aes(x = Iteration, y = Pop2_freq, color = "EM Pop2")) +
      # Bayesian results
      geom_ribbon(data = bayes_results,
                  aes(x = Iteration, 
                      ymin = Pop1_lower, 
                      ymax = Pop1_upper),
                  alpha = 0.2) +
      geom_ribbon(data = bayes_results,
                  aes(x = Iteration, 
                      ymin = Pop2_lower, 
                      ymax = Pop2_upper),
                  alpha = 0.2) +
      geom_line(data = bayes_results, 
                aes(x = Iteration, y = Pop1_mean, color = "Bayes Pop1")) +
      geom_line(data = bayes_results, 
                aes(x = Iteration, y = Pop2_mean, color = "Bayes Pop2")) +
      geom_hline(yintercept = input$true_freq1, linetype = "dashed") +
      geom_hline(yintercept = input$true_freq2, linetype = "dashed") +
      labs(title = "EM vs Bayesian Estimation",
           subtitle = "Shaded areas show 95% credible intervals",
           x = "Iteration", 
           y = "Allele Frequency") +
      theme_minimal()
  })
  
  # Add visualization of how responsibilities change
  output$responsibility_evolution <- renderPlot({
    req(results())
    data <- results()$data
    em_results <- results()$em_results
    
    # Create data frame of responsibilities across iterations
    resp_df <- data.frame(
      Individual = rep(1:nrow(data), times = 10),
      Iteration = rep(1:10, each = nrow(data)),
      Responsibility = unlist(lapply(em_results$responsibilities, function(x) x[,1])),
      AlleleCount = rep(data$AlleleCount/20, times = 10)
    )
    
    ggplot(resp_df, aes(x = AlleleCount, y = Responsibility, color = factor(Iteration))) +
      geom_point(alpha = 0.5) +
      geom_smooth(method = "loess", se = FALSE) +
      labs(title = "Evolution of Population Assignments",
           x = "Observed Allele Frequency",
           y = "Responsibility (γ) for Population 1",
           color = "Iteration") +
      theme_minimal()
  })
  
  # Add summary text
  output$convergence_summary <- renderText({
    req(results())
    data <- results()$data
    em_results <- results()$em_results
    
    # Get final and true assignments
    final_resp <- em_results$responsibilities[[10]]
    true_pop1 <- data$TruePopulation == 1
    
    # Calculate accuracy
    final_assign <- final_resp > 0.5
    accuracy <- mean(final_assign == true_pop1)
    
    paste0(
      "Convergence Summary:\n",
      "\nInitial state: Uncertain assignments (γ ≈ 0.5)",
      "\nFinal state: ", round(mean(abs(final_resp - 0.5) > 0.4) * 100, 1), 
      "% of individuals strongly assigned (|γ - 0.5| > 0.4)",
      "\nAccuracy: ", round(accuracy * 100, 1), "% match with true populations",
      "\n\nCompare with known assignments:",
      "\nEM final α₁/β₁: ", round(tail(em_results$alpha1, 1), 1), "/", 
      round(tail(em_results$beta1, 1), 1),
      "\nTrue α₁/β₁: ", round(2 + sum(data$AlleleCount[true_pop1]), 1), "/",
      round(2 + sum(20 - data$AlleleCount[true_pop1]), 1)
    )
  })
  
  output$comparison_text <- renderText({
    req(results())
    em_results <- results()$em_results$history
    bayes_results <- results()$bayes_results
    
    em_final <- tail(em_results, 1)
    bayes_final <- tail(bayes_results, 1)
    
    paste0(
      "Final Estimates Comparison:\n",
      "EM Pop1: ", round(em_final$Pop1_freq, 3), "\n",
      "Bayes Pop1: ", round(bayes_final$Pop1_mean, 3), 
      " (", round(bayes_final$Pop1_lower, 3), 
      " - ", round(bayes_final$Pop1_upper, 3), ")\n",
      "EM Pop2: ", round(em_final$Pop2_freq, 3), "\n",
      "Bayes Pop2: ", round(bayes_final$Pop2_mean, 3),
      " (", round(bayes_final$Pop2_lower, 3), 
      " - ", round(bayes_final$Pop2_upper, 3), ")"
    )
  })
  
  
  # Add these to the server:
  output$prior_viz <- renderPlot({
    # Generate Beta distribution curves
    x <- seq(0, 1, length.out = 200)
    
    # Create data frame for different Beta distributions
    df <- data.frame(
      x = rep(x, 4),
      density = c(
        dbeta(x, input$alpha, input$beta),
        dbeta(x, 1, 1),  # Uniform (no prior knowledge)
        dbeta(x, 10, 10),  # Strong belief in 0.5
        dbeta(x, 2, 8)   # Belief skewed towards lower values
      ),
      Prior = rep(c(
        "Current Prior",
        "Uniform (No Prior)",
        "Strong Central",
        "Skewed"
      ), each = 200)
    )
    
    ggplot(df, aes(x = x, y = density, color = Prior)) +
      geom_line(size = 1) +
      labs(title = "Beta Distribution Prior Shapes",
           x = "Allele Frequency",
           y = "Density",
           subtitle = sprintf("Current: α = %d, β = %d", input$alpha, input$beta)) +
      theme_minimal() +
      theme(legend.position = "bottom")
  })
  
  output$prior_interpretation <- renderText({
    total_strength <- input$alpha + input$beta
    mean_belief <- input$alpha / total_strength
    
    paste0(
      "Prior Interpretation:\n",
      "\nStrength of Prior (α + β): ", total_strength,
      "\n- Small values (<5): Weak prior, data dominates",
      "\n- Large values (>20): Strong prior, resistant to change",
      "\n\nMean of Prior (α/(α+β)): ", round(mean_belief, 3),
      "\n- Close to 0.5: Centered belief",
      "\n- <0.5: Skewed towards lower frequencies",
      "\n- >0.5: Skewed towards higher frequencies",
      "\n\nIn Population Genetics Terms:",
      "\n- α: 'Pseudo-counts' of reference allele",
      "\n- β: 'Pseudo-counts' of alternate allele",
      "\n- Total (α+β): Sample size of prior knowledge"
    )
  })
  
  # Mathematical explanation
  output$math_explanation <- renderUI({
    withMathJax(
      HTML(paste0(
        "<h3>EM Algorithm Mathematical Details</h3>",
        "<h4>Model:</h4>",
        "<p>For each individual i:</p>",
        "\\[ X_i \\sim \\text{Binomial}(n, \\theta_k) \\]",
        "<p>where k ∈ {1,2} is the population and θ<sub>k</sub> is the allele frequency</p>",
        
        "<h4>E-step (Expectation):</h4>",
        "<p>Calculate responsibility (γ<sub>ik</sub>) - probability individual i belongs to population k:</p>",
        "\\[ \\gamma_{ik} = \\frac{\\pi_k\\text{Binomial}(x_i|n, \\theta_k)}{\\sum_j \\pi_j\\text{Binomial}(x_i|n, \\theta_j)} \\]",
        "<p>where π<sub>k</sub> is the mixing proportion for population k</p>",
        
        "<h4>M-step (Maximization):</h4>",
        "<p>Update allele frequencies and mixing proportions:</p>",
        "\\[ \\theta_k^{\\text{new}} = \\frac{\\sum_{i=1}^N \\gamma_{ik}x_i}{n\\sum_{i=1}^N \\gamma_{ik}} \\]",
        "\\[ \\pi_k^{\\text{new}} = \\frac{1}{N}\\sum_{i=1}^N \\gamma_{ik} \\]",
        
        "<h3>Bayesian Population Structure</h3>",
        "<h4>Model with Priors:</h4>",
        "<p>For each population k:</p>",
        "\\[ \\theta_k \\sim \\text{Beta}(\\alpha_k, \\beta_k) \\]",
        "\\[ \\pi \\sim \\text{Beta}(a, b) \\text{ or } \\text{Dirichlet}(\\alpha) \\text{ for K>2} \\]",
        "\\[ X_i|\\theta_k \\sim \\text{Binomial}(n, \\theta_k) \\]",
        
        "<h4>Posterior Updates:</h4>",
        "<p>Due to conjugacy, we get:</p>",
        "\\[ \\theta_k|X \\sim \\text{Beta}(\\alpha_k + \\sum_i \\gamma_{ik}x_i, \\beta_k + \\sum_i \\gamma_{ik}(n-x_i)) \\]",
        "\\[ \\pi|X \\sim \\text{Beta}(a + \\sum_i \\gamma_{i1}, b + \\sum_i \\gamma_{i2}) \\]",
        
        "<h4>Convergence Factors:</h4>",
        "<ul>",
        "<li><strong>Separation of populations:</strong> More distinct θ<sub>k</sub> values converge faster</li>",
        "<li><strong>Sample size:</strong> More data generally leads to faster convergence</li>",
        "<li><strong>Prior strength:</strong> Strong informative priors can speed up convergence</li>",
        "<li><strong>Initial values:</strong> Starting closer to true values speeds convergence</li>",
        "<li><strong>Mixing proportions:</strong> Very uneven proportions can slow convergence</li>",
        "</ul>",
        
        "<h4>Key Differences:</h4>",
        "<ul>",
        "<li>EM provides point estimates</li>",
        "<li>Bayesian approach provides full posterior distributions</li>",
        "<li>Bayesian approach incorporates prior knowledge</li>",
        "<li>Uncertainty quantification is natural in Bayesian framework</li>",
        "</ul>"
      ))
    )
  })
  
  # Dynamic UI for scenario controls
  output$scenario_controls <- renderUI({
    switch(input$demo_scenario,
           "separation" = tagList(
             sliderInput("sep_diff", "Frequency Difference",
                        min = 0.1, max = 0.8, value = 0.4, step = 0.1)
           ),
           "sample_size" = tagList(
             sliderInput("demo_n", "Number of Individuals",
                        min = 50, max = 1000, value = c(100, 500), step = 50)
           ),
           "prior_strength" = tagList(
             sliderInput("demo_alpha", "Prior Strength (α+β)",
                        min = 2, max = 40, value = c(2, 20), step = 2)
           ),
           "init_values" = tagList(
             sliderInput("init_dist", "Distance from True Values",
                        min = 0.1, max = 0.5, value = 0.3, step = 0.1)
           ),
           "mixing" = tagList(
             sliderInput("demo_mix", "Population 1 Proportion",
                        min = 0.1, max = 0.9, value = c(0.2, 0.5), step = 0.1)
           )
    )
  })
  
  # Run convergence demo
  demo_results <- reactiveVal(NULL)
  
  observeEvent(input$run_demo, {
    # Get parameters based on scenario
    params <- switch(input$demo_scenario,
      "separation" = list(
        close = list(
          freq1 = 0.6,
          freq2 = 0.4,
          n_inds = 1000,
          mix_prop = 0.5
        ),
        far = list(
          freq1 = 0.9,
          freq2 = 0.1,
          n_inds = 1000,
          mix_prop = 0.5
        )
      ),
      "sample_size" = list(
        small = list(
          freq1 = 0.8,
          freq2 = 0.2,
          n_inds = input$demo_n[1],
          mix_prop = 0.5
        ),
        large = list(
          freq1 = 0.8,
          freq2 = 0.2,
          n_inds = input$demo_n[2],
          mix_prop = 0.5
        )
      ),
      "prior_strength" = list(
        weak = list(
          freq1 = 0.8,
          freq2 = 0.2,
          n_inds = 1000,
          mix_prop = 0.5,
          alpha = input$demo_alpha[1],
          beta = input$demo_alpha[1]
        ),
        strong = list(
          freq1 = 0.8,
          freq2 = 0.2,
          n_inds = 1000,
          mix_prop = 0.5,
          alpha = input$demo_alpha[2],
          beta = input$demo_alpha[2]
        )
      ),
      "init_values" = list(
        good = list(
          freq1 = 0.8,
          freq2 = 0.2,
          n_inds = 1000,
          mix_prop = 0.5,
          init_freq1 = 0.7,
          init_freq2 = 0.3
        ),
        poor = list(
          freq1 = 0.8,
          freq2 = 0.2,
          n_inds = 1000,
          mix_prop = 0.5,
          init_freq1 = 0.3,
          init_freq2 = 0.7
        )
      ),
      "mixing" = list(
        balanced = list(
          freq1 = 0.8,
          freq2 = 0.2,
          n_inds = 1000,
          mix_prop = input$demo_mix[2]
        ),
        unbalanced = list(
          freq1 = 0.8,
          freq2 = 0.2,
          n_inds = 1000,
          mix_prop = input$demo_mix[1]
        )
      )
    )
    
    # Run EM for both cases
    results <- list()
    metrics <- list()
    
    for(case in names(params)) {
      # Generate data
      data <- generate_data(
        n_inds = params[[case]]$n_inds,
        freq1 = params[[case]]$freq1,
        freq2 = params[[case]]$freq2,
        mix_prop = params[[case]]$mix_prop
      )
      
      # Run EM with history tracking
      em_results <- run_em_with_history(
        data = data,
        init_freq1 = if(!is.null(params[[case]]$init_freq1)) params[[case]]$init_freq1 else 0.5,
        init_freq2 = if(!is.null(params[[case]]$init_freq2)) params[[case]]$init_freq2 else 0.5,
        alpha = if(!is.null(params[[case]]$alpha)) params[[case]]$alpha else 1,
        beta = if(!is.null(params[[case]]$beta)) params[[case]]$beta else 1,
        max_iter = 100,
        tol = 1e-6
      )
      
      results[[case]] <- em_results
      
      # Calculate metrics
      metrics[[paste0("case", if(case %in% c("close", "small", "weak", "poor", "unbalanced")) 1 else 2)]] <- list(
        n_until_stable = which(diff(em_results$history$Pop1_freq) < 1e-6)[1],
        final_change = abs(tail(diff(em_results$history$Pop1_freq), 1))
      )
    }
    
    demo_results(list(
      results = results,
      metrics = metrics
    ))
  })
  
  # Helper function to run EM with history
  run_em_with_history <- function(data, init_freq1, init_freq2, alpha, beta, max_iter, tol) {
    freq1 <- init_freq1
    freq2 <- init_freq2
    history <- data.frame(
      Iteration = 0,
      Pop1_freq = freq1,
      Pop2_freq = freq2,
      Mix_prop = 0.5
    )
    
    for(iter in 1:max_iter) {
      # E-step: Calculate responsibilities
      resp1 <- dbinom(data$AlleleCount, size = 20, prob = freq1)
      resp2 <- dbinom(data$AlleleCount, size = 20, prob = freq2)
      gamma <- resp1 / (resp1 + resp2)
      
      # M-step: Update frequencies
      freq1_new <- (sum(gamma * data$AlleleCount) + alpha - 1) / (sum(gamma * 20) + alpha + beta - 2)
      freq2_new <- (sum((1-gamma) * data$AlleleCount) + alpha - 1) / (sum((1-gamma) * 20) + alpha + beta - 2)
      mix_prop <- mean(gamma)
      
      # Store history
      history <- rbind(history, data.frame(
        Iteration = iter,
        Pop1_freq = freq1_new,
        Pop2_freq = freq2_new,
        Mix_prop = mix_prop
      ))
      
      # Check convergence
      if(max(abs(freq1_new - freq1), abs(freq2_new - freq2)) < tol) break
      
      freq1 <- freq1_new
      freq2 <- freq2_new
    }
    
    list(
      freq1 = freq1,
      freq2 = freq2,
      history = history
    )
  }
  
  # Plot demo results
  output$demo_plot <- renderPlot({
    req(demo_results())
    
    results <- demo_results()$results
    scenario <- input$demo_scenario
    
    # Combine histories into one data frame
    plot_data <- do.call(rbind, lapply(names(results), function(case) {
      history <- results[[case]]$history
      history$Case <- case
      history
    }))
    
    # Create base plot
    p <- ggplot(plot_data, aes(x = Iteration)) +
      theme_minimal() +
      labs(title = switch(scenario,
        "separation" = "Effect of Population Separation on Convergence",
        "sample_size" = "Effect of Sample Size on Convergence",
        "prior_strength" = "Effect of Prior Strength on Convergence",
        "init_values" = "Effect of Initial Values on Convergence",
        "mixing" = "Effect of Population Mixing Proportions on Convergence"
      ))
    
    # Add appropriate geoms based on scenario
    p <- p + switch(scenario,
      "separation" = {
        list(
          geom_line(aes(y = Pop1_freq, color = Case, linetype = "Pop 1")),
          geom_line(aes(y = Pop2_freq, color = Case, linetype = "Pop 2")),
          scale_color_manual(values = c("close" = "#E69F00", "far" = "#56B4E9"),
                           labels = c("close" = "Close Frequencies", "far" = "Separated Frequencies")),
          scale_linetype_manual(values = c("Pop 1" = "solid", "Pop 2" = "dashed")),
          labs(y = "Allele Frequency", color = "Scenario", linetype = "Population")
        )
      },
      "sample_size" = {
        list(
          geom_line(aes(y = Pop1_freq, color = Case, linetype = "Pop 1")),
          geom_line(aes(y = Pop2_freq, color = Case, linetype = "Pop 2")),
          scale_color_manual(values = c("small" = "#E69F00", "large" = "#56B4E9"),
                           labels = c("small" = paste("N =", input$demo_n[1]), 
                                    "large" = paste("N =", input$demo_n[2]))),
          scale_linetype_manual(values = c("Pop 1" = "solid", "Pop 2" = "dashed")),
          labs(y = "Allele Frequency", color = "Sample Size", linetype = "Population")
        )
      },
      "prior_strength" = {
        list(
          geom_ribbon(aes(ymin = Pop1_freq - 1/sqrt(input$demo_alpha[1]), 
                         ymax = Pop1_freq + 1/sqrt(input$demo_alpha[1]),
                         fill = "weak", group = "weak"), alpha = 0.2),
          geom_ribbon(aes(ymin = Pop1_freq - 1/sqrt(input$demo_alpha[2]), 
                         ymax = Pop1_freq + 1/sqrt(input$demo_alpha[2]),
                         fill = "strong", group = "strong"), alpha = 0.2),
          geom_line(aes(y = Pop1_freq, color = Case)),
          scale_color_manual(values = c("weak" = "#E69F00", "strong" = "#56B4E9"),
                           labels = c("weak" = paste("α =", input$demo_alpha[1]), 
                                    "strong" = paste("α =", input$demo_alpha[2]))),
          scale_fill_manual(values = c("weak" = "#E69F00", "strong" = "#56B4E9")),
          labs(y = "Allele Frequency", color = "Prior Strength", fill = "Prior Strength")
        )
      },
      "init_values" = {
        list(
          geom_line(aes(y = Pop1_freq, color = Case, linetype = "Pop 1")),
          geom_line(aes(y = Pop2_freq, color = Case, linetype = "Pop 2")),
          scale_color_manual(values = c("good" = "#E69F00", "poor" = "#56B4E9"),
                           labels = c("good" = "Good Initial Values", 
                                    "poor" = "Poor Initial Values")),
          scale_linetype_manual(values = c("Pop 1" = "solid", "Pop 2" = "dashed")),
          labs(y = "Allele Frequency", color = "Initialization", linetype = "Population")
        )
      },
      "mixing" = {
        list(
          geom_line(aes(y = Mix_prop, color = Case)),
          scale_color_manual(values = c("balanced" = "#E69F00", "unbalanced" = "#56B4E9"),
                           labels = c("balanced" = paste("Mix Prop =", input$demo_mix[2]), 
                                    "unbalanced" = paste("Mix Prop =", input$demo_mix[1]))),
          labs(y = "Mixing Proportion", color = "Scenario")
        )
      }
    )
    
    print(p)
  })
  
  # Generate summary text
  output$demo_summary <- renderText({
    req(demo_results())
    
    metrics <- demo_results()$metrics
    scenario <- input$demo_scenario
    
    # Create summary based on scenario
    summary_text <- switch(scenario,
      "separation" = sprintf(
        "With close frequencies, convergence took %d iterations (final change: %.4f).\nWith separated frequencies, convergence took %d iterations (final change: %.4f).",
        metrics$case1$n_until_stable, metrics$case1$final_change,
        metrics$case2$n_until_stable, metrics$case2$final_change
      ),
      "sample_size" = sprintf(
        "With N=%d, convergence took %d iterations (final change: %.4f).\nWith N=%d, convergence took %d iterations (final change: %.4f).",
        input$demo_n[1], metrics$case1$n_until_stable, metrics$case1$final_change,
        input$demo_n[2], metrics$case2$n_until_stable, metrics$case2$final_change
      ),
      "prior_strength" = sprintf(
        "With weak prior (α=%.1f), convergence took %d iterations (final change: %.4f).\nWith strong prior (α=%.1f), convergence took %d iterations (final change: %.4f).",
        input$demo_alpha[1], metrics$case1$n_until_stable, metrics$case1$final_change,
        input$demo_alpha[2], metrics$case2$n_until_stable, metrics$case2$final_change
      ),
      "init_values" = sprintf(
        "With good initialization, convergence took %d iterations (final change: %.4f).\nWith poor initialization, convergence took %d iterations (final change: %.4f).",
        metrics$case1$n_until_stable, metrics$case1$final_change,
        metrics$case2$n_until_stable, metrics$case2$final_change
      ),
      "mixing" = sprintf(
        "With balanced mixing (%.2f), convergence took %d iterations (final change: %.4f).\nWith unbalanced mixing (%.2f), convergence took %d iterations (final change: %.4f).",
        input$demo_mix[2], metrics$case1$n_until_stable, metrics$case1$final_change,
        input$demo_mix[1], metrics$case2$n_until_stable, metrics$case2$final_change
      )
    )
    
    paste("Convergence Summary:\n", summary_text)
  })
}

# Run the application
shinyApp(ui = ui, server = server)