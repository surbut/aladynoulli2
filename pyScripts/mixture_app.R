library(shiny)
library(ggplot2)
library(dplyr)
library(gridExtra)

# Define UI
ui <- fluidPage(
  titlePanel("Understanding Mixture Distributions through Population Structure"),
  sidebarLayout(
    sidebarPanel(
      sliderInput("n_pops", "Number of Populations",
                  min = 2, max = 4, value = 2, step = 1),
      sliderInput("n_inds", "Number of Individuals",
                  min = 100, max = 500, value = 200, step = 50),
      sliderInput("mix_prop", "Mixing Proportion",
                  min = 0.1, max = 0.9, value = 0.5, step = 0.1),
      actionButton("update", "Update Visualization")
    ),
    mainPanel(
      tabsetPanel(
        tabPanel("Data Generation",
                 plotOutput("true_structure"),
                 plotOutput("observed_data")),
        tabPanel("EM Algorithm",
                 plotOutput("em_progress"),
                 htmlOutput("em_explanation")),
        tabPanel("Mathematical Details",
                 withMathJax(htmlOutput("math_details")))
      )
    )
  )
)

# Define server logic
server <- function(input, output) {
  
  # Generate data
  generate_data <- function(n_pops, n_inds, mix_prop) {
    # Create population means
    means <- seq(-2, 2, length.out = n_pops)
    
    # Create mixing proportions
    props <- rep(mix_prop, n_pops)
    props <- props / sum(props)
    
    # Assign individuals to populations
    pop_assign <- sample(1:n_pops, n_inds, replace = TRUE, prob = props)
    
    # Generate data
    data <- data.frame(
      Individual = 1:n_inds,
      Population = factor(pop_assign),
      Value = rnorm(n_inds, mean = means[pop_assign], sd = 0.5)
    )
    
    return(data)
  }
  
  # EM algorithm visualization
  em_algorithm <- function(data, n_pops) {
    # Initialize parameters
    means <- seq(min(data$Value), max(data$Value), length.out = n_pops)
    sds <- rep(0.5, n_pops)
    props <- rep(1/n_pops, n_pops)
    
    # Store history for visualization
    history <- list()
    
    # EM iterations
    for (iter in 1:10) {
      # E-step: Calculate responsibilities
      responsibilities <- matrix(0, nrow = nrow(data), ncol = n_pops)
      for (k in 1:n_pops) {
        responsibilities[,k] <- props[k] * dnorm(data$Value, means[k], sds[k])
      }
      responsibilities <- responsibilities / rowSums(responsibilities)
      
      # M-step: Update parameters
      for (k in 1:n_pops) {
        props[k] <- mean(responsibilities[,k])
        means[k] <- sum(responsibilities[,k] * data$Value) / sum(responsibilities[,k])
        sds[k] <- sqrt(sum(responsibilities[,k] * (data$Value - means[k])^2) / sum(responsibilities[,k]))
      }
      
      # Store current state
      history[[iter]] <- list(
        means = means,
        sds = sds,
        props = props,
        responsibilities = responsibilities
      )
    }
    
    return(history)
  }
  
  # Create true structure plot
  output$true_structure <- renderPlot({
    data <- generate_data(input$n_pops, input$n_inds, input$mix_prop)
    
    ggplot(data, aes(x = Value, fill = Population)) +
      geom_histogram(bins = 30, alpha = 0.7) +
      scale_fill_brewer(palette = "Set1") +
      labs(title = "True Population Structure",
           subtitle = "Each color represents a different population",
           x = "Value", y = "Count") +
      theme_minimal()
  })
  
  # Create observed data plot
  output$observed_data <- renderPlot({
    data <- generate_data(input$n_pops, input$n_inds, input$mix_prop)
    
    ggplot(data, aes(x = Value)) +
      geom_histogram(bins = 30, fill = "gray70") +
      labs(title = "Observed Data",
           subtitle = "Population labels are hidden - can you identify the populations?",
           x = "Value", y = "Count") +
      theme_minimal()
  })
  
  # Create EM progress plot
  output$em_progress <- renderPlot({
    data <- generate_data(input$n_pops, input$n_inds, input$mix_prop)
    history <- em_algorithm(data, input$n_pops)
    
    # Create sequence for plotting densities
    x_seq <- seq(min(data$Value) - 1, max(data$Value) + 1, length.out = 200)
    
    # Create data frame for plotting
    plot_data <- data.frame()
    for (iter in 1:length(history)) {
      state <- history[[iter]]
      for (k in 1:input$n_pops) {
        density <- state$props[k] * dnorm(x_seq, state$means[k], state$sds[k])
        plot_data <- rbind(plot_data, data.frame(
          x = x_seq,
          y = density,
          Component = factor(k),
          Iteration = iter
        ))
      }
    }
    
    ggplot(plot_data, aes(x = x, y = y, color = Component)) +
      geom_line() +
      facet_wrap(~ Iteration, ncol = 2) +
      scale_color_brewer(palette = "Set1") +
      labs(title = "EM Algorithm Progress",
           subtitle = "Each panel shows one iteration of the EM algorithm",
           x = "Value", y = "Density") +
      theme_minimal()
  })
  
  # Create EM explanation
  output$em_explanation <- renderUI({
    HTML(paste0(
      "<h3>EM Algorithm Steps:</h3>",
      "<ol>",
      "<li><strong>Initialization:</strong> Start with random guesses for component parameters</li>",
      "<li><strong>E-step (Expectation):</strong> Calculate how likely each data point belongs to each component</li>",
      "<li><strong>M-step (Maximization):</strong> Update component parameters based on these probabilities</li>",
      "<li><strong>Repeat:</strong> Until parameters stop changing significantly</li>",
      "</ol>",
      "<p>Watch how the algorithm improves its estimates with each iteration!</p>"
    ))
  })
  
  # Create mathematical details
  output$math_details <- renderUI({
    HTML(paste0(
      "<h3>Mathematical Formulation:</h3>",
      "<p>A mixture model combines multiple distributions:</p>",
      "$$p(x) = \\sum_{k=1}^K \\pi_k f_k(x|\\theta_k)$$",
      "<p>Where:</p>",
      "<ul>",
      "<li>\\(K\\) is the number of components (populations)</li>",
      "<li>\\(\\pi_k\\) are the mixing weights (population proportions)</li>",
      "<li>\\(f_k(x|\\theta_k)\\) are the component densities</li>",
      "</ul>",
      "<h4>EM Algorithm:</h4>",
      "<p><strong>E-step:</strong> Calculate responsibilities</p>",
      "$$\\gamma_{ik} = \\frac{\\pi_k f_k(x_i|\\theta_k)}{\\sum_{j=1}^K \\pi_j f_j(x_i|\\theta_j)}$$",
      "<p><strong>M-step:</strong> Update parameters</p>",
      "$$\\pi_k^{new} = \\frac{1}{n}\\sum_{i=1}^n \\gamma_{ik}$$",
      "$$\\mu_k^{new} = \\frac{\\sum_{i=1}^n \\gamma_{ik}x_i}{\\sum_{i=1}^n \\gamma_{ik}}$$",
      "$$(\\sigma_k^{new})^2 = \\frac{\\sum_{i=1}^n \\gamma_{ik}(x_i-\\mu_k^{new})^2}{\\sum_{i=1}^n \\gamma_{ik}}$$"
    ))
  })
}

# Run the application
shinyApp(ui = ui, server = server) 