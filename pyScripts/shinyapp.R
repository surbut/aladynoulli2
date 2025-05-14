library(shiny)
library(ggplot2)
library(flexdashboard)
library(dplyr)

# Generalized Gamma PDF function
dgg <- function(t, beta, sigma, lambda) {
  # Calculate components
  abs_lambda <- abs(lambda)
  lambda_sq_inv <- 1/(lambda^2)
  exp_beta_t <- (exp(-beta) * t)
  
  # Calculate numerator and denominator parts
  num1 <- abs_lambda * (lambda_sq_inv)^(lambda_sq_inv)
  denom1 <- sigma * t * gamma(lambda_sq_inv)
  exp_part1 <- (exp_beta_t)^(1/(sigma*lambda))
  exp_part2 <- exp(-lambda_sq_inv * (exp_beta_t)^(lambda/sigma))
  
  # Return PDF value
  return((num1/denom1) * exp_part1 * exp_part2)
}

# Generalized Gamma survival function
sgg <- function(t, beta, sigma, lambda) {
  # For simplicity, calculate 1 - CDF through integration of PDF
  # (This is an approximation for illustration purposes)
  result <- numeric(length(t))
  for (i in 1:length(t)) {
    if (t[i] <= 0) {
      result[i] <- 1
    } else {
      # Numerical integration from t to a large value
      result[i] <- integrate(function(x) dgg(x, beta, sigma, lambda), t[i], 100)$value
      # Ensure result is between 0 and 1
      result[i] <- max(0, min(1, result[i]))
    }
  }
  return(result)
}

# Hazard function
hgg <- function(t, beta, sigma, lambda) {
  dgg(t, beta, sigma, lambda) / sgg(t, beta, sigma, lambda)
}

# UI
ui <- fluidPage(
  titlePanel("Generalized Gamma Distribution Explorer"),
  
  sidebarLayout(
    sidebarPanel(
      h3("Distribution Parameters"),
      
      # β parameter (location)
      sliderInput("beta", 
                  "β (beta) - Location Parameter:", 
                  min = 0.1, 
                  max = 5.0, 
                  value = 2.0,
                  step = 0.1),
      helpText("Controls the central tendency/median of survival time."),
      helpText("Higher β = longer expected survival."),
      
      # σ parameter (scale)
      sliderInput("sigma", 
                  "σ (sigma) - Scale Parameter:", 
                  min = 0.1, 
                  max = 3.0, 
                  value = 1.0,
                  step = 0.1),
      helpText("Controls the spread/dispersion of survival times."),
      helpText("Higher σ = more variability, gradual curve."),
      
      # λ parameter (shape)
      sliderInput("lambda", 
                  "λ (lambda) - Shape Parameter:", 
                  min = -1.0, 
                  max = 1.0, 
                  value = 0.5,
                  step = 0.1),
      helpText("Controls the form of the hazard function."),
      helpText("λ < 0: Decreasing hazard"),
      helpText("λ ≈ 0: Arc-shaped hazard"),
      helpText("λ > 0: Increasing hazard"),
      
      # Special cases
      selectInput("preset", "Special Cases:", 
                  choices = c("Custom" = "custom",
                              "Weibull (λ = 1)" = "weibull",
                              "Gamma (σ = λ)" = "gamma",
                              "Log-Normal (λ = 0)" = "lognormal",
                              "Exponential (σ = λ = 1)" = "exponential")),
      
      hr(),
      
      # Clinical examples
      selectInput("example", "Clinical Examples:", 
                  choices = c("None" = "none",
                              "Post-surgery recovery" = "surgery",
                              "Age-related disease" = "age",
                              "Treatment with initial risk" = "treatment",
                              "Diabetes case study" = "diabetes"))
    ),
    
    mainPanel(
      tabsetPanel(
        tabPanel("Survival Function", 
                 plotOutput("survivalPlot", height = "400px"),
                 hr(),
                 helpText("The survival function shows the probability of surviving beyond time t.")),
        
        tabPanel("Hazard Function", 
                 plotOutput("hazardPlot", height = "400px"),
                 hr(),
                 helpText("The hazard function shows the instantaneous risk of the event at time t.")),
        
        tabPanel("Probability Density", 
                 plotOutput("densityPlot", height = "400px"),
                 hr(),
                 helpText("The probability density function shows the relative likelihood of the event at time t.")),
        
        tabPanel("Information", 
                 h3("About the Generalized Gamma Distribution"),
                 p("The Generalized Gamma distribution is a flexible parametric model that includes many common survival distributions as special cases:"),
                 tags$ul(
                   tags$li("Weibull distribution: λ = 1"),
                   tags$li("Gamma distribution: σ = λ"),
                   tags$li("Log-Normal distribution: λ = 0"),
                   tags$li("Exponential distribution: σ = λ = 1")
                 ),
                 h3("Parameter Interpretations"),
                 tags$ul(
                   tags$li(strong("β (beta) - Location parameter:"), " Controls the central tendency of the distribution. Related to median survival time by β ≈ log(median) + constant."),
                   tags$li(strong("σ (sigma) - Scale parameter:"), " Controls the spread/dispersion of survival times. Higher values lead to more variable survival times."),
                   tags$li(strong("λ (lambda) - Shape parameter:"), " Controls the form of the hazard function. Different values create decreasing, arc-shaped, or increasing hazards.")
                 ),
                 h3("Clinical Relevance"),
                 p("Different hazard shapes represent different clinical scenarios:"),
                 tags$ul(
                   tags$li(strong("Decreasing hazard (λ < 0):"), " Risk is highest early and then decreases. Common in post-surgery recovery."),
                   tags$li(strong("Arc-shaped hazard (λ ≈ 0):"), " Risk increases, peaks, then falls. Common in treatments with initial risk but long-term benefit."),
                   tags$li(strong("Increasing hazard (λ > 0):"), " Risk grows over time. Common in age-related diseases.")
                 ),
                 h3("In the Hierarchical Bayesian Model"),
                 p("β is personalized for each patient, while σ and λ are specific to subgroups. This allows for personalized survival curves with appropriate hazard shapes for different patient populations.")
        )
      )
    )
  )
)

# Server
server <- function(input, output, session) {
  
  # Update parameters based on preset selection
  observe({
    if (input$preset != "custom") {
      if (input$preset == "weibull") {
        updateSliderInput(session, "lambda", value = 1.0)
      } else if (input$preset == "lognormal") {
        updateSliderInput(session, "lambda", value = 0.0)
      } else if (input$preset == "exponential") {
        updateSliderInput(session, "sigma", value = 1.0)
        updateSliderInput(session, "lambda", value = 1.0)
      } else if (input$preset == "gamma") {
        # Set sigma = lambda
        updateSliderInput(session, "sigma", value = input$lambda)
      }
    }
  })
  
  # Update parameters based on clinical examples
  observe({
    if (input$example != "none") {
      if (input$example == "surgery") {
        # Post-surgery: Decreasing hazard
        updateSliderInput(session, "beta", value = 2.0)
        updateSliderInput(session, "sigma", value = 2.8)
        updateSliderInput(session, "lambda", value = -0.5)
      } else if (input$example == "age") {
        # Age-related: Increasing hazard
        updateSliderInput(session, "beta", value = 3.0)
        updateSliderInput(session, "sigma", value = 1.5)
        updateSliderInput(session, "lambda", value = 0.5)
      } else if (input$example == "treatment") {
        # Treatment with initial risk: Arc-shaped
        updateSliderInput(session, "beta", value = 2.5)
        updateSliderInput(session, "sigma", value = 1.5)
        updateSliderInput(session, "lambda", value = 0.1)
      } else if (input$example == "diabetes") {
        # From the paper: Add both curves
        updateSliderInput(session, "beta", value = 2.0)
        updateSliderInput(session, "sigma", value = 1.5)
        updateSliderInput(session, "lambda", value = -0.1)
      }
    }
  })
  
  # Survival Plot
  output$survivalPlot <- renderPlot({
    beta <- input$beta
    sigma <- input$sigma
    lambda <- input$lambda
    
    # Generate time points
    t_values <- seq(0, 10, by = 0.1)
    
    # Calculate survival for main parameters
    surv_values <- sapply(t_values, function(t) sgg(t, beta, sigma, lambda))
    
    # Create data frame
    df <- data.frame(Time = t_values, Survival = surv_values, Group = "Current Parameters")
    
    # Add diabetes comparison if selected
    if (input$example == "diabetes") {
      # Non-diabetic parameters from the paper (σ=2.77, λ=-0.54)
      non_diabetic <- sapply(t_values, function(t) sgg(t, beta, 2.77, -0.54))
      # Diabetic parameters from the paper (σ=1.52, λ=-0.13)
      diabetic <- sapply(t_values, function(t) sgg(t, beta, 1.52, -0.13))
      
      df_diabetes <- data.frame(
        Time = c(t_values, t_values),
        Survival = c(non_diabetic, diabetic),
        Group = c(rep("Non-diabetic", length(t_values)), 
                  rep("Diabetic", length(t_values)))
      )
      
      df <- df_diabetes
    }
    
    # Create plot
    p <- ggplot(df, aes(x = Time, y = Survival, color = Group)) +
      geom_line(size = 1.2) +
      theme_minimal() +
      labs(title = "Survival Function",
           x = "Time",
           y = "Survival Probability",
           color = "Group") +
      ylim(0, 1) +
      theme(text = element_text(size = 14),
            plot.title = element_text(hjust = 0.5, size = 16))
    
    if (input$example != "diabetes") {
      p <- p + scale_color_manual(values = c("Current Parameters" = "blue"))
    } else {
      p <- p + scale_color_manual(values = c("Non-diabetic" = "blue", 
                                             "Diabetic" = "red"))
    }
    
    return(p)
  })
  
  # Hazard Plot
  output$hazardPlot <- renderPlot({
    beta <- input$beta
    sigma <- input$sigma
    lambda <- input$lambda
    
    # Generate time points
    t_values <- seq(0.1, 10, by = 0.1)  # Start at 0.1 to avoid division by zero
    
    # Calculate hazard for main parameters
    hazard_values <- sapply(t_values, function(t) dgg(t, beta, sigma, lambda) / sgg(t, beta, sigma, lambda))
    
    # Create data frame
    df <- data.frame(Time = t_values, Hazard = hazard_values, Group = "Current Parameters")
    
    # Add diabetes comparison if selected
    if (input$example == "diabetes") {
      # Non-diabetic parameters from the paper (σ=2.77, λ=-0.54)
      non_diabetic <- sapply(t_values, function(t) dgg(t, beta, 2.77, -0.54) / sgg(t, beta, 2.77, -0.54))
      # Diabetic parameters from the paper (σ=1.52, λ=-0.13)
      diabetic <- sapply(t_values, function(t) dgg(t, beta, 1.52, -0.13) / sgg(t, beta, 1.52, -0.13))
      
      df_diabetes <- data.frame(
        Time = c(t_values, t_values),
        Hazard = c(non_diabetic, diabetic),
        Group = c(rep("Non-diabetic", length(t_values)), 
                  rep("Diabetic", length(t_values)))
      )
      
      df <- df_diabetes
    }
    
    # Create plot
    p <- ggplot(df, aes(x = Time, y = Hazard, color = Group)) +
      geom_line(size = 1.2) +
      theme_minimal() +
      labs(title = "Hazard Function",
           x = "Time",
           y = "Hazard Rate",
           color = "Group") +
      theme(text = element_text(size = 14),
            plot.title = element_text(hjust = 0.5, size = 16))
    
    if (input$example != "diabetes") {
      p <- p + scale_color_manual(values = c("Current Parameters" = "blue"))
    } else {
      p <- p + scale_color_manual(values = c("Non-diabetic" = "blue", 
                                             "Diabetic" = "red"))
    }
    
    return(p)
  })
  
  # Density Plot
  output$densityPlot <- renderPlot({
    beta <- input$beta
    sigma <- input$sigma
    lambda <- input$lambda
    
    # Generate time points
    t_values <- seq(0.1, 10, by = 0.1)
    
    # Calculate density for main parameters
    pdf_values <- sapply(t_values, function(t) dgg(t, beta, sigma, lambda))
    
    # Create data frame
    df <- data.frame(Time = t_values, Density = pdf_values, Group = "Current Parameters")
    
    # Add diabetes comparison if selected
    if (input$example == "diabetes") {
      # Non-diabetic parameters from the paper (σ=2.77, λ=-0.54)
      non_diabetic <- sapply(t_values, function(t) dgg(t, beta, 2.77, -0.54))
      # Diabetic parameters from the paper (σ=1.52, λ=-0.13)
      diabetic <- sapply(t_values, function(t) dgg(t, beta, 1.52, -0.13))
      
      df_diabetes <- data.frame(
        Time = c(t_values, t_values),
        Density = c(non_diabetic, diabetic),
        Group = c(rep("Non-diabetic", length(t_values)), 
                  rep("Diabetic", length(t_values)))
      )
      
      df <- df_diabetes
    }
    
    # Create plot
    p <- ggplot(df, aes(x = Time, y = Density, color = Group)) +
      geom_line(size = 1.2) +
      theme_minimal() +
      labs(title = "Probability Density Function",
           x = "Time",
           y = "Density",
           color = "Group") +
      theme(text = element_text(size = 14),
            plot.title = element_text(hjust = 0.5, size = 16))
    
    if (input$example != "diabetes") {
      p <- p + scale_color_manual(values = c("Current Parameters" = "blue"))
    } else {
      p <- p + scale_color_manual(values = c("Non-diabetic" = "blue", 
                                             "Diabetic" = "red"))
    }
    
    return(p)
  })
}

# Run the app
shinyApp(ui = ui, server = server)