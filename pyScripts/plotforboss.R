### for Pradeep

model_data = readRDS("modelsforplotting.rds")
ascvd = model_data[model_data$Disease%in%"ASCVD",]
prevent = 0.644

# Create data frame for plotting with correct order
plot_data = data.frame(
  Method = factor(c("FRS 30-year", "PREVENT 30-year", "MSGene lifetime", "Aladynoulli Dynamic"), 
                 levels = c("FRS 30-year", "PREVENT 30-year", "MSGene lifetime", "Aladynoulli Dynamic")),
  C_index = c(0.530, 0.644, 0.699, 0.856),
  CI_lower = c(0.528, 0.640, 0.696, 0.845),
  CI_upper = c(0.533, 0.648, 0.703, 0.867)
)

# Create the plot with deeper, nicer colors
library(ggplot2)

ggplot(plot_data, aes(x = Method, y = C_index)) +
  geom_bar(stat = "identity", 
           fill = c("#2E86AB", "#A23B72", "#F18F01", "#C73E1D"), 
           alpha = 0.8) +
  geom_errorbar(aes(ymin = CI_lower, ymax = CI_upper), width = 0.2, color = "black") +
  coord_flip() +  # Horizontal bars
  ylim(0, 0.9) +
  labs(
    title = "Comparison of prediction models using bootstrap",
    y = "C-index value",
    x = ""
  ) +
  theme_minimal() +
  theme(
    axis.text.y = element_text(size = 12),
    axis.title.y = element_text(size = 14, angle = 90),
    plot.title = element_text(size = 14, hjust = 0.5),
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank()
  ) +
  annotate("text", x = 0.5, y = 0.1, label = "D", size = 6, fontface = "bold")
