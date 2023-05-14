# 1st subtask...............

#a...............

# Install and load necessary packages
install.packages(c("readxl", "stats", "base", "dplyr"))
library(readxl)
library(stats)
library(base)
library(dplyr)

# Read in the data
vehicles_df <- read_excel("vehicles.xlsx")
View(head(vehicles_df))

# Remove Samples Column
vehicles_df <- vehicles_df[, -which(names(vehicles_df) == "Samples")]
View(head(vehicles_df))

# Select only numeric columns
numeric_cols <- sapply(vehicles_df, is.numeric)
vehicles_df_numeric <- vehicles_df[, numeric_cols]

# Identify and remove outliers using z-score method
z_scores <- as.matrix(scale(vehicles_df[, numeric_cols]))
outliers <- apply(z_scores, 1, function(x) any(abs(x) > 3))
clean_df <- vehicles_df[!outliers,]
View(head(clean_df))

# scale the numeric columns
num_cols <- sapply(clean_df, is.numeric)
clean_df[, num_cols] <- scale(clean_df[, num_cols])
View (head(clean_df[, num_cols]))

# Remove the column class
clean_df <- clean_df[, -which(names(clean_df) == "Class")]
View(head(clean_df))

#b............... 

install.packages("NbClust")
library(NbClust)

# Determine the number of clusters using NbClust
set.seed(123)
nb_clusters <- NbClust(clean_df, distance = "euclidean", min.nc=2, max.nc=10, method="kmeans")
table(nb_clusters$Best.nc[1,])

library(tidyverse)
library(cluster)
library(ggplot2)

# Elbow method to determine the optimal number of clusters
set.seed(123)  
wss <- sapply(1:10, 
              function(k) {
                kmeans(clean_df, k, nstart = 10, iter.max = 50)$tot.withinss
              })

# Plot the elbow curve
elbow_plot <- tibble(k = 1:10, wss = wss) %>% 
  ggplot(aes(x = k, y = wss)) +
  geom_line() +
  geom_point() +
  scale_x_continuous(breaks = 1:10) +
  labs(x = "Number of Clusters (k)", y = "Within-Cluster Sum of Squares (WSS)") +
  theme_minimal()

# Display the plot
elbow_plot

install.packages("factoextra")
library(factoextra)

# Gap statistic to determine the optimal number of clusters
set.seed(123) 
gap_stat <- clusGap(clean_df, FUN = kmeans, nstart = 25, K.max = 10, B = 50)

# Plot the gap statistic
fviz_gap_stat(gap_stat) + labs(title = "Gap Statistic Plot")

# Display the results
gap_stat

# Determine the optimal number of cluster centers using silhouette method
set.seed(123)  # for reproducibility
silhouettemethod <- fviz_nbclust(clean_df, FUNcluster = kmeans, method = "silhouette", nstart = 25, k.max = 10, verbose = FALSE)

# Plot the silhouette plot
plot(silhouettemethod)


#c...............

set.seed(123) 
kmeans_model <- kmeans(clean_df, centers = 3, nstart = 25, iter.max = 50)

# Print the cluster assignments
kmeans_model$cluster


# Set the seed for reproducibility
set.seed(123)

# Calculate the within-cluster sum of squares (WSS)
wss <- sum(kmeans_model$withinss)
wss

# Calculate the between-cluster sum of squares (BSS)
bss <- sum(kmeans_model$betweenss)
bss

# Print the ratio of between_cluster_sums_of_squares (BSS) over total_sum_of_Squares (TSS)
cat("\n\nBSS/TSS ratio:\n")
cat(kmeans_model$betweenss / kmeans_model$totss)

library(factoextra)
# Visualize the clusters using the first two principal components
fviz_cluster(kmeans_model, geom = "point", data = clean_df, stand = FALSE, palette = "jco", ggtheme = theme_minimal(), main = "Kmeans Clustering Results")

#d...............

library(cluster)

# Calculate silhouette widths
silwidths <- silhouette(kmeans_model$cluster, dist(clean_df))
head(silwidths)

# Plot the silhouette widths
plot(silwidths, main = "Silhouette Plot for Kmeans Clustering", border = NA, col = 1:3)

# Average silhouette width
mean(silwidths[,3])



#2nd subtask...............

# e...............

# Performing PCA on clean_df
PCAVehicles <- prcomp(clean_df, center = TRUE, scale = FALSE)
eigenvalue <- PCAVehicles$sdev^2
eigenvalue
eigenvector <- PCAVehicles$rotation
head(eigenvector)

# Identify the PCAs which satisfies the threshold
cum_score <- cumsum(PCAVehicles$sdev^2 / sum(PCAVehicles$sdev^2))
print(cum_score)

# number of PCAs that reaches the cumulative score of 92
num_PCAs <- which(cum_score >= 0.92)[1]
num_PCAs

# Create the newdataset to store the PCAs
transformed <- as.data.frame(PCAVehicles$x[, 1:num_PCAs])
view(head(transformed))

#e...............

# NbCLust method
set.seed(123)
PCA_nbClust <- NbClust(transformed, distance = "euclidean", min.nc = 2, max.nc = 15, method = "kmeans", index = "all")

# Elbow method
fviz_nbclust(transformed, kmeans, method= "wss") + labs(subtitle = "Elbow method")

# Gap statistics method
fviz_nbclust(transformed, kmeans, nstart = 25,  method = "gap_stat", nboot = 50)+labs(subtitle = "Gap statistic method")

# Silhoutte method
fviz_nbclust(transformed, kmeans, method = "silhouette") + labs(subtitle = "Silhouette method")

#f...............

# kmeans analysis
set.seed(123) 
PCA_kmeans_model <- kmeans(transformed, centers = 3, nstart = 25, iter.max = 50)
PCA_kmeans_model$cluster

# Set the seed for reproducibility
set.seed(123)
# Calculate the within-cluster sum of squares (WSS)
PCA_wss <- sum(PCA_kmeans_model$withinss)
PCA_wss

# Calculate the between-cluster sum of squares (BSS)
PCA_bss <- sum(PCA_kmeans_model$betweenss)
PCA_bss

# Print the ratio of between_cluster_sums_of_squares (BSS) over total_sum_of_Squares (TSS)
cat("\n\nBSS/TSS ratio:\n")
cat(PCA_kmeans_model$betweenss / PCA_kmeans_model$totss)


#g...............

# Calculate silhouette widths
PCA_silwidths <- silhouette(PCA_kmeans_model$cluster, dist(transformed))
head(PCA_silwidths)

# Plot the silhouette widths
plot(PCA_silwidths, main = "Silhouette Plot for Kmeans Clustering", border = NA, col = 1:3)


#h...............

kmeans_func <- function(x, k) {
  kmeans(x, centers = k, nstart = 25)
}

# Visualization of Calinski-Harabasz Index
calinski_harabasz <- clusGap(transformed, kmeans_func, K.max = length(PCA_kmeans_model$size))
fviz_gap_stat(calinski_harabasz) + labs(subtitle = "Calinski-Harabasz Index")
