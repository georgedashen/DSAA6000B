library(magrittr)
library(dplyr)
library(readr)
library(tidyr)
library(ggplot2)
library(ggrepel)

#load
setwd("D:/WorkFile/HKUST/GNN6000B/assignment/results/")
df <- list.files("./") %>% lapply(read_csv) %>% bind_rows
#remove duplicate, keep higher AUC
df1 <- df %>% filter(EPOCH==750) %>% group_by(hid_dim, n_layer) %>% top_n(n=1, AUC)
df1$n_layer <- as.factor(df1$n_layer)

#plot1: metrics
ggplot(df1, aes(hid_dim, AUC, label=formatC(AUC, digits = 4))) + 
  geom_point(aes(color=n_layer),size=6) + geom_line(aes(group=n_layer, color=n_layer)) +
  geom_text_repel(hjust=-0.3, vjust=0.4) + 
  theme_bw(base_size = 20) + xlim(c(5,52))
ggsave(filename = "../metrics.png", width = 10, height = 8, dpi = 300)

#plot2: time
ggplot(df1, aes(hid_dim, avg_time_per_epoch, label=formatC(avg_time_per_epoch, digits = 3))) + 
  geom_point(aes(color=n_layer),size=6) + geom_line(aes(group=n_layer, color=n_layer)) +
  geom_text_repel(hjust=-0.75, vjust=0.4) + 
  theme_bw(base_size = 20) + xlim(c(0,52))
ggsave(filename = "../time.png", width = 10, height = 8, dpi = 300)
