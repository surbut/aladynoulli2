library(tidyverse)
library(viridis)
library(reshape2)
library(patchwork)
library(grid)
library(dendextend)

# Load the data


mgb_checkpoint <- readRDS("~/Library/CloudStorage//Dropbox/mgb_model.rds")
aou_checkpoint <- readRDS("~/Library/CloudStorage//Dropbox/aou_model.rds")
ukb_checkpoint <- readRDS("~/Library/CloudStorage//Dropbox/ukb_model.rds")


mgb_params=readRDS("~/Library/CloudStorage//Dropbox/mgb_params.rds")
aou_params=readRDS("~/Library/CloudStorage/Dropbox/aou_params.rds")
param=ukb_params=readRDS("~/Library/CloudStorage/Dropbox/ukb_params.rds")

dimnames(param$phi)=list(c(0:20),ukb_checkpoint$disease_names[,1],c(30:81))
dimnames(aou_params$phi)=list(c(0:20),aou_checkpoint$disease_names,c(30:80))
dimnames(mgb_params$phi)=list(c(0:20),mgb_checkpoint$disease_names,c(30:80))


