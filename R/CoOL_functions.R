######################## Causes of Outcome Learning #############################
######################## Version 09.09.2020

########## Minor functions ############

#' Function used as part of other functions
#'
#' Function used as part of other functions
#'
#' @param r rows in matrix
#' @param c columns in matrix
#' @param v mean
#' @export

random <- function(r,c,v) {
  w1 <- matrix(NA,nrow = r, ncol = c)
  w1 <- sapply(w1,function(x){rnorm(1,v,.01)})
  w1 <- matrix(w1,nrow = r, ncol = c)
  return(w1)
}

#' Function used as part of other functions
#'
#' Function used as part of other functions
#'
#' @param input input in the relu function
#' @export

relu <- function(input) {
  #  return(ifelse(input<0,0,input))
  return((input>0)*input)
}


########## Wrappers ##############


#' CoOL working example with sex, drug A, and drug B
#'
#' To reproduce the CoOL working example with sex, drug A, and drug B.
#'
#' @param n number of observations for the synthetic data

CoOL_0_working_example <- function(n) {
  drug_a = sample(1:0,n,prob=c(0.2,0.8),replace=TRUE)
  sex = sample(1:0,n,prob=c(0.5,0.5),replace=TRUE)
  drug_b = sample(1:0,n,prob=c(0.2,0.8),replace=TRUE)
  Y <-  sample(1:0,n,prob=c(0.05,0.95),replace = TRUE)
  for (i in 1:n) {
    if (sex[i] == 0 & drug_a[i] == 1 & sample(1:0,1,prob=c(.15,0.8)) ) {
      Y[i] <- 1
    }
    if (sex[i] == 1 & drug_b[i] == 1 & sample(1:0,1,prob=c(.15,0.85)) ) {
      Y[i] <- 1
    }
  }
  data <- data.frame(Y,sex,drug_a,drug_b) #,C)
  for (i in 1:ncol(data))   data[,i] <- as.numeric(data[,i])
  return(data)
}



#' Binary encode exposure data
#'
#' This function binary encodes the exposure data set so that each category is coded 0 and 1 (e.g. the variable sex will be two variables men (1/0) and women (1/)).
#'
#' @param expusure_data The exposure data set
#' @export
#' @details
#' @examples
#' #See the example under CoOL_0_synthetic_data
#'

CoOL_0_binary_encore_exposure_data <- function(exposure_data) {
  for (i in 1:ncol(exposure_data)) {exposure_data[,i] <- factor(exposure_data[,i])}
  exposure_data <- one_hot(data.table::as.data.table(exposure_data))
  return(exposure_data)
}





#' Initiates a non-negative neural network
#'
#' This function initiates a monotonistc neural network. The one-hidden layer monotonistic neural network is designed to resemble a DAG with hidden synergistic components. With the model, we intend to learn the various synergistic interactions between the exposures and outcome. The model needs to be monotonistic and estimate the risk on an additive scale. Neural networks include hidden activation functions (if the sum of the input exceeds a threshold, information is passed on), which can model minimum threshold values of interactions between exposures. We need to specify the upper limit of the number of possible hidden activation functions and through model fitting, the model may be able to learn both stand-alone and synergistically interacting factors.
#'
#' @param inputs The number of exposures.
#' @param output The outbut variable is used to calcualte the mean of it used to initiate the baseline risk.
#' @param hidden Number of hidden nodes.
#' @param confounder Allows to control away a confounder (connected to the output layer)
#' @export
#' @details
#'
#' The monotonistic neural network can be denoted as:
#' \deqn{
#' P(Y=1|X^+)=\sum_{j}\Big(w_{j,k}^+ReLU_j\big(\sum_{i}(w_{i,j}^+X_i^+) + b_j^-\big)\Big) + R^{b}
#' }
#'
#' @examples
#' #See the example under CoOL_0_synthetic_data
#'


CoOL_1_initiate_neural_network <- function(inputs,output,hidden=10,confounder=FALSE) {
  # Weight initiation
  w1 <- abs(random(inputs,hidden,0.01))
  b1 <- -abs(random(1,hidden,0.01))
#  w2 <- abs(random(hidden,1,0.01))
  w2 <- matrix(1,nrow=hidden)
  b2 <- abs(random(1,1,0.01))
  if (confounder==TRUE)  c2 <- abs(random(1,1,0.01))
  performance <- NA
  best_epoch <- NA
  weight_performance <- NA
  epochs <- NA
  b2 <- as.matrix(mean(output))
  if (confounder == FALSE)   return(list(w1,b1,w2,b2,performance,epochs,best_epoch))
  if (confounder == TRUE)   return(list(w1,b1,w2,b2,c2,performance,epochs,best_epoch))

}








#' Training the monotonistic neural network
#'
#' This function trains the monotonistic neural network. Fitting the model is done in a step-wise procedure one individual at a time, where the model estimates individual's risk of the disease outcome, estimates the prediction's residual error and adjusts the model parameters to reduce this error. By iterating through all individuals for multiple epochs (one complete iterations through all individuals is called an epoch), we end with parameters for the model, where the errors are smallest possible for the full population. The model fit follows the linear expectation that synergism is a combined effect larger than the sum of independent effects. The initial values, derivatives, and learning rates are described in further detail in the Supplementary material. The monotonistic model ensures that the predicted value cannot be negative. The model does not prevent estimating probabilities above 1, but this would be unlikely, as risks of disease and mortality even for high risk groups in general are far below 1. The use of a test dataset does not seem to assist deciding on the optimal number of epochs possibly due to the constrains due to the monotonicity assumption. We suggest splitting data into a train and test data set, such that findings from the train data set can be confirmed in the test data set before developing hypotheses.
#'
#' @param X The exposure data
#' @param Y The outcome data
#' @param model The fitted monotonistic neural network
#' @param lr Learning rate (several LR can be provided, such that the model training will train for each LR and continue to the next)
#' @param epochs Epochs
#' @param patience The number of epochs allowed without an improvement in performance.
#' @param plot_and_evaluation_frequency The interval for plotting the performance and checking the patience
#' @param IPCW Inverse probability of censoring weights (Warning: not yet correctly implemented)
#' @details
#' For each individual:\deqn{
#' P(Y=1|X^+)=R^b+\sum_iR^X_i
#' }
#' The below procedure is conducted for all individuals in a one by one fashion. The baseline risk, $R^b$, is simply parameterised in the model. The decomposition of the risk contributions for exposures, $R^X_i$, takes 3 steps:
#'
#' Step 1 - Subtract the baseline risk, $R^b$:
#' \deqn{
#' R^X_k =  P(Y=1|X^+)-R^b
#' }
#' Step 2 - Decompose to the hidden layer:
#' \deqn{
#' R^{X}_j =  \frac{H_j w_{j,k}}{\sum_j(H_j w_{j,k})} R^X_k
#' }
#' Where $H_j$ is the value taken by each of the $ReLU()_j$ functions for the specific individual.
#'
#' Step 3 - Hidden layer to exposures:
#' \deqn{
#' R^{X}_i = \sum_j \Big(\frac{X_i^+ w_{i,j}}{\sum_i( X_i^+ w_{i,j})}R^X_j\Big)
#' }
#' This creates a dataset with the dimensions equal to the number of individuals times the number of exposures plus a baseline risk value, which can be termed a risk contribution matrix. Instead of exposure values, individuals are given risk contributions, R^X_i.
#'
#' @export
#' @examples
#' #See the example under CoOL_0_synthetic_data


CoOL_2_train_neural_network <- function(X_train, Y_train, X_test, Y_test, model, lr = c(1e-4,1e-5,1e-6),
                            epochs = 50000, patience = 100,
                            plot_and_evaluation_frequency = 50, IPCW = NA, L1=0.00001, spline_df=10) {
  X_test = X_train
  Y_test = Y_train
  if (is.na(IPCW)) IPCW <- rep(1,nrow(X_train))
for (lr_set in lr) {
  print(paste0("############################## Learning rate: ",lr_set," ##############################"))
  performance = model$train_performance
  performance_test = model$test_performance
  weight_performance = model$weight_performance
  baseline_risk_monitor = model$baseline_risk_monitor
  par(mfrow=c(1,3));par(mar=c(3,5,3,1))
    for(rounds in 1:ceiling(c(epochs/plot_and_evaluation_frequency))) {
      model <- CoOL_cpp_train_network_relu(x=as.matrix(X_train),y=as.matrix(Y_train),testx=as.matrix(X_test),testy=as.matrix(Y_test),
              lr = lr_set, maxepochs  = plot_and_evaluation_frequency, W1_input = model[[1]],B1_input = model[[2]],
              W2_input = model[[3]],B2_input = model[[4]], IPCW = IPCW, L1=L1)
      performance <- c(performance,model$train_performance)
      performance_test <- c(performance_test,model$test_performance)
      weight_performance <- c(weight_performance,model$weight_performance)
      baseline_risk_monitor <- c(baseline_risk_monitor,model$baseline_risk_monitor)
      plot(performance, type='l',yaxs='i', ylab="Mean squared error",
           xlab="Epochs",main="Performance on training data set")
      points(smooth.spline(performance, df = spline_df),col="red",type='l',lwd=2)
#      plot(performance_test, type='l',yaxs='i', ylab="Mean squared error",
#           xlab="Epochs",main="Performance on test data set")
     plot(log(weight_performance), type='l', ylab="log of mean squared weight difference",
          xlab="Epochs",main="Log mean squared weight difference")
     points(smooth.spline(log(weight_performance), df = spline_df),col="red",type='l',lwd=2)
     plot(baseline_risk_monitor,type='l', main="Estimated baseline risk by epoch")
     abline(h=mean(Y_train),lty=2)
     points(smooth.spline(baseline_risk_monitor, df = spline_df),col="red",type='l',lwd=2)
      if(length(performance)-which.min(performance)>patience) break
    }
  model$train_performance <- c(performance)
  model$test_performance <-  c(performance_test)
  model$weight_performance <-  c(weight_performance)
  model$baseline_risk_monitor <- c(baseline_risk_monitor)
  model$epochs = epochs
}
  par(mfrow=c(1,1))
  return(model)
}



#' Training the monotonistic neural network with a confounder connected to the output layer
#'
#' This function trains the monotonistic neural network with a confounder connected to the output layer. This functions allows one to divide the training process into several steps.
#'
#' @param X The exposure data
#' @param Y The outcome data
#' @param C The confounder data
#' @param model The fitted monotonistic neural network
#' @param lr Learning rate
#' @param epochs Epochs
#' @param patience The number of epochs allowed without an improvement in performance.
#' @param plot_and_evaluation_frequency The interval for plotting the performance and checking the patience
#' @details
#' @export
#' @examples
#' #See the example under CoOL_0_synthetic_data

CoOL_2_train_neural_network_with_confounder <- function(X, Y, C, model, lr = 0.01,
                                      epochs = 50000, patience = 500,
                                      plot_and_evaluation_frequency = 50) {
  C <- as.matrix(C)
  performance = model$train_performance
  par(mfrow=c(1,1));par(mar=c(3,5,3,1))
  for(rounds in 1:ceiling(c(epochs/plot_and_evaluation_frequency))) {
    model <- CoOL_cpp_train_network_relu_with_confounder(as.matrix(X),as.matrix(Y),as.matrix(C),as.matrix(X),as.matrix(Y),as.matrix(C),
           lr = lr, maxepochs  = plot_and_evaluation_frequency, W1_input = model[[1]],B1_input = model[[2]],W2_input = model[[3]],B2_input = model[[4]],C2_input = model[[5]])
    performance <- c(performance,model$train_performance)
    plot(performance, type='l',yaxs='i', ylab="Mean squared error",
         xlab="Epochs",main="Performance")
    if(length(performance)-which.min(performance)>patience) break
  }
  model$train_performance <- c(performance)
  model$test_performance <- NA
  return(model)
}



#' Plotting the monotonistic neural network
#'
#' This function plots the monotonistic neural network
#'
#' @param model The fitted monotonistic neural network
#' @param names Labels of each exposure
#' @param arrow_size defines the arrow_size for the model illustration in the reported training progress.
#' @export
#' @examples
#' #See the example under CoOL_0_synthetic_data

CoOL_3_plot_neural_network <- function(model,names,arrow_size = 2, title = "Model") {
  par(mar=c(0,0,2,0))
  plot(0,0,type='n',xlim=c(0,4),ylim=c(-max(nrow(model[[1]]),nrow(model[[3]]))-1,0),axes=FALSE,ylab="",xlab="",main=title)
  #abline(h=0)
  points(rep(1,nrow(model[[1]])),-c(1:nrow(model[[1]])),cex=10)
  points(rep(2,ncol(model[[1]])),-c(1:ncol(model[[1]])),cex=10)
  points(3,-(ncol(model[[1]])+1)/2,cex=10)
  # Static edges first in grey
  for (g in 1:nrow(model[[3]])) {
    arrows(x0=2,x1=3,y0=-g,y1= -(ncol(model[[1]])+1)/2,lwd=abs(model[[3]][g,1])*5,col=ifelse(model[[3]][g,1]>0,"grey","white"),length=0)
    #   text(2,-g,round(model[[3]][g,1],2),pos=3)
  }
  # Trained edges
  for (g in 1:nrow(model[[1]])) {
    for (h in 1:ncol(model[[1]])) {
      arrows(x0=1,x1=2,y0=-g,y1=-h,lwd=abs(model[[1]][g,h])*arrow_size,col=ifelse(model[[1]][g,h]>0,"green","white"),length=0)
      #      text(1,-g,round(model[[1]][g,h],2),pos=3)
    }
  }
  for (i in 1:nrow(model[[1]])) {
    text(rep(1,nrow(model[[1]]))[i],-c(1:nrow(model[[1]]))[i],names[i])
  }
  text(rep(2,ncol(model[[1]])),-c(1:ncol(model[[1]])),paste0("b=",round(model[[2]][1,],2)),pos=1)
  text(3,-(ncol(model[[1]])+1)/2,paste0("B=",round(model[[4]][1,1],2)),pos=1)
  par(mar=c(5.1,4.1,4.1,2.1))
  #  points(3,-(ncol(model[[1]])+1)/2+1,cex=10)
  #  arrows(x0=3,x1=3,y0=-(ncol(model[[1]])+1)/2+1,y1= -(ncol(model[[1]])+1)/2,lwd=abs(model[[5]][1,1])*arrow_size,col=ifelse(model[[5]][1,1]>0,"green","white"),length=0)
}


#' Predict the risk of the outcome using the fitted monotonistic neural network
#'
#' Predict the risk of the outcome using the fitted monotonistic neural network.
#'
#' @param X The exposure data
#' @param model The fitted the monotonistic neural network
#' @export
#' @examples
#' #See the example under CoOL_0_synthetic_data


CoOL_4_predict_risks <- function(X,model) {
  H <- relu(t(t(as.matrix(X) %*% as.matrix(model[[1]])) + as.vector(model[[2]])))
  o = relu(as.vector(H %*% model[[3]][,1] + as.vector(model[[4]][1,1])))
  if(max(o)>1) print("Warning: Some predicted risks are above 1")
  return(o)
}

#' Predict the risk of the outcome using the fitted monotonistic neural network
#'
#' Predict the risk of the outcome using the fitted monotonistic neural network.
#'
#' @param X The exposure data
#' @param model The fitted the monotonistic neural network
#' @export
#' @examples
#' #See the example under CoOL_0_synthetic_data

CoOL_4_AUC <- function(outcome_data,exposure_data,model,title="Accuracy") {
library(pROC)
pred <- CoOL_4_predict_risks(exposure_data,model)
plot(roc(outcome_data,pred),print.auc=TRUE,main=title)
}




#' Layer-wise relevance propagation of the fitted monotonistic neural network
#'
#' Calculates risk contributions for each exposure and a baseline using layer-wise relevance propagation of the fitted monotonistic neural network and data.
#'
#' @param X The exposure data
#' @param model The fitted the monotonistic neural network
#' @export
#' @examples
#' #See the example under CoOL_0_synthetic_data


CoOL_5_layerwise_relevance_propagation <- function(X,model) {
  #model <- model_2_c
  #X = X_flip

  labels <- colnames(X)
  X = as.matrix(X)

  # Forward
  R_X <- matrix(0,ncol=ncol(X),nrow=nrow(X))
  U_B = NULL

  H_all <- relu(t(t(as.matrix(X) %*% as.matrix(model[[1]])) + as.vector(model[[2]])))
  o_all = as.vector(H_all %*% model[[3]][,1] + as.vector(model[[4]][1,1]))

  for (i in 1:nrow(X)) {
    if (i / 1000 == i %/% 1000) {print(i)}
    H <- H_all[i,]
    o <- as.numeric(o_all[i])

    # Layer-wise relevance propagation (LRP)
    Pos1 = model[[3]][,1]
    Pos1 = ifelse(Pos1>0,Pos1,0)
    Pos1_sum = sum(H*Pos1) #+ifelse(model[[5]][1,1] * C[i]>0,model[[5]][1,1] * C[i],0)
    Pos1_sum <- ifelse(is.na(Pos1_sum)|Pos1_sum==0,1,Pos1_sum)
    Pos1 = (H*Pos1)/Pos1_sum

    model[[3]][,1] < 0 # ensure none are below 0


    o <- as.numeric(o * (1-  (relu(model[[4]]) / o))) # Subtracting the first U_B
    R_H = Pos1 * o

    Pos2 = model[[1]]
    Pos2 = ifelse(Pos2>0,Pos2,0)
    model[[1]] < 0 # ensure none are below 0

    for (g in 1:length(H)) {
      Pos2_sum = sum(X[i,]*Pos2[,g])
      Pos2_sum <- ifelse(is.na(Pos2_sum)|Pos2_sum==0,1,Pos2_sum)
      R_X[i,] = R_X[i,] + (X[i,]*Pos2[,g])/Pos2_sum * R_H[g] #+ sum((((X[i,g][X[i,g]>0])*Neg2[g,])/Neg2_sum) * (-b) * R_H)
    }

    U_B[i] <- model[[4]]
    #R_X[i,] <- ((o_all[i]*(1-(U_B[i]/o_all[i])))/sum(R_X[i,]))*R_X[i,]
    if (sum(R_X[i,])==0 | is.na(sum(R_X[i,]))) R_X[i,] = 0
  }


  Baseline_risk <- U_B
  R_X <- data.frame(cbind(R_X,Baseline_risk))
  colnames(R_X) <- c(labels,"Baseline_risk")

  #Sanity check
  if (max(o_all-rowSums(R_X)) > 1e-6) print("WARNING: Some risk contributions do not sum to the predicted value")
  return(R_X)
}


#' Dendrogram and sub-groups
#'
#' Calculates presents a dendrogram coloured by the pre-defined number of sub-groups and provides the vector with sub-groups.
#'
#' @param risk_contributions The risk contributions
#' @param number_of_subgroups The number of sub-groups chosen (Visual inspection is necessary)
#' @export
#' @examples
#' #See the example under CoOL_0_synthetic_data



CoOL_6_dendrogram <- function(risk_contributions,number_of_subgroups=3, title = "Dendrogram") {
  library(ClustGeo)
  p <- cbind(risk_contributions)
  p <- plyr::count(p)
  pfreq <- p$freq
  p <- p[,-c(ncol(p))]
  p_h_c <- hclustgeo(dist(p,method = "manhattan"), wt=pfreq)
  pclus <- cutree(p_h_c, number_of_subgroups)
  id <- 1:nrow(risk_contributions)
  temp <- merge(cbind(id,risk_contributions),cbind(p,pclus))
  clus <- temp$pclus[order(temp$id)]
  table(clus)
  library(ggtree)
  library(ggplot2)
  library(wesanderson)
  colours <- c("grey",wes_palette("Darjeeling1"))
  print(ggtree(p_h_c,layout="equal_angle") +
          geom_tippoint(size=sqrt(pfreq)/2, alpha=.2, color=colours[pclus])+
          ggtitle(title) +
          theme(plot.title = element_text(size = 15, face = "bold")))
}



#' Assign sub-groups
#'
#' Calculates presents a dendrogram coloured by the pre-defined number of sub-groups and provides the vector with sub-groups.
#'
#' @param risk_contributions The risk contributions
#' @param number_of_subgroups The number of sub-groups chosen (Visual inspection is necessary)
#' @export
#' @examples
#' #See the example under CoOL_0_synthetic_data



CoOL_6_sub_groups <- function(risk_contributions,number_of_subgroups=3) {
  library(ClustGeo)
  p <- cbind(risk_contributions)
  p <- plyr::count(p)
  pfreq <- p$freq
  p <- p[,-c(ncol(p))]
  p_h_c <- hclustgeo(dist(p,method = "manhattan"), wt=pfreq)
  pclus <- cutree(p_h_c, number_of_subgroups)
  id <- 1:nrow(risk_contributions)
  temp <- merge(cbind(id,risk_contributions),cbind(p,pclus))
  clus <- temp$pclus[order(temp$id)]
  table(clus)
  return(clus)
}


#' Prevalence and mean risk plot
#'
#' This plot shows the prevalence and mean risk for each sub-group. Its destribution hits at sub-groups with great public health potential.
#'
#' @param risk_contributions The risk contributions
#' @param sub_groups The vector with the sub-groups
#' @param title The title of the plot
#' @export
#' @examples
#' #See the example under CoOL_0_synthetic_data


CoOL_7_prevalence_and_mean_risk_plot <- function(risk_contributions,sub_groups,title="Prevalence and mean risk\nof sub-groups") {
  library(wesanderson)
  par(mar=c(5,3,2,2))
  colours <- c("grey",wes_palette("Darjeeling1"))
risk_max = 0
    for (i in 1:max(sub_groups)) {
    risk <- sum(colMeans(as.matrix(risk_contributions[sub_groups==i,])))
    risk_max = max(risk_max,risk)
  }
  plot(0,0,type='n',xlim=c(0,1),ylim=c(0,risk_max*1.1),xaxs='i',yaxs='i',
       axes=FALSE,ylab="Risk",xlab="Prevalence",frame.plot=FALSE,main=title)
  axis(1,seq(0,1,.2));axis(2,seq(0,1,.05))
  rect(0,0,1,1)
  prev0 = 0; total = 0
  for (i in 1:max(sub_groups)) {
    prev <- sum(sub_groups==i)/length(sub_groups)
    risk <- sum(colMeans(as.matrix(risk_contributions[sub_groups==i,])))
    rect(xleft = prev0,ybottom = 0,xright = prev+prev0,ytop = risk, col=colours[i])
    prev0 = prev + prev0
    total = total + risk * prev
  }
  arrows(x0=0,x1=1,y0=mean(risk_contributions$Baseline_risk),lty=2,length=0)
}


#' Mean risk contributions by sub-groups
#'
#' Table with the mean risk contributions by sub-groups.
#'
#' @param risk_contributions The risk contributions
#' @param sub_groups The vector with the sub-groups
#' @param title The title of the plot
#' @export
#' @examples
#' #See the example under CoOL_0_synthetic_data

CoOL_8_mean_risk_contributions_by_sub_group <- function(risk_contributions,sub_groups) {
  library(wesanderson)
  colours <- c("grey",wes_palette("Darjeeling1"))
  prev0 = 0; total = 0
  for (i in 1:max(sub_groups)) {
    prev <- sum(sub_groups==i)/length(sub_groups)
    risk <- sum(colMeans(as.matrix(risk_contributions[sub_groups==i,])))
    prev0 = prev + prev0
    total = total + risk * prev
  }
  st <- 1
  d <- data.frame(matrix(NA, nrow=ncol(risk_contributions)))
  for (g in 1:max(sub_groups)) {
    for (i in 1:nrow(d)) {
      d[i,g] <- mean(risk_contributions[sub_groups==g,i])
    }}
  d <- t(d)
  rownames(d) <- paste("Group",1:max(sub_groups))
  colnames(d) <- names(risk_contributions)
  par(mar=c(0,0,0,0))
  plot(0,0,type='n',xlim=c(-ncol(d)-6,0),ylim=c(-nrow(d)-1,1),axes=F)
  text(c(-ncol(d)):c(-1),0,rev(colnames(d)),srt=25,cex=st)
  text(-ncol(d)-6,0,"F) Mean risk contributions by sub-group (SD)\n[mean risk contribution if other exposures are set to 0]",pos=4,cex=st)
  for (i in 1:max(sub_groups)) {
    prev <- sum(sub_groups==i)/length(sub_groups)
    risk <- sum(colMeans(as.matrix(risk_contributions[sub_groups==i,])))
    risk_obs <- mean(outcome_data[sub_groups==i])
    text(-ncol(d)-6,-i,paste0("Sub-group ",i,": ","n=",sum(sub_groups==i),", e=",sum(outcome_data[sub_groups==i]),",Prev=",format(round(prev*100,1),nsmall=1),"%, risk=",format(round(risk*100,1),nsmall=1),"%,\nexcess=",
                              format(round(prev*(risk-mean(risk_contributions$Baseline_risk))/total*100,1),nsmall=1),
                              "%, Obs risk=",format(round(risk_obs*100,1),nsmall=1),"% (",
                              paste0(format(round(prop.test(sum(outcome_data[sub_groups==i]),length(t(outcome_data)[sub_groups==i]))$conf.int*100,1),nsmall=1),collapse="-"),
                              "%)\n",
                              "Risk based on the sum of individual effects =",
                              format(round(mean(CoOL_6_sum_of_individual_effects(exposure_data,model)[sub_groups==i])*100,1),nsmall=1),
                              "%"),pos=4,col=colours[i])
  }
  m <- max(d)
  ind_effect_matrix <- CoOL_6_individual_effects_matrix(exposure_data,model)
  for(g in 1:ncol(d)) { for (i in 1:nrow(d)){
    value <- paste0(format(round(as.numeric(d[i,g])*100,1),nsmall=),"%\n(",
                    format(round(sd(risk_contributions[sub_groups==i,g])*100,1),nsmall=1),"%)\n[",
                    format(round(mean(ind_effect_matrix[sub_groups==i,g]*100),1),nsmall=1),"%]"
    )
    text(-g,-i,value,col=adjustcolor(colours[i],d[i,g]/m),cex=st*d[i,g]/m)
  }}
  }



#' Predict the risk based on the sum of individual effects
#'
#' By summing the through the risk as if each individual had been exposed to only one exposure, with the value the individual actually had.
#'
#' @param X The exposure data
#' @param model The fitted the monotonistic neural network
#' @export
#' @examples
#' #See the example under CoOL_0_synthetic_data

CoOL_6_sum_of_individual_effects <- function(X,model) {
# All individuals has the baseline risk
    sum_of_individial_effects = rep(as.vector(model[[4]][1,1]),nrow(X))
  # Loop through each exposure with the actual values by the individuals
    for (i in 1:ncol(X)) {
      X_temp <- as.data.frame(matrix(0,nrow = nrow(X), ncol=ncol(X)))
      X_temp[,i] <- X[,i]
      sum_of_individial_effects = sum_of_individial_effects +
        rowSums(relu(t(t(as.matrix(X_temp) %*% as.matrix(model[[1]])) + as.vector(model[[2]]))))
}
return(sum_of_individial_effects)
}


#' Risk contribution matrix based on individual effects (had all other exposures been set to zero)
#'
#' Estimating the risk contribution for each exposure if each individual had been exposed to only one exposure, with the value the individual actually had.
#'
#' @param X The exposure data
#' @param model The fitted the monotonistic neural network
#' @export
#' @examples
#' #See the example under CoOL_0_synthetic_data

CoOL_6_individual_effects_matrix <- function(X,model) {
  ind_effect_matrix <- as.data.frame(matrix(0,nrow = nrow(X), ncol=ncol(X)+1)) # +1 for the baseline risk
    # Loop through each exposure with the actual values by the individuals
  for (i in 1:ncol(X)) {
    X_temp <- as.data.frame(matrix(0,nrow = nrow(X), ncol=ncol(X)))
    X_temp[,i] <- X[,i]
    ind_effect_matrix[,i] <- rowSums(relu(t(t(as.matrix(X_temp) %*% as.matrix(model[[1]])) + as.vector(model[[2]]))))
  }
  # All individuals has the baseline risk
  ind_effect_matrix[,ncol(X)+1] <- rep(as.vector(model[[4]][1,1]),nrow(X))
  return(ind_effect_matrix)
}
