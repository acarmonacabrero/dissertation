args = commandArgs(trailingOnly=TRUE)  # year, names, run_id, LHS/RS, n_gsa_runs
# args = c('1991', ' ', '1', 'LHS', '3')
library(sensitivity)
library(lhs)

RSsobolSample = function(data=data_gsa[, -1], n, order=2, type=sobolEff){
  # Type can be sobol, sobol2007 or sobolEff
  X1 = data.frame(matrix(nrow=n, ncol=dim(data)[2]))
  X2 = data.frame(matrix(nrow=n, ncol=dim(data)[2]))
  var = 0
  for (i in 1:(dim(data)[2])){
    var = var + 1
    set.seed(strtoi(args[3])*var)
    xhist = hist(data[, var], freq=FALSE)
    bins = with(xhist, sample(length(mids), n, p=density, replace=TRUE)) # choose a bin
    X1[,var] = runif(length(bins), xhist$breaks[bins], xhist$breaks[bins+1]) # sample a uniform in it
    set.seed(strtoi(args[3])*var*20)
    xhist = hist(data[, var], freq=FALSE)
    bins = with(xhist, sample(length(mids), n, p=density, replace=TRUE)) # choose a bin
    X2[,var] = runif(length(bins), xhist$breaks[bins], xhist$breaks[bins+1]) # sample a uniform in it
    write.csv(X1, 'X1.csv', row.names = FALSE)
    write.csv(X2, 'X2.csv', row.names = FALSE)
  }
  xx = sobolEff(model=NULL, X1, X2, order=order, nboot=100, conf=0.95)
  return(xx)
} 

empirical_RSsobolSample = function(data, n, order=2, type=sobolEff){
  # Type can be sobol, sobol2007 or sobolEff
  X1 = data.frame(matrix(nrow=n, ncol=dim(data)[2]))
  X2 = data.frame(matrix(nrow=n, ncol=dim(data)[2]))
  var = 0
  for (i in 1:(dim(data)[2])){
    var = var + 1
    set.seed(strtoi(args[3])*var)
    X1[, var] = sample(data[, var], n, replace=TRUE)
    set.seed(strtoi(args[3])*var*2)
    X2[, var] = sample(data[, var], n, replace=TRUE)
    write.csv(X1, 'X1.csv', row.names = FALSE)
    write.csv(X2, 'X2.csv', row.names = FALSE)
  }
  xx = sobolEff(model=NULL, X1, X2, order=order, nboot=100, conf=0.95)
  return(xx)
} 

empirical_LHSsobolSample = function(data, n, order=2, type=sobol){
  # Type can be sobol, sobol2007 or sobolEff
  k = dim(data)[2]
  N = dim(data)[1]
  for (i in 1:n){
    if (i==1){
      set.seed(strtoi(args[3]))
      lhs_r = sample(1:N)
      
      set.seed((strtoi(args[3]) + 1)*2)
      lhs_r2 = sample(1:N)
      
      lhs_sample = data[lhs_r, ]
      lhs_sample2 = data[lhs_r2, ]
    } else {
      set.seed(strtoi(args[3]) * i * 10000)
      lhs_r = sample(1:N)
      
      lhs_sample_i = data[lhs_r, ]
      
      set.seed((strtoi(args[3]) + 1) * i * 20000)
      lhs_r2 = sample(1:N)
      
      lhs_sample_i2 = data[lhs_r2, ]
      
      
      lhs_sample = rbind(lhs_sample, lhs_sample_i)
      lhs_sample2 = rbind(lhs_sample2, lhs_sample_i2)
    }
  }
  write.csv(lhs_sample, 'X1.csv', row.names = FALSE)
  write.csv(lhs_sample2, 'X2.csv', row.names = FALSE)
  xx = sobolEff(model=NULL, lhs_sample, lhs_sample2, order=order, nboot=100, conf=0.95)
  return(xx)
}

# n_lhs = args[5]  # LHS
# n_rs = args[5]  # RS

data = read.csv('year_permutation.csv')
X = data
X$target = NULL
Y = data$target
options(scipen=99999)

k = dim(X)[2]

if (args[4] == 'LHS'){
  n = args[5]  # LHS
  xx = LHSsobolSample(X, order=1, strtoi(args[5]), type=sobolEff)
}

if (args[4] == 'RS'){
  n = args[5]  # RS
  xx = RSsobolSample(X, order=1, strtoi(args[5]), type=sobolEff)
}

if (args[4] == 'eRS'){
  n = args[5]  # eRS
  xx = empirical_RSsobolSample(X, order=1, strtoi(args[5]), type=sobolEff)
}

gsa_sample = xx$X
# dim(gsa_sample)
colnames(gsa_sample) = colnames(X)
write.csv(gsa_sample, row.names=FALSE, 'gsa_sample.csv')

