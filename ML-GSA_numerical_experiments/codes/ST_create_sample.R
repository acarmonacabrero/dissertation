args = commandArgs(trailingOnly=TRUE)  # funct, n_gsa, order, number_of_inputs
# args = c('ishigami', '6000', '2', '3')
# args = c('sobolG', '6000', '8')
library(sensitivity)
library(lhs)
funct = args[1]
n_gsa = strtoi(args[2])
k = strtoi(args[3])
order=0

RSsobol = function(funct, n_gsa, order=0, k){
  if (funct == 'ishigami'){
    # set.seed(strtoi(seed)*var)
    X1 = data.frame(matrix(runif(k*n_gsa, min=-pi, max=pi), nrow=n_gsa))
    # set.seed(strtoi(seed)*var)
    X2 = data.frame(matrix(runif(k*n_gsa, min=-pi, max=pi), nrow=n_gsa))
  }
  if (funct == 'sobolG'){
    # set.seed(strtoi(seed)*var)
    X1 = data.frame(matrix(runif(k*n_gsa, min=0, max=1), nrow=n_gsa))
    # set.seed(strtoi(seed)*var)
    X2 = data.frame(matrix(runif(k*n_gsa, min=0, max=1), nrow=n_gsa))
  }
  write.csv(X1, 'X1.csv', row.names = FALSE)
  write.csv(X2, 'X2.csv', row.names = FALSE)
  xx = sobolEff(model=NULL, X1, X2, order=order, nboot=100)
  return(xx)
}


xx = RSsobol(funct, n_gsa, order, k)

gsa_sample = xx$X

if (funct == 'ishigami'){
  colnames(gsa_sample) = c('X1', 'X2', 'X3')
}

if (funct == 'sobolG'){
  c_names = c()
  for (i in 1:k){
    c_names = append(c_names, paste('X', i, sep=''))
  }
  colnames(gsa_sample) = c_names
}

write.csv(gsa_sample, row.names=FALSE, 'gsa_sample.csv')
