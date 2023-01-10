args = commandArgs(trailingOnly=TRUE)  # order
# args = 2
library(sensitivity)

X1 = read.csv('X1.csv')
X2 = read.csv('X2.csv')
xx = sobol(model=NULL, X1, X2, order=strtoi(args[1]), nboot=100)
gsa_runs = read.csv('gsa_runs.csv')
tell(xx, as.matrix(gsa_runs))
Sindexes = data.frame(xx$S$original)
colnames(Sindexes) = 1
Sindexes2 = data.frame(t(Sindexes))
write.csv(Sindexes2, 'S_indexes.csv', row.names = FALSE)
