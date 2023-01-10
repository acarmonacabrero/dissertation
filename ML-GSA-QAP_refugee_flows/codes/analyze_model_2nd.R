args = commandArgs(trailingOnly=TRUE)
library(sensitivity)

X1 = read.csv('X1.csv')
X2 = read.csv('X2.csv')
xx = sobolEff(model=NULL, X1, X2, order=2, nboot=100, conf=0.95)
gsa_runs = read.csv('gsa_runs.csv')
gsa_runs2 = as.matrix(gsa_runs)
tell(xx, gsa_runs2)
Sindexes = data.frame(xx$S)

Sindexes2 = data.frame(t(Sindexes))
# colnames(Sindexes2) = 0:54
write.csv(Sindexes2, 'S_indexes.csv')
