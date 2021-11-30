"
Este script estima la red de interdependencias entre indicadores.
"


library("sparsebn")
library("stringi")




filep = "ruta_de_acceso/data/red/changes.csv"
print(filep)
S <- t(data.matrix(read.csv(filep, header=FALSE, sep = ",")))
data <- sparsebnData(S, type = "continuous")
dags.estimate <- estimate.dag(data)
dags.param <- estimate.parameters(dags.estimate, data=data)
selected.lambda <- select.parameter(dags.estimate, data=data)
dags.final.net <- dags.estimate[[selected.lambda]]
dags.final.param <- dags.param[[selected.lambda]]
adjMatrix <- dags.final.param$coefs
write.table(as.matrix(adjMatrix), file="ruta_de_acceso/data/red/sparsebn.csv", row.names=FALSE, col.names=FALSE)



















