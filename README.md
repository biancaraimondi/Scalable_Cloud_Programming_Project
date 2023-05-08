# Scalable and Cloud Programming Project

Questa repository contiene il codice sorgente per l'esame relativo alla materia "Scalable and Cloud programming"
effettuato nell'anno accademico 2022-23.
L'obiettivo del progetto consiste nella implementazione di due algoritmi di Machine Learning con il pattern MapReduce;
dopo aver effettuato l'implementazione di tali algoritmi, gli eseguibili vengono pubblicati sulla piattaforma
Google Cloud Platform per venire eseguiti in cloud.

## Modelli implementati

I modelli di Machine Learning che abbiamo deciso di implementare sono due:

- Gaussian Naive Bayes
- K-Nearest Neighbours

Entrambi i modelli si prestano bene all'impiego del pattern map-reduce, e beneficiano molto dell'uso della
parallelizzazione

## Deployment su GCP

Per effettuare il deployment del progetto su Google Cloud Platofrm abbiamo eseguito i seguenti step:

1. Effettuato il build del progetto mediante il tool `sbt`.
2. Creato un bucket su GCP, nel quale abbiamo inserito il dataset interessato e l'eseguibile `.jar`.
3. Creato un cluster su Google Dataproc.
4. Eseguito multipli job per testare diverse configurazioni ed effettuare cross validation dei risultati anche
   successivi ad un cold boot della machina remota.

Infine, è importante notare che, prima di eseguire gli step precedentemente esposti, ci siamo assicurati che le versioni
di Scala e Spark utilizzate su GCP corrispondessero a quelle utilizzate su ambienti locali.