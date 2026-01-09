# Poor Person’s Neural Network
### Projet de programmation parallèle MPI / OpenMP

**Étudiant :** Thomas  
**Niveau :** M2  
**Sujet :** Poor person’s neural network  

---

## 1. Objectif du projet

L’objectif de ce projet est d’implémenter le *forward pass* d’un réseau de neurones simple en utilisant une approche hybride MPI + OpenMP.  
Le but n’est pas d’entraîner un modèle, mais d’illustrer comment les calculs internes d’un réseau de neurones peuvent être parallélisés efficacement sur une architecture HPC.

---

## 2. Modèle de réseau de neurones

Le réseau implémenté est un réseau entièrement connecté composé de :
- une couche d’entrée
- une couche cachée
- une couche de sortie

Chaque neurone calcule une somme pondérée suivie d’une fonction d’activation ReLU.  
Aucune phase d’apprentissage ou de rétropropagation n’est implémentée, conformément aux consignes du projet.

---

## 3. Stratégie de parallélisation

### MPI
MPI est utilisé pour répartir le calcul de la couche cachée entre plusieurs processus.  
Chaque processus MPI calcule un sous-ensemble des neurones cachés. Les résultats partiels sont combinés à l’aide de `MPI_Allreduce`.

### OpenMP
À l’intérieur de chaque processus MPI, OpenMP est utilisé pour paralléliser les boucles de calcul (produits matrice-vecteur).

Le modèle de threading utilisé est **MPI_THREAD_FUNNELED**, ce qui garantit que seuls les threads maîtres effectuent des appels MPI.

---

## 4. Mesure des performances

Le temps d’exécution est mesuré avec `MPI_Wtime()` et sauvegardé automatiquement dans un fichier `results.csv`.  
Chaque ligne du fichier contient :
- le nombre de processus MPI
- le nombre de threads OpenMP
- le temps d’exécution

---

## 5. Compilation et exécution

### Compilation
```bash
mpicxx -fopenmp main.cpp -o main
