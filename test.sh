for t in 1 2 4 8
do
  export OMP_NUM_THREADS=$t
  mpirun -np 4 ./main
done