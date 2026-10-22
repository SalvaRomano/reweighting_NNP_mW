for i in N_*; do cd $i/rerun_NNP;  echo $i; mpirun -np 12 ~/n2p2-committee-modified/bin/lmp_mpi -in md.lmp; cd ../../; done
