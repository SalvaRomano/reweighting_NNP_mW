for i in N*/rho*; do cd $i; 
	mpirun -np 12 ~/n2p2-committee-modified/bin/lmp_mpi -in md.lmp	
cd ../../; done
