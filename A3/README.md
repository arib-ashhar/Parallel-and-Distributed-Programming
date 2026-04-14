module purge
module load compiler/gcc/9.1/mpich/3.3.1

make clean
make

mpirun -np 16 ./main input_report.txt output.txt
bash benchmark.sh
