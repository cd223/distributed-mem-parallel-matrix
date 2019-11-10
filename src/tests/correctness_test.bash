#!/bin/bash

# Initialise variables
prec=0.1
repeat=25
nprocs=(2 5 10 15 20 30 50)
dims=(10 20 50 100 200 500 1000 2000)
pass=1

for dim in "${dims[@]}"
do
    # Set expected output for current dimension (using n=1 as benchmark)
    mpirun -np 1 ./distributedrelax -d $dim -p $prec -i > in.txt

    # Check output matched by increasing number of processes
    for ncount in "${nprocs[@]}"
    do
        # Repeat each test to ensure some confidence in reproducibility
        for (( i=1; i<=$repeat; i++ )) 
        do  
            mpirun -np $ncount ./distributedrelax -d $dim -p $prec -i > out.txt

            # If files differ, failure!
            if [[ $(diff in.txt out.txt) ]]; then 
                pass=0
                echo "FAILED TEST for d=$dim, n=$ncount"
            fi
        done
        # Else if no trials fail, print success for this test case
        if [ "$pass" -eq "1" ]; then
            echo "PASSED TEST for d=$dim, n=$ncount"
        fi
    done
done

if [ "$pass" -eq "1" ]; then
   echo "All passed.";
   exit;
fi