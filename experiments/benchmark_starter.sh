for problem_dim in {1..5}; do
    for file in $(ls runners/); do
        if [ "${file: -3}" == ".py" ]
        then
            python3 runner.py runners.${file%???} 20 data.random_points 0.1 $problem_dim
        fi
    done
    echo "Batch $problem_dim finished."
done
