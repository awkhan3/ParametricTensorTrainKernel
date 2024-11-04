#!/bin/bash 

# Remove the existing experiments_out directory if it exists
rm -r experiments_out

# Create the experiments_out directory if it doesn't exist
mkdir -p experiments_out


# Loop to process experiments from 1 to 5
for k in {1..5}
do
  # Copy the experiment file into the current directory
  cp experiments/experiment_${k}.py .

  # Run the Python script from the current directory
  python experiment_${k}.py > experiments_out/experiment_${k}_result.txt

  # Delete the experiment file after running it
  rm -r experiment_${k}.py

  # Print message to indicate completion
  echo "Ran experiment_${k}.py, output saved to experiments_out/experiment_${k}_result.txt and the script was deleted."
done


