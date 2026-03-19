# Embedding Experiments
Accelerating embedding learning is priority #1 given that we have so few steps to achieve the lowest loss possible
## Embed Optimizer
Instead of using adamw for the embedding optimizer, let's use an optimizer that 
1. has a momentum buffer, and then takes the row-wise RMS to produce the update
2. Additionally, it should regularize the embedding if the size gets too big. Basically if rms >=1.25, normalize. You can do this in parallel
3. Because it's tied, we can add a scalar to the output projection head to capture scale if we need to

Then detach the embed lr from the rest if it's not already, and 

## RMS SSL
This one is complicated

## Untie