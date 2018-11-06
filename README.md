# Tensorflow Tutor Code

This repo contains the tutor code for tensorflow beginners. The documentation is at [this link](https://docs.google.com/document/d/16nMMqGOytPgfUZRUFAmBeR-Gec2_t_6DAsVRqhrJbFE/edit?usp=sharing).

The test environment is 
```
python 2.7.14
tensorflow 1.5.0
```
Clone or fork the code to your own machine. Type in
```
python mnist.py
```
It will automatically train the model and print out loss values and test accuracies. Feel free to change the code and play around with it.

Have fun!

## For dll-0x Server Users (Author: Yuwei Hu)

1. Try with the default environment. It should work. There is one Tesla p100 on dll-00, and four GTX 1080 Ti on dll-01. The CUDA version is 9.0 on both machines.
```
python 2.7.14
tensorflow 1.6.0
CUDA 9.0
```

2. If the tutor code can not run in the default environment, go through the following steps.

Step 1: Connect to the GPU server.
Type the following command in the terminal. Replace ``<username>`` with your own username. It should be your Cornell ID.
```
ssh <username>@dll-00.ece.cornell.edu
```
or
```
ssh <username>@dll-01.ece.cornell.edu
```

Step 2: Configure the environment.
Run the following commands to configure environment variables:
```
export PATH=/usr/local/cuda-9.0/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:${LD_LIBRARY_PATH}
```

Step 3: Run the example.
Now, if you want to run some example codes, they are at ``/usr/local/cuda-9.0/samples/``
Copy the samples to your personal folder first:
```
cp -rp /usr/local/cuda-9.0/samples/ ~/cuda_samples/
```
Then, cd to a folder that contains a specific example you want to run.
```
cd ~/cuda_samples/<path_to_the_sample>
```
A good one to start is ``deviceQuery``, which shows you the properties of the GPU and whether it's working properly. Find it at ``~/cuda_samples/1_Utilities/deviceQuery/``, and type in
```
make
```
The example will be compiled. Run the excutable file using
```
./<name_of_the_binary>
```

## Authors

* **Yichi Zhang** -  [Web](http://zhang.ece.cornell.edu/people/yichi-zhang/)
