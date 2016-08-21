### To run

You need to install tensorflow, numpy and scipy

To install tensorflow and numpy, follow this guide:
[tensorflow](https://www.tensorflow.org/versions/r0.10/get_started/os_setup.html)

It is suggested to install the system on Virtualenv
[Virtualenv](https://virtualenv.pypa.io/en/stable/)

To run the system, and if you are using Vritualenv, you have to execute the following command:
```
path to virtualenv installation/bin/activate
```
Then you should run the FNN_learning.py file in FNN_python folder:
Execute the following
```
python FNN_learning.py
```
### Tensorboard
To view the training data, run the following code:
tensorboard --logdir=log path
the log path is the one that includes the log files, in this case, it is dqn_log_files
