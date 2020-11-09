# Imitation Learning

## Introduction

In this baseline we train a small squeezenet model on expert trajectories to simply clone the behavior of the expert.
Using only the expert trajectories would result in a model unable to recover from non-optimal positions; Instead, we use a technique called DAgger: a dataset aggregation technique with mixed policies between expert and model.

## Quick start

Use the jupyter notebook notebook.ipynb to quickly start training and testing the imitation learning Dagger.

## Detailed Steps

### Clone the repo

Clone this [repo](https://github.com/duckietown/gym-duckietown):

$ git clone https://github.com/duckietown/gym-duckietown.git

$ cd gym-duckietown

### Installing Packages

$ pip3 install -e .

## Training

$ python -m learning.train

### Arguments

* `--episode` or `-i` an integer specifying the number of episodes to train the agent, defaults to 10.
* `--horizon` or `-r` an integer specifying the length of the horizon in each episode, defaults to 64.
* `--learning-rate` or `-l` integer specifying the index from the list [1e-1, 1e-2, 1e-3, 1e-4, 1e-5] to select the learning rate, defaults to 2.
* `--decay` or `-d` integer specifying the index from the list [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95] to select the initial probability to choose the teacher, the learner.
* `--save-path` or `-s` string specifying the path where to save the trained model, models will be overwritten to keep latest episode, defaults to a file named iil_baseline.pt on the project root.
* `--map-name` or `-m` string  specifying which map to use for training, defaults to loop_empty.
* `--num-outputs` integer specifying the number of outputs the model will have, can be modified to train only angular speed, defaults to 2 for both linear and angular speed.
* `--domain-rand` or `-dr` a flag to enable domain randomization for the transferability to real world from simulation.
* `--randomize-map` or `-rm` a flag to randomize training maps on reset.

## Testing

$ python -m learning.test

### Arguments

* `--model-path` or `-mp` string specifying the path to the saved model to be used in testing.
* `--episode` or `-i` an integer specifying the number of episodes to test the agent, defaults to 10.
* `--horizon` or `-r` an integer specifying the length of the horizon in each episode, defaults to 64.
* `--save-path` or `-s` string specifying the path where to save the trained model, models will be overwritten to keep latest episode, defaults to a file named iil_baseline.pt on the project root.
* `--num-outputs` integer specifying the number of outputs the model has, defaults to 2.
* `--map-name` or `-m` string  specifying which map to use for training, defaults to loop_empty.

## Submitting 
* Copy trained model files into submission/models directory and then use [duckietown shell](https://github.com/duckietown/duckietown-shell) to submit. 
* For more information on submitting check [duckietown shell documentation](https://docs.duckietown.org/DT19/AIDO/out/cli.html).

## Acknowledgment

* We started from previous work done by Manfred Díaz as a boilerplate, and we would like to thank him for his full support with code and answering our questions.

## Authors

* [Mostafa ElAraby ](https://www.mostafaelaraby.com/)
  + [Linkedin](https://linkedin.com/in/mostafaelaraby)
* Ramon Emiliani
  + [Linkedin](https://www.linkedin.com/in/ramonemiliani)

## References

``` 

@phdthesis{diaz2018interactive,
  title={Interactive and Uncertainty-aware Imitation Learning: Theory and Applications},
  author={Diaz Cabrera, Manfred Ramon},
  year={2018},
  school={Concordia University}
}

@inproceedings{ross2011reduction,
  title={A reduction of imitation learning and structured prediction to no-regret online learning},
  author={Ross, St{\'e}phane and Gordon, Geoffrey and Bagnell, Drew},
  booktitle={Proceedings of the fourteenth international conference on artificial intelligence and statistics},
  pages={627--635},
  year={2011}
}

@article{loquercio2018dronet,
  title={Dronet: Learning to fly by driving},
  author={Loquercio, Antonio and Maqueda, Ana I and Del-Blanco, Carlos R and Scaramuzza, Davide},
  journal={IEEE Robotics and Automation Letters},
  volume={3},
  number={2},
  pages={1088--1095},
  year={2018},
  publisher={IEEE}
}
```
