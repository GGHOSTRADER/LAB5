
Plain Vanilla DQN implementation on Atari


Goal:  Use Plain vanilla DQN to train for defeating the opponent by bouncing the ball past them

Constrains
- My machine has core i9-13900HX,32gb ram and geforce RTX 4060 GPU
- Must use CNN as Q-Function Approximator
- Must do preprocessing of the input frames (Grayscale, resize and stack frames)
- Suggest number of hidden layers for DQN according to machine
- use epsilon-greedy policy
- implement Experience Replay with uniform sampling
- log and evaluate performance while training
- Accordingly, to get full score for this part, your DQN shall steadily achieve an average score above 19
(over 20 evaluation episodes)
- All the modules or classes (listed below) are already provided in the file dqn.py. Please do not change
the structure of the modules.

Model
actions: 0: Noop , 1: Fire , 2:Right , 3: Left , 4: RightFire , 5:LeftFire
Reward: +1 agent scores , -1 opponent scores
State: 210x160 RBG Image

Breakdwon

2.2 Task 2: Vanilla DQN with Visual Observations on Atari
Goal: Extend your DQN implementation to work on high-dimensional visual input using the Pong-v5 environment from the Arcade Learning Environment suite, also known as Atari. You can find the detailed definitions of
the states, actions, and rewards of each environment at the official website https://ale.farama.org/index.
html.
Requirements:
• Preprocess the input frames (grayscale, resize, and stack frames)
• Use a convolutional neural network (CNN) as the Q-function approximator
• Evaluate and plot the total episodic rewards versus environment steps (preferably via Weight and Bias)