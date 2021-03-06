import numpy as np
import numpy.random as rand
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

def build_trans_mat_gridworld():
  # 5x5 gridworld laid out like:
  # 0  1  2  3  4
  # 5  6  7  8  9
  # 10 11 12 13 14
  # 15 16 17 18 19
  # 20 21 22 23 24
  # where 24 is a goal state that always transitions to a
  # special zero-reward terminal state (25) with no available actions
  trans_mat = np.zeros((26,4,26))

  # NOTE: the following iterations only happen for states 0-23.
  # This means terminal state 25 has zero probability to transition to any state,
  # even itself, making it terminal, and state 24 is handled specially below.

  # Action 0 = down
  for s in range(24):
    if s < 20:
      trans_mat[s,0,s+5] = 1
    else:
      trans_mat[s,0,s] = 1

  # Action 1 = up
  for s in range(24):
    if s >= 5:
      trans_mat[s,1,s-5] = 1
    else:
      trans_mat[s,1,s] = 1

  # Action 2 = left
  for s in range(24):
    if s%5 > 0:
      trans_mat[s,2,s-1] = 1
    else:
      trans_mat[s,2,s] = 1

 # Action 3 = right
  for s in range(24):
    if s%5 < 4:
      trans_mat[s,3,s+1] = 1
    else:
      trans_mat[s,3,s] = 1

  # Finally, goal state always goes to zero reward terminal state
  for a in range(4):
    trans_mat[24,a,25] = 1

  return trans_mat



def calcMaxEntPolicy(trans_mat, horizon, r_weights, state_features):
  """
  For a given reward function and horizon, calculate the MaxEnt policy that gives equal weight to equal reward trajectories

  trans_mat: an S x A x S' array of transition probabilites from state s to s' if action a is taken
  horizon: the finite time horizon (int) of the problem for calculating state frequencies
  r_weights: a size F array of the weights of the current reward function to evaluate
  state_features: an S x F array that lists F feature values for each state in S

  return: an S x A policy in which each entry is the probability of taking action a in state s
  """
  n_states = np.shape(trans_mat)[0]
  n_actions = np.shape(trans_mat)[1]
  policy = np.zeros((n_states,n_actions))

  za = np.zeros([n_states, n_actions])
  zs = np.zeros(n_states)
  zs[n_states-1] = 1
  for i in range(horizon):
    za = np.sum(trans_mat * np.expand_dims(np.expand_dims(np.exp(np.dot(state_features, r_weights)), axis=-1), axis=-1) * zs, axis=-1)
    zs = (np.sum(za, axis=-1))
    zs[n_states-1] += 1

  policy = za/np.expand_dims(zs, axis=-1)

  return policy

def calcExpectedStateFreq(trans_mat, horizon, start_dist, policy):
  """
  Given a MaxEnt policy, begin with the start state distribution and propagate forward to find the expected state frequencies over the horizon

  trans_mat: an S x A x S' array of transition probabilites from state s to s' if action a is taken
  horizon: the finite time horizon (int) of the problem for calculating state frequencies
  start_dist: a size S array of starting start probabilities - must sum to 1
  policy: an S x A array array of probabilities of taking action a when in state s

  return: a size S array of expected state visitation frequencies
  """

  state_freq = np.zeros(len(start_dist))
  # state_freq = np.copy(start_dist)
  dist = np.copy(start_dist)
  for i in range(horizon):
    state_freq += dist
    dist = np.sum(np.sum(np.expand_dims(np.expand_dims(dist, axis=-1) * policy, axis=-1) * trans_mat, axis=1), axis=0)
  return state_freq



def maxEntIRL(trans_mat, state_features, demos, seed_weights, n_epochs, horizon, learning_rate):
  """
  Compute a MaxEnt reward function from demonstration trajectories

  trans_mat: an S x A x S' array that describes transition probabilites from state s to s' if action a is taken
  state_features: an S x F array that lists F feature values for each state in S
  demos: a list of lists containing D demos of varying lengths, where each demo is series of states (ints)
  seed_weights: a size F array of starting reward weights
  n_epochs: how many times (int) to perform gradient descent steps
  horizon: the finite time horizon (int) of the problem for calculating state frequencies
  learning_rate: a multiplicative factor (float) that determines gradient step size

  return: a size F array of reward weights
  """

  n_features = np.shape(state_features)[1]
  r_weights = np.zeros(n_features)
  n_state = np.shape(state_features)[0]
  n_demo = len(demos)
  start_dist = np.zeros(n_state)

  # compute initial distribution and average feature
  demo_len, f_tilde = 0, 0
  for i, demo in enumerate(demos):
    start_dist[demo[0]] += 1
    demo_len += len(demo)
    for s in demo:
        f_tilde += state_features[s]
  f_tilde /= n_demo
  start_dist /= n_demo

  # plot for step size
  plot_reward = False
  reward_list = []
  step_size_list = []

  for i in range(n_epochs):
    for j in range(horizon):
      # compute maxent policy
      max_ent_pol = calcMaxEntPolicy(trans_mat, horizon, r_weights, state_features)
      # compute state frequency
      state_freq = calcExpectedStateFreq(trans_mat, horizon, start_dist, max_ent_pol)

    # compute gradient of log likelihood
    g = f_tilde - np.sum(np.expand_dims(state_freq, axis=-1) * state_features, axis=0)
    # r_weights += learning_rate * g

    # line search
    r_old = 0
    for demo in demos:
        r_old += computeReward(demo, r_weights, state_features)
    step_size = learning_rate
    for i in range(10):
        new_weights = r_weights + step_size * g
        r_new = 0
        for demo in demos:
            r_new += computeReward(demo, new_weights, state_features)
        if r_old < r_new:
            print("Step Size Ok")
            break
        else:
            print("Step Size Shrink")
            step_size *= 0.5
    step_size_list.append(step_size)
    reward_list.append(r_new)
    r_weights = new_weights

  # plot
  if plot_reward:
      plt.subplot(2,1,1)
      plt.plot(reward_list, '-*', markersize=5, c='b')
      plt.ylabel('Reward')
      plt.subplot(2,1,2)
      plt.plot(step_size_list, '-*', markersize=5, c='b')
      plt.ylabel('Step Size')
      plt.xlabel('iters')
      plt.show()

  return r_weights

def computeReward(demo, weight, state_features):
    rew = 0
    for s in demo:
        rew += np.dot(state_features[s], weight)
    return rew

if __name__ == '__main__':

  # Build domain, features, and demos
  trans_mat = build_trans_mat_gridworld()
  state_features = np.eye(26,25)  # Terminal state has no features, forcing zero reward
  demos = [[0,1,2,3,4,9,14,19,24,25],
          [0,5,10,15,20,21,22,23,24,25],
          [0,5,6,11,12,17,18,23,24,25],
          [0,1,6,7,12,13,18,19,24,25]]

  # Appen demos to prevent underfitting
  add_demo = False
  if add_demo:
      demos.append([0,5,10,11,16,17,18,23,24,25])
      demos.append([0,1,6,11,16,21,22,23,24,25])

  seed_weights = np.zeros(25)

  # Parameters
  n_epochs = 100
  horizon = 10
  learning_rate = 1.0

  # Main algorithm call
  r_weights = maxEntIRL(trans_mat, state_features, demos, seed_weights, n_epochs, horizon, learning_rate)

  # Print out reward parameters
  print("Reward")
  print("-----------------------------")
  for i in range(5):
      print("%.2f|%.2f|%.2f|%.2f|%.2f"%(r_weights[5*i],
                                        r_weights[5*i+1],
                                        r_weights[5*i+2],
                                        r_weights[5*i+3],
                                        r_weights[5*i+4]))
  print("-----------------------------")

  # Construct reward function from weights and state features
  reward_fxn = []
  for s_i in range(25):
    reward_fxn.append( np.dot(r_weights, state_features[s_i]) )
  reward_fxn = np.reshape(reward_fxn, (5,5))

  # Plot reward function
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  X = np.arange(0, 5, 1)
  Y = np.arange(0, 5, 1)
  X, Y = np.meshgrid(X, Y)
  surf = ax.plot_surface(X, Y, reward_fxn, rstride=1, cstride=1, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
  plt.show()
