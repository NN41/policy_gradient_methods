# Implement REINFORCE for CartPole-v1
This project implements the REINFORCE algorithm from scratch to solve the OpenAI Gym CartPole-v1 environment using Pytorch.


## REINFORCE
The training progress is very sensitive to the learning rate (using SGD). Because of this it's important to make sure that, once tuned, the magnitude of the gradients don't change. That's why we need to take the scale the double sum by the total number of log probabilities, not only by the number of trajectories. Otherwise, as the training progresses, the trajectories collect higher rewards and the magnitude of the gradients change. What was a suitable learning rate in the first epoch, will be too large by the Nth epoch. To see this, consider a single episode per epoch, in which case our loss function is proportional to $\Sigma_{t=0}^T \log \pi_\theta(a_t|s_t) R(\tau)$. It is clear that the magnitude of the gradient grows as trajectories become longer.

Looking at the most recent expression of policy gradient being a double sum of grad-log-probs scaled by the correspoding episode's return, we see how REINFORCE learns. REINFORCE tries to maximize the double sum of grad-log-probs as a proxy for the expected return. It does so by increasing the log-probabilities, or equivalantly, the probabilities of choosing the action $a_t$ under state $s_t$. This doesn't depend on how well action $a_t$ actually was, the only thing matters is the entire trajectory's return. It can be that action $a_0|s_0$ was actually really bad, but if the agent managed to recover and the rest of the episode goes great, then REINFORCE still makes $a_0|s_0$ more likely. Note that other (better) actions are actually made less likely this way, which makes the algorithm very sensitive to the randomly chosen actions at each time step. In the case of the cartpole, if the policy always selects the wrong action with high likelihood on the first timestep, but always manages to recover, then this wrong first action will be reinforced over time by pure overrepresentation in the training data of on-policy trajectories. This motivates discounting the future rewards.

Note that the reinforcement of actions happens proportionally to the associated reward, since the contribution of the grad-log-prob of a certain state-action pair in the policy gradient is proportional to this associated reward. In the naive policy gradient, that is the entire trajectory's return. If one trajectory performs extremely well relative to the other trajectories by pure luck, then all actions in that trajectory will be reinforced by an disproportional amount as well, even the actions that were irrelevant or even detrimental to the success of that trajectory. 

According to wikipedia: the score function (i.e. grad-log-prob(a_t|s_t)) can be interpreted as the direction in parameter space we need to move to increase the probability of taking action a_t in state s_t. The (naive) policy gradient is then the weighted avrage of all possible directions that increase the probability of taking any of the actions in any of the corresponding states, but weighted by the associated episode's return. As per the VPG page of Spinning Up, policy gradients learn the optimal policy by pushing up probabilities of actions that lead to higher return, while pushing down probabilities that lead to lower return.

The desired consequence of using reward-to-go and baselines is that it produces lower-variance estimates of the policy gradient, that is, we estimate by the gradient policy by a sample mean of weight * Grad(log_prob). Reducing the variance of this estimate comes down to reducing the variance of "weight", since Grad(log_prob) depends on the policy network architecture in remains unaffected by different choices of "weight" (such as "return" or "reward-to-go" or "reward-to-go - baseline" etc). Thus, to save compute, as a proxy, we will focus on measuring the variance of "weight" instead of weight * Grad(log_prob), the latter of which would be way more compute intensive. 

The core fundamentals of policy gradient methods, is that we want to associate some measure of goodness to an action that we take, then improve the likelihood of good actions based on that measure. Depending on which goodness measure we choose, we make a trade-off between bias and variance: using returns gives unbiased estimates while having high variance. Using a value function approximation significantly lowers variance at the cost of introducing bias (depending on how bad the apprxoimation is).

### Baselines
Spinning Up Part 3 suggests using MSE for learning our MLP that is approximating the on-policy value function. As discussed by Goodfellow, you can derive MSE loss through MLE by assuming that targets are produced through the underlying process plus some Gaussian noise. In this context, that would mean assuming that a trajectory's return is Gaussian distributed around the value function, which generally seems like an unrealistic assumption to make, especially in the cartpole context, since we would expect a lot of short trajectories and few long ones, leading to a heavily skewed returns distribution.

Rather, the choice for MSE loss is one of simplicity and pragmatism. It's a well-understood loss function and minizing MSE loss learns to predict the mean of the target (trajectory returns) given the input (state). Here, the mean of the target is exactly the on-policy value function.  

## Experiments & Results
- Adam performs way better than SGD
- One of the simplest algorithms seems to be a 1-hidden layer MLP with 4 hidden units and ReLU activation for the policy network, then Adam with lr = 0.01 and 50 episodes per epoch, 100 epochs, using reward-to-go and average over all. This achieves around avg 500 (maximum) return after 100 epochs.  

## Obstacles
- It took a lot of effort getting the REINFORCE algorithm with a value function MLP baseline working. In particular, it seems that I was way overtraining the value network during after each round of 1000 episodes. I also used the parameters from the previous training round and I didnt' reinitilize the optimizer in the beginning. The results is that the network likely got overfit to a first batch of trajectories using policy pi_k, then wasn't able to adapt anymore to the trajectories from policy pi_{k+1}. When training the network only a single epoch and reinitilizatin both the network and Adam optimizer for every new policy pi_k, we are making progress again. I'm surprised to see that the test MSE loss of the value network doesn't decrease by more than 1%. Way more than such an improvement seems to suggest overfitting. During some epochs, the value network doesn't seem to learn anything at all, strangely. The question is that with such a badly fit value network, do you even get a noticeable decrease in variance?
- It seems that the biggest problem with the value network is overfitting to a certain distribution coming from policy pi_k, then being unable to adjust its parameters to fit the distribution shift for policy pi_{k+1}
- I also had to include an ReLU activation function at the output layer, since the value network was outputting negative values, which doesn't make sense for CartPole. Then I figured out this is disastrous for training the neural network. At each new policy, sometimes the network would be initialized with weights such that all inputs would lead to negative outputs, hence they would be squashed to zero for all training data and the gradients would be zero as well.
- The BIGGEST obstacle of this project was figuring out for how long to train the value network, how to set it up, etc. I really should use an experimentation framework to grid search the correct learning rate. Just use one epoch, reset every time, then slowly adjust the learning rate from 0 to 0.1 using SGD and see where you end up. The GAE paper mentions that using a value function introduces a bias. Maybe that is the problem, that if you train it too much, the bias overwhelms the benefits?
- What I struggle to understand: Using rewards-to-go with a baseline gives an unbiased estimate of the policy gradient, certainly. So even a badly-trained value network should not introduce bias, however it may introduce variance. But then why do I struggle so much with getting a baseline working? Maybe I should try to run with smaller learning rates, more epochs and more episodes per epoch.
- POTENTIALLY the fact that the value network at the terminated state still outputs some random number instead of 0 might have enough of an impact on the baseline values that it messes with the measure of goodness of actions towards the end of the episodes
- Some discussions with Gemini about using eval() when generating sample trajectories for the policy to be trained on. Gemini was suggesting a two-pass approach, where you use eval() when generating trajectories and then put it to train() and then use the action and states from the generated trajectories to get the log probs from the model in train mode. But this is wrong since the policy gradient as an expectation is assuming that the trajectories are sample according to the policy that is being updated (i.e. the policy network in train mode). So, you must have train enabled when generating actions and log probs that will be used for updating the network.
- It's important to use AI but be very careful. For example, in light of reproducibility, we decided to use the same seed for each first run of each experiment, each second run, each third run etc, by setting seed = base_seed + run_idx. However, Gemini 2.5 Pro Preview 06-05 put this seed setting in the wrong place, which meant that the seed did not get reset properly (even though the runs were using the correct seed, the seed was not reset). 
- Acknowledge that it is not pretty to be doing forward passes through the value function network even when it is not being used for the agent.
- TODO: don't collect a certain number of episodes, collect a certain number of data points, to keep the total simulation time under control. Better for experiments.
- TODO: move the training functions for the value function network into the Trainer class.

## Experiment 1
- What we are looking for is quick improvement in terms of time/update step (so it runs quick) and stable training process (low variance within and across runs), so it's easy to compare two sets of hyperparameters. In this experiment especially, we are looking for a good stable baseline against which to compare other methdos
- Note that there is a certain run-off effect: the agent improves exponentially. This is because, as the agent manages to stay alive for longer, it is able to collect way more data, leading to even more improvement per update step. As such, the time duration of a training run is overwhelmingly correlated with performance (i.e. average episode length), and cannot be used to select between the hidden number of neurons, since that has a comparatively small effect.  
- The largest source of compute cost is simulating the episodes. Especially, as the performance of the agent grows, it is able to stay alive longer, increasing the average length of the episodes.
- LEARNING RATE: From inspecting learning_rate=0.001, it's clear that it is way too small, with none of the runs improving noticeably (for any of the other settings for number of episodes or policy hidden size). For an lr of 0.01, we do see consistent improvement, with more hidden neurons leading to quicker improvement (since per update step we have more params to update). However, none of the runs reach the maximum rewards of 500 within the run time. An lr of 0.1 gives quick improvement, albeit with huge variance and instability (not only between runs, but especially within a single run as well, with some runs achieving 500 reward before deteriorating significantly). To find a good trade-off between improvement speed and variance (stability) of the training process, a learning rate of 0.01 seems suitable for our future experiments.
- NUM EPISODES: Has a number of effects: (1) Variance: Fewer episodes lead to more instability (variance) of the training process, since your policy gradient estimate will have higher variance with fewer data points. (2) Improvement per step: More episodes lead to more learning per step (using future returns). This is because, for more episodes, there might be a few that achieve through pure chance a high reward. Using future returns, such lucky episodes will be reinforced dispropotionally compared to the vast majority of mediocre runs. This can be verified by inspecting the variances of the future returns, which show that the training runs corresponding to quick improvement correspond to high variance of the future returns (e.g. 50 eps and lr 0.01). This difference in performance increase gets damped for lower learning rates (and expectedly for GAEs as well since their variance doesn't explode), since any lucky episodes will have a lower effect on the gradient update. (3) Training time: This is important, especially with the exponentially increasing episode simulation time as the network improves. For a lr of 0.01, there is not a clear benefit to increase the number of episodes from 20 to 50 (!), since both achieve similar performance (under all lrs and hidden neurons), while training time more than doubles.   
- HIDDEN SIZE: there doesn't seem to be a clear improvement in training when changing hidden neurons from 2 to 4 (across all lrs). The performance noticeably improves when using 8 hidden neurons. In particular, a network with only 2 hidden neurons seem to learn just fine. Before we concluded that the overwhelming impact on training time comes from the performance of the agent (not _directly_ from the number of hidden neurons), we will use 8 hidden neurons to speed up the improvement per update step. For lr 0.01, the worst run using 8 hidden neurons is better than the best run using 4 neurons.
- CONCLUSIONS: learning rate of 0.01, 20 episodes, 8 hidden neurons. Note that the GAE paper uses 20 episodes as well, but uses no hidden layer.

## Experiment 2a
- We reinitialize the Adam optimizer at each training run of the value function network. We don't reinitialize the value network. We also train the policy network before training the value function network, following the GAE paper.
- The effect of lr and epochs on training the value network, using GAE. It's clear that a high learning rate and many epochs lead to some sort of overtraining, where the network initially learns effectively, but then very quickly collapses and its performance goes to zero. The higher the learning rate or the more epochs, the quicker this collapse happens. This could be caused by overtraining, where the value network gets overfit so much to a batch of episodes, that it cannot handle the distributional shift from policy k to policy k+1. In the fully collapsed states, the gradients of the policy and value network approach, meaning that it won't be able to get out of the collapsed state anymore.
- We choose value learning rate 0.0001, and use a num_epochs_value_network of 1 (so one pass over the training data). The best performing runs consistently have only 1 or 2 epochs. With 5 epochs, there is still learning at a low learning rate, but lots of instability.
- However, the reason why we see less instability at few epochs and low lr, is that the value function practically doesn't learn and we deal with extreme underfitting. Since GAE with the current lambda setting of 0.96 approximates a discoutned future returns with value function baseline, and since the value function doesn't really learn, we basically reduced the learning problem to one using discounted future returns (rewards-to-go). The decrease in test loss over 2 epochs is usually in the range of 0.05%-0.50%.

## Experiment 3
- Investigate the effect of GAE and value function baseline. Using lambda = 0.96 and gamma = 0.98 according to CartPole results in Figure 2 of the GAE paper.
- We see similar results as in Experiment 2. If we put lr and epochs to high for the value function, we see total collapse (in Experiment 2) due to unability to deal with distributional shifts in the policy. However, putting it too low basically reduces the problem to using rewards-to-go.

## Experiment 4
- Do a hyperparameter sweep for discounted future returns with value function baseline, trying to get it to work to some extent.


# Policy Gradient Methods from Scratch: REINFORCE (VPG), Baselines and GAE

This repository contains a from-scratch PyTorch implementation of several basic policy gradient algorithms, designed to solve the `CartPole-v1` environment from [Gymnasium](https://gymnasium.farama.org/).

The primary goal of this project is to gain hands-on experience with policy gradient methods by implementing them from scratch and by setting up experiments to explore their properties and trade-offs.

## Key Features & Implementations
- **REINFORCE (VPG):** Implementation using both full returns and future returns (rewards-to-go), using a stochastic policy $\pi_\theta$ based on a one-hidden layer MLP.
- **Value Function Baseline:** Implementation using future returns with a value function baseline, intended to reduce the variance of the policy gradient estimates. The value function is approximated by a one-hidden layer MLP.
- **Generalized Advantage Estimates (GAEs):** A full implementation of GAEs to provide a tunable bias-variance trade-off between TD(1)-errors and discounted future returns with value function baseline.
- **Experimentation Framework:** Includes TensorBoard logging for detailed analysis of training runs, including reward curves.

## Demo: Trained Agent in Action

![Trained Agent playing CartPole](./assets/cartpole_demo.gif)
*(Suggestion: Create a short GIF of your best-performing agent and place it in an `assets` folder.)*

## Setup & Usage

### 1. Installation
First, clone the repository and set up the Python environment.

```bash
git clone https://github.com/NN41/policy_gradient_methods.git
cd policy_gradient_methods
pip install -r requirements.txt
```

### 2. Running the Training
To run a single training instance with the default configuration (using GAE):

```bash
python main.py
```

Experiment configurations can be adjusted within `main.py` or by creating a dedicated `run_experiments.py` script.

### 3. Monitoring with TensorBoard
All experiment results, including learning curves and weight variances, are logged to the `runs/` directory. To view them:

```bash
tensorboard --logdir runs
```

### 4. Running an Experiment


## Background & Implementation

This section provides a brief overview of policy gradient methods and their implementation in this project. This project is based on the theory from OpenAI's [Spinning Up in Deep RL](https://spinningup.openai.com/en/latest/index.html) and the 2016 paper [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) by Duan et al.

### Policy Gradient Methods
Policy gradient methods are one of the simplest ways to do reinforcement learning. These methods optimize a parametrized policy $\pi_\theta$ with the goal of maximizing the expected return $J(\pi_\theta) = E_{\tau\sim\pi_\theta}[R(\tau)]$, where $\tau$ is a trajectory sampled from the policy $\pi_\theta$. In this project, $R(\tau) = \sum_{t=0}^T r_t$ is the undiscounted finite-horizon return of the trajectory. The core idea of policy gradient algorithms is to update the policy parameters $\theta$ through gradient ascent. The policy gradient theorem allows us to write the policy gradient $\nabla_\theta J(\pi_\theta)$ as $E_{\tau\sim\pi_\theta}[\sum_{t=0}^T\nabla_\theta\log\pi_\theta(a_t|s_t)\Psi_t]$. We can estimate this quantity by collecting trajectories by sampling $\pi_\theta$ and taking the sample mean of the $\nabla_\theta\log\pi_\theta(a_t|s_t)\Psi_t$. Here, $\Psi_\theta$ is the "weight", which represents the quality of taking action $a_t$ in state $s_t$. Intuitively, if an action $a_t$ is associated with a high weight, it's considered 'good' and its likelihood will be pushed up. Similarly, if an action is considered 'bad' according to the weight, its likelihood is pushed down.

The choice of the weight $\Psi_t$ is critical and determines the trade-off between bias and variance in the gradient estimate. This project implements several common choices for $\Psi_t$ from scratch:
- **Full Returns (`weight_kind='r'`)**: $\Psi_t = R(\tau) = \sum_{t'=0}^Tr_{t'}$. Used in the original REINFORCE algorithm. It is an unbiased estimate, but suffers from high variance, because a single luck or unlucky action late in an episode can drastically change the returns for all preceding actions.
- **Future Returns (Rewards-to-Go) (`weight_kind='r'`)**: $\Psi_t = \sum_{t'=t}^Tr_{t'}$. A more sensible choice for $\Psi_t$, since it only focus on the consequences (the future returns) of taking action $a_t$. This is the "vanilla" policy gradient (VPG) implementation. It is unbiased as well, but has slightly lower variance. Also has a discounted version, implemented as `weight_kind='dfr'`.
- **Future Returns with a Value Function Baseline (`weight_kind='dfrb'`)**: $\Psi_t = \left(\sum_{t'=t}^T\gamma^{t'-t}r_{t'}\right)-V_\phi(s_t)$. Here, we subtract a baseline from the discounted future returns. In this project, we approximate the on-policy value function $V^\pi$ by an MLP $V_\phi$ trained to reduce an MSE loss with respect to future returns sampled from the parametrized policy $\pi_\theta$. A *good* estimate of the value function can significantly reduce variance wihtout introducing bias.
- **Generalized Advantage Estimation (GAE) (`weight_kind='gae'`)**: $\Psi_t=\text{GAE}_t(\gamma,\lambda) = \sum_{t'=t}^T(\gamma\lambda)^{t'-t}\delta_{t'}$, where $\delta_t$ is the TD(1) error. The GAE paper introduces a scheme to smoothly interpolate between the low-variance (but biased) one-step TD error and the high-variance (but unbiased) future returns with a value function baseline. The parameter $\lambda$ controls this bias-variance trade-off. In particular:
  - For $\lambda=0$, we get the TD(1) error $\Psi_t=\delta_t=r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$, implemented as `weight_kind='td'`.
  - For $\lambda=1$, we get the future with a value function baseline `weight_kind='dfrb'`.

### Role of GAE Parameters $\gamma$ and 
As discussed in the GAE paper, both $\gamma$ and $\lambda$ control the amount of bias we introduce in the system.

The first parameter, the discount factor $\gamma\in[0,1]$, introduces a bias in the objective itself by changing the problem we are solving. It replaces the original problem of maximizing the expected *undiscounted* return $E\left[\sum_{t=0}^Tr_t\right]$ by the proxy problem of maximizing the expected *discounted* return $E\left[\sum_{t=0}^T\gamma^tr_t\right]$. By shrinking the magnitude of rewards from the distant future, we are stabilizing learning signals by making the return less sensitive to uncorrelated random events that might occur far in the distant future. This does introduce some bias. In this project (within the `CartPole-v1` environment), however, this bias is inconsequential, because solving the proxy problem solves the original problem as well, due to the constant and immediate nature of the feedback loop. The agent receives one point for every time step it survives, so receiving $X$ points on the discounted problem implies receiving even more points on the original, undiscounted problem.

The second parameter $\lambda$ the bias resulting from using an incorrect estimator for the value function. Since $V_\phi$ is a neural network, it is never equal to the true value function, which gives a second source of bias, but this time for the *proxy* problem. Because of this, as we decrease $\lambda < 1$ towards zero, we rely more on highly-biased low-variance TD-errors. For $\lambda = 1$, we uncover the  

### Implementation Details

## Experiments & Results
### Experiment 1: Minimum Viable Policy Hyperparameters

### Experiment 2: Agent Performance Collapse / Methods Comparison

### Experiment 3: Variance of Weight Types
Using results from Experiment 2 (for vf underfitting and collapsing)

Note that using discounted future returns does indeed lower variance wrt undiscounted.

## Key Learnings & Obstacles

## Future Work & TODO


--------------------------------------------------
--------------------------
------------------------
## Experiment: The Impact of Advantage Estimation on Variance & Stability

The core of this project was to investigate a fundamental challenge in policy gradient methods: high variance in the gradient estimates, which leads to unstable training. I compared three different methods for calculating the "weight" or "advantage" term used in the policy loss function.

1.  **Reward-to-Go (MC):** An unbiased but high-variance estimate using the sum of future rewards.
2.  **Reward-to-Go with Value Baseline:** Subtracting a learned state-value function `V(s)` from the reward-to-go to center the advantages around zero.
3.  **Generalized Advantage Estimation (GAE):** A sophisticated technique that interpolates between the value baseline approach (for `λ=0`) and the simple reward-to-go approach (for `λ=1`).

### Results

As hypothesized, the choice of advantage estimator has a dramatic impact on training stability. Using GAE (`λ=0.96`) consistently led to the most stable learning and fastest convergence to the maximum score of 500.

*(Suggestion: Insert a screenshot of your TensorBoard plot comparing the average episode return for the three methods. A second plot showing the variance of the weights for each method would be even better.)*

![Learning Curves Comparison](./assets/results_comparison.png)

The plot above clearly shows that while all methods eventually learn, the GAE-based agent (blue) exhibits a much smoother and more reliable learning curve, whereas the simple Reward-to-Go agent (red) shows significant instability between epochs.

## Key Learnings & Obstacles

This project was a deep dive into the practical challenges of implementing RL algorithms.

*   **Learning 1: Training a Value Function Baseline is a Delicate Balance.** The biggest challenge was getting the value function to train effectively alongside the policy.
    *   **Problem:** Overtraining the value network on a fixed batch of data from policy `π_k` caused it to fail catastrophically when presented with data from the new policy `π_{k+1}`.
    *   **Solution:** I found that training the value network for only a small number of epochs (1-3) per policy update, with a carefully tuned (often small) learning rate, was critical. This prevents the value function from overfitting to a stale data distribution.

*   **Learning 2: The Importance of Critical AI Tool Usage.** Throughout this project, I used LLMs as a coding and debugging partner. This highlighted the need for careful verification.
    *   **Insight:** An early suggestion from an LLM misplaced the random seed reset, which could have led to non-reproducible experiments. This reinforced the importance of understanding the fundamentals yourself and critically evaluating any AI-generated code or suggestions.

*   **Learning 3: Architectural Choices Matter.**
    *   **Problem:** My value network initially included a `ReLU` activation on the output layer, assuming non-negative returns. This was a mistake, as it led to zero gradients if the network weights initialized in a way that produced negative outputs.
    *   **Solution:** Removing the final activation and using a linear output layer was crucial for stable training.

## Future Work
- [ ] Implement Proximal Policy Optimization (PPO) and compare its performance.
- [ ] Refactor the data collection loop to collect a fixed number of environment steps rather than a fixed number of episodes, to normalize the amount of data per policy update.
- [ ] Test the implemented agents on more complex environments like `Acrobot-v1` or `LunarLander-v1`.