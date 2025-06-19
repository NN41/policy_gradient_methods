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
- **Future Returns (Rewards-to-Go) (`weight_kind='fr'`)**: $\Psi_t = \sum_{t'=t}^Tr_{t'}$. A more sensible choice for $\Psi_t$, since it only focus on the consequences (the future returns) of taking action $a_t$. This is the "vanilla" policy gradient (VPG) implementation. It is unbiased as well, but has slightly lower variance. Also has a discounted version, implemented as `weight_kind='dfr'`.
- **Future Returns with a Value Function Baseline (`weight_kind='dfrb'`)**: $\Psi_t = \left(\sum_{t'=t}^T\gamma^{t'-t}r_{t'}\right)-V_\phi(s_t)$. Here, we subtract a baseline from the discounted future returns. In this project, we approximate the on-policy value function $V^\pi$ by an MLP $V_\phi$ trained to reduce an MSE loss with respect to future returns sampled from the parametrized policy $\pi_\theta$. A *good* estimate of the value function can significantly reduce variance wihtout introducing bias.
- **Generalized Advantage Estimation (GAE) (`weight_kind='gae'`)**: $\Psi_t=\text{GAE}_t(\gamma,\lambda) = \sum_{t'=t}^T(\gamma\lambda)^{t'-t}\delta_{t'}$, where $\delta_t$ is the TD(1) error. The GAE paper introduces a scheme to smoothly interpolate between the low-variance (but biased) one-step TD error and the high-variance (but unbiased) future returns with a value function baseline. The parameter $\lambda$ controls this bias-variance trade-off. In particular:
  - For $\lambda=0$, we get the TD(1) error $\Psi_t=\delta_t=r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$, implemented as `weight_kind='td'`.
  - For $\lambda=1$, we get the future with a value function baseline `weight_kind='dfrb'`.

### Role of GAE Parameters
As discussed in the GAE paper, both $\gamma$ and $\lambda$ control the amount of bias we introduce in the system. However, they control different types of biases.

The first bias is introduced by changing the problem we are solving and is controlled by the discount factor $\gamma\in[0,1]$. When $\gamma<1$, it replaces the original problem of maximizing the expected *undiscounted* return $E\left[\sum_{t=0}^Tr_t\right]$ by the proxy problem of maximizing the expected *discounted* return $E\left[\sum_{t=0}^T\gamma^tr_t\right]$. By shrinking the magnitude of rewards from the distant future, we are stabilizing learning signals by making the return less sensitive to uncorrelated random events that might occur far in the distant future. This does introduce some bias. In this project (within the `CartPole-v1` environment), however, this bias is inconsequential, because solving the proxy problem solves the original problem as well, due to the constant and immediate nature of the feedback loop. The agent receives one point for every time step it survives, so receiving $X$ points on the discounted problem implies receiving even more points on the original, undiscounted problem.

The second bias is introduced by using an incorrect estimator for the true value function. Looking at the GAE formule, we have to replace the true value function $V^{\pi_\theta}$ a neural network $V_\phi$, which is always going to be an approximation. Therefore, the TD errors we calculate are not the true TD errors, i.e. it's a biased estimate. The parameter $\lambda\in[0,1]$ provides a sliding scale between reducing variance or reducing this bias.
- When $\lambda=1$, the GAE estimate reduces to the discounted future return with a baseline $V_\phi$. It is well-known that we may include baselines without changing the expectation of the policy gradient estimate. As such, no matter how bad your approximation $V_\phi(s_t)$ is, the estimate remains unbiased for the *proxy* problem. However, the estimate has high variance, especially if $V_\phi$ is insufficiently or particularly badly trained.
- When $\lambda=0$, the GAE estimate reduces to the one-step TD error. As we mentioned above, this is a biased estimate, so the final GAE estimate is biased as well. At his, however, very low variance. By varying $\lambda$ between $0$ and $1$, you can choose the trade-off between this variance and the bias. The GAE paper finds generally finds best performance when $\lambda \in [0.9, 0.99]$. 

### Implementation Details
In this project, we parametrize the policy and value function by one-hidden layer MLPs with hidden-layer ReLU activation. For the value function network, we use MSE loss with respect to discounted future returns. We optimize both networks using Adam.

## Experiments & Results
The hyperparameters and training metrics of each training run are stored in the folder `\runs\experiment_group\run`, where `experiment_group` and `run` are respectively chosen by the user or created at runtime. 

### Experiment 1: Policy Hyperparameters Grid Search
We perform a grid search to find the hyperparameters that allow us to train the policy network to achieve good performance in a reasonable amount of training time. The objective is to find a good benchmark that exhibits stable but noticeable improvement with a manageable training time.

We train an agent to solve the environment through REINFORCE (VPG), using undiscounted future returns as the weights (`weight_kind=fr`). Using this bare-minimum set up gives us a benchmark to compare other methods against.

We perform a grid search by choosing the number of episodes per epoch from $[10,20,50]$, the learning rate from $[0.001,0.01,0.1]$ and the number of hidden neurons from $[2,4,8]$. We run 50 epochs. In the following figure taken from Tensorboard, we show a single run for each parameter combo from the Cartesian product, each run using the same seed.

![Experiment 1](assets\image.png)

Each color corresponds to a tuple (learning rate, hidden size), with multiple lines of the same color representing varying degrees of episodes per epoch. We note the following:
- The upper band (pink, blue, black) represent a learning rate of 0.1 and quickly achieves maximum performance of 500 return after roughly 30 epochs, albeit with massive variance within and across runs. Due to this instability, we discard this learning rate.
- The lower band (orange, black, blue) represents a learning rate of 0.001. It is clear that this learning rate is too little to make any meaningful progress and is discarded as well.
- The middle band (green, purple, yellow) represents a learning rate of 0.01, giving us a good trade-off between stability and noticeable improvement over time. 

In all cases, increasing the number of hidden neurons increases the improvement per epoch, since at each gradient ascent step we have more network weights to update. In this project, the biggest driver of training time is not the complexity of the networks, but the total number of steps we have to simulate per epoch. As such, to allow for maximum flexibility, we choose 8 hidden neurons for the policy network, corresponding to the green line. The three green lines correspond to varying number of episodes per epoch. In general, using 10 episodes lead to slightly more instability and using 50 episodes leads to slightly quicker performance improvement due to increased likelihood of highly lucky episodes (at the cost of higher training time), which have an disproportionally positive effect on training due to using future returns as weights. We choose the middle ground of 20 episodes per epoch. The GAE paper also chooses this number in their CartPole experiment. 

As mentioned above, the biggest driver of training time is the number of steps we have to simulate. As a side effect, more successful runs require more simulation steps and thus require way more time to train. As an agent becomes better, it becomes more time-consuming to train it. This is an extra argument to choose the green line, giving a good balance between quick training time and decent performance. 

### Experiment 2: Agent Performance Collapse / Methods Comparison

### Experiment 3: Variance of Weight Types
Using results from Experiment 2 (for vf underfitting and collapsing)

Note that using discounted future returns does indeed lower variance wrt undiscounted.

## Key Learnings & Obstacles

## Future Work
- [ ] dfs
- [ ] Implement Proximal Policy Optimization (PPO) and compare its performance.
- [ ] Refactor the data collection loop to collect a fixed number of environment steps rather than a fixed number of episodes, to normalize the amount of data per policy update.
- [ ] Test the implemented agents on more complex environments like `Acrobot-v1` or `LunarLander-v1`.


--------------------------------------------------
--------------------------
------------------------


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