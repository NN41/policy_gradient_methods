# Implement REINFORCE for CartPole-v1
This project implements the REINFORCE algorithm from scratch to solve the OpenAI Gym CartPole-v1 environment using Pytorch.

## Project Goals
- Implement REINFORCE from scratch with an MLP policy.
- Investigate the impact of a value function baseline.
- Experiment with network architecture.

## REINFORCE
The training progress is very sensitive to the learning rate (using SGD). Because of this it's important to make sure that, once tuned, the magnitude of the gradients don't change. That's why we need to take the scale the double sum by the total number of log probabilities, not only by the number of trajectories. Otherwise, as the training progresses, the trajectories collect higher rewards and the magnitude of the gradients change. What was a suitable learning rate in the first epoch, will be too large by the Nth epoch. To see this, consider a single episode per epoch, in which case our loss function is proportional to $\Sigma_{t=0}^T \log \pi_\theta(a_t|s_t) R(\tau)$. It is clear that the magnitude of the gradient grows as trajectories become longer.

Looking at the most recent expression of policy gradient being a double sum of grad-log-probs scaled by the correspoding episode's return, we see how REINFORCE learns. REINFORCE tries to maximize the double sum of grad-log-probs as a proxy for the expected return. It does so by increasing the log-probabilities, or equivalantly, the probabilities of choosing the action $a_t$ under state $s_t$. This doesn't depend on how well action $a_t$ actually was, the only thing matters is the entire trajectory's return. It can be that action $a_0|s_0$ was actually really bad, but if the agent managed to recover and the rest of the episode goes great, then REINFORCE still makes $a_0|s_0$ more likely. Note that other (better) actions are actually made less likely this way, which makes the algorithm very sensitive to the randomly chosen actions at each time step. In the case of the cartpole, if the policy always selects the wrong action with high likelihood on the first timestep, but always manages to recover, then this wrong first action will be reinforced over time by pure overrepresentation in the training data of on-policy trajectories. This motivates discounting the future rewards.

Note that the reinforcement of actions happens proportionally to the associated reward, since the contribution of the grad-log-prob of a certain state-action pair in the policy gradient is proportional to this associated reward. In the naive policy gradient, that is the entire trajectory's return. If one trajectory performs extremely well relative to the other trajectories by pure luck, then all actions in that trajectory will be reinforced by an disproportional amount as well, even the actions that were irrelevant or even detrimental to the success of that trajectory. 

### Baselines
Spinning Up Part 3 suggests using MSE for learning our MLP that is approximating the on-policy value function. As discussed by Goodfellow, you can derive MSE loss through MLE by assuming that targets are produced through the underlying process plus some Gaussian noise. In this context, that would mean assuming that a trajectory's return is Gaussian distributed around the value function, which generally seems like an unrealistic assumption to make, especially in the cartpole context, since we would expect a lot of short trajectories and few long ones, leading to a heavily skewed returns distribution.

Rather, the choice for MSE loss is one of simplicity and pragmatism. It's a well-understood loss function and minizing MSE loss learns to predict the mean of the target (trajectory returns) given the input (state). Here, the mean of the target is exactly the on-policy value function.  

## Experiments & Results
- Adam performs way better than SGD