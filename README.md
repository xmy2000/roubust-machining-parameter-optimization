## Robust Machining Parameter Optimization Considering Uncertainties in Aviation Part Cutting Processes

This repository is the implementation of the paper: "**Robust Machining Parameter Optimization Considering Uncertainties
in Aviation Part Cutting Processes**".
This repository contains the implementation of the **machining indicator estimation model** and the **machining
parameter optimization model** proposed in the paper.

As this paper is currently under review, only the core implementation of the method is provided at this stage. Once the
manuscript is accepted, we will promptly supplement all details.

### Introduction

The aviation part cutting process involves significant uncertainties, leading to quality fluctuations and low first-pass
yield even under identical machining configurations. On-site dynamic control of machining parameters is crucial for
stable quality. However, the control process currently relies on manual experience and trial-and-error, which hinders
the ability to respond promptly to uncertainty-induced disturbances and balance quality with efficiency. This study
proposes an integrated “modeling-quantification-optimization” parameter optimal control method. First, we formally model
the aleatoric and epistemic uncertainties in the cutting process using a graph structure. Second, we construct a
machining indicator estimation model to separately quantify two types of uncertainty, thereby generating prediction
intervals. Based on a graph neural network architecture that integrates causal structure mining, the estimation model
enables accurate and stable prediction under incomplete, distribution-shifted field-collected data. Third, we design a
deep reinforcement learning-based agent for robust optimization of machining parameters. A virtual machining environment
incorporating uncertainty is constructed using the trained estimation model for interactive agent training. The agent's
decision robustness under aleatoric and epistemic uncertainty disturbances is enhanced through observation variable
perturbation and the introduction of an Adversary network. Finally, experimental verification in the turning of
aero-engine casings demonstrates the effectiveness of the proposed method. The estimation model performs optimally in
terms of prediction accuracy and uncertainty quantification quality. The strategy obtained by the optimization model
approximates the Pareto frontier while maintaining robustness under uncertainty disturbances. Deployed in the real
cutting environment, the agent improves machining quality by an average of 9% and efficiency by 35% compared to manual
control, offering a practical solution for high-end aviation equipment manufacturing.