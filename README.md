This is the repository for the paper "Temporal Regularization for Markov Decision Process" https://arxiv.org/abs/1811.00429 that will be presented at NIPS 2018.
If you use this repository please cite the paper:

@article{thodoroff2018temporal,
  title={Temporal Regularization in Markov Decision Process},
  author={Thodoroff, Pierre and Durand, Audrey and Pineau, Joelle and Precup, Doina},
  journal={arXiv preprint arXiv:1811.00429},
  year={2018}
}

Simple experiments:

For now those can be found in the "exp" folder however, a clean version using google colab will be published soon.

Deep Reinforcement learning:

The codebase was forked around February 2018 from the open AI baselines repository (https://github.com/openai/baselines).
The baselines repo has now changed significantly. I will update the code in this repository with the newer open AI implementation in the near future and perform benchmarks. 

The implementation of temporal regularization is straightforward. We modify the target of PPO using exponential smoothing. The modifications can be found in the file baselines/ppo1/pposgd_simple.py at the function add_vtarg_and_adv.
