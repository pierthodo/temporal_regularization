#!/usr/bin/env python

from mpi4py import MPI
from baselines.common import set_global_seeds
from baselines import bench
import os.path as osp
import gym, logging
from baselines import logger
from baselines.common.atari_wrappers import make_atari, wrap_deepmind

def train(env_id, num_timesteps, seed,theta,name,decay,time_param):
    if name != "":
        name = name+"theta-"+str(theta)+"-decay-"+str(decay) + "-timeparam-" + str(time_param) + '/'
    from baselines.ppo1 import pposgd_simple, cnn_policy
    import baselines.common.tf_util as U
    rank = MPI.COMM_WORLD.Get_rank()
    sess = U.single_threaded_session()
    sess.__enter__()
    import os.path
    import datetime
    tmp = os.path.join('/project/rrg-dprecup/pthodo/LOG/'+name+'atari_'+str(env_id)+'/',
        str(seed)+'--'+datetime.datetime.now().strftime("openai-%Y-%m-%d-%H-%M-%S-%f"))
    if rank == 0:
        logger.configure(dir=tmp)
    else:
        logger.configure(dir=tmp,format_strs=[])
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed)
    env = make_atari(env_id)
    def policy_fn(name, ob_space, ac_space): #pylint: disable=W0613
        return cnn_policy.CnnPolicy(name=name, ob_space=ob_space, ac_space=ac_space)
    env = bench.Monitor(env, tmp and
        osp.join(tmp, str(rank)))
    env.seed(workerseed)
    gym.logger.setLevel(logging.WARN)

    env = wrap_deepmind(env)
    env.seed(workerseed)

    pposgd_simple.learn(env, policy_fn,
        max_timesteps=int(num_timesteps * 1.1),
        timesteps_per_actorbatch=256,
        clip_param=0.2, entcoeff=0.01,
        optim_epochs=4, optim_stepsize=1e-3, optim_batchsize=64,
        gamma=0.99, lam=0.95,
        schedule='linear',theta = theta,decay=decay,time_param=time_param
    )
    env.close()

def main():
    import argparse
    import datetime
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='PongNoFrameskip-v4')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(10e6))
    parser.add_argument('--theta', type=float, default=0)
    parser.add_argument('--decay',type=float,default=0)
    parser.add_argument('--timeparam',type=float,default=0)    
    parser.add_argument('--name',default=datetime.datetime.now().strftime("openai-%Y-%m-%d-%H-%M-%S-%f"))

    args = parser.parse_args()
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed,theta=args.theta,name=args.name,decay=args.decay,time_param=args.timeparam)

if __name__ == '__main__':
    main()
