#!/usr/bin/env python
from baselines.common import set_global_seeds, tf_util as U
from baselines import bench
import gym, logging
from baselines import logger

def train(env_id, num_timesteps, seed,theta,name,decay,lr,time_param):
    import os.path
    import datetime
    print(name)
    if name != "":
        name =name+ "theta-"+str(theta)+"-decay-"+str(decay) + "-lr-"+str(lr)+"-time_param-"+str(time_param)+'/'
    print(name)
    from baselines.ppo1 import mlp_policy, pposgd_simple
    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(seed)
    env = gym.make(env_id)
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2)
    tmp = os.path.join('./../LOG/'+name+'mujoco_'+str(env_id)+'/',
        str(seed)+'--'+datetime.datetime.now().strftime("openai-%Y-%m-%d-%H-%M-%S-%f"))
    print(tmp)
    logger.configure(dir=tmp)
    env = bench.Monitor(env, tmp)
    env.seed(seed)
    gym.logger.setLevel(logging.WARN)
    print("Starting training")
    pposgd_simple.learn(env, policy_fn,
            max_timesteps=num_timesteps,
            timesteps_per_actorbatch=2048,
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=10, optim_stepsize=lr, optim_batchsize=64,
            gamma=0.99, lam=0.95, schedule='linear',theta=theta,decay=decay,time_param=time_param
        )
    env.close()

def main():
    import argparse
    import datetime
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='Swimmer-v2')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(1e6))
    parser.add_argument('--theta', type=float, default=0)
    parser.add_argument('--decay',type=float,default=0)
    parser.add_argument('--timeparam',type=float,default=0)
    parser.add_argument('--lr',type=float,default=3e-4)
    parser.add_argument('--name',default=datetime.datetime.now().strftime("openai-%Y-%m-%d-%H-%M-%S-%f"))
    print("Import done")
    args = parser.parse_args()
    logger.configure()
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed,theta=args.theta,name=args.name,decay=args.decay,lr=args.lr,time_param=args.timeparam)


if __name__ == '__main__':
    print("Starting main")
    main()
