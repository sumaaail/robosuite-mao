from code.pytorch.recovery_rl.arg_utils import get_args
from code.pytorch.recovery_rl.experiment import Experiment
from code.pytorch.recovery_rl.env.obj_extraction import *
from mujoco_py import GlfwContext
GlfwContext(offscreen=True)


if __name__ == '__main__':
    # # Get user arguments and construct config
    exp_cfg = get_args()
    # print("cnn :::", exp_cfg.cnn)
    
    # # Create experiment and run it
    experiment = Experiment(exp_cfg)
    
    # # experiment.run()
    # experiment.pretrain_critic_recovery()
    
    env = ObjExtraction()

    obs = env.reset()
    print("obs :", obs)
    
    while True:
        env.render()
