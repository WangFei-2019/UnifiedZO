# trainer/__init__.py

from .base_zo_trainer import BaseZOTrainer
from .mezo_trainer import MeZOTrainer
from .lozo_trainer import LoZOTrainer
from .hizoo_trainer import HiZOOTrainer
from .pzo_trainer import PZOTrainer
from .fzoo_trainer import FZooTrainer
from .zo_adamu_trainer import ZOAdaMUTrainer
# [AdaLeZO Integration] Import the new trainer
from .adalezo_trainer import AdaLeZOTrainer
from .schedulers import zo_lr_scheduler, hessian_smooth_scheduler

def get_trainer_class(args):
    """
    Factory to return the correct Trainer class based on args.
    """
    if args.trainer == "mezo":
        return MeZOTrainer
    elif args.trainer == "zoadamu":
        return ZOAdaMUTrainer
    elif args.trainer == "lozo":
        return LoZOTrainer
    elif args.trainer == "hizoo":
        return HiZOOTrainer
    elif args.trainer == "pzo":
        return PZOTrainer
    elif args.trainer == "fzoo":
        return FZooTrainer
    # [AdaLeZO Integration] Return AdaLeZO class
    elif args.trainer == "adalezo":
        return AdaLeZOTrainer
    else:
        # Fallback to base or standard Trainer if "regular"
        return BaseZOTrainer