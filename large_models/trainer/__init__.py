# trainer/__init__.py

from .base_zo_trainer import BaseZOTrainer
from .mezo_trainer import MeZOTrainer
from .lozo_trainer import LoZOTrainer
from .hizoo_trainer import HiZOOTrainer
from .pzo_trainer import PZOTrainer
from .fzoo_trainer import FZooTrainer
from .zo_adamu_trainer import ZOAdaMUTrainer
from .adalezo_trainer import AdaLeZOTrainer
from .schedulers import zo_lr_scheduler, hessian_smooth_scheduler

# AdaLeZO Plus
from .adalozo_trainer import AdaLoZOTrainer
from .adazoadamu_trainer import AdaZOAdaMUTrainer
from .adahizoo_trainer import AdaHiZOOTrainer
from .adapzo_trainer import AdaPZOTrainer
from .adafzoo_trainer import AdaFZooTrainer

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
    elif args.trainer == "adalezo":
        return AdaLeZOTrainer

    # AdaLeZO Plus
    elif args.trainer == "adazoadamu":
        return AdaZOAdaMUTrainer
    elif args.trainer == "adalozo":
        return AdaLoZOTrainer
    elif args.trainer == "adahizoo":
        return AdaHiZOOTrainer
    elif args.trainer == "adapzo":
        return AdaPZOTrainer
    elif args.trainer == "adafzoo":
        return AdaFZooTrainer
    else:
        # Fallback to base or standard Trainer if "regular"
        return BaseZOTrainer