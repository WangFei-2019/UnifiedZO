from .base_zo_trainer import BaseZOTrainer
from .mezo_trainer import MeZOTrainer
from .lozo_trainer import LoZOTrainer
from .hizoo_trainer import HiZOOTrainer
from .pzo_trainer import PZOTrainer
from .fzoo_trainer import FZooTrainer
from .zo_adamu_trainer import ZOAdaMUTrainer
from .dizo_trainer import DiZOTrainer
from .mezo_svrg_trainer import MeZOSVRGTrainer
from .adalezo_trainer import AdaLeZOTrainer
from .schedulers import zo_lr_scheduler, hessian_smooth_scheduler

# Quantized ZO
from .qzo_trainer import QZOTrainer
from .lqzo_trainer import LQZOTrainer

# AdaLeZO Plus
from .adalozo_trainer import AdaLoZOTrainer
from .adazoadamu_trainer import AdaZOAdaMUTrainer
from .adahizoo_trainer import AdaHiZOOTrainer
from .adapzo_trainer import AdaPZOTrainer
from .adafzoo_trainer import AdaFZooTrainer
from .adadizo_trainer import AdaDiZOTrainer
from .adamezosvrg_trainer import AdaMeZOSVRGTrainer

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
    elif args.trainer == "dizo":
        return DiZOTrainer
    elif args.trainer == "mezo_svrg":
        return MeZOSVRGTrainer
    elif args.trainer == "adalezo":
        return AdaLeZOTrainer
        
    # Quantized
    elif args.trainer == "qzo":
        return QZOTrainer
    elif args.trainer == "lqzo":
        return LQZOTrainer

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
    elif args.trainer == "adadizo":
        return AdaDiZOTrainer
    elif args.trainer == "adamezosvrg":
        return AdaMeZOSVRGTrainer
    else:
        # Fallback to base or standard Trainer if "regular"
        return BaseZOTrainer