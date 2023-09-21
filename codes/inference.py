import torch
import torch.nn as nn
import torchvision.models as tvm

from data_loader import get_data
from extended_config import (cfg as conf, key_maps, CN, update_from_dict)
from utils import Learner, synchronize
from mdl import get_default_net, RetinaBackBone, SSDBackBone, ZSGNet
from loss import get_default_loss
from evaluator import get_default_eval
import ssd_vgg

import pdb

def learner_init(uid: str, cfg: CN) -> Learner:
    device = torch.device('cuda')
    data = get_data(cfg)

    # Ugly hack because I wanted ratios, scales
    # in fractional formats
    if type(cfg['ratios']) != list:
        ratios = eval(cfg['ratios'], {})
    else:
        ratios = cfg['ratios']
    if type(cfg['scales']) != list:
        scales = cfg['scale_factor'] * np.array(eval(cfg['scales'], {}))
    else:
        scales = cfg['scale_factor'] * np.array(cfg['scales'])

    num_anchors = len(ratios) * len(scales)
    mdl = get_default_net(num_anchors=num_anchors, cfg=cfg)
    mdl.to(device)
    if cfg.do_dist:
        mdl = torch.nn.parallel.DistributedDataParallel(
            mdl, device_ids=[cfg.local_rank],
            output_device=cfg.local_rank, broadcast_buffers=True,
            find_unused_parameters=True)
    elif not cfg.do_dist and cfg.num_gpus:
        # Use data parallel
        mdl = torch.nn.DataParallel(mdl)

    loss_fn = get_default_loss(ratios, scales, cfg)
    loss_fn.to(device)

    eval_fn = get_default_eval(ratios, scales, cfg)
    # eval_fn.to(device)
    opt_fn = partial(torch.optim.Adam, betas=(0.9, 0.99))

    learn = Learner(uid=uid, data=data, mdl=mdl, loss_fn=loss_fn,
                    opt_fn=opt_fn, eval_fn=eval_fn, device=device, cfg=cfg)
    return learn

pdb.set_trace()
cfg = conf
kwargs = {}
cfg = update_from_dict(cfg, kwargs, key_maps)

device = torch.device('cuda')


# learn = Learner(uid=uid, data=data, mdl=mdl, loss_fn=loss_fn,
#                     opt_fn=opt_fn, eval_fn=eval_fn, device=device, cfg=cfg)
# learn = learner_init(uid, cfg)

mdl = get_default_net(num_anchors=9, cfg=cfg)
mdl.to(device)
mdl.eval()

# with torch.no_grad():
