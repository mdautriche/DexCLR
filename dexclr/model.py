import os
import torch

def load_optimizer(args, model):

    if args.lr == None:
        lr=3e-4

    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        raise NotImplementedError

    return optimizer


def save_model(args, model, optimizer):
    out = os.path.join(args.model_path, "checkpoint_{}.tar".format(args.current_epoch))

    # To save a DataParallel model generically, save the model.module.state_dict().
    # This way, you have the flexibility to load the model any way you want to any device you want.
    try:
        os.makedirs(args.model_path)
    except:
        pass

    if isinstance(model, torch.nn.DataParallel):
        torch.save(model.module.state_dict(), out)
    else:
        torch.save(model.net.state_dict(), out)
