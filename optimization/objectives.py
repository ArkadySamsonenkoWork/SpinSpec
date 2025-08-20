import torch


class BaseObjectiveFunction:
    def __init__(self):
        pass

    def __call__(self, pred: torch.Tensor, target: torch.Tensor):
        pass


class MSEObjective(BaseObjectiveFunction):
    def __call__(self, pred: torch.Tensor, target: torch.Tensor):
        return torch.nn.functional.mse_loss(pred, target)


class MAEObjective(BaseObjectiveFunction):
    def __call__(self, pred: torch.Tensor, target: torch.Tensor):
        return torch.nn.functional.l1_loss(pred, target)


class CrossCorrelation(BaseObjectiveFunction):
    def __call__(self, pred: torch.Tensor, target: torch.Tensor):
        vx = pred - pred.mean(dim=-1, keepdim=True)
        vy = target - target.mean(dim=-1, keepdim=True)
        corr =\
            torch.sum(vx * vy, dim=-1) / (torch.sqrt(torch.sum(vx ** 2, dim=-1)) * torch.sqrt(torch.sum(vy ** 2, dim=-1)))
        return 1 - corr


class CosineSimilarity(BaseObjectiveFunction):
    def __call__(self, pred: torch.Tensor, target: torch.Tensor):
        cos_sim = torch.nn.functional.cosine_similarity(pred, target, dim=-1)
        loss = 1 - cos_sim.mean()
        return loss