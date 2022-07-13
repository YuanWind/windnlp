# -*- encoding: utf-8 -*-
'''
@File    :   Adversarial.py
@Time    :   2022/04/18 09:03:12
@Author  :   Yuan Wind
@Desc    :   None
'''
import logging
logger = logging.getLogger(__name__.replace('_', ''))
import torch


class AWP:
    def __init__(
        self,
        model,
        adv_param="weight",
        awp_a=1.0,
        awp_e= 0.01,
        awp_k=1,
    ):
        """
        https://www.kaggle.com/competitions/nbme-score-clinical-patient-notes/discussion/323095
        """
        self.model = model
        self.adv_param = adv_param
        self.adv_lr = awp_a
        self.adv_eps = awp_e
        self.adv_step = awp_k
        self.backup = {}
        self.backup_eps = {}

    # def attack_backward(self, tokens, attention_mask, token_type_ids, label, epoch):
    #     if (self.adv_lr == 0) or (epoch < self.start_epoch):
    #         return None

    #     self._save()
    #     for _ in range(self.adv_step):
    #         self._attack_step()
    #         with torch.cuda.amp.autocast():

    #             out = self.model(tokens, attention_mask=attention_mask, token_type_ids=token_type_ids).view(-1, 1)
    #             adv_loss = self.criterion(out, label.view(-1, 1))
    #             adv_loss = torch.masked_select(adv_loss, label.view(-1, 1) != -1).mean()

    #         self.optimizer.zero_grad()
    #     return adv_loss

    def attack(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name][0]), self.backup_eps[name][1]
                    )
                # param.data.clamp_(*self.backup_eps[name])

    def save(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}
class EMA():
    def __init__(self, model, decay):
        """
        https://www.cnblogs.com/sddai/p/14646581.html
        """
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.is_register = False
        
    def register(self):
        """
        把原来模型中需要求梯度的参数克隆一份
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
        self.is_register = True
        logger.info('Now EMA registered the model parameters.')
    def update(self):
        """
        将更新梯度后的当前参数和更新前的参数进行滑动平均保存在shadow weight中. shadow_w = (1-decay) * cur_weight + decay * shadow_w
        """
        if self.is_register:
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    assert name in self.shadow
                    new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name].to(param.device)
                    self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """
        复制shadow weight到模型参数中去，使用shadow weight中的参数进行评测，能够使模型更鲁棒
        """
        if self.is_register:
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    assert name in self.shadow
                    self.backup[name] = param.data
                    param.data = self.shadow[name].to(param.device)

    def restore(self):
        """
        恢复模型原来的参数
        """
        if self.is_register and len(self.backup) > 0:
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    assert name in self.backup
                    param.data = self.backup[name].to(param.device)
            self.backup = {}
        
class FGM():
    """
    https://zhuanlan.zhihu.com/p/91269728
    """
    def __init__(self, module):
        self.module = module
        self.backup = {}

    def attack(
        self,
        epsilon=1.,
        emb_name='emb'
    ):
        for name, param in self.module.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(
        self,
        emb_name='emb'
    ):
        for name, param in self.module.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class PGD():
    """
    https://zhuanlan.zhihu.com/p/91269728
    """
    def __init__(self, module):
        self.module = module
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(
        self,
        epsilon=1.,
        alpha=0.3,
        emb_name='emb',
        is_first_attack=False
    ):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.module.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)

    def restore(self, emb_name='emb'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.module.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.module.named_parameters():
            if param.requires_grad:
                if  param.grad is not None:
                    self.grad_backup[name] = param.grad.clone()
                # else:
                #     logger.warning(f'{name} 的 param 没有梯度！')

    def restore_grad(self):
        for name, param in self.module.named_parameters():
            if param.requires_grad and name in self.grad_backup:
                param.grad = self.grad_backup[name]