"""
Layer-wise Adaptive Moments optimizer (LAMB)

This is a minimal PyTorch implementation of the LAMB optimizer commonly used for large-batch
training. It extends Adam with a layer-wise trust ratio computed as ||w|| / ||update|| which
stabilizes training when scaling batch sizes.

Notes and limitations
- Sparse gradients are not supported (matches common reference implementations).
- Weight decay is applied in a decoupled AdamW-style manner via adding to the update.
- Trust ratio falls back to 1.0 when either weight or update norms are zero.

References
- You, Zhang, Hsieh, Demmel, Keutzer (2019): Large Batch Optimization for Deep Learning: Training BERT in 76 minutes.
"""

import torch


class Lamb(torch.optim.Optimizer):
    """PyTorch Optimizer implementing LAMB.

    Args:
        params: Iterable of parameters to optimize.
        lr (float): Learning rate.
        betas (tuple[float, float]): Exponential averaging coefficients for moments.
        eps (float): Numerical stability epsilon.
        weight_decay (float): Decoupled L2 weight decay coefficient.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(Lamb, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Returns:
            Optional[torch.Tensor]: The loss if a closure was provided.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Lamb does not support sparse gradients, consider SparseAdam instad.')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                m_t = exp_avg / bias_correction1
                v_t = exp_avg_sq / bias_correction2
                torch.sqrt_(v_t)

                update = m_t / (v_t + group['eps'])

                ratio = 1.0
                if group['weight_decay'] > 0:
                    update.add_(p.data, alpha=group['weight_decay'])

                    g_norm = torch.norm(update.flatten())
                    w_norm = torch.norm(p.data.flatten())

                    if w_norm > 0.0 and g_norm > 0.0:
                        ratio = w_norm / g_norm

                p.data.add_(update, alpha=-group['lr'] * ratio)

        return loss
