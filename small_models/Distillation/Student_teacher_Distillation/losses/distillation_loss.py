import torch.nn.functional as F
from losses.kl_divergence import kl_div

def distillation_loss(student_logits, teacher_logits, labels, alpha=0.6, T=2.0):
    ce_loss = F.cross_entropy(student_logits, labels)
    kd_loss = kl_div(student_logits, teacher_logits, T)

    return alpha * ce_loss + (1 - alpha) * kd_loss, ce_loss, kd_loss