import torch.nn.functional as F

def kl_div(student_logits, teacher_logits, T=2.0):
    student_log_prob = F.log_softmax(student_logits / T, dim=-1)
    teacher_prob = F.softmax(teacher_logits / T, dim=-1)

    return F.kl_div(student_log_prob, teacher_prob, reduction="batchmean") * (T * T)