import torch
from losses.distillation_loss import distillation_loss

def train_step(student, teacher, batch, optimizer, T=2.0, alpha=0.6, device="cpu"):
    student.train()
    teacher.eval()

    x, y = batch
    x, y = x.to(device), y.to(device)

    with torch.no_grad():
        teacher_logits = teacher(x)

    student_logits = student(x)

    loss, ce, kd = distillation_loss(
        student_logits,
        teacher_logits,
        y,
        alpha=alpha,
        T=T
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), ce.item(), kd.item()