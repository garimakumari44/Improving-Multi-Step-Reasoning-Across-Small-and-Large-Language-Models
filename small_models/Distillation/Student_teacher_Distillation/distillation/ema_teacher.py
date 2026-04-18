@torch.no_grad()
def update_ema(student, teacher, beta=0.99):
    for ps, pt in zip(student.parameters(), teacher.parameters()):
        pt.data = beta * pt.data + (1 - beta) * ps.data