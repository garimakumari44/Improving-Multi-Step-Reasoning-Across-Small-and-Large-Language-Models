import torch
from training.train_step import train_step
from data.dataloader import get_loader
from models.teacher import TeacherModel
from models.student import StudentModel

def train(epochs=5, lr=3e-4, device="cpu"):

    loader = get_loader()

    teacher = TeacherModel().to(device)
    student = StudentModel().to(device)

    # freeze teacher
    for p in teacher.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(student.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0

        for batch in loader:
            loss, ce, kd = train_step(
                student, teacher, batch, optimizer, device=device
            )
            total_loss += loss

        print(f"Epoch {epoch+1} | Loss: {total_loss:.4f}")

    return student, teacher