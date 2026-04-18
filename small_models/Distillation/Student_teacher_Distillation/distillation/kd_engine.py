class DistillationEngine:
    def __init__(self, student, teacher):
        self.student = student
        self.teacher = teacher

    def forward(self, x):
        with torch.no_grad():
            teacher_logits = self.teacher(x)

        student_logits = self.student(x)

        return student_logits, teacher_logits