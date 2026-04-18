from models.base_model import BaseModel

class StudentModel(BaseModel):
    def __init__(self):
        super().__init__(input_dim=128, hidden_dim=128, output_dim=10)