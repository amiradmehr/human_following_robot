

class PID:
    def __init__(self, P, I, D):
        self.P = P
        self.I = I
        self.D = D
        self.prev_error = 0
        self.integral = 0

    def update(self, error):
        self.integral += error
        derivative = (error - self.prev_error)
        self.prev_error = error
        return self.P * error + self.I * self.integral + self.D * derivative


