import random
import plotly.express as px
class Die:
    def __init__(self, num_sides1, num_sides2, times):
        self.num_sides1 = num_sides1
        self.num_sides2 = num_sides2
        self.times = times
    def poss_process_draw(self):
        frequences = list()
        frequences_count = list()
        a = 0
        while a <= self.times:
            b = random.choice([value for value in range(1, self.num_sides1 + 1)])
            c = random.choice([value for value in range(1, self.num_sides2 + 1)])
            sum_number = b + c
            frequences.append(sum_number)
            a += 1
        for value in range(2, self.num_sides1 + self.num_sides2 + 1):
            frequences_count.append(frequences.count(value))
        fig = px.bar(x = [value for value in range(2, self.num_sides1 + self.num_sides2 + 1)], y = frequences_count)
        fig.update_layout(xaxis_dtick = 1)
        fig.show()

real = Die(6, 12,5000)
real.poss_process_draw()
