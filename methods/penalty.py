def penalty_grade(lam: float, epoch: int, num_epochs: int):
    k = 6
    x = (epoch/num_epochs)**(1/k)
    return lam*(1 - (1/(1 + ((1-x)/(x))**k)))