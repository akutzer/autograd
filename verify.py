import torch

torch.set_printoptions(precision=10)
def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    tmp = (x.log() + (-x) * y - y.sin())
    tmp2 = tmp
    for i in range(1, 5):
        tmp = tmp * i
    return tmp / ((2 * x).cos() + tmp2).exp()

    tmp = (x.log() + (-x) * y - y.sin())
    tmp2 = tmp
    tmp = tmp * 2
    return tmp / ((2 * x).cos() + tmp2).exp()



x = torch.tensor(2., requires_grad=True)
y = torch.tensor(5., requires_grad=True)

out = f(x, y)
out.backward()
print(out)
print(x.grad)
print(y.grad)