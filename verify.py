import torch


torch.set_printoptions(precision=10)
def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    tmp = (x.log() + (-x) * y - y.sin())
    if (tmp * 2) < 0:
        tmp = tmp * tmp
    tmp2 = tmp
    for i in range(1, 5):
        tmp = tmp * ((y - x) / i).exp()
    return tmp / ((2 * x).cos().abs() + tmp2)


if __name__ == "__main__":
    x = torch.tensor(2., requires_grad=True)
    y = torch.tensor(5., requires_grad=True)

    out = f(x, y)
    out.backward()
    print(out)
    print(x.grad)
    print(y.grad)
