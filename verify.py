import torch
import torch.autograd as autograd
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

    df_dx = autograd.grad(out, x, create_graph=True)[0]
    df_dy = autograd.grad(out, y, create_graph=True)[0]
    print(df_dx)
    print(df_dy)

    dff_dxx = autograd.grad(df_dx, x, retain_graph=True, materialize_grads=True)[0]
    dff_dxy = autograd.grad(df_dx, y, retain_graph=True, materialize_grads=True)[0]
    dff_dyx = autograd.grad(df_dy, x, retain_graph=True, materialize_grads=True)[0]
    dff_dyy = autograd.grad(df_dy, y, retain_graph=True, materialize_grads=True)[0]
    print(dff_dxx)
    print(dff_dxy)
    print(dff_dyx)
    print(dff_dyy)
