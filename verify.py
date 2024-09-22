import torch
import torch.autograd as autograd
torch.set_printoptions(precision=10)



def f_xy(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    tmp = (x.log() + (-x) * y - y.sin())
    if (tmp * 2) < 0:
        tmp = tmp * tmp
    tmp2 = tmp
    for i in range(1, 5):
        tmp = tmp * ((y - x) / i).exp()
    return tmp / ((2 * x).cos().abs() + tmp2)


# Function with one input (x)
def f_x(x: torch.Tensor) -> torch.Tensor:
    a = x.log()
    b = a.cos()
    c = a.exp()
    d = b + c
    e = b.exp()
    return d

if __name__ == "__main__":
    print("Testing f(x):")
    x = torch.tensor(2., requires_grad=True)

    out_x = f_x(x)

    df_dx = autograd.grad(out_x, x, retain_graph=True, create_graph=True)[0]
    print("df/dx for f(x):", df_dx)

    ddf_dxx = autograd.grad(df_dx, x, retain_graph=True, create_graph=True, materialize_grads=True)[0]
    print("d²f/dx² for f(x):", ddf_dxx)

    dddf_dxxx = autograd.grad(ddf_dxx, x, retain_graph=True, materialize_grads=True)[0]
    print("d³f/dx³ for f(x):", dddf_dxxx)


    print("\n\nTesting f(x, y):")
    x = torch.tensor(2., requires_grad=True)
    y = torch.tensor(5., requires_grad=True)

    out_xy = f_xy(x, y)

    df_dx = autograd.grad(out_xy, x, retain_graph=True, create_graph=True)[0]
    df_dy = autograd.grad(out_xy, y, create_graph=True)[0]
    print("df/dx for f(x, y):", df_dx)
    print("df/dy for f(x, y):", df_dy)

    dff_dxx = autograd.grad(df_dx, x, retain_graph=True, materialize_grads=True)[0]
    dff_dxy = autograd.grad(df_dx, y, retain_graph=True, materialize_grads=True)[0]
    dff_dyx = autograd.grad(df_dy, x, retain_graph=True, materialize_grads=True)[0]
    dff_dyy = autograd.grad(df_dy, y, retain_graph=True, materialize_grads=True)[0]
    print("d²f/dx² for f(x, y):", dff_dxx)
    print("d²f/dxdy for f(x, y):", dff_dxy)
    print("d²f/dydx for f(x, y):", dff_dyx)
    print("d²f/dy² for f(x, y):", dff_dyy)

