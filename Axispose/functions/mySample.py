import glob
import torch
import tqdm
import torchvision.utils as tvu
import os
import numpy as np

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

def generalized_steps(x, seq, model, b, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to('cuda')
            et = model(xt, t)
            x0_t = (xt[:,:3,:,:] - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))
            c1 = (
                kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to('cpu'))

    return xs, x0_preds

def sample_image(x, model, args, betas,last=True):
    try:
        skip = args.skip
    except Exception:
        skip = 1
    num_timesteps = betas.shape[0]
    if args.sample_type == "generalized":
        if args.skip_type == "uniform":
            skip = num_timesteps // args.timesteps
            seq = range(0, num_timesteps, skip)
        elif args.skip_type == "quad":
            seq = (
                    np.linspace(
                        0, np.sqrt(num_timesteps * 0.8), args.timesteps
                    )
                    ** 2
            )
            seq = [int(s) for s in list(seq)]
        else:
            raise NotImplementedError

        xs = generalized_steps(x, seq, model, betas, eta=args.eta)
        x = xs

    if last:
        x = x[0][-1]
    return x

def inverse_data_transform(config, X):
    if hasattr(config, "image_mean"):
        X = X + config.image_mean.to(X.device)[None, ...]

    if config.data.logit_transform:
        X = torch.sigmoid(X)
    elif config.data.rescaled:
        X = (X + 1.0) / 2.0

    return torch.clamp(X, 0.0, 1.0)

def sample_fid(model, args, config,rgb, beta, last=True):
    image_folder = os.path.join(args.exp, "image_samples_lyt", args.image_folder)
    img_id = len(glob.glob(f"{image_folder}/*"))
    print(f"starting from image {img_id}")

    with torch.no_grad():
        n = config.sampling.batch_size
        x = torch.randn(
            n,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=rgb.device,
        )
        x = torch.cat((x, rgb), dim=1)
        x = sample_image(x, model, args, betas=beta)
        x = inverse_data_transform(config, x)

        for i in range(n):
            tvu.save_image(
                x[i], os.path.join(image_folder, f"{img_id}.png")
            )
            img_id += 1