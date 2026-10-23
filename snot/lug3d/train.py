"""Train S-NOT on lug3d. See README.md for data and usage."""

from run_config import parse_run_args, prepare_run, load_train_index


def main():
    args = parse_run_args("snot", "lug3d")
    prepare_run(args, "snot", "lug3d")
    import time
    from snot.trainer import torch_trainer
    from snot.modules.point_position_embedding import PosEmbLinear, encode_position, position_encoding_channels
    from snot.modules.transformer import SelfAttentionBlocks, MLP, CrossAttentionBlocks, sinusoidal_positional_encoding
    import os
    from torch.utils.data import DataLoader
    import torch.nn.functional as F
    import torch.nn as nn
    import torch
    import random
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    import matplotlib.pyplot as plt
    import numpy as np
    np.random.seed(args.seed)
    from torch.utils.data import Dataset, DataLoader, TensorDataset
    from sklearn.model_selection import train_test_split
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    class COPLoss(nn.Module):
        """
        COP(y_true, y_pred) = sum((y_true - y_pred)^2)
                             / (batch_size * Var(flatten(y_true)))
        Mirrors the TensorFlow version:
          var_true = batch_size * reduce_variance(flatten(y_true))
          data_loss = sum(squared_error) / var_true
        """

        def __init__(self):
            super(COPLoss, self).__init__()

        def forward(self, y_pred, y_true):
            yt = y_true.view(-1)
            yp = y_pred.view(-1)
            sqr_err = (yt - yp).pow(2).sum()
            var_flat = yt.var(unbiased=False)
            var_true = y_true.size(0) * var_flat
            return sqr_err / var_true
    data_loc = str(args.data_dir)
    data_peeq = np.load(os.path.join(data_loc, 'peeq_all.npy')).astype(np.float32)
    data_ystress = np.load(os.path.join(data_loc, 'ystress_all.npy')).astype(np.float32) / 1000000.0
    Heat_Amp = np.load(os.path.join(data_loc, 'data_amp_5000.npy')).astype(np.float32)
    xy_train_testing = np.load(os.path.join(data_loc, 'xyz_coords.npy')).astype(np.float32)
    xy_train_testing_mm = xy_train_testing * 1000.0
    (xy_mean, xy_std) = (np.mean(xy_train_testing, axis=0)[None], np.std(xy_train_testing, axis=0)[None])
    xy_train_testing = (xy_train_testing - xy_mean) / xy_std
    n_cases = args.num_samples
    Heat_Amp = Heat_Amp[:n_cases]
    data_peeq = data_peeq[:n_cases].transpose(0, 2, 1)[..., 1:]
    data_ystress = data_ystress[:n_cases].transpose(0, 2, 1)[..., 1:]
    Nsteps = data_peeq.shape[2]
    print('data shapes: ', xy_train_testing.shape, Heat_Amp.shape, data_peeq.shape, data_ystress.shape)
    N = len(data_peeq)
    n_train = int(0.8 * N)
    np.random.seed(args.seed)
    train_case = np.random.choice(N, n_train, replace=False)
    test_case = np.setdiff1d(np.arange(N), train_case)
    np.save('train_indices.npy', train_case)
    np.save('test_indices.npy', test_case)
    # Fit on training targets only; use the same transform for every case.
    (sigma_mean, sigma_std) = (np.mean(data_ystress[train_case]), np.std(data_ystress[train_case]))
    sigma_std = sigma_std if sigma_std != 0 else 1.0
    sigma_norm = (data_ystress - sigma_mean) / sigma_std
    (pe_mean, pe_std) = (np.mean(data_peeq[train_case]), np.std(data_peeq[train_case]))
    pe_std = pe_std if pe_std != 0 else 1.0
    pe_norm = (data_peeq - pe_mean) / pe_std
    s_norm = np.stack([sigma_norm, pe_norm], axis=-1)
    print('s_norm shape: ', s_norm.shape)

    def solu_inv(y):
        """
        y: numpy array of shape (samples, nodes, 2),
           where [:,:,0] is normalized stress and [:,:,1] is normalized peeq
        """
        y0 = y[..., 0] * sigma_std + sigma_mean
        y1 = y[..., 1] * pe_std + pe_mean
        return np.stack([y0, y1], axis=-1)
    u0_all = Heat_Amp[:, :, None]
    u0_all = torch.from_numpy(u0_all)
    s_all = torch.from_numpy(s_norm)
    xy_train_testing = torch.from_numpy(xy_train_testing)
    print('n_train = ', n_train)
    u0_train = u0_all[train_case]
    u0_test = u0_all[test_case]
    s_train = s_all[train_case]
    s_test = s_all[test_case]
    dataset_train = TensorDataset(u0_train, s_train)
    dataset_test = TensorDataset(u0_test, s_test)
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=args.eval_batch_size, shuffle=False)
    print('setup model...')

    class Branch(nn.Module):

        def __init__(self, input_dim=1, embed_dim=128, self_attn_layers=4, emb_hidden_dims=[128, 128], num_heads=4):
            super(Branch, self).__init__()
            self.gru1 = nn.GRU(input_size=input_dim, hidden_size=256, batch_first=True, bidirectional=False)
            self.gru2 = nn.GRU(input_size=256, hidden_size=128, batch_first=True, bidirectional=False)
            self.time_distributed = nn.Linear(128, embed_dim)
            self.weighted_sum = nn.Linear(2 * embed_dim, embed_dim, bias=False)
            self.pos_encoding = sinusoidal_positional_encoding(length=2048, d_model=embed_dim)
            self.pos_encoding = self.pos_encoding[None, :, :].to(device)
            self.attention_blocks = SelfAttentionBlocks(width=embed_dim, heads=num_heads, layers=self_attn_layers)

        def forward(self, x):
            (out, _) = self.gru1(x)
            (out, _) = self.gru2(out)
            x = self.time_distributed(out)
            (B, S, embed_dim) = x.shape
            x = x * torch.sqrt(torch.tensor(embed_dim, dtype=torch.float32, device=x.device))
            x = torch.cat([x, self.pos_encoding[:, :S, :].repeat(B, 1, 1)], dim=-1)
            x = self.weighted_sum(x)
            x = self.attention_blocks(x)
            return x

    class Truck(nn.Module):

        def __init__(self, branch, embed_dim=128, cross_attn_layers=4, num_heads=4, in_channels=3, out_channels=1, emd_version='nerf'):
            super(Truck, self).__init__()
            self.branch = branch
            self.emd_version = emd_version
            d = position_encoding_channels(emd_version)
            self.Q_encoder = nn.Sequential(nn.Linear(d * in_channels, 2 * embed_dim), nn.SiLU(), nn.Linear(2 * embed_dim, 2 * embed_dim), nn.SiLU(), nn.Linear(2 * embed_dim, embed_dim))
            self.cross_att_blocks = CrossAttentionBlocks(width=embed_dim, heads=num_heads, layers=cross_attn_layers)
            out_channels = out_channels * Nsteps
            self.output_proj = nn.Sequential(nn.Linear(embed_dim, 2 * embed_dim), nn.SiLU(), nn.Linear(2 * embed_dim, 2 * embed_dim), nn.SiLU(), nn.Linear(2 * embed_dim, out_channels))

        def forward(self, inp_fun, x):
            """
            inp_fun: (B, S, in_channels)
            x: (nx, nd)
            """
            B = inp_fun.shape[0]
            br_out = self.branch(inp_fun)
            x = encode_position(self.emd_version, position=x)
            x = self.Q_encoder(x)
            x = x[None].expand(B, -1, -1)
            x = self.cross_att_blocks(x, br_out)
            x = self.output_proj(x)
            return x.view(B, -1, Nsteps, 2)
    model_params = {'nh': 1, 'satt': 3, 'catt': 6, 'embed_dim': 128}
    branch = Branch(input_dim=1, embed_dim=model_params['embed_dim'], self_attn_layers=model_params['satt'], emb_hidden_dims=[128, 128], num_heads=model_params['nh']).to(device)
    snot = Truck(branch, embed_dim=model_params['embed_dim'], cross_attn_layers=model_params['catt'], num_heads=model_params['nh'], in_channels=3, out_channels=2).to(device)
    xy_train_testing = xy_train_testing.to(device)

    class TRAINER(torch_trainer.TorchTrainer):

        def __init__(self, models, device, filebase):
            super().__init__(models, device, filebase)

        def evaluate_losses(self, data):
            inp_fun = data[0].to(self.device)
            y_true = data[1].to(self.device)
            y_pred = self.models[0](inp_fun, xy_train_testing)
            loss = nn.L1Loss()(y_true, y_pred)
            loss_dic = {'loss': loss.item()}
            return (loss, loss_dic)

        def predict(self, data_loader):
            y_pred = []
            y_true = []
            self.models[0].eval()
            with torch.no_grad():
                for data in data_loader:
                    inp_fun = data[0].to(self.device)
                    y_true_batch = data[1].to(self.device)
                    pred = self.models[0](inp_fun, xy_train_testing)
                    y_pred.append(pred.cpu().detach().numpy())
                    y_true.append(y_true_batch.cpu().detach().numpy())
            y_pred = np.concatenate(y_pred, axis=0)
            y_true = np.concatenate(y_true, axis=0)
            return (y_pred, y_true)
    epochs = args.epochs
    file_base = './checkpoints'
    trainer = TRAINER(snot, device, file_base)
    optimizer = torch.optim.AdamW(trainer.parameters(), lr=0.001)
    checkpoint = torch_trainer.ModelCheckpoint(monitor='val_loss', save_best_only=True)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=80)
    trainer.compile(optimizer=optimizer, lr_scheduler=lr_scheduler, checkpoint=checkpoint, scheduler_metric_name='val_loss')
    h = trainer.fit(dataloader_train, val_loader=dataloader_test, epochs=epochs)
    trainer.save_logs()
    print('load weights and predict...')
    trainer.load_weights(device=device)
    start_time = time.time()
    (y_pred, y_true) = trainer.predict(dataloader_test)
    print(f'Prediction time: {time.time() - start_time:.2f} seconds, each sample: {(time.time() - start_time) / len(y_pred):.4f} seconds')
    y_pred = solu_inv(y_pred)
    y_true = solu_inv(y_true)
    (n_samples, n_nodes, _, _) = y_true.shape
    error_s = np.linalg.norm(y_true[:, :, :, 0] - y_pred[:, :, :, 0], axis=(1, 2)) / (np.linalg.norm(y_true[:, :, :, 0], axis=(1, 2)) + 1e-08)
    print(f'Mean L2 error (stress) for test data: {np.mean(error_s)}, std: {np.std(error_s)}, max: {np.max(error_s)}')
    abs_error_p = np.abs(y_pred[:, :, :, 1] - y_true[:, :, :, 1])
    abs_error_p_samp = np.mean(abs_error_p, axis=(1, 2))
    print('mean of absolute error of Peeq: {:.4e}'.format(np.mean(abs_error_p_samp)))
    print('std of absolute error of Peeq: {:.4e}'.format(np.std(abs_error_p_samp)))
    print('max of absolute error of Peeq: {:.4e}'.format(np.max(abs_error_p_samp)))
    stress_true_test = y_true[..., 0]
    stress_pred_test = y_pred[..., 0]
    peeq_true_test = y_true[..., 1]
    peeq_pred_test = y_pred[..., 1]
    np.savez_compressed('predictions.npz', a=peeq_true_test, b=stress_true_test, c=peeq_pred_test, d=stress_pred_test, e=xy_train_testing_mm)


if __name__ == "__main__":
    main()
