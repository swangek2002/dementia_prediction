import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import wandb

class FCNet(nn.Module):
    """Fully Connected Neural Network with Dynamic Hidden Layers"""

    def __init__(self, input_dim, hidden_dim, output_dim, output_act, device="cpu"):

        super().__init__()

        self.device = device
        logging.debug(f"FCNet: Using {self.device} as the device")

        match hidden_dim:
            case None:
                hidden_dim = []
            case int():
                hidden_dim = [hidden_dim]
            case list():
                pass
            case _:
                logging.warning(f"Invalid type {type(hidden_dim)} for hidden_dim")
                raise NotImplementedError
        
        layers = []
        for lyr in range(len(hidden_dim)):
            layers.append(nn.Linear(input_dim, hidden_dim[lyr]))
            layers.append(nn.ReLU())
            input_dim = hidden_dim[lyr]

        layers.append(nn.Linear(input_dim, output_dim))
        layers.append(output_act)
        
        layers = nn.ModuleList(layers)
        self.mapping = nn.Sequential(*layers)

    def forward(self, x):
        return self.mapping(x)


class CondODENet(nn.Module):
    """Conditional ODE Neural Network"""

    def __init__(self, input_dim, hidden_dim, output_dim, device="cpu", n=15, modified=True):

        super().__init__()

        self.device = device #torch.device("cuda:0" if device == "gpu" and torch.cuda.is_available() else "cpu")
        logging.debug(f"CondODENet: Using {self.device} as the device")
        self.modified = modified

        self.output_dim = output_dim
        self.n = n
        u, w = np.polynomial.legendre.leggauss(n)
        self.u = nn.Parameter(torch.tensor(u, device=self.device, dtype=torch.float32)[None, :], requires_grad=False)
        self.w = nn.Parameter(torch.tensor(w, device=self.device, dtype=torch.float32)[None, :], requires_grad=False)

        self.BaseNet = FCNet(input_dim, hidden_dim, output_dim, nn.Softplus(), device)
        self.BaseNet = self.BaseNet.to(self.BaseNet.device)

    def ode_mapping(self, x, t):

        z = torch.cat((x, t), 1)
        return self.BaseNet(z)

    def _sparse_basenet(self, z, col_indices):
        """Forward through BaseNet computing only specific output columns.
        Avoids the massive (M, 108K) matmul by only computing needed columns."""
        hidden = z
        layers = list(self.BaseNet.mapping)
        for layer in layers[:-2]:
            hidden = layer(hidden)
        last_linear = layers[-2]
        out_act = layers[-1]
        w = last_linear.weight[col_indices]
        b = last_linear.bias[col_indices]
        sparse_out = F.linear(hidden, w, b)
        return out_act(sparse_out)

    def ode_mapping_sparse(self, x, t, col_indices):
        z = torch.cat((x, t), 1)
        return self._sparse_basenet(z, col_indices)

    def forward_sparse(self, x, t, col_indices):
        """Like forward() but only computes specific output columns."""
        tau = torch.matmul(t / 2, 1 + self.u)
        tau_ = torch.flatten(tau).unsqueeze(1)
        reppedx = x.repeat_interleave(self.n, dim=0)
        dudt = self.ode_mapping_sparse(reppedx, tau_, col_indices)
        n_cols = len(col_indices)
        f = dudt.reshape((*tau.shape, n_cols))
        pred = t / 2 * ((self.w.unsqueeze(2) * f).sum(dim=1))
        pred = pred.to(self.device)
        if self.modified:
            return 1 - torch.exp(-pred)
        else:
            return torch.tanh(pred)

    def forward(self, x, t):

        # Compute quadrature points (scaling between 0 and t)
        tau = torch.matmul(t / 2, 1 + self.u)
        tau_ = torch.flatten(tau).unsqueeze(1)
        reppedx = x.repeat_interleave(self.n, dim=0)

        dudt = self.ode_mapping(reppedx, tau_)
        f = dudt.reshape((*tau.shape, self.output_dim))
        pred = t / 2 * ((self.w.unsqueeze(2) * f).sum(dim=1))
        pred = pred.to(self.device)

        if self.modified:
            return 1 - torch.exp(-pred)
        else:
            return torch.tanh(pred)
            

class ODESurvSingle(nn.Module):
    def __init__(self, 
                 cov_dim, 
                 hidden_dim,
                 device="cpu",
                 n=15, 
                 modified=True):

        super().__init__()

        input_dim = cov_dim + 1
        self.net = CondODENet(input_dim, hidden_dim, 1, device=device, n=n, modified=modified)
        self.net = self.net.to(self.net.device)
        self.modified = modified

        self.lr = 1e-3
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x, t):
        x = x.to(self.net.device)
        t = t.to(self.net.device)
        t = t.unsqueeze(1)
        return self.net.forward(x, t).squeeze()

    def predict(self, x, t):
        # wrap back for original optimize code
        return self.forward(x, t)

    def loss(self, x, t, k):
        # print(f"x: {x.shape}, t:{t.shape} k:{k.shape}")
        x = x.to(self.net.device)
        t = t.to(self.net.device)
        k = k.to(self.net.device)

        t = t.unsqueeze(1)
        eps = 1e-8

        censterm = torch.tensor(0)
        cens_ids = torch.where(k == 0)[0]
        if torch.numel(cens_ids) != 0:
            cdf_cens = self.net.forward(x[cens_ids, :], t[cens_ids, :]).squeeze()
            censterm = torch.log(1 - cdf_cens + eps).sum()

        uncensterm = torch.tensor(0)
        uncens_ids = torch.where(k == 1)[0]
        if torch.numel(uncens_ids) != 0:
            cdf_uncens = self.net.forward(x[uncens_ids, :], t[uncens_ids, :]).squeeze()
            if not self.modified:
                cdf_uncens = cdf_uncens ** 2
            dudt_uncens = self.net.ode_mapping(x[uncens_ids, :], t[uncens_ids, :]).squeeze()
            uncensterm = (torch.log(1 - cdf_uncens + eps) + torch.log(dudt_uncens + eps)).sum()

        return -(censterm + uncensterm)

    def optimize(self, data_loader, n_epochs, logging_freq=10, data_loader_val=None,
                 max_wait=20):
        batch_size = data_loader.batch_size

        if data_loader_val is not None:
            best_val_loss = np.inf
            wait = 0

        for epoch in range(n_epochs):

            train_loss = 0.0

            for batch_idx, (x, t, k) in enumerate(data_loader):
                argsort_t = torch.argsort(t)
                x_ = x[argsort_t,:]
                t_ = t[argsort_t]
                k_ = k[argsort_t]

                self.optimizer.zero_grad()
                loss = self.loss(x_,t_,k_)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            if epoch % logging_freq == 0:
                print(f"\tEpoch: {epoch:2}. Total loss: {train_loss:11.2f}")
                if data_loader_val is not None:
                    val_loss = 0
                    for batch_idx, (x, t, k) in enumerate(data_loader_val):
                        argsort_t = torch.argsort(t)
                        x_ = x[argsort_t,:]
                        t_ = t[argsort_t]
                        k_ = k[argsort_t]

                        loss = self.loss(x_,t_,k_)
                        val_loss += loss.item()

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        wait = 0
                        print(f"best_epoch: {epoch}")
                        torch.save(self.state_dict(), "low")
                    else:
                        wait += 1

                    if wait > max_wait:
                        state_dict = torch.load("low")
                        self.load_state_dict(state_dict)
                        return

                    print(f"\tEpoch: {epoch:2}. Total val loss: {val_loss:11.2f}")
        if data_loader_val is not None:
            state_dict = torch.load("low")
            self.load_state_dict(state_dict)
        

class ODESurvMultiple(nn.Module):
    def __init__(self, 
                 cov_dim, 
                 hidden_dim,
                 num_risks,
                 device="cpu",
                 n=15):
        super().__init__()

        input_dim = cov_dim + 1
        self.pinet = FCNet(cov_dim, hidden_dim, num_risks, nn.Softmax(dim=1), device)
        self.pinet = self.pinet.to(self.pinet.device)

        self.odenet = CondODENet(input_dim, hidden_dim, num_risks, device, n)
        self.odenet = self.odenet.to(self.odenet.device)

        self.K = num_risks

        self.lr = 1e-3
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def get_pi(self, x):
        return self.pinet(x)

    def forward(self, x, t):

        t = t.unsqueeze(1)
        pi = self.get_pi(x)
        preds = pi * self.odenet.forward(x, t)

        # print(f"x shape {x.shape}")
        # print(f"with preds {preds[0,:]}")

        return preds, pi

    def predict(self, x, t):
        # wrap back for original optimize code
        return self.forward(x, t)

    def loss(self, x, t, k):

        t = t.unsqueeze(1)
        eps = 1e-8

        censterm = torch.tensor(0, device=x.device, dtype=x.dtype)
        cens_ids = torch.where(k == 0)[0]
        if torch.numel(cens_ids) != 0:
            cif_cens = self.forward(x[cens_ids, :], t[cens_ids, 0])[0]
            cdf_cens = cif_cens.sum(dim=1)
            censterm = torch.log(1 - cdf_cens + eps).sum()

        # Sparse uncensored term: instead of computing all 108K output columns,
        # only compute the columns corresponding to events that actually occurred.
        # Reduces the main matmul from (M, 32)x(32, 108K) to (M, 32)x(32, ~500).
        uncensterm = torch.tensor(0, device=x.device, dtype=x.dtype)
        uncens_mask = k > 0
        if uncens_mask.any():
            x_u = x[uncens_mask]
            t_u = t[uncens_mask]
            k_raw = (k[uncens_mask] - 1).long()

            unique_events, inverse_idx = torch.unique(k_raw, return_inverse=True)

            # ODE CDF and dudt: sparse computation (~500 cols instead of 108K)
            cdf_sparse = self.odenet.forward_sparse(x_u, t_u, unique_events)
            dudt_sparse = self.odenet.ode_mapping_sparse(x_u, t_u, unique_events)

            cdf_sel = cdf_sparse[torch.arange(len(x_u), device=x.device), inverse_idx]
            dudt_sel = dudt_sparse[torch.arange(len(x_u), device=x.device), inverse_idx]

            pi = self.get_pi(x_u)
            pi_sel = pi[torch.arange(len(x_u), device=x.device), k_raw]

            uncensterm = (torch.log(1 - cdf_sel + eps)
                          + torch.log(dudt_sel + eps)
                          + torch.log(pi_sel + eps)).sum()

        return -(censterm + uncensterm)

    def optimize(self, data_loader, n_epochs, logging_freq=10, data_loader_val=None,
                 max_wait=20):
   
        batch_size = data_loader.batch_size

        if data_loader_val is not None:
            best_val_loss = np.inf
            wait = 0

        for epoch in range(n_epochs):

            train_loss = 0.0

            for batch_idx, (x, t, k) in enumerate(data_loader):
                argsort_t = torch.argsort(t)
                x_ = x[argsort_t,:].to(self.odenet.device)
                t_ = t[argsort_t].to(self.odenet.device)
                k_ = k[argsort_t].to(self.odenet.device)

                self.optimizer.zero_grad()
                loss = self.loss(x_,t_,k_)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            if epoch % logging_freq == 0:
                print(f"\tEpoch: {epoch:2}. Total loss: {train_loss:11.2f}")
                if data_loader_val is not None:
                    val_loss = 0
                    for batch_idx, (x, t, k) in enumerate(data_loader_val):
                        argsort_t = torch.argsort(t)
                        x_ = x[argsort_t,:].to(self.odenet.device)
                        t_ = t[argsort_t].to(self.odenet.device)
                        k_ = k[argsort_t].to(self.odenet.device)

                        loss = self.loss(x_,t_,k_)
                        val_loss += loss.item()

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        wait = 0
                        print(f"best_epoch: {epoch}")
                        torch.save(self.state_dict(), "low_")
                    else:
                        wait += 1

                    if wait > max_wait:
                        state_dict = torch.load("low_")
                        self.load_state_dict(state_dict)
                        return

                    print(f"\tEpoch: {epoch:2}. Total val loss: {val_loss:11.2f}")
        if data_loader_val is not None:
            print("loading low_")
            state_dict = torch.load("low_")
            self.load_state_dict(state_dict)
            return
