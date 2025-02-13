import torch
import torch.nn as nn
from src import utils


class head_attention(nn.Module):
    def __init__(self, d, d_head, drop=0.0):
        super().__init__()
        self.Q = nn.Linear(d, d_head)
        self.K = nn.Linear(d, d_head)
        self.E = nn.Linear(d, d_head)
        self.V = nn.Linear(d, d_head)
        self.sqrt_d = torch.sqrt(torch.tensor(d_head))
        self.drop_att = nn.Dropout(drop)
        self.Ni = nn.Linear(d, d_head)
        self.Nj = nn.Linear(d, d_head)
    def forward(self, x, e, node_mask):
        x_mask = node_mask.unsqueeze(-1) # [bs, n, 1]
        e_mask_1 = x_mask.unsqueeze(2)   # [bs, n, 1, 1]
        e_mask_2 = x_mask.unsqueeze(1)  # [bs, 1, n, 1]
        Q = self.Q(x) * x_mask # [bs, n, d_head]
        K = self.K(x) * x_mask # [bs, n, d_head]
        V = self.V(x) * x_mask # [bs, n, d_head]
        Q = Q.unsqueeze(2) # [bs, n, 1, d_head]
        K = K.unsqueeze(1) # [bs, 1, n, d_head]
        E = self.E(e) * e_mask_1 * e_mask_2 # [bs, n, n, d_head]
        Ni = self.Ni(x).unsqueeze(2) * e_mask_1 # [bs, n, 1, d_head]
        Nj = self.Nj(x).unsqueeze(1) * e_mask_2 # [bs, 1, n, d_head]
        e = (Ni + Nj + E) * e_mask_1 * e_mask_2 # [bs, n, n, d_head]
        Att = (Q * e * K).sum(dim=3) / self.sqrt_d # [bs, n, n]
        att_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
        att_mask = att_mask.bool()
        Att = Att.masked_fill(~att_mask, -1e9)
        Att = torch.softmax(Att, dim=1)            # [bs, n, n]
        Att = self.drop_att(Att)                   # [bs, n, n]
        x = (Att @ V) * x_mask                     # [bs, n, d_head]
        return x, e                  # [bs, n, d_head], [bs, n, n, d_head]


class MHA(nn.Module):
    def __init__(self, d, num_heads, drop=0.0):  
        super().__init__()
        d_head = d // num_heads
        self.heads = nn.ModuleList([head_attention(d, d_head, drop) for _ in range(num_heads)])
        self.WOx = nn.Linear(d, d)
        self.WOe = nn.Linear(d, d)
        self.drop_x = nn.Dropout(drop)
        self.drop_e = nn.Dropout(drop)
    def forward(self, x, e, node_mask):
        x_mask = node_mask.unsqueeze(-1) # [bs, n, 1]
        e_mask_1 = x_mask.unsqueeze(2)  # [bs, n, 1, 1]
        e_mask_2 = x_mask.unsqueeze(1)  # [bs, 1, n, 1]
        x_MHA = []
        e_MHA = []    
        for head in self.heads:
            x_HA, e_HA = head(x,e, node_mask) # [bs, n, d_head], [bs, n, n, d_head]
            x_MHA.append(x_HA)
            e_MHA.append(e_HA)
        x = self.WOx(torch.cat(x_MHA, dim=2)) # [bs, n, d]
        x = x * x_mask                        # [bs, n, d]
        x = self.drop_x(x)                    # [bs, n, d]
        e = self.WOe(torch.cat(e_MHA, dim=3)) # [bs, n, n, d]
        e = e * e_mask_1 * e_mask_2           # [bs, n, n, d]
        e = self.drop_e(e)                    # [bs, n, n, d]
        return x, e                           # [bs, n, d], [bs, n, n, d]


class BlockGT(nn.Module):
    def __init__(self, d, num_heads, drop=0.0):  
        super().__init__()
        self.LNx = nn.LayerNorm(d)
        self.LNe = nn.LayerNorm(d)
        self.LNx2 = nn.LayerNorm(d)
        self.LNe2 = nn.LayerNorm(d)
        self.MHA = MHA(d, num_heads, drop)
        self.MLPx = nn.Sequential(nn.Linear(d, 4*d), nn.LeakyReLU(), nn.Linear(4*d, d))
        self.MLPe = nn.Sequential(nn.Linear(d, 4*d), nn.LeakyReLU(), nn.Linear(4*d, d))
        self.drop_x_mlp = nn.Dropout(drop)
        self.drop_e_mlp = nn.Dropout(drop)
    def forward(self, x, e, node_mask):
        x_mask = node_mask.unsqueeze(-1) # [bs, n, 1]
        e_mask_1 = x_mask.unsqueeze(2)  # [bs, n, 1, 1]
        e_mask_2 = x_mask.unsqueeze(1)  # [bs, 1, n, 1]
        x = self.LNx(x)                 # [bs, n, d]
        e = self.LNe(e)                 # [bs, n, n, d]
        x_MHA, e_MHA = self.MHA(x, e, node_mask)   # [bs, n, d], [bs, n, n, d]
        x = x + x_MHA                   # [bs, n, d]
        x_mlp = self.MLPx(self.LNx2(x)) # [bs, n, d]
        x_mlp = x_mlp * x_mask          # [bs, n, d]
        x = x + x_mlp                   # [bs, n, d]
        x = self.drop_x_mlp(x)          # [bs, n, d]
        e = e + e_MHA                       # [bs, n, n, d]
        e_mlp = self.MLPe(self.LNe2(e))     # [bs, n, n, d]
        e_mlp = e_mlp * e_mask_1 * e_mask_2 # [bs, n, n, d]
        e = e + e_mlp                       # [bs, n, n, d]
        e = self.drop_e_mlp(e)              # [bs, n, n, d]
        return x, e                         # [bs, n, d], [bs, n, n, d]


class GraphTransformer(nn.Module):
    """
    n_layers : int -- number of layers
    dims : dict -- contains dimensions for each feature type
    """
    def __init__(self, n_layers: int, input_dims: dict, hidden_mlp_dims: dict, hidden_dims: dict,
                 output_dims: dict, act_fn_in=nn.ReLU(), act_fn_out=nn.ReLU()):
        super().__init__()
        self.n_layers = n_layers
        self.out_dim_X = output_dims['X']
        self.out_dim_E = output_dims['E']
        self.out_dim_y = output_dims['y']
        hidden_dims["dx"] = hidden_dims["de"]
        self.mlp_in_X = nn.Sequential(
            nn.Linear(input_dims['X'], hidden_mlp_dims['X']), act_fn_in,
            nn.Linear(hidden_mlp_dims['X'], hidden_dims['dx']), act_fn_in
        )
        self.mlp_in_E = nn.Sequential(
            nn.Linear(input_dims['E'], hidden_mlp_dims['E']), act_fn_in,
            nn.Linear(hidden_mlp_dims['E'], hidden_dims['de']), act_fn_in
        )
        self.mlp_in_y = nn.Sequential(
            nn.Linear(input_dims['y'], hidden_mlp_dims['y']), act_fn_in,
            nn.Linear(hidden_mlp_dims['y'], hidden_dims['dy']), act_fn_in
        )
        self.tf_layers = nn.ModuleList(
            [
                BlockGT(
                    d=hidden_dims['dx'],
                    num_heads=hidden_dims['n_head'],
                    drop=0.0
                )
                for _ in range(n_layers)
            ]
        )
        self.mlp_out_X = nn.Sequential(
            nn.Linear(hidden_dims['dx'], hidden_mlp_dims['X']), act_fn_out,
            nn.Linear(hidden_mlp_dims['X'], output_dims['X'])
        )
        self.mlp_out_E = nn.Sequential(
            nn.Linear(hidden_dims['de'], hidden_mlp_dims['E']), act_fn_out,
            nn.Linear(hidden_mlp_dims['E'], output_dims['E'])
        )
        self.mlp_out_y = nn.Sequential(
            nn.Linear(hidden_dims['dy'], hidden_mlp_dims['y']), act_fn_out,
            nn.Linear(hidden_mlp_dims['y'], output_dims['y'])
        )

    def forward(self, X, E, y, node_mask):
        bs, n = X.shape[0], X.shape[1]

        diag_mask = torch.eye(n)
        diag_mask = ~diag_mask.type_as(E).bool()
        diag_mask = diag_mask.unsqueeze(0).unsqueeze(-1).expand(bs, -1, -1, -1)

        X_to_out = X[..., :self.out_dim_X]
        E_to_out = E[..., :self.out_dim_E]
        y_to_out = y[..., :self.out_dim_y]

        new_E = self.mlp_in_E(E)
        new_E = (new_E + new_E.transpose(1, 2)) / 2
        after_in = utils.PlaceHolder(X=self.mlp_in_X(X), E=new_E, y=self.mlp_in_y(y)).mask(node_mask)
        X, E, y = after_in.X, after_in.E, after_in.y

        for layer in self.tf_layers:
            X, E = layer(X, E, node_mask)

        X = self.mlp_out_X(X)
        E = self.mlp_out_E(E)
        y = self.mlp_out_y(y)

        X = (X + X_to_out)
        E = (E + E_to_out) * diag_mask
        y = y + y_to_out

        E = 1/2 * (E + torch.transpose(E, 1, 2))

        return utils.PlaceHolder(X=X, E=E, y=y).mask(node_mask)
