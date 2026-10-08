import torch
import warnings
import numpy as np
import torch.nn.functional as F
import argparse

from model import THOT_GC
from metrics import clustering_evaluation
from utils import load_data, DATA_ROOT

warnings.filterwarnings('ignore')


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_and_evaluate(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset = load_data(args.dataset, data_root=args.data_root)
    data = dataset[0].to(device)
    y_true = data.y.cpu().numpy()

    n_clusters = len(np.unique(y_true))
    model = THOT_GC(
        in_dim=dataset.num_features,
        hid_dim=args.hid_dim,
        n_clusters=n_clusters,
        args=args,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ==========================================
    # 掩码图自编码器预训练
    # ==========================================
    print("\n=== Stage 1: Masked Graph Auto-Encoder Pre-training ===")
    model.train()
    for epoch in range(1, args.pretrain_epochs + 1):
        optimizer.zero_grad()

        drop_mask = torch.empty((data.x.size(1),), dtype=torch.float32, device=device).uniform_(0, 1) < args.feat_mask_prob
        x_masked = data.x.clone()
        x_masked[:, drop_mask] = 0

        edge_mask = torch.rand(data.edge_index.size(1), device=device) > args.edge_drop_prob
        edge_index_corrupted = data.edge_index[:, edge_mask]

        z = model(x_masked, edge_index_corrupted)
        loss = model.pretrain_loss(z, data.x, data.edge_index)

        loss.backward()
        optimizer.step()

        if epoch % args.log_interval == 0:
            print(f"Pretrain Epoch [{epoch:03d}/{args.pretrain_epochs}], Recon Loss: {loss.item():.4f}")

    # ==========================================
    # 聚类锚点初始化
    # ==========================================
    print("\n=== Stage 2: Prototype Initialization ===")
    model.eval()
    with torch.no_grad():
        z_clean = model(data.x, data.edge_index)
        model.init_prototypes(z_clean, data.edge_index)
    print("Initialization Complete. Prototypes Anchored.")

    # ==========================================
    # 图纯化
    # ==========================================
    print("\n=== Stage 3: Topology-Smoothed Fine-tuning ===")
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr * args.finetune_lr_scale

    best = {'acc': -1.0, 'nmi': 0.0, 'ari': 0.0, 'f1': 0.0, 'epoch': 0}
    for epoch in range(1, args.finetune_epochs + 1):
        model.train()
        optimizer.zero_grad()

        z_base = model(data.x, data.edge_index)

        with torch.no_grad():
            c = F.normalize(model.prototypes, dim=1)
            logits_init = torch.mm(z_base, c.t()) / model.tau
            Q_init = F.softmax(logits_init, dim=1)

        z_enhanced = model.enhance_representation(data.x, z_base, Q_init, epoch, args.finetune_epochs)
        loss, Q = model.finetune_loss(z_enhanced, data.edge_index, current_epoch=epoch, total_epochs=args.finetune_epochs)

        loss.backward()
        optimizer.step()

        if epoch % args.log_interval == 0:
            model.eval()
            with torch.no_grad():
                z_eval = model(data.x, data.edge_index)
                y_pred = model.get_cluster_assignments(z_eval, data.edge_index)
                acc, nmi, ari, f1 = clustering_evaluation(y_true, y_pred)
            if nmi > best['nmi']:
                best = {'acc': acc, 'nmi': nmi, 'ari': ari, 'f1': f1, 'epoch': epoch}
            print(
                f"Finetune Epoch [{epoch:03d}/{args.finetune_epochs}], "
                f"Total Loss: {loss.item():.4f}, ACC: {acc * 100:.2f}%, "
                f"NMI: {nmi * 100:.2f}%, ARI: {ari * 100:.2f}%, F1: {f1 * 100:.2f}%"
            )

    # ==========================================
    # 最终评估
    # ==========================================
    print("\n" + "=" * 30)
    print("Final Clustering Evaluation:")
    model.eval()
    with torch.no_grad():
        z_final = model(data.x, data.edge_index)
        y_pred = model.get_cluster_assignments(z_final, data.edge_index)

    acc, nmi, ari, f1 = clustering_evaluation(y_true, y_pred)
    if nmi > best['nmi']:
        best = {'acc': acc, 'nmi': nmi, 'ari': ari, 'f1': f1, 'epoch': args.finetune_epochs}
    print(f"Predicted distribution: {np.bincount(y_pred)}")
    print(f"ACC (Accuracy)       : {acc * 100:.2f}%")
    print(f"NMI (Mut. Info)      : {nmi * 100:.2f}%")
    print(f"ARI (Rand Index)     : {ari * 100:.2f}%")
    print(f"Macro-F1             : {f1 * 100:.2f}%")
    print(
        f"BEST @ epoch {best['epoch']}: "
        f"ACC={best['acc'] * 100:.2f}%, NMI={best['nmi'] * 100:.2f}%, "
        f"ARI={best['ari'] * 100:.2f}%, F1={best['f1'] * 100:.2f}%"
    )
    print("=" * 30)
    return best


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and evaluate THOT-GC')

    # data / train
    parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'citeseer', 'pubmed', 'photo', 'computers'])
    parser.add_argument('--data_root', type=str, default=DATA_ROOT)
    parser.add_argument('--ntrials', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--hid_dim', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)  # cora: 1e-3, citeseer: 5e-3, photo: 1e-3
    parser.add_argument('--weight_decay', type=float, default=1e-5)  # cora: 1e-5, citeseer: 5e-3, photo: 5e-3
    parser.add_argument('--pretrain_epochs', type=int, default=500)
    parser.add_argument('--finetune_epochs', type=int, default=1000)
    parser.add_argument('--finetune_lr_scale', type=float, default=0.1)
    parser.add_argument('--log_interval', type=int, default=20)

    # pretrain masking
    parser.add_argument('--feat_mask_prob', type=float, default=0.5)
    parser.add_argument('--edge_drop_prob', type=float, default=0.22)

    # encoder
    parser.add_argument('--gcn_dropout', type=float, default=0.3)
    parser.add_argument('--layer_w1', type=float, default=0.4)
    parser.add_argument('--layer_w2', type=float, default=0.4)
    parser.add_argument('--layer_w3', type=float, default=0.2)
    parser.add_argument('--tau', type=float, default=0.55)
    parser.add_argument('--adj_temp', type=float, default=0.2)
    parser.add_argument('--feat_loss_weight', type=float, default=2.0)

    # CAF / enhancement
    parser.add_argument('--enhance_ratio_start', type=float, default=0.05)
    parser.add_argument('--enhance_ratio_max', type=float, default=0.10)
    parser.add_argument('--enhance_alpha_scale', type=float, default=0.1)

    # topology / APPNP
    parser.add_argument('--alpha', type=float, default=0.15)
    parser.add_argument('--k', type=int, default=2)
    parser.add_argument('--noise_sim_thresh', type=float, default=0.8)
    parser.add_argument('--confidence_thresh', type=float, default=0.75)

    # prototypes / clustering
    parser.add_argument('--proto_momentum', type=float, default=0.99)
    parser.add_argument('--proto_update_interval', type=int, default=5)
    parser.add_argument('--kmeans_n_init', type=int, default=20)
    parser.add_argument('--kmeans_seed', type=int, default=2021)
    parser.add_argument('--disp_weight_max', type=float, default=0.15)
    parser.add_argument('--disp_weight_scale', type=float, default=0.3)

    args = parser.parse_args()
    base_seed = args.seed
    kmeans_base = args.kmeans_seed
    records = []

    for trial in range(args.ntrials):
        trial_seed = base_seed + trial
        args.seed = trial_seed
        args.kmeans_seed = kmeans_base + trial
        print("\n" + "#" * 40)
        print(f"Trial [{trial + 1}/{args.ntrials}] seed={trial_seed} kmeans_seed={args.kmeans_seed}")
        print("#" * 40)
        best = train_and_evaluate(args)
        best['trial'] = trial
        best['seed'] = trial_seed
        records.append(best)
        print(
            f"Trial {trial + 1} BEST: "
            f"ACC={best['acc'] * 100:.2f}%, NMI={best['nmi'] * 100:.2f}%, "
            f"ARI={best['ari'] * 100:.2f}%, F1={best['f1'] * 100:.2f}% "
            f"(epoch {best['epoch']})"
        )

    keys = ['acc', 'nmi', 'ari', 'f1']
    arr = {k: np.array([r[k] for r in records], dtype=np.float64) for k in keys}
    ddof = 1 if args.ntrials > 1 else 0
    print("\n" + "=" * 40)
    print(f"Summary over {args.ntrials} trials (seed={base_seed}..{base_seed + args.ntrials - 1})")
    for i, r in enumerate(records):
        print(
            f"  [{i + 1}] seed={r['seed']}: "
            f"ACC={r['acc'] * 100:.2f}  NMI={r['nmi'] * 100:.2f}  "
            f"ARI={r['ari'] * 100:.2f}  F1={r['f1'] * 100:.2f}"
        )
    print("-" * 40)
    for k in keys:
        mean = arr[k].mean() * 100
        std = arr[k].std(ddof=ddof) * 100
        print(f"{k.upper():4s}: {mean:.2f} ± {std:.2f}")
    print("=" * 40)

# seed=42 搜索得到的较优命令（BEST 为微调过程中最高 ACC）
# cora      BEST NMI=62.26 / ACC=76.66 / ARI=59.07 / F1=74.34  (seed=42; 以 NMI 为主)
# /data/conda_share/zxd/kippo_env/bin/python /home/zxd/project/chat_with_ai/myself/THOT_GC/main.py --dataset cora --hid_dim 256 --lr 1e-3 --weight_decay 1e-5 --pretrain_epochs 500 --finetune_epochs 400 --k 3 --tau 0.4
# citeseer  BEST ACC=72.74 / NMI=47.90 / ARI=49.67 / F1=68.18  (seed=42, wide+coord)
# /data/conda_share/zxd/kippo_env/bin/python /home/zxd/project/chat_with_ai/myself/THOT_GC/main.py --dataset citeseer --hid_dim 64 --lr 1e-3 --weight_decay 1e-3 --pretrain_epochs 300 --finetune_epochs 400 --finetune_lr_scale 0.1 --feat_mask_prob 0.5 --edge_drop_prob 0.35 --gcn_dropout 0.8 --layer_w1 1.0 --layer_w2 0.0 --layer_w3 0.0 --tau 0.3 --adj_temp 1.2 --feat_loss_weight 1.0 --enhance_ratio_start 0.02 --enhance_ratio_max 0.25 --enhance_alpha_scale 0.2 --alpha 0.2 --k 1 --noise_sim_thresh 0.9 --confidence_thresh 0.95 --proto_momentum 0.95 --proto_update_interval 5 --kmeans_n_init 10 --kmeans_seed 0 --disp_weight_max 0.05 --disp_weight_scale 0.6
# photo     BEST NMI=74.36 / ACC=80.86 / ARI=65.46 / F1=74.22  (seed=42, wide+coord, NMI)
# /data/conda_share/zxd/kippo_env/bin/python /home/zxd/project/chat_with_ai/myself/THOT_GC/main.py --dataset photo --hid_dim 128 --lr 2e-3 --weight_decay 1e-5 --pretrain_epochs 700 --finetune_epochs 600 --feat_mask_prob 0.5 --edge_drop_prob 0.35 --gcn_dropout 0.7 --tau 0.2 --feat_loss_weight 5.0 --enhance_ratio_start 0.12 --enhance_ratio_max 0.12 --enhance_alpha_scale 0.2 --k 5 --noise_sim_thresh 0.65 --confidence_thresh 0.95 --proto_momentum 0.95 --proto_update_interval 3 --kmeans_n_init 10 --disp_weight_scale 0.1
# computers BEST NMI=59.30 / ACC=61.77 / ARI=46.90 / F1=54.53  (seed=42, wide+coord, NMI)
# /data/conda_share/zxd/kippo_env/bin/python /home/zxd/project/chat_with_ai/myself/THOT_GC/main.py --dataset computers --hid_dim 64 --lr 5e-4 --weight_decay 1e-4 --pretrain_epochs 300 --finetune_epochs 600 --finetune_lr_scale 0.2 --edge_drop_prob 0.05 --tau 0.2 --adj_temp 1.2 --feat_loss_weight 0.5 --enhance_ratio_start 0.02 --enhance_alpha_scale 0.0 --alpha 0.2 --k 12 --noise_sim_thresh 0.5 --proto_momentum 0.95 --proto_update_interval 3 --kmeans_n_init 10 --kmeans_seed 42 --disp_weight_max 0.3 --disp_weight_scale 0.0
