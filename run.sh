#cora
python main.py --dataset cora --hid_dim 256 --lr 1e-3 --weight_decay 1e-5 --pretrain_epochs 500 --finetune_epochs 400 --k 3 --tau 0.4

# citeseer
python main.py --dataset citeseer --ntrials 10 --hid_dim 64 --lr 1e-3 --weight_decay 1e-3 --pretrain_epochs 300 --finetune_epochs 400 --edge_drop_prob 0.35 --gcn_dropout 0.8 --layer_w1 1.0 --layer_w2 0.0 --layer_w3 0.0 --tau 0.3 --adj_temp 1.2 --feat_loss_weight 1.0 --enhance_ratio_start 0.02 --enhance_ratio_max 0.25 --enhance_alpha_scale 0.2 --alpha 0.4 --k 1 --noise_sim_thresh 0.9 --confidence_thresh 0.95 --proto_momentum 0.95 --kmeans_n_init 10 --kmeans_seed 0 --disp_weight_max 0.05 --disp_weight_scale 0.6

# photo
python main.py --dataset photo --ntrials 10 --hid_dim 128 --lr 2e-3 --weight_decay 1e-5 --pretrain_epochs 700 --finetune_epochs 600 --edge_drop_prob 0.35 --gcn_dropout 0.7 --tau 0.2 --feat_loss_weight 5.0 --enhance_ratio_start 0.12 --enhance_ratio_max 0.12 --enhance_alpha_scale 0.2 --k 5 --noise_sim_thresh 0.65 --confidence_thresh 0.95 --proto_momentum 0.95 --proto_update_interval 3 --kmeans_n_init 10 --disp_weight_scale 0.1

# computers
python main.py --dataset computers --ntrials 10 --hid_dim 64 --lr 5e-4 --weight_decay 1e-4 --pretrain_epochs 300 --finetune_epochs 600 --finetune_lr_scale 0.2 --edge_drop_prob 0.05 --tau 0.2 --adj_temp 1.2 --feat_loss_weight 0.5 --enhance_ratio_start 0.02 --enhance_alpha_scale 0.0 --alpha 0.2 --k 12 --noise_sim_thresh 0.5 --proto_momentum 0.95 --proto_update_interval 3 --kmeans_n_init 10 --kmeans_seed 42 --disp_weight_max 0.3 --disp_weight_scale 0.0

