# CGGC

**CGGC** (*From Cluster Cores to Graph-Guided Self-Training: A Framework for Graph Clustering*) is an unsupervised method for clustering nodes on attributed graphs.

It first pretrains a dual-stream encoder (GCN topology + MLP attributes) with a masked graph autoencoder (**JRPI**), reconstructing both features and edges. Cluster prototypes are then initialized by KMeans on APPNP-smoothed embeddings.

Fine-tuning combines two modules:

- **CCRE** selects high-confidence cluster cores agreed by the raw and graph views, pulls representations toward those cores, and adds a core-alignment loss.
- **RAGS** drops dissimilar edges, adds semantic top-\(k\) neighbors, and runs cosine-weighted APPNP. The smoothed assignment is sharpened into a DEC-style target \(P\); the encoder is trained with \(\mathrm{KL}(Q\|P)\) plus a cluster-dispersion term.

Inference uses the encoder embedding, confidence-based edge purification, and APPNP. The main metric is **NMI**.

```bash
python main.py --dataset cora --ablation full
```
