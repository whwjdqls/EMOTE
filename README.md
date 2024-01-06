EMOTE

---

1. create conda environment by :

```
conda create --name <env> --file environmenets.txt
```

2. Add submodule for lip reading loss

```
bash pull_submodeuls.sh
```

3. Download assets for EMOTE, FLINT, lip reading model, video emotion recognition model

```
bash download_assets.sh
```

For trainig EMOTE, run :

```
python main/train.py
```
