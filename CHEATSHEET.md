# Minecraft House Generator – Cheat Sheet

## Quick Start

1. `python get_started.py`  
   Check setup

2. `python test_vae_model.py`  
   Test model

3. `python train_vae.py --epochs 50`  
   Train (quick test)

4. `python generate_houses.py --checkpoint checkpoints/best_model.pth --num_samples 10`

5. `python analyze_houses.py generated_houses/`

6. `python visualize_houses.py generated_houses/ --save_individual`

## Analysis and Visualization

### All Houses – Text Analysis

```bash
python analyze_houses.py generated_houses/
python analyze_houses.py generated_houses/ -v
````

### All Houses – PyCharm Mode (each house = separate plot)

```bash
python visualize_houses.py generated_houses/ --pycharm
python visualize_houses.py generated_houses/ --pycharm --max_houses 10
```

### All Houses – Visualization (recommended for >10 houses)

```bash
python visualize_houses.py generated_houses/ --save_individual --output_dir viz
```

### All Houses – Comparison (max 20)

```bash
python visualize_houses.py generated_houses/ --max_houses 20
```

### Save Comparison

```bash
python visualize_houses.py generated_houses/ --save --max_houses 20
```

### Single House – Detailed

```bash
python visualize_houses.py generated_houses/house_001.npy --mode both
```

## Training

### Quick test (50 epochs)

```bash
python train_vae.py --epochs 50 --batch_size 4
```

### Standard (100 epochs)

```bash
python train_vae.py --epochs 100 --batch_size 4 --kl_weight 0.001
```

### More variation (lower kl_weight)

```bash
python train_vae.py --epochs 100 --kl_weight 0.0005
```

### More structure (higher kl_weight)

```bash
python train_vae.py --epochs 100 --kl_weight 0.005
```

### TensorBoard monitoring

```bash
tensorboard --logdir runs
```

Then open:

```
http://localhost:6006
```

## Generation

### Random houses

```bash
python generate_houses.py --checkpoint checkpoints/best_model.pth \
    --num_samples 10 --mode random
```

### Interpolation (smooth transition between 2 houses)

```bash
python generate_houses.py --checkpoint checkpoints/best_model.pth \
    --num_samples 10 --mode interpolate
```

### Variations (similar houses)

```bash
python generate_houses.py --checkpoint checkpoints/best_model.pth \
    --num_samples 5 --mode variations
```

### With seed (reproducible results)

```bash
python generate_houses.py --checkpoint checkpoints/best_model.pth \
    --num_samples 10 --seed 42
```

## Tests

```bash
python test_vae_model.py
python test_house_dataset.py
python test_unified_parser.py houses/
python test_litematic_parser.py houses/litematic/
```

## Key Parameters

| Parameter      | Description                                            |
|----------------|--------------------------------------------------------|
| `--epochs`     | Number of epochs (50=test, 100–200=standard, 500=long) |
| `--batch_size` | Batch size (2=small GPU/CPU, 4=standard, 8=large GPU)  |
| `--kl_weight`  | Critical parameter (0.0001–0.01, sweet spot: 0.001)    |
| `--latent_dim` | Latent size (128/256/512, default 256 is fine)         |
| `--lr`         | Learning rate (0.0005–0.002, default 0.001)            |
| `--threshold`  | Generation threshold (0.3–0.7, default 0.5)            |

## File Structure

```
houses/
├── litematic/
├── schem/
└── schematic/

checkpoints/
├── best_model.pth
├── final_model.pth
└── checkpoint_epoch_*.pth

generated_houses/
runs/
visualizations/
```

## Troubleshooting

| Problem                     | Fix                                    |
|-----------------------------|----------------------------------------|
| CUDA out of memory          | Use `--batch_size 2` or `--device cpu` |
| Generations look like noise | Increase `--kl_weight` to 0.005        |
| Generations empty/full      | Decrease `--kl_weight` to 0.0005       |
| Loss not decreasing         | More epochs, higher lr, check data     |
| Too little data             | Collect more houses (20–50+)           |

## Pro Tips

1. Always run `analyze_houses.py` before visualization.
2. For more than 10 houses, use `--save_individual`.
3. Monitor TensorBoard during training.
4. `kl_weight` is the most important parameter.
5. More diverse data gives better results.
6. Start with 50 epochs for testing.
7. Save checkpoints every 10 epochs.
8. Density between 10–60% usually produces good houses.