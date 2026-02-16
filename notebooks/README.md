# SPADE Exploration Notebooks

This directory contains Jupyter notebooks for exploring and understanding the SPADE pipeline.

## Notebooks

### `explore_pipeline.ipynb`

A comprehensive exploration notebook that breaks down the entire SPADE pipeline into explorable cells:

1. **Setup and Imports** - Environment setup and library imports
2. **Load Configuration** - Load dataset and training configuration
3. **Load Training Data (Normal Images)** - Visualize normal training images
4. **Synthetic Anomaly Generation Visualization** - See how CutPaste and Perlin anomalies are generated
5. **Patch-Level Label Generation** - Understand how pixel masks are converted to patch labels
6. **Training Data with 80/20 Normal/Synthetic Split** - Visualize the training data distribution
7. **Compare Synthetic vs Real Test Anomalies** - Side-by-side comparison of synthetic and real anomalies
8. **Model Inference and Heatmap Visualization** - Run inference and visualize anomaly heatmaps
9. **Detailed Patch-Level Analysis** - Compare predictions vs ground truth at patch level
10. **Training Data Statistics** - Analyze the training data distribution

## Usage

1. **Start Jupyter**:
   ```bash
   cd /home/owais/spade
   jupyter notebook notebooks/explore_pipeline.ipynb
   ```
   
   Or use JupyterLab:
   ```bash
   jupyter lab notebooks/explore_pipeline.ipynb
   ```

2. **Run cells sequentially** - Each section builds on the previous one

3. **Modify parameters** - Adjust `category`, `synthetic_prob`, or other config values in the cells

## Requirements

Make sure you have:
- Jupyter installed: `pip install jupyter notebook jupyterlab`
- All SPADE dependencies installed: `pip install -r requirements.txt`
- MVTec dataset downloaded and configured in `config/data.yaml`

## Notes

- The notebook automatically detects if a trained checkpoint exists and loads it for inference
- If no checkpoint is found, model inference sections will skip gracefully
- All visualizations are interactive and can be modified in the cells






