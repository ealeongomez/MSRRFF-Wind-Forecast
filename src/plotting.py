"""
Plotting utilities for Optuna hyperparameter optimization results.
Simple functions that work directly with importances.
"""


import matplotlib.pyplot as plt
import numpy as np
import re
from typing import Dict

# ========================================================================================
# Plot grouped importances
# ========================================================================================
def plot_grouped_importances(importances_dict: Dict[str, Dict[str, float]], 
                             output_path: str = "grouped_importances.png") -> None:
    """
    Plot grouped bar chart comparing hyperparameter importances across studies.
    
    Args:
        importances_dict: Dictionary like {'Study1': {'param1': 0.5, 'param2': 0.3}, 'Study2': {...}}
        output_path: Path where to save the output image
        
    Example:
        importances = {
            'RNN': {'learning_rate': 0.5, 'hidden_size': 0.3},
            'GRU': {'learning_rate': 0.4, 'hidden_size': 0.4},
            'LSTM': {'learning_rate': 0.6, 'hidden_size': 0.2}
        }
        plot_grouped_importances(importances, 'output.png')
    """
    
    # Get all unique parameters
    all_params = sorted(set().union(*[imp.keys() for imp in importances_dict.values()]))
    study_names = list(importances_dict.keys())

    print(study_names)
    
    # Create importance matrix
    importance_matrix = []
    for study_name in study_names:
        study_importances = [importances_dict[study_name].get(param, 0) for param in all_params]
        importance_matrix.append(study_importances)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Configure bars
    x = np.arange(len(all_params))
    width = 0.8 / len(study_names)
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    
    # Draw bars
    for i, (study_name, importances) in enumerate(zip(study_names, importance_matrix)):
        offset = width * (i - len(study_names)/2 + 0.5)
        name = re.sub(r'Tuning-', ' ', study_name)
        name = name.replace(' Beijing', '')
        name = name.replace(' Argone', '')
        name = name.replace(' Chengdu', '')
        name = name.replace(' Netherland', '')
        name = name.replace('-0', '')
        name = name.replace('-1', '')
        name = name.replace('-2', '')

        ax.bar(x + offset, importances, width, label=name, 
               color=colors[i % len(colors)], alpha=0.85, 
               edgecolor='black', linewidth=0.8)
    
    # Style
    ax.set_xlabel('Hyperparameters', fontsize=30, fontweight='bold')
    ax.set_ylabel('Importance', fontsize=30, fontweight='bold')
    #ax.set_title('Hyperparameter Importances Comparison', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(all_params, rotation=25, ha='right', fontsize=30)
    ax.legend(fontsize=20, loc='upper right', framealpha=0.95)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


# ========================================================================================
# Plot contour search (individual or comparison)
# ========================================================================================
def plot_contour_search(studies_data: Dict[str, Dict[str, np.ndarray]],
                       param_x: str,
                       param_y: str,
                       output_path: str = "contour_search.png",
                       grid_size: int = 250,
                       smooth_sigma: float = 1.5) -> None:
    """
    Plot contour maps for hyperparameter search (supports single or multiple studies).
    Uses RBF interpolation + Gaussian smoothing for complete coverage.
    """
    from scipy.interpolate import Rbf
    from scipy.ndimage import gaussian_filter

    n_studies = len(studies_data)
    fig, axes = plt.subplots(1, n_studies, figsize=(7 * n_studies, 6))
    axes = [axes] if n_studies == 1 else axes

    for ax, (study_name, data) in zip(axes, studies_data.items()):
        x = data[param_x]
        y = data[param_y]
        z = data["value"]

        mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
        x, y, z = x[mask], y[mask], z[mask]

        if len(x) < 3:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=ax.transAxes)
            continue

        x_margin = max((x.max() - x.min()) * 0.08, 1e-6)
        y_margin = max((y.max() - y.min()) * 0.08, 1e-6)

        xi = np.linspace(x.min() - x_margin, x.max() + x_margin, grid_size)
        yi = np.linspace(y.min() - y_margin, y.max() + y_margin, grid_size)
        xi, yi = np.meshgrid(xi, yi)

        eps = 0.25 * ((x.max() - x.min()) + (y.max() - y.min()))
        rbf = Rbf(x, y, z, function='multiquadric', epsilon=max(eps, 1e-6))
        zi = rbf(xi, yi)
        zi_smooth = gaussian_filter(zi, sigma=smooth_sigma)

        z_min, z_max = np.nanmin(zi_smooth), np.nanmax(zi_smooth)
        if np.isclose(z_min, z_max):
            z_max = z_min + 1e-6
        levels = np.linspace(z_min, z_max, 25)

        contour = ax.contourf(xi, yi, zi_smooth, levels=levels, cmap='viridis', extend='both', alpha=0.9)
        ax.contour(xi, yi, zi_smooth, levels=levels, colors='black', linewidths=0.3, alpha=0.3)

        ax.scatter(x, y, c=z, s=45, cmap='viridis', edgecolor='white', linewidth=0.8, zorder=10)
        best_idx = np.argmin(z)
        ax.scatter(x[best_idx], y[best_idx], marker='*', c='red', s=280, edgecolor='white', linewidth=1.8, zorder=15, label='Best')

        cbar = plt.colorbar(contour, ax=ax, pad=0.02, fraction=0.05)
        cbar.set_label('Objective', fontsize=10, fontweight='bold')

        clean_name = study_name.replace('RFF-', '').replace('Tuning-', '').replace('Beijing', '').strip()
        ax.set_title(clean_name, fontsize=14, fontweight='bold')
        ax.set_xlabel(param_x.replace('params_', '').replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.set_ylabel(param_y.replace('params_', '').replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.25)
        ax.tick_params(labelsize=10)
        ax.legend(loc='upper right', fontsize=9)
        ax.set_xlim(x.min() - x_margin, x.max() + x_margin)
        ax.set_ylim(y.min() - y_margin, y.max() + y_margin)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

# ========================================================================================
# Plot multiple contours comparison
def plot_contours_comparison(studies_data: Dict[str, Dict[str, np.ndarray]],
                            param_x: str,
                            param_y: str,
                            output_path: str = "contours_comparison.png",
                            grid_size: int = 200,
                            smooth_sigma: float = 1.2) -> None:
    """
    Plot multiple contour maps comparing hyperparameter search across studies
    using RBF interpolation + Gaussian smoothing for smoother surfaces.
    """
    from scipy.interpolate import Rbf
    from scipy.ndimage import gaussian_filter

    n_studies = len(studies_data)
    fig, axes = plt.subplots(1, n_studies, figsize=(7 * n_studies, 6))
    axes = [axes] if n_studies == 1 else axes

    for ax, (study_name, data) in zip(axes, studies_data.items()):
        x = data[param_x]
        y = data[param_y]
        z = data["value"]

        # Limpiar NaN
        mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
        x, y, z = x[mask], y[mask], z[mask]

        if len(x) < 3:
            ax.text(0.5, 0.5, "Insufficient data",
                    ha="center", va="center", transform=ax.transAxes)
            continue

        # Márgenes
        x_margin = max((x.max() - x.min()) * 0.08, 1e-6)
        y_margin = max((y.max() - y.min()) * 0.08, 1e-6)

        xi = np.linspace(x.min() - x_margin, x.max() + x_margin, grid_size)
        yi = np.linspace(y.min() - y_margin, y.max() + y_margin, grid_size)
        xi, yi = np.meshgrid(xi, yi)

        # RBF (multiquadric es robusto; ajustar epsilon según escala)
        eps = 0.25 * ((x.max() - x.min()) + (y.max() - y.min()))
        rbf = Rbf(x, y, z, function='multiquadric', epsilon=max(eps, 1e-6))
        zi = rbf(xi, yi)

        # Suavizado gaussiano
        zi_smooth = gaussian_filter(zi, sigma=smooth_sigma)

        z_min, z_max = np.nanmin(zi_smooth), np.nanmax(zi_smooth)
        if np.isclose(z_min, z_max):
            z_max = z_min + 1e-6
        levels = np.linspace(z_min, z_max, 25)

        contour = ax.contourf(xi, yi, zi_smooth, levels=levels,
                              cmap='viridis', extend='both', alpha=0.9)
        ax.contour(xi, yi, zi_smooth, levels=levels,
                   colors='black', linewidths=0.3, alpha=0.3)

        # Puntos reales
        ax.scatter(x, y, c=z, s=45, cmap='viridis',
                   edgecolor='white', linewidth=0.8, zorder=10)

        best_idx = np.argmin(z)
        ax.scatter(x[best_idx], y[best_idx], marker='*', c='red', s=280,
                   edgecolor='white', linewidth=1.8, zorder=15, label='Best')

        cbar = plt.colorbar(contour, ax=ax, pad=0.02, fraction=0.05)
        cbar.set_label('Objective', fontsize=10, fontweight='bold')

        clean_name = study_name.replace('RFF-', '').replace('Tuning-', '').replace('Beijing', '').strip()
        ax.set_title(clean_name, fontsize=14, fontweight='bold')
        ax.set_xlabel(param_x.replace('params_', '').replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.set_ylabel(param_y.replace('params_', '').replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.25)
        ax.tick_params(labelsize=10)
        ax.legend(loc='upper right', fontsize=9)

        ax.set_xlim(x.min() - x_margin, x.max() + x_margin)
        ax.set_ylim(y.min() - y_margin, y.max() + y_margin)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()