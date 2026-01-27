from pathlib import Path
import matplotlib.pyplot as plt


def _safe_name(name: str) -> str:
    return name.replace(' ', '_').replace('/', '_').lower()


def save_line_figure(figure_dir, test_name, times, values, figsize=(10, 2)):
    """Save a single-line figure. Returns the saved Path or None."""
    if figure_dir is None:
        return None
    try:
        fig_dir = Path(figure_dir)
    except Exception:
        return None
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(times, values, linewidth=1)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title(test_name)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    filepath = fig_dir / f"{_safe_name(test_name)}.png"
    plt.savefig(filepath, dpi=100, bbox_inches='tight')
    plt.close()
    return filepath


def save_comparison_figure(figure_dir, test_name, times, original, filtered, figsize=(14, 6)):
    """Save a before/after comparison figure. Returns the saved Path or None."""
    if figure_dir is None:
        return None
    try:
        fig_dir = Path(figure_dir)
    except Exception:
        return None
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    axes[0].plot(times, original, 'b-', linewidth=1, label='Original')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title(f'{test_name} - Original')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(times, filtered, 'g-', linewidth=1, label='Filtered')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Amplitude')
    axes[1].set_title(f'{test_name} - Filtered')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    filepath = fig_dir / f"{_safe_name(test_name)}.png"
    plt.savefig(filepath, dpi=100, bbox_inches='tight')
    plt.close()
    return filepath
