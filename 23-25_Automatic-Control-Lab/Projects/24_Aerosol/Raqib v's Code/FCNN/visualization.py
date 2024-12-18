import matplotlib.pyplot as plt

def plot_real_vs_pred(actual, predicted, time):
    """Plot the real vs predicted strain values."""
    plt.figure(figsize=(10, 6))
    plt.plot(time, actual, label='Real Strain', color='blue', marker='o', markersize=0.01)
    plt.plot(time, predicted, label='Predicted Strain', color='red', linestyle='--')
    plt.title('Real vs Predicted Strain over Time')
    plt.xlabel('Time (min)')
    plt.ylabel('Strain')
    plt.legend()
    plt.grid(True)
    plt.show()

