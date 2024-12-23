import matplotlib.pyplot as plt


def plot_real_vs_pred(actual, predicted, time):
    """Plot real vs predicted strain over time."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(time, actual, label='Real Strain', color='blue', linewidth=1)
    plt.plot(time, predicted, label='Predicted Strain', color='red', linewidth=1)
    plt.title('Real vs Predicted Strain over Time')
    plt.xlabel('Time (min)')
    plt.ylabel('Strain')
    plt.legend()
    plt.grid()
    plt.show()


