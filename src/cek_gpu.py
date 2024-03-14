from jtop import jtop
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def update_plot(frame):
    global gpu_data, ax
    with jtop() as jetson:
        gpu_usage = jetson.stats['GPU']
        gpu_data.append(gpu_usage)
        ax.clear()
        ax.plot(gpu_data)
        ax.set_title('GPU Usage (%)')
        ax.set_xlabel('Time')
        ax.set_ylabel('Usage (%)')


if __name__ == "__main__":
    gpu_data = []
    fig, ax = plt.subplots()
    # Update plot every second
    ani = FuncAnimation(fig, update_plot, interval=1000)
    plt.show()
