from jtop import jtop
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def restart_jtop_service():
    subprocess.run(['sudo', 'systemctl', 'restart', 'jtop.service'])
def update_plot(frame):
    global gpu_data, mem_data, ax1, ax2
    with jtop() as jetson:
        gpu_usage = jetson.stats['GPU']
        gpu_data.append(gpu_usage)
        mem_usage = jetson.stats['RAM']
        mem_data.append(mem_usage)
        ax1.clear()
        ax1.plot(gpu_data)
        ax1.set_title('GPU Usage (%)')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Usage (%)')
        ax2.clear()
        ax2.plot(mem_data)
        ax2.set_title('Memory Usage (%)')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Usage (%)')

if __name__ == "__main__":
    gpu_data = []
    mem_data = []
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.tight_layout(pad=3.0)  # Adjust layout to accommodate subplots

    # Update plot every second
    ani = FuncAnimation(fig, update_plot, interval=1000)

    plt.show()
    # Restart jtop service every 10 seconds
    while True:
        restart_jtop_service()
        time.sleep(10)
# with jtop() as jetson:
#     print(jetson.stats)