from jtop import jtop
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import subprocess
import time
import sys
import csv
import os 
import signal
import psutil
    # Function to find and kill processes by name
def kill_processes_by_name(process_names):
    for name in process_names:
        # Use pgrep to find the PID of the process by name
        pid_list = subprocess.run(['pgrep', '-f', name], capture_output=True, text=True).stdout.split()
        for pid in pid_list:
            # Kill the process using its PID
            subprocess.run(['kill', pid])
            print(f"Killed process with name {name} and PID {pid}")
# Check if the correct number of command-line arguments is provided
if len(sys.argv) != 3:
    print("Usage: python3 cek_gpu.py gpu_mem.csv plot.png")
    sys.exit(1)

gpu_mem_file = sys.argv[1]
plot_image_file = sys.argv[2]
number =1
# Function to save data to CSV file
def save_to_csv(data, filename):    
    fields = ['number', 'gpu_usage', 'mem_usage']
    with open(filename, mode='w') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        # writing headers (field names)
        writer.writeheader()
    
        # writing data rows
        writer.writerows(gpu_mem)

def restart_jtop_service():
    subprocess.run(['sudo', 'systemctl', 'restart', 'jtop.service'])


def update_plot(frame):
    global gpu_mem,gpu_data, mem_data, ax1, ax2, start_time, number
    with jtop() as jetson:
        gpu_usage = jetson.stats['GPU']
        mem_usage = jetson.stats['RAM'] * 100
        gpu_mem.append({'number':number, 'gpu_usage': gpu_usage, 'mem_usage': mem_usage})
        number=number+1
        gpu_data.append(gpu_usage)
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

        # Check if 3 minutes have elapsed
        if time.time() - start_time >= 400:
            ani.event_source.stop()  # Stop the animation
            save_to_csv(gpu_mem, gpu_mem_file)
            plt.savefig(plot_image_file) 
            # plt.close()  # Close the plot window
            # os.kill(os.getpid(), signal.SIGKILL)
            # List of process names to kill
            process_names = ["image_view-2", "tracker_node-1"]

            # Kill processes by name
            kill_processes_by_name(process_names)
            sys.exit(1)

if __name__ == "__main__":
    # Restart jtop service every 10 seconds
    
    restart_jtop_service()
    gpu_mem = []
    gpu_data = []
    mem_data = []
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.tight_layout(pad=3.0)  # Adjust layout to accommodate subplots

    start_time = time.time()  # Record the start time

    # Update plot every second
    ani = FuncAnimation(fig, update_plot, interval=1)

    plt.show()
    while True:
        restart_jtop_service()
        time.sleep(10)
