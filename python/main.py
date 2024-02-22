import subprocess
import matplotlib.pyplot as plt
import numpy as np  # Import numpy for standard deviation calculation

class HyperLogLogSimulator:
    def __init__(self, cpp_executable, num_executions):
        self.cpp_executable = cpp_executable
        self.num_executions = num_executions
        self.estimated_numbers = []
        self.actual_numbers = []
        self.accuracy = []
    
    def collect_data(self, threads_per_block, shift_bits):
        for _ in range(self.num_executions):
            process = subprocess.Popen([self.cpp_executable, str(threads_per_block), str(shift_bits)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()

            if process.returncode != 0:
                print("Error:", stderr.decode())
                return
            output = stdout.decode().strip().splitlines()
            self.estimated_numbers.append(float(output[0].split()[-1]))
            self.actual_numbers.append(float(output[1].split()[-1]))
            self.accuracy.append(abs(self.estimated_numbers[-1] - self.actual_numbers[-1]) / self.actual_numbers[-1] * 100)

    def plot_results(self):
        # Compute standard error
        standard_error = np.std(self.accuracy)

        plt.figure(figsize=(10, 5))
        self._plot_error(standard_error)
        self._plot_estimations()
        plt.tight_layout()
        plt.savefig('simulation_results.png')
        plt.show()

    def _plot_error(self, standard_error):
        plt.subplot(2, 1, 1)
        plt.scatter(range(1, self.num_executions + 1), self.accuracy, label='Error')
        plt.axhline(y=standard_error, color='r', linestyle='-', label=f'Standard Error: {standard_error:.2f}%')
        plt.xlabel('Execution')
        plt.ylabel('Error Rate (%)')
        plt.title('Accuracy Overview')
        plt.legend()

    def _plot_estimations(self):
        plt.subplot(2, 1, 2)
        plt.scatter(range(1, self.num_executions + 1), self.estimated_numbers, label='Estimated Count')
        plt.scatter(range(1, self.num_executions + 1), self.actual_numbers, color='r', label='Actual Count')
        plt.xlabel('Execution')
        plt.ylabel('Count')
        plt.title('Estimated vs. Actual Count')
        plt.legend()

if __name__ == "__main__":
    cpp_executable = "../HyperLogLog"  # Assuming C++ exec file location
    threads_per_block = input("Number of CUDA thread's per thread block: ")
    shifts = input("Shift factor for main.cpp: ")
    num_executions = int(input("Desired test samples: "))

    assert 1 <= int(threads_per_block) <= 1024
    assert int(shifts) <= 28
    assert 1 <= int(num_executions) <= 20

    sim = HyperLogLogSimulator(cpp_executable, num_executions)
    sim.collect_data(threads_per_block, shifts)
    sim.plot_results()
