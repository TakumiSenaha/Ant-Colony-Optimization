import matplotlib.pyplot as plt
import csv
import numpy as np

class LogAnalyzer:
	def __init__(self, file_path: str) -> None:
		self.file_path = file_path
		self.log_data = self.load_log()
		self.summary_log = self.summarize_log()

	def load_log(self) -> list:
		log_data = []
		with open(self.file_path, "r") as f:
				reader = csv.reader(f)
				for row in reader:
						log_data.append([int(val) for val in row])
		return log_data

	def summarize_log(self) -> list:
		summary_log = []
		for i in range(len(self.log_data)):
				if len(self.log_data[i]) > 0:
						tmp_list = [i, len(self.log_data[i]), round(sum(self.log_data[i])/len(self.log_data[i]), 1)]
						summary_log.append(tmp_list)
				else:
						summary_log.append([i, 0, 0])
		return summary_log

	def plot_results(self, title: str) -> None:
		x = []
		reach_rate = []
		ave_wid = []
		for i in range(len(self.summary_log)):
				x.append(i * 10)
				reach_rate.append((self.summary_log[i][1] / 50) * 100)
				ave_wid.append(self.summary_log[i][2])

		plt.scatter(x, ave_wid)
		plt.ylim(0, 100)
		plt.xlabel("The number of released ant")
		plt.ylabel("Average maximum capacity path of reaching interest")
		plt.title(f"{title} - Average Maximum Capacity Path")
		plt.show()

		plt.scatter(x, reach_rate)
		plt.ylim(0, 100)
		plt.xlabel("The number of released ant")
		plt.ylabel("Probability of reaching interest [%]")
		plt.title(f"{title} - Probability of Reaching Interest")
		plt.show()

if __name__ == "__main__":
	# シミュレーション結果のログファイルを読み込む
	interest_analyzer = LogAnalyzer("./simulation_result/log_interest.csv")
	rand_analyzer = LogAnalyzer("./simulation_result/log_rand.csv")

	# ログデータを要約し、グラフを表示する
	interest_analyzer.plot_results("Interest Log")
	rand_analyzer.plot_results("Rand Log")
