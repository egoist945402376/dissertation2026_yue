import matplotlib.pyplot as plt

labels = ['n/10', 'n/5', 'n/2', 'n', 'n/0.5', 'n/0.1']
paper = [1.124, 1.231, 1.518, 2.035, 3.056, 11.061]
mine = [1.1535, 1.2505, 1.5486, 2.0480, 3.0477, 11.0475]

x = range(len(labels))

plt.figure(figsize=(8, 5))
plt.plot(x, paper, marker='o', color='blue', label='Paper')
plt.plot(x, mine, marker='o', color='gold', label='My replication')

for xi, yp, ym in zip(x, paper, mine):
    plt.annotate(f'{yp:.3f}', (xi, yp), textcoords='offset points', xytext=(0, 10), ha='center', fontsize=8, color='blue')
    plt.annotate(f'{ym:.3f}', (xi, ym), textcoords='offset points', xytext=(0, -14), ha='center', fontsize=8, color='gold')

plt.xticks(x, labels)
plt.xlabel('lambda')
plt.ylabel('bound value')
plt.title('Wasserstein bound: paper vs replication')
plt.legend()
plt.tight_layout()
plt.savefig('bound_comparison.png')
plt.show()
