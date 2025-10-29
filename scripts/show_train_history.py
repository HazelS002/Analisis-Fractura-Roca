from matplotlib import pyplot as plt
import seaborn as sns
import json

with open("./models/Noise2Void/Noise2Void.json", 'r') as f:
    history = json.load(f)

for key in history.keys():
    plt.plot(history[key], label=key)

plt.legend()
plt.show()