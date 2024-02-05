from utils import K_Means

k_means = K_Means("data/train.csv", 3)

k_means.run()

print(k_means.centroids)
