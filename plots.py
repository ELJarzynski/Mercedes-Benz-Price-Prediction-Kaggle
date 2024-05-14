from datasetPreparing import *
import matplotlib.pyplot as plt
# Histogramy dla wszystkich kolumn
df.hist(bins=50, figsize=(20, 15))
plt.show()
# Scatter plot dla Price a Mileage
plt.figure(figsize=(12, 8))
plt.scatter(df['Mileage'], df['Price'], alpha=0.2)
plt.xlabel('Mileage')
plt.ylabel('Price')
plt.grid(True)
plt.show()

# Scatter plot dla Price a Rating
plt.figure(figsize=(12, 8))
plt.scatter(df['Rating'], df['Price'], alpha=0.5)
plt.xlabel('Rating')
plt.ylabel('Price')
plt.grid(True)
plt.show()

# Scatter plot dla Price a Review Count
plt.figure(figsize=(12, 8))
plt.scatter(df['Review Count'], df['Price'], alpha=0.5)
plt.xlabel('Review Count')
plt.ylabel('Price')
plt.grid(True)
plt.show()

# Scatter plot dla Price a Year Build
plt.figure(figsize=(12, 8))
plt.scatter(df['Year Build'], df['Price'], alpha=0.5)
plt.xlabel('Year Build')
plt.ylabel('Price')
plt.grid(True)
plt.show()
