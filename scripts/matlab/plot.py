import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt



experiment =pd.read_csv('/home/claudia/catkin_ws/src/dVRL/baselines/logs/dVRL_HER_2/progress.csv')
plt.figure(figsize=(15, 7))
sns.palplot(sns.color_palette("husl", 8))
plt.set_title('Training success rate')

trainingplt = sns.lineplot(
    x="epoch",
    y="training_success_rate",
    hue='year',
    data=nyc_df
).set_title('Success rate of training')
plt.show()


