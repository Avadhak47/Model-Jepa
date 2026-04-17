import pandas as pd
import plotly.express as px

df = pd.DataFrame({'x': [1,2], 'y': [3,4], 'Cluster': ['A', 'B'], 'Grid_ID': ['TR_1', 'TR_2'], 'Slot_ID': [0, 1]})
fig = px.scatter(df, x='x', y='y', color='Cluster', hover_data=['Grid_ID', 'Slot_ID'])
print("Customdata trace 0:", getattr(fig.data[0], 'customdata', 'MISSING!'))
print("Customdata trace 1:", getattr(fig.data[1], 'customdata', 'MISSING!'))
