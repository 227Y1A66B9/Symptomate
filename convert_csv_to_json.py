import pandas as pd

# Specify encoding — try 'ISO-8859-1' or 'latin1'
df = pd.read_csv("more_extended_disease_symptoms.csv", encoding='ISO-8859-1')

df.to_json("more_extended_disease_symptoms.json", orient="records", indent=4)

print("✅ CSV converted to JSON successfully!")

