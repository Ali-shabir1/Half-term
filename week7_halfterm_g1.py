#Do not change the function names provided—this ensures your solutions can be automatically tested.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def load_data(filepath):
    data=pd.read_csv(filepath)
    return data
data=load_data("titanic.csv")
print(data.head()) 



def survival_by_age_group(df):
    bins = [0, 12, 18, 35, 60, np.inf]
    labels = ['Children', 'Teen', 'Young Adults', 'Middle-aged', 'Elderly']
    
    df['Age Group'] = pd.cut(df['Age'], bins=bins, labels=labels)
    
    age_group_counts = df['Age Group'].value_counts().sort_index()
    age_group_survivors = df[df['Survived'] == 1]['Age Group'].value_counts().sort_index()
    
    survival_rate = (age_group_survivors / age_group_counts).fillna(0)
    
    highest_survival_group = survival_rate.idxmax()
    
    return survival_rate, age_group_counts, age_group_survivors, highest_survival_group



def gender_survival_by_class(df):
    survival_rates = df.groupby(['Pclass', 'Sex'])['Survived'].mean().unstack()
    
    survival_rate_diff = survival_rates.max() - survival_rates.min()
    largest_diff_gender = survival_rate_diff.idxmax()
    evidence_statement = (
         {largest_diff_gender} 

    )
    
    return survival_rates, largest_diff_gender, evidence_statement





def survival_by_family_size(df):
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    survival_rates = df.groupby('FamilySize')['Survived'].mean()
    
    plt.figure(figsize=(10, 6))
    survival_rates.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title('Survival Rates by Family Size', fontsize=16)
    plt.xlabel('Family Size', fontsize=14)
    plt.ylabel('Survival Rate', fontsize=14)
    plt.xticks(rotation=0, fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
    
    if survival_rates[1] > survival_rates.max():
        description = "Passengers who are traveling alone had the highest survival rate."
    else:
        description = (
            "Passengers traveling with smaller family had better survival chances, "
            "while larger families had lower survival rates"
        )
    
    return survival_rates, description




def survival_by_fare_quartile(df):
    df['Fare_Quartile'] = pd.qcut(df['Fare'], 4, labels=['Q1 (Lowest)', 'Q2', 'Q3', 'Q4 (Highest)'])
    
    
    survival_rates = df.groupby('Fare_Quartile')['Survived'].mean()
    survival_rates.plot(kind='bar', color='skyblue', figsize=(8, 6))
    plt.title('Survival Rates by Fare Quartile')
    plt.xlabel('Fare Quartile')
    plt.ylabel('Survival Rate')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    description = (
        "The visualization shows that survival rates tend to increase with higher fare quartiles. "
        "This suggests that passengers who paid higher fares had better chances of survival, likely due to better access "
        "to lifeboats, which may have been prioritized during the evacuation."
    )
    
    return survival_rates, description

