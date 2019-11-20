# %%

import os

import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd
import seaborn as sns
from IPython.core.interactiveshell import InteractiveShell

# Setting styles
InteractiveShell.ast_node_interactivity = "all"
sns.set(style="whitegrid", color_codes=True, rc={"figure.figsize": (12.7, 9.27)})

# %%

df = pd.read_csv(
    os.path.join("data", "processed", "data_sentiment.csv"), low_memory=False
)

# %%

df.sentiment_afinn.describe()

sns.distplot(df.sentiment_afinn)
plt.title("Afinn sentiment distribution")
plt.show()

# %%

df.sentiment_vader.describe()

sns.distplot(df.sentiment_vader)
plt.title("Vader sentiment distribution")
plt.show()

# %%


ax = sns.scatterplot(df.sentiment_afinn, df.sentiment_vader, alpha=0.005)
plt.show()

# %%

print("POSITIVE:")
for i, text in enumerate(
    df.sort_values(by="sentiment_afinn", ascending=False)["message"][:5]
):
    print(i + 1, "most positive")
    print(text[:200])
    print("")

print("")
print("NEGATIVE:")
for i, text in enumerate(
    df.sort_values(by="sentiment_afinn", ascending=True)["message"][:5]
):
    print(i + 1, "most negative")
    print(text[:200])
    print("")

# %%

print("POSITIVE:")
for i, text in enumerate(
    df.sort_values(by="sentiment_vader", ascending=False)["message"][:5]
):
    print(i + 1, "most positive")
    print(text[:200])
    print("")

print("")
print("NEGATIVE:")
for i, text in enumerate(
    df.sort_values(by="sentiment_vader", ascending=True)["message"][:5]
):
    print(i + 1, "most negative")
    print(text[:200])
    print("")

# %%

afinn_positive_wc = WordCloud(background_color="white").generate(
    " ".join(df.sort_values(by="sentiment_afinn", ascending=False)["message"][:500])
)
plt.imshow(afinn_positive_wc)
plt.title("Wordcloud of 500 most positive messages by Afinn")
plt.show()

afinn_negative_wc = WordCloud(background_color="white").generate(
    " ".join(df.sort_values(by="sentiment_afinn", ascending=True)["message"][:500])
)
plt.imshow(afinn_negative_wc)
plt.title("Wordcloud of 500 most negative messages by Afinn")
plt.show()

# %%

vader_positive_wc = WordCloud(background_color="white").generate(
    " ".join(df.sort_values(by="sentiment_vader", ascending=False)["message"][:500])
)
plt.imshow(vader_positive_wc)
plt.title("Wordcloud of 500 most positive messages by Vader")
plt.show()

vader_negative_wc = WordCloud(background_color="white").generate(
    " ".join(df.sort_values(by="sentiment_vader", ascending=True)["message"][:500])
)
plt.imshow(vader_negative_wc)
plt.title("Wordcloud of 500 most negative messages by Vader")
plt.show()


# %%

df["created_at"] = pd.to_datetime(df.created_at)
df["updated_at"] = pd.to_datetime(df.updated_at)

df["created_at_date"] = df.created_at.dt.date
df["updated_at_date"] = df.updated_at.dt.date

df["created_at_month"] = df["created_at"] - pd.offsets.MonthBegin(1, normalize=True)
df["updated_at_month"] = df["updated_at"] - pd.offsets.MonthBegin(1, normalize=True)
df["created_at_month"] = df["created_at_month"].dt.date
df["updated_at_month"] = df["updated_at_month"].dt.date

# %%

df.groupby("created_at_date")["sentiment_afinn"].mean().plot()
plt.show()

df.groupby("created_at_date")["sentiment_vader"].mean().plot()
plt.show()

# %%

sns.boxplot(x="created_at_month", y="sentiment_afinn", data=df)
plt.xticks(rotation=90)
plt.ylim([-15, 15])
plt.show()

# %%

sns.boxplot(x="created_at_month", y="sentiment_vader", data=df)
plt.xticks(rotation=90)
plt.show()

# %%

df_sample = df.sample(10000)
plt.scatter(
    x=df_sample.created_at_date,
    y=df_sample.sentiment_afinn,
    c=df_sample.sentiment_afinn,
    s=df_sample.impressions * 100,
    alpha=0.01,
)
plt.xticks(rotation=90)
plt.xlim([pd.Timestamp("2018-01-01"), pd.Timestamp("2019-01-01")])
plt.ylim([-30, 30])
plt.show()

# %%

df_sample = df.sample(10000)
plt.scatter(
    x=df_sample.created_at_date,
    y=df_sample.sentiment_vader,
    c=df_sample.sentiment_vader,
    s=df_sample.impressions * 100,
    alpha=0.01,
)
plt.xticks(rotation=90)
plt.xlim([pd.Timestamp("2018-01-01"), pd.Timestamp("2019-01-01")])
plt.show()
