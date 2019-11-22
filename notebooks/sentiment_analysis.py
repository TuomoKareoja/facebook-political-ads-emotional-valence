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

df.sentiment.describe()

sns.distplot(df.sentiment)
plt.title("Afinn sentiment distribution")
plt.show()

# %%

df.sentiment_norm.describe()

sns.distplot(df.sentiment_norm)
plt.title("Afinn normalized sentiment distribution")
plt.show()

# %%


ax = sns.scatterplot(df.sentiment, df.sentiment_norm, alpha=0.005)
plt.title("Relationship between Afinn and Afinn normalized")
plt.show()

# %% [markdown]

# # What do the most positive and negative messages look like
#
# * For pure afinn sentiment values it is clear that really long and messages get the
# top spots
# * With normalized (divided by the number of words) sentiment values short and clear
# messages get the top spots. This seems much better
# * The normalized sentiment values have no outliers and the values seems more
# realistic and intuitive. We only used normalized values for the rest of the
# analysis

# %%

print("AFINN")
print("")

print("POSITIVE:")
for i, text in enumerate(
    df.sort_values(by="sentiment", ascending=False)["message"][:5]
):
    print(i + 1, "most positive")
    print(text[:200])
    print("")

print("")
print("NEGATIVE:")
for i, text in enumerate(df.sort_values(by="sentiment", ascending=True)["message"][:5]):
    print(i + 1, "most negative")
    print(text[:200])
    print("")

print("AFINN normalized")
print("")


print("POSITIVE:")
for i, text in enumerate(
    df.sort_values(by="sentiment_norm", ascending=False)["message"][:5]
):
    print(i + 1, "most positive")
    print(text[:200])
    print("")

print("")
print("NEGATIVE:")
for i, text in enumerate(
    df.sort_values(by="sentiment_norm", ascending=True)["message"][:5]
):
    print(i + 1, "most negative")
    print(text[:200])
    print("")

# %%

afinn_positive_wc = WordCloud(background_color="white").generate(
    " ".join(df.sort_values(by="sentiment_norm", ascending=False)["message"][:500])
)
plt.imshow(afinn_positive_wc)
plt.title("Wordcloud of 500 most positive messages")
plt.show()

afinn_negative_wc = WordCloud(background_color="white").generate(
    " ".join(df.sort_values(by="sentiment_norm", ascending=True)["message"][:500])
)
plt.imshow(afinn_negative_wc)
plt.title("Wordcloud of 500 most negative messages")
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

df.groupby("created_at_date")["sentiment_norm"].mean().plot()
plt.show()

# %%

sns.boxplot(
    x="created_at_date", y="sentiment_norm", data=df.sort_values(by="created_at_date")
)
plt.title("Distribution of Afinn be date")
plt.xticks(rotation=90)
plt.ylim([-0.4, 0.4])
plt.show()

# %%

df_sample = df.sample(10000)
plt.scatter(
    x=df_sample.created_at_date,
    y=df_sample.sentiment_norm,
    c=df_sample.sentiment_norm,
    # s=df_sample.impressions * 100,
    alpha=0.01,
)
plt.xticks(rotation=90)
plt.ylim([-0.5, 0.5])
plt.xlim([pd.Timestamp("2018-09-15"), pd.Timestamp("2018-11-15")])
plt.show()

# %%

top_segments = df.target_segment.value_counts().index[0:5]
df_top_segments = df[df["target_segment"].isin(top_segments)]

sns.lineplot(
    x="created_at_date", y="sentiment_norm", hue="target_segment", data=df_top_segments
)
plt.xlim([pd.Timestamp("2018-09-15"), pd.Timestamp("2018-11-15")])
plt.show()

# %%

sns.boxplot(x="target_segment", y="sentiment_norm", data=df_top_segments)
plt.show()

# %%

df_sample = df.sample(10000)
plt.scatter(
    x=df_sample.message.str.len(),
    y=df_sample.sentiment_norm,
    c=df_sample.sentiment_norm,
    # s=df_sample.impressions * 100,
    alpha=0.01,
)
plt.ylim([-0.5, 0.5])
plt.xlim([0, 1000])
plt.xticks(rotation=90)
plt.show()

# %%

sns.boxplot(x=df.targetedness, y=df.sentiment_norm)
plt.ylim([-0.5, 0.5])
# plt.xticks(rotation=90)
plt.show()


# %%

df.dtypes

# %%

df.advertiser.value_counts().plot.bar()

# %%
df.paid_for_by.value_counts().plot.bar()

# %%

top_advertisers = df.advertiser.value_counts().index[:10]
top_payers = df.paid_for_by.value_counts().index[:10]

# %%

sns.boxplot(
    x="advertiser", y="sentiment_norm", data=df[df["advertiser"].isin(top_advertisers)]
)
plt.show()
sns.countplot(x="advertiser", data=df[df["advertiser"].isin(top_advertisers)])
plt.show()

# %%

sns.boxplot(
    x="paid_for_by", y="sentiment_norm", data=df[df["paid_for_by"].isin(top_payers)]
)
plt.xticks(rotation=90)
plt.show()

sns.countplot(x="paid_for_by", data=df[df["paid_for_by"].isin(top_payers)])
plt.xticks(rotation=90)
plt.show()


# %%
