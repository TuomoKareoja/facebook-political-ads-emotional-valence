# %%

import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud
from src.text.process_text import strip_html_tags
from IPython.core.interactiveshell import InteractiveShell
import ast

# Setting styles
InteractiveShell.ast_node_interactivity = "all"
sns.set(style="whitegrid", color_codes=True, rc={"figure.figsize": (12.7, 9.27)})

# setting seed to keep the sample the same
random_state = 123
random.seed(random_state)

# %% [markdown]

# # Reading in the data
#
# * The data is quite big so we only read a sample of it for the analysis

# %%

total_n = 164600
sample_n = 10000
# adding one so that header will not be included in the skip list
skip = sorted(random.sample(range(1, total_n + 1), total_n - sample_n))
df = pd.read_csv(os.path.join("data", "raw", "fbpac-ads-en-US.csv"), skiprows=skip)

# %% [markdown]

# # Datatypes and missing values
#
# * Crucially there seems to be some ads with no title. This might mess up our
# processing down the line if we don't take this into account

# %%

df.dtypes
df.isnull().sum()
df.head()

# %% [markdown]

# # Changing times to actual datetime

# %%

df["created_at"] = pd.to_datetime(df.created_at)
df["updated_at"] = pd.to_datetime(df.updated_at)

# %% [markdown]

# Making sure that there are no known faulty rows in the dataset
# Checking that all adds are in english

# %%

df.suppressed.value_counts()

# %%

df.lang.value_counts()

# %% [markdown]

# Is ID unique? YES

# %%

df.id.duplicated().sum()

# %% [markdown]

# ## Distribution of votes for "politicalness"
#
# * Highly skewed and hard to make sense of in a normal histogram
# * Cumulative plot works better. We see that higher proportion of ads have
# no not-political votes

# %%

sns.distplot(df.political, hist=False, label="political")
sns.distplot(df.not_political, hist=False, label="non-political")
plt.legend()
plt.xlabel("number of votes")
plt.ylabel("% of votes")
plt.show()

# %%

political_cumulative = (
    np.cumsum(df.political.value_counts().sort_index()) * 100 / len(df)
)
not_political_cumulative = (
    np.cumsum(df.not_political.value_counts().sort_index()) * 100 / len(df)
)

political_cumulative.plot(label="political")
not_political_cumulative.plot(label="non-political")
plt.legend()
plt.ylabel("% lower or equal number of votes")
plt.xlabel("Number of votes")
plt.ylim([0, 100])
plt.xlim([0, 30])


# %% [markdown]

# # Correlation between number of votes
#
# * Vote types are somewhat correlated (R2 around 0.13)
# * Not suprising as ads are seen more often probably get more of both votes

# %%

ax = sns.lmplot(x="political", y="not_political", scatter_kws={"alpha": 0.2}, data=df)
ax.fig.text(
    0.60,
    0.60,
    f"Correlation: {np.round(np.corrcoef(df.political, df.not_political)[0, 1], 2)}",
    fontsize=11,
)
plt.ylabel("Number of not-political votes")
plt.ylabel("Number of political votes")
plt.xlim([0, 60])
plt.ylim([0, 35])
plt.show()

# %% [markdown]

# # Political probability distribution and connection to votes
#
# * Both votes correlated positively to probability of being political
# * There are some adds where predicted probability is less than 0.7, we should drop
# these as these are outliers compared to rest of the data
# * There is a possibility of actually dropping even more adds based
# on predicted probability. This is something that we should keep in mind

# %% political probability distribution and connection to votes

sns.distplot(df.political_probability)
plt.title("Distribution of political probability")
plt.plot()

ax = sns.lmplot(
    x="political", y="political_probability", scatter_kws={"alpha": 0.2}, data=df
)
ax.fig.text(
    0.60,
    0.60,
    f"Correlation: {np.round(np.corrcoef(df.political, df.not_political)[0, 1], 2)}",
    fontsize=11,
)
plt.ylabel("Predicted probability of being political")
plt.xlabel("Number of political votes")
plt.xlim([0, 60])
plt.ylim([0, 1])
plt.show()

ax = sns.lmplot(
    x="not_political", y="political_probability", scatter_kws={"alpha": 0.2}, data=df
)
ax.fig.text(
    0.60,
    0.60,
    f"Correlation: {np.round(np.corrcoef(df.political, df.not_political)[0, 1], 2)}",
    fontsize=11,
)
plt.ylabel("Predicted probability of being political")
plt.xlabel("Number of not-political votes")
plt.xlim([0, 35])
plt.ylim([0, 1])
plt.show()

# %% [markdown]

# # Is it really political?
#
# * The percent of ads that have as much or more political than not political votes
# goes up when the predicted politicalness goes up. This is as expected
# * We can cut the adds below 0.99 probability and still lose only few ads.
# This should make the dataset more uniformly about actual political adds
# * We also should add a check that the add has at least as many political
# as not political votes. Not more because so many adds have zero votes

# %%

sns.distplot(df.political_probability, label="Distribution of political probability")
sns.lineplot(
    np.round(df.political_probability, 2),
    (df.political >= df.not_political) * 100,
    label="political votes >= not political votes",
)
plt.legend()
plt.title("Distribution of political probability")
plt.show()

sns.distplot(
    df.political_probability, bins=100, label="Distribution of political probability"
)
sns.lineplot(
    np.round(df.political_probability, 2),
    (df.political >= df.not_political) * 100,
    label="political votes >= not political votes",
)
plt.legend()
plt.title("Distribution of political probability 0.9-1")
plt.xlim([0.9, 1])
plt.show()


# %% [markdown]

# # When was the ad created and last seen
#
# * There seems to be some problem with the data in the beginning of 2019
# as there almost ads created in jan, feb and apr
# * Lets keep only the data up to end of 2019 and focus on 2018 november elections

# %%

df.created_at.dt.date.value_counts().plot()
plt.title("created_at")
plt.show()

df.updated_at.dt.date.value_counts().plot()
plt.title("updated_at")
plt.show()

# %% [markdown]
#
# # Distribution of created_at and updated_at combined with impressions
#
# * The end of 2017 seems like an outlier as huge number of impressions (times seen)
# compared to the rest of the dataset. Maybe due to the buzz created by the launch
# of the monitoring extension attracted users that shortly stopped using the extension?
# * Data before 2018 should be dropped from the analysis as it seems clearly different
# than the data after it

# %%

df.groupby(df.created_at.dt.date)["impressions"].sum().plot()
plt.title("created_at")
plt.show()

df.groupby(df.updated_at.dt.date)["impressions"].sum().plot()
plt.title("updated_at")
plt.show()


# %% [markdown]

# # Ad lifetime
#
# * Most of the adds disappear after 11 days of their appearance
# * Some adds live very long

# %% ad lifetime

df = df.assign(days_up=(df["updated_at"] - df["created_at"]).dt.days)
print(df.days_up.describe())
sns.distplot(df.days_up, bins=100)
plt.title("Days between ad first and last seen")
plt.show()

# %% [markdown]

# # Wordcloud of the title and the message
#
# * Looks like that either facebook adds or the users of the extensions are
# pretty liberal (planned parenthood, Elizabeth Warren, Internation Rescue...)
# * Words is message convey that ads really try to get people to vote and show
# up
# * Interestingly word donation is nowhere to be seen. Asking for money
# either is not so common or is made in more veiled form

# %% cleaning messages and title

df["clean_message"] = [strip_html_tags(message) for message in df["message"]]
# Title has some missing values and we need to take them into account
# be making the datatype string (null are floats!?)
df["clean_title"] = [strip_html_tags(str(title)) for title in df["title"]]

# %%

title_wc = WordCloud(background_color="white").generate(" ".join(df["clean_title"]))
plt.imshow(title_wc, interpolation="bilinear")
plt.title("Ad title")
plt.show()

message_wc = WordCloud(background_color="white").generate(" ".join(df["clean_message"]))
plt.imshow(message_wc, interpolation="bilinear")
plt.title("Ad message")
plt.show()

# %% [markdown]

# # Length of the message
#
# * The average wordcount is 312 words and even the median is 210. The messages are
# really long compared to normal adds (from personal experience)
# * There are some extremely long messages with max word count being 13465, which
# is really a length comparable to a long news article (who is this for)
# * Minimum word length is 3
# * Because the length of the message varies so much we should
# take this into account in the afinn sentiment analysis. A message
# with 2000 words and one highly positive word should not get a high score

# %%


print(df.clean_message.str.len().describe())
print("median wordcount:", df.clean_message.str.len().median())
sns.distplot(df.clean_message.str.len(), bins=100)
plt.title("Distribution of the message wordcount")
plt.show()

# %% [markdown]

# * Shortest message is TBD and seems like a placeholder
# * longest mesage is an excerpt from a novel and definitely not
# a political add

# %%

print("shortest message:")
print(list(df[df.clean_message.str.len() == 3]["clean_message"]))
print("")

print("longest message:")
print(list(df[df.clean_message.str.len() == 13563]["clean_message"]))
print("")

# %% [markdown]

# # Relation of message length to its politicalness
#
# * When the length of the message grows from 0 to 1000, the proportion
# of ads with equal or more political than not political votes increases
# * adds which are very long, are predicted as political, but have
# their votes all over the place (much variation because of small number
# of votes?)
# %%

sns.lineplot(
    df.clean_message.str.len().round(-1),
    df.political_probability * 100,
    label="political_probability",
)
sns.lineplot(
    df.clean_message.str.len().round(-1),
    (df.political >= df.not_political) * 100,
    label="political votes >= not political votes",
)
plt.ylabel('%')
plt.legend()
# plt.title("Distribution of political probability")
plt.show()

# %% [markdown]

# # Actual look at the long messages
#
# * These seem mostly political. Lets keep these in the analysis

# %%

for message in df[df.clean_message.str.len() > 2000]["clean_message"]:
    print(message)
    print("")



# %% [markdown]

# # Who is the advertiser
#
# * About information as the same as title
# * Clearly has missing information with the nan markings
# * We saw at the beginning that this has around 25 % of values
# missing

# %%

# need to change type as missing are floats
advertiser_wc = WordCloud(background_color="white").generate(
    " ".join(df["advertiser"].apply(str))
)
plt.imshow(advertiser_wc, interpolation="bilinear")
plt.title("Advertiser")
plt.show()

# %% [markdown]

# # Who paid for the add
#
# * Very much like advertiser
# * We saw at the beginning that this has around 30 % of values
# missing, so even more than advertiser

# %%

# need to change type as missing are floats
df["paid_for_by_clean"] = [strip_html_tags(str(title)) for title in df["paid_for_by"]]
who_paid_wc = WordCloud(background_color="white").generate(
    " ".join(df["paid_for_by_clean"])
)
plt.imshow(who_paid_wc, interpolation="bilinear")
plt.title("Who paid?")
plt.show()


# %% [markdown]

# # Targetedness
#
# * Has around 30 % of the values missing
# * High values are rare as are zero values. Zero makes sense because why
# would you not make an add that is targeted in Facebook?

# %%

sns.countplot(df.targetedness.dropna())
plt.title("Ad targetness distribution")
plt.show()


# %% [markdown]

# # Who was targeted most when and how many times
#
# Information about why the ad was seen in is packed in to list of dictionaries
# with target and segment separeted. To use this data we need to unpack the
# the values to separate columns by type
#
# There are few values when the number of segments in the same target type
# is more than one. We are going to ignore this fact and just keep
# one value (the last one) for each target segment. E. g. ad was seen
# because location city x and district x we only keep one of these values.
# This is rare, so does not affect the results much, but makes analyzing
# the data much easier
#
# There are also some target categories that contain no values.
# Good example are if somebody saw somethign because they liked something.
# This category was probably removed because it contains too personal information.
# To keep these in the analysis we substitute these values with
# unknown


# %%

all_targets = [
    [target["target"] for target in ast.literal_eval(targeting)]
    for targeting in df["targets"].dropna()
]
flattened_targets = [target for targets in all_targets for target in targets]

unique_targets = list(set(flattened_targets))


df["targets"].fillna("empty", inplace=True)
df["targets"].replace("[]", "empty", inplace=True)
for target_cat in unique_targets:
    target_segments = []
    for target in df["targets"]:
        if target != "empty":
            segment = None
            for target_dict in ast.literal_eval(target):
                if target_dict["target"] == target_cat:
                    # if there are no segments found unknown is
                    # our fallback value
                    segment = "unknown"
                    if "segment" in target_dict:
                        # we only keep the last found segment
                        segment = target_dict["segment"]
            target_segments.append(segment)
        else:
            target_segments.append(None)

    # keeping only columns where there are segment information
    # example likes have no information (maybe too personal?)
    if sum(segment is not None for segment in target_segments) > 0:
        df["target_" + target_cat.lower()] = target_segments

# %% [markdown]

# # Checking the values of the created segment
#
# * Most of the new columns are almost completely empty with age and region
# state being outlier in their wide use
# * Most import categories seem to be comes fom United States and is over 18. As these
# are by far the most used categories in the most used target types
# * Likes are not actually widely used as the
# * The political leanings target confirms our suspicion that we have an
# overrepresentation of liberal voters (from California also :) )

# %%

# find all created target columns
target_columns = [column for column in df.columns if "target_" in column]

df[target_columns].isnull().mean().plot.bar()
plt.title("Number of missing values in target columns")
plt.ylabel("Proportion missing")
plt.show()

for target in target_columns:
    # showing only ten most common categories
    df[target].value_counts()[:10].plot.bar()
    plt.title(target)
    plt.show()

# %%

# # Entities
#
# The data also contains pretranscribed 'things' that are mentioned
# in the data

# %%


all_entity_types = [
    [entity["entity_type"] for entity in ast.literal_eval(entities)]
    for entities in df["entities"].dropna()
]
flattened_entity_types = [
    entity for entities in all_entity_types for entity in entities
]

unique_entity_types = list(set(flattened_entity_types))


df["entities"].fillna("empty", inplace=True)
df["entities"].replace("[]", "empty", inplace=True)
for entity_type in unique_entity_types:
    target_entities = []
    for entity in df["entities"]:
        if entity != "empty":
            entity_values = []
            for entity_dict in ast.literal_eval(entity):
                if entity_dict["entity_type"] == entity_type:
                    if "entity" in entity_dict:
                        entity_values.append(entity_dict["entity"])
            target_entities.append(entity_values)
        else:
            target_entities.append([])

    df["entity_" + entity_type.lower()] = target_entities

