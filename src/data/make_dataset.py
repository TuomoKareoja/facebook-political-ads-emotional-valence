# -*- coding: utf-8 -*-
import logging
import os
from pathlib import Path

import pandas as pd
from afinn import Afinn
from dotenv import find_dotenv, load_dotenv
import src.text.process_text as process_text
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)

    logger.info("Reading in dataset keeping interesting columns")
    df = pd.read_csv(
        os.path.join("data", "raw", "fbpac-ads-en-US.csv"),
        usecols=[
            "id",
            "political",
            "not_political",
            "title",
            "message",
            "created_at",
            "updated_at",
            "advertiser",
            "paid_for_by",
            "impressions",
            "political_probability",
            "targets",
            "targetedness",
        ],
        # for testing purposes
        # nrows=100,
    )

    logger.info("Dropping adds where predicted probability of being political < 0.7")
    df.query("political_probability >= 0.7", inplace=True)

    logger.info("Add columns for days up")
    df["created_at"] = pd.to_datetime(df.created_at)
    df["updated_at"] = pd.to_datetime(df.updated_at)

    logger.info("Keeping only 2018 data")
    df = df[df["created_at"].dt.year == 2018]

    logger.info("Calculating the days the ad was up")
    df = df.assign(days_up=(df["updated_at"] - df["created_at"]).dt.days)

    logger.info("Expand different target types to to different columns")
    df = process_text.expand_targets(df)

    logger.info("Preprocessing text")
    df["message"] = [process_text.strip_html_tags(message) for message in df["message"]]
    # str() deals with missing values that are actually of type float!
    df["title"] = [process_text.strip_html_tags(str(title)) for title in df["title"]]

    logger.info("Coding text sentiment")
    af = Afinn()
    df["sentiment_afinn"] = [af.score(text) for text in df["message"]]
    analyzer = SentimentIntensityAnalyzer()
    df["sentiment_vader"] = [
        analyzer.polarity_scores(message)["compound"] for message in df["message"]
    ]

    logger.info("Saving processed data")
    df.to_csv(os.path.join("data", "processed", "data_sentiment.csv"), index=False)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automatically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
