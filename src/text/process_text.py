# -*- coding: utf-8 -*-
import ast

from bs4 import BeautifulSoup


# TODO:
# add docstring


def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text()
    return stripped_text


def expand_targets(df):
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

    df.drop(columns=["targets"], inplace=True)

    return df
