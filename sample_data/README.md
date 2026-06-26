# Sample Data

This folder contains a small sample of the YouTube metadata used in the Nepal Gen Z Digital Mobilization project. The sample is provided solely to illustrate the expected input format for the preprocessing and analysis scripts.

## File Description

### `youtube_metadata_sample.csv`

This file contains a small subset of publicly available YouTube metadata. It is intended for testing the preprocessing pipeline and demonstrating the required input format.

The complete dataset is not included in this repository because it is being used in multiple ongoing research projects. A representative sample is provided to demonstrate the expected input format and enable users to run the preprocessing pipeline.

## Metadata Description

| Column           | Description                                         |
| ---------------- | --------------------------------------------------- |
| `video_title`    | Original YouTube video title.                       |
| `video_ID`       | Unique YouTube video identifier.                    |
| `channel_name`   | Name of the YouTube channel.                        |
| `channel_id`     | Unique YouTube channel identifier.                  |
| `likes`          | Number of likes at the time of data collection.     |
| `views`          | Number of views at the time of data collection.     |
| `comments_count` | Number of comments at the time of data collection.  |
| `share_link`     | Public URL of the YouTube video.                    |
| `published_at`   | Video publication timestamp (UTC, ISO 8601 format). |
