# Preprocessing

This folder contains scripts for preparing YouTube metadata for downstream modeling.

Current scripts include:

* Cleaning video titles (removing emojis, hashtags, and extra whitespace)
* Automatic language detection
* Heuristic correction for romanized Hindi/Nepali text
* English translation of multilingual titles

The preprocessing script expects an input CSV containing at least a `video_title` column.
