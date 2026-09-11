# Step 1 Findings: Data Ingestion & Schema Understanding

## 1. Dataset Overview & Inventory
- **Raw File**: `twcs.csv` (492.58 MB / 516,508,641 bytes)
- **Total Rows**: 2,811,774
- **Total Columns**: 7 (`tweet_id`, `author_id`, `inbound`, `created_at`, `text`, `response_tweet_id`, `in_response_to_tweet_id`)
- **Duplicate `tweet_id`s**: 0 (0% duplicate rate; `tweet_id` is a unique primary key)
- **Author Demographics**:
  - Total unique authors: 702,777
  - Unique brand (outbound) handles: 108 (e.g., `AmazonHelp`, `AppleSupport`, `Uber_Support`, `SpotifyCares`, `Delta`)
  - Unique customer (inbound) IDs: 702,669 (anonymized numeric strings like `115712`, `173519`)
  - Overlap between brands and customers: 0 (clean separation between customer and support agent identities)
- **Inbound Distribution**:
  - Inbound (Customer): 1,537,843 (54.69%)
  - Outbound (Brand): 1,273,931 (45.31%)

---

## 2. Schema Semantics & Verified Meaning of Columns

| Column | Data Type | Null Count | Null % | Verified Semantic Meaning |
| :--- | :--- | :--- | :--- | :--- |
| `tweet_id` | `int64` | 0 | 0.00% | Unique global identifier for the tweet. Primary key. |
| `author_id` | `object` (str) | 0 | 0.00% | Name of the brand handle (for brands) or anonymized numeric ID (for customers). |
| `inbound` | `bool` | 0 | 0.00% | `True` if tweet was sent by a customer to a brand; `False` if sent by a brand/agent. |
| `created_at` | `object` (str) | 0 | 0.00% | Timestamp in UTC format `%a %b %d %H:%M:%S +0000 %Y` (2008-05-08 20:13:59 to 2017-12-03 23:14:01). |
| `text` | `object` (str) | 0 | 0.00% | Raw tweet text content, including user mentions (`@handle`), links (`https://t.co/...`), and agent sign-offs (e.g., `^PA`, `*KittyG`, `/AP`). |
| `response_tweet_id` | `object` (str) | 1,040,629 | 37.01% | Forward pointer: The ID(s) of tweets that replied directly to this tweet. Can contain multiple comma-separated IDs. |
| `in_response_to_tweet_id` | `object` (str) | 794,335 | 28.25% | Backward pointer: The single ID of the parent tweet this tweet is responding to. |

---

## 3. Empirical Thread Linkage & Coverage Analysis

### 3.1 Forward Linkage: `inbound` → `response_tweet_id` (Task 3e)
- **Total Inbound Tweets**: 1,537,843
- **Inbound with Non-Null `response_tweet_id`**: 1,303,829 (84.78%)
- **Resolvability (First Response ID)**: 1,280,997 / 1,303,829 (98.25%) exist in the dataset.
- **Brand Resolution**: 1,137,425 / 1,280,997 (88.79%) of resolved response tweets belong to a brand author (`inbound=False`).
- **All Response IDs Coverage**: Across all 1,555,781 referenced IDs in comma-separated strings, 1,450,335 (93.22%) exist in the dataset.

### 3.2 Backward Linkage: `brand` → `in_response_to_tweet_id` (Task 3d & 3f)
- **Total Brand Tweets (`inbound=False`)**: 1,273,931
- **Brand Tweets with Non-Null `in_response_to_tweet_id`**: 1,266,942 (99.45%)
- **Brand Resolvability**: 1,265,281 / 1,266,942 (99.87%) point to a tweet that exists in the dataset.
- **Customer Parent Resolution**: 1,261,888 / 1,265,281 (99.73%) point to an inbound customer tweet.
- **Random 500 Sample Resolution (Task 3f)**: 500/500 (100.00%) of referenced parent tweets exist in the dataset, while 0/500 (0.00%) are missing (external/uncollected parent tweets).
- **Overall Dataset `in_response_to_tweet_id` Resolution**: 2,013,577 / 2,017,439 (99.81%).

---

## 4. Multi-Turn Threads & Structural Verification (Task 3g)

Threads are not limited to single question-answer pairs; customer conversations frequently continue across multiple turns:

```text
Turn 1 [Tweet 5] Customer (115712): @sprintcare I did.
  └── Turn 2 [Tweet 4] Brand (sprintcare): @115712 Please send us a Private Message so that we can further assist you. Just click ‘Message’ at the top of your profile.
        └── Turn 3 [Tweet 3] Customer (115712): @sprintcare I have sent several private messages and no one is responding as usual
              └── Turn 4 [Tweet 1] Brand (sprintcare): @115712 I understand. I would like to assist you. We would need to get you into a private secured link to further assist.
```

```text
Turn 1 [Tweet 16] Customer (115713): @sprintcare Since I signed up with you....Since day 1
  └── Turn 2 [Tweet 15] Brand (sprintcare): @115713 We understand your concerns and we'd like for you to please send us a Direct Message, so that we can further assist you. -AA
        └── Turn 3 [Tweet 12] Customer (115713): @sprintcare You gonna magically change your connectivity for me and my whole family ? 🤥 💯
              └── Turn 4 [Tweet 11] Brand (sprintcare): @115713 This is saddening to hear. Please shoot us a DM, so that we can look into this for you. -KC
```

```text
Turn 1 [Tweet 16] Customer (115713): @sprintcare Since I signed up with you....Since day 1
  └── Turn 2 [Tweet 15] Brand (sprintcare): @115713 We understand your concerns and we'd like for you to please send us a Direct Message, so that we can further assist you. -AA
        └── Turn 3 [Tweet 12] Customer (115713): @sprintcare You gonna magically change your connectivity for me and my whole family ? 🤥 💯
              └── Turn 4 [Tweet 13] Brand (sprintcare): @115713 I would really like to work with you to have this resolved. Kindly send us a DM. I'm here for you! -ResolutionSup SR
```

---

## 5. Recommended Join Direction for Thread Reconstruction

**Primary Recommendation: Backward Linkage (`in_response_to_tweet_id`) supplemented by Forward Indexing**

### Reasoning:
1. **Determinism and 1-to-1 Relationship**: `in_response_to_tweet_id` is strictly scalar (single parent ID), whereas `response_tweet_id` contains comma-separated lists (222,426 rows, 7.91%) when multiple agents or tweets respond to the same customer query.
2. **High Resolution**: 99.87% of brand tweets with `in_response_to_tweet_id` successfully resolve to their immediate parent in the dataset (99.73% directly to a customer tweet).
3. **Natural DAG / Tree Reconstruction**: By indexing children via `in_response_to_tweet_id`, we can construct full conversation trees from root inbound tweets (`in_response_to_tweet_id IS NULL` and `inbound = True`) through all subsequent agent replies and customer follow-ups in strict chronological order.

---

## 6. Key Data Quirks Discovered
1. **Comma-Separated `response_tweet_id`**: 222,426 rows (7.91%) contain multiple IDs (e.g. `5,7` or `9,6,10`). Splitting by comma is required if navigating forward.
2. **Missing Parent Tweets**: ~0.0% of `in_response_to_tweet_id` references point to tweets not present in the dataset (deleted tweets, private mentions, or conversations that started before the data collection window).
3. **Agent Signatures**: Agent tweets consistently feature sign-off codes (e.g., `^PA`, `^NK`, `*KittyG`, `/AP`, `-Sam`) and standardized redirect links (`https://t.co/...` to DMs or help portals).
4. **Character Lengths**: Inbound customer tweets average 109.9 chars (median 109), while brand replies average 118.7 chars (median 120, with max lengths up to 343).

---

## 7. Parquet Checkpoint
- Full dataset checkpoint successfully created at `data/processed/twcs_full.parquet` (251.53 MB).
