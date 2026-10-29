# Your Assignment Code Style Pattern (learned from your files)

## EXACT PATTERN YOU FOLLOW:

### CELL 1 [MARKDOWN] → Main Title
# IT462 - Exploratory Data Analysis - Project
## Title of Project

### CELL 2 [CODE] → All imports + load data (NO comments inside)
import pandas as pd
...
df = pd.read_csv(...)

### CELL 3 [MARKDOWN] → Section heading
### Section Name | Description

### CELL 4 [MARKDOWN] → Sub-question or task description
**a) What we are doing in bold**

### CELL 5 [CODE] → Clean code, minimal inline comments only where needed
df.head()

### CELL 6 [MARKDOWN] → Observation
**Observation:** What we found in 1-2 sentences.

## RULES LEARNED:
- No comments inside code (or very minimal like # StandardScaler)
- Bold text for question/task labels in markdown
- "Observation:" or "Conclusion:" keyword for insights
- Short 1-3 sentence observations, not long paragraphs
- Section heading format: ### Section | Description
- Sub-task format: **a) Task description**
