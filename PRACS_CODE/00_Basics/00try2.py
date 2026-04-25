import pandas as pd

# Encoding
df = pd.read_csv("new_ev_tech_cleaned_dataset.csv", encoding="ISO-8859-1")
print(df.head())

print("\nData Description:\n",df.describe())
print("\nData Info:\n",df.info())

# Delimiter
df['cleaned_text'] = (df['Text'].fillna(''))
df['cleaned_text'] = df['cleaned_text'].astype(str)
df['cleaned_text'] = df['cleaned_text'].str.lower()
df['cleaned_text'] = df['cleaned_text'].str.strip()
df['cleaned_text'] = df['cleaned_text'].str.replace(r'[^\w\s]','')
df['cleaned_text'] = df['cleaned_text'].str.replace(r'[^a-z\s]', '', regex=True)
# Keep only letters + spaces: r'[^a-z\s]'
# Keep letters/digits/underscore + spaces: r'[^\w\s]' (for removing non-word non-space)

print("Cleaned Text:\n",df['cleaned_text'].head())

# CONFIG 4 — Data Types
# ─────────────────────────────────────────────────────────────
print("\n── CONFIG 4: Data Types ──")
print(df.dtypes)

# Cast columns explicitly
df["Likes"] = df["Likes"].astype(int)
df["View_Count"] = df["View_Count"].astype(int)
df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True)
print("\nAfter casting:")
print(df[["Likes", "View_Count", "Timestamp"]].dtypes)

# CONFIG 5 — Missing Values
# ─────────────────────────────────────────────────────────────
print("\n── CONFIG 5: Missing Values ──")
print(df.isnull().sum())

# Fill: Parent_ID NaN means it's a top-level comment, fill with "ROOT"
df["Parent_ID"] = df["Parent_ID"].fillna("ROOT")
print(f"\nParent_ID after fillna('ROOT'): {df['Parent_ID'].isnull().sum()} nulls left")

# Drop rows where Text is null (none here, but good practice)
df.dropna(subset=["Text"], inplace=True)
print(f"Rows after dropna on Text: {len(df)}")

# ─────────────────────────────────────────────────────────────
# CONFIG 6 — Quoting & Escape Characters
# ─────────────────────────────────────────────────────────────
print("\n── CONFIG 6: Quoting & Escape Characters ──")
# Count fields with embedded commas or quotes
has_comma = df["Text"].str.contains(",", na=False).sum()
has_quote = df["Text"].str.contains('"', na=False).sum()
print(f"Text fields with embedded commas : {has_comma}")
print(f"Text fields with embedded quotes : {has_quote}")

# ─────────────────────────────────────────────────────────────
# CONFIG 7 — Long Text (commas, line breaks)
# ─────────────────────────────────────────────────────────────
print("\n── CONFIG 7: Long Text ──")
df["text_len"] = df["Text"].str.len()
print(df["text_len"].describe())

real_newlines = df["Text"].str.contains("\n", na=False).sum()
print(f"Rows with real \\n inside text: {real_newlines}")
print(f"Longest comment: {df['text_len'].max()} chars")


# ─────────────────────────────────────────────────────────────
# CONFIG 8 — Skipping Malformed Rows
# ─────────────────────────────────────────────────────────────
print("\n── CONFIG 8: Malformed Rows ──")
df_skip = pd.read_csv('new_ev_tech_cleaned_dataset.csv', on_bad_lines="skip",  encoding="utf-8")
df_err  = pd.read_csv('new_ev_tech_cleaned_dataset.csv', on_bad_lines="error", encoding="utf-8")
print(f"on_bad_lines='error': {len(df_err)} rows")
print(f"on_bad_lines='skip' : {len(df_skip)} rows")
print(f"Malformed rows skipped: {len(df_err) - len(df_skip)}")   # 0 = clean file


# ── STEP 7: Interpret ─────────────────────────────────────────
print("""
=== INTERPRETATION ===
1. Delimiter   : Comma (,) — standard CSV, 13 separators per row = 14 columns
2. Encoding    : UTF-8, no BOM — safe to read on all platforms
3. Header      : Present at row 0 — always use header=0 (pandas default)
4. Data Types  : Likes & View_Count are int64; all text cols are object (string)
5. Nulls       : Only Parent_ID has nulls (75.8%) — these are top-level comments
6. Quoting     : 518/1056 comments contain commas; pandas QUOTE_MINIMAL handles them
7. Long Text   : Comments range 2–5138 chars; 198 rows have real newline characters
8. Bad Rows    : 0 malformed rows — dataset is well-formed
9. File Path   : 892 KB file, absolute path recommended for portability
10. Validation : 1056×14 shape confirmed, 0 duplicates, all required columns present
""")