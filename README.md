# 🎯 War Prediction Using Social Media Analytics

<p align="center">
  <b>Analyzing YouTube Data to Understand Public Sentiment on the Russia-Ukraine War</b>
</p>

---

## 📌 What This Project Does

This project collects and analyzes YouTube comments and videos about the Russia-Ukraine war to understand public opinion and predict sentiment trends. By examining what people say on social media, we can:

- **Track public sentiment** (positive, negative, or neutral) over time
- **Identify opinion patterns** related to war events
- **Analyze engagement** on war-related content
- **Build a foundation** for predicting future sentiment shifts

### 🎓 Academic Context
**Experiment 2: Data Collection, Cleaning, and Storage**  
**Learning Outcome (LO1)**: Master the complete data pipeline from collection to storage

---

## 🚀 Quick Start Guide

### Step 1: Get Your YouTube API Key
1. Visit [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project
3. Enable "YouTube Data API v3"
4. Create credentials → API Key
5. Copy your API key

### Step 2: Install Required Libraries
```bash
pip install google-api-python-client pandas numpy matplotlib seaborn textblob nltk sqlalchemy pymongo
```

### Step 3: Run the Project
1. Open `project.ipynb` in Jupyter or VS Code
2. Find `API_KEY = 'YOUR_API_KEY'` and paste your key
3. Run all cells from top to bottom
4. Wait for data collection and analysis to complete

### Step 4: View Results
- Check the `data/` folder for collected data files
- View visualizations directly in the notebook
- Query the SQLite database for custom analytics

---

## 📊 How It Works

### The Complete Process

```
Step 1: COLLECT DATA          →  YouTube videos and comments about the war
Step 2: CLEAN DATA            →  Remove noise, duplicates, and spam
Step 3: ANALYZE SENTIMENT     →  Calculate how positive/negative each comment is
Step 4: VISUALIZE TRENDS      →  Create charts showing sentiment patterns
Step 5: STORE DATA            →  Save in multiple formats for future use
```

### What Data is Collected?

**From YouTube Videos:**
- Video title and description
- View count, like count, comment count
- Publication date
- Engagement metrics

**From YouTube Comments:**
- Comment text
- Author name
- Publication date
- Likes and replies
- **Sentiment scores** (added by our analysis)

---

## 🛠️ Technical Details

## 🛠️ Technical Details

### Technologies Used

| Technology | What It Does | Why We Use It |
|------------|--------------|---------------|
| **YouTube Data API v3** | Access YouTube data | Collect videos and comments legally |
| **pandas** | Organize data in tables | Make data easy to work with |
| **TextBlob** | Analyze sentiment | Understand if comments are positive/negative |
| **nltk** | Process text | Clean and prepare text for analysis |
| **matplotlib & seaborn** | Create charts | Visualize sentiment trends |
| **SQLite** | Store data | Save data for future analysis |

### Project Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  YOUTUBE DATA SOURCE                     │
│         (Videos & Comments about the war)                │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│              1. DATA COLLECTION                          │
│   • Search for war-related videos                        │
│   • Collect video metadata                               │
│   • Gather comments from viewers                         │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│              2. DATA CLEANING                            │
│   • Remove URLs and special characters                   │
│   • Filter out spam and duplicates                       │
│   • Standardize text format                              │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│              3. SENTIMENT ANALYSIS                       │
│   • Calculate sentiment score (-1 to +1)                 │
│   • Categorize as positive/negative/neutral              │
│   • Measure subjectivity                                 │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│              4. DATA VISUALIZATION                       │
│   • Sentiment distribution charts                        │
│   • Engagement analysis graphs                           │
│   • Time-series trend plots                              │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│              5. DATA STORAGE                             │
│   • CSV files (easy viewing)                             │
│   • JSON files (flexible format)                         │
│   • SQLite database (for queries)                        │
└─────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
Social_Media_Analytics/
│
├── project.ipynb                 # 📓 Main notebook - Run this!
│
├── README.md                     # 📖 This documentation file
│
├── data/                         # 💾 Generated data (created when you run)
│   ├── videos_data.csv          # Video information
│   ├── videos_data.json         # Video data (JSON format)
│   ├── comments_data.csv        # All comments
│   ├── comments_data.json       # Comments (JSON format)
│   ├── youtube_war_analysis.db  # SQLite database
│   └── data_summary.json        # Summary statistics
│
└── Copy_of_Untitled5.ipynb      # 📝 Backup/draft file
```

---

## 📈 What You'll Get

### 1. Cleaned Data Files
- **videos_data.csv**: All video information in spreadsheet format
- **comments_data.csv**: All comments with sentiment scores
- **youtube_war_analysis.db**: Database for advanced queries

### 2. Sentiment Analysis
Each comment is analyzed for:
- **Polarity**: From -1 (very negative) to +1 (very positive)
- **Subjectivity**: From 0 (factual) to 1 (opinionated)
- **Category**: Classified as positive, negative, or neutral

### 3. Visualizations
- Pie chart showing positive/negative/neutral distribution
- Sentiment trend over time
- Video engagement analysis
- Most controversial topics

### 4. Analytics Capabilities
Run SQL queries like:
```sql
-- Find most negative comments
SELECT comment_text, sentiment_polarity 
FROM comments 
WHERE sentiment_category = 'negative'
ORDER BY sentiment_polarity ASC;

-- Average sentiment by video
SELECT video_id, AVG(sentiment_polarity) 
FROM comments 
GROUP BY video_id;
```

---

## 🎓 Understanding the Code

### Main Components

**1. YouTubeDataCollector Class**
```python
# What it does:
- Searches for videos on YouTube
- Gets video details (views, likes, etc.)
- Collects comments from videos
- Organizes everything into tables
```

**2. DataCleaner Class**
```python
# What it does:
- Cleans messy text data
- Removes duplicates and spam
- Analyzes sentiment of each comment
- Adds useful features (word count, etc.)
```

**3. DataStorage Class**
```python
# What it does:
- Saves data in multiple formats
- Creates database with proper structure
- Enables easy data loading
- Generates summary reports
```

---

## 📊 Sample Output

### Sentiment Distribution Example:
```
Positive Comments: 245 (42%)
Negative Comments: 178 (31%)
Neutral Comments:  157 (27%)

Average Sentiment: 0.12 (slightly positive)
```

### Top Engaged Video:
```
Title: "Russia-Ukraine War Update"
Views: 2,450,000
Likes: 45,000
Comments: 3,200
Engagement Rate: 1.96%
Average Sentiment: -0.23 (negative)
```

---

## 🔧 Customization Options

### Change Target Channel
```python
# In project.ipynb, modify:
CHANNEL_ID = 'UCknLrEdhRCp1aegoMqRaCZg'  # DW News (default)

# Try other news channels:
# Al Jazeera: UCNye-wNBqNL5ZzHSJj3l8Eg
# Sky News: UCoMdktPbSTixAyNGwb-UYkQ
```

### Adjust Data Collection
```python
MAX_VIDEOS = 5              # Number of videos to analyze
MAX_COMMENTS_PER_VIDEO = 50 # Comments per video
SEARCH_QUERY = 'Russia Ukraine war'  # Search terms
```

---

## ⚠️ Important Notes

### API Limitations
- **Daily Quota**: 10,000 units per day
- **Cost per search**: 100 units
- **Cost per video**: 1 unit
- **Cost per comment list**: 1 unit

**Example**: Analyzing 5 videos ≈ 110 units (you can run ~90 times per day)

### Best Practices
- ✅ Start with small datasets (5-10 videos)
- ✅ Test with one channel first
- ✅ Monitor your API quota usage
- ✅ Save your results regularly
- ❌ Don't exceed quota limits
- ❌ Don't spam API requests

### Privacy & Ethics
- Only collects **public** data
- Respects YouTube terms of service
- No personal user tracking
- For **educational purposes** only

---

## 🐛 Troubleshooting

### Problem: "API Key Error"
**Solution**: 
- Verify your API key is correct
- Check that YouTube Data API v3 is enabled
- Make sure there are no extra spaces in the key

### Problem: "Quota Exceeded"
**Solution**:
- Wait 24 hours for quota reset
- Request quota increase in Google Cloud Console
- Reduce MAX_VIDEOS or MAX_COMMENTS

### Problem: "No Comments Found"
**Solution**:
- Some videos have comments disabled
- Try different videos or channels
- Check video privacy settings

### Problem: "Module Not Found"
**Solution**:
- Run the first cell to install all libraries
- Restart the notebook kernel
- Install libraries manually: `pip install <library-name>`

---

## 📚 What Can You Learn?

### Skills Developed
1. **API Integration**: Working with real-world APIs
2. **Data Cleaning**: Handling messy text data
3. **Sentiment Analysis**: NLP and text analysis
4. **Data Visualization**: Creating meaningful charts
5. **Database Management**: SQL and data storage
6. **Python Programming**: Object-oriented design

### Real-World Applications
- Social media monitoring
- Brand sentiment tracking
- Political opinion analysis
- Market research
- Crisis management
- Trend prediction

---

## 🔮 Future Enhancements

### Planned Features (Next Experiments)

**Experiment 3**: Advanced Analysis
- Topic modeling (what are people talking about?)
- Named entity recognition (who is mentioned?)
- Keyword extraction
- Language detection

**Experiment 4**: Time-Series Analysis
- Predict future sentiment trends
- Correlate with real war events
- Identify sentiment shifts
- Early warning system

**Experiment 5**: Machine Learning
- Train prediction models
- Classify war-related events
- Detect misinformation
- Automated categorization

**Experiment 6**: Interactive Dashboard
- Real-time data updates
- Interactive visualizations
- Alert notifications
- Web-based interface

---

## 📖 Additional Resources

### Documentation
- [YouTube Data API Guide](https://developers.google.com/youtube/v3)
- [TextBlob Tutorial](https://textblob.readthedocs.io/)
- [Pandas Documentation](https://pandas.pydata.org/)
- [NLTK Book](https://www.nltk.org/book/)

### Learning Materials
- Google's API Python Quickstart
- Sentiment Analysis with Python course
- Data Cleaning Best Practices
- SQL Query Examples

---

## ✅ Learning Outcomes Achieved

By completing this project, you will demonstrate:

- ✅ **Data Collection**: Successfully gather data from social media APIs
- ✅ **Data Cleaning**: Process and prepare raw data for analysis  
- ✅ **Data Storage**: Implement multiple storage solutions
- ✅ **Data Analysis**: Perform sentiment analysis on text data
- ✅ **Data Visualization**: Create meaningful visual representations
- ✅ **Documentation**: Write clear technical documentation

---

## 👤 Project Information

**Project Name**: Social Media Analytics - War Prediction  
**Experiment**: 2 (Data Collection, Cleaning, and Storage)  
**Repository**: AnshulParkar/Social_Media_Analytics  
**Date**: February 2026  
**Version**: 1.0  

---

## 📝 License & Usage

This project is for **educational purposes only**. 

- Respects YouTube Terms of Service
- Uses official APIs only
- No web scraping or unauthorized access
- Public data only

---

**Need Help?** Check the troubleshooting section or review the inline comments in `project.ipynb`

**Questions?** Review the documentation links or YouTube Data API official guides 
