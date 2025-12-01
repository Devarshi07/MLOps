# Lab 6: Advanced Querying and Visualization in BigQuery
## Premier League Analysis

## Objective
In this lab, you will learn how to:
- Write complex SQL queries to analyze football match data using BigQuery.
- Utilize BigQuery's built-in visualization tools for sports analytics insights.
- Export and share your query results for further analysis or reporting.

## Prerequisites
Before you begin, ensure that you:
- Have a dataset loaded into BigQuery (from previous labs).
- Are familiar with basic SQL querying and data visualization concepts.
- Understand fundamental football/soccer statistics (goals, points, wins, draws, losses).

## Dataset Overview
The dataset contains **139 matches** with **131 columns** including:
- **Match Information**: Division, Date, Time, HomeTeam, AwayTeam
- **Results**: Full-Time Home Goals (FTHG), Full-Time Away Goals (FTAG), Full-Time Result (FTR)
- **Half-Time Stats**: HTHG, HTAG, HTR
- **Match Statistics**: Shots (HS/AS), Shots on Target (HST/AST), Fouls (HF/AF), Corners (HC/AC), Cards (HY/AY/HR/AR)
- **Betting Odds**: Multiple bookmaker odds (B365, BFD, BMGM, etc.) for various markets including Asian Handicap and Over/Under

## Steps

### Step 1: Perform Data Analysis Using Advanced SQL Queries

Access your existing BigQuery dataset and table at `mlops-479921.lab6_dataset.lab6_table`.

Write complex SQL queries to perform comprehensive football league analysis using aggregate functions, CASE statements, and UNION ALL to combine home and away statistics.

**Sample Query: League Standings Table**
```sql
WITH TeamMatches AS (
    -- Step 1: Get Stats for Home Games
    SELECT 
        HomeTeam AS Team,
        FTHG AS GoalsFor,
        FTAG AS GoalsAgainst,
        CASE WHEN FTR = 'H' THEN 3 WHEN FTR = 'D' THEN 1 ELSE 0 END AS Points,
        CASE WHEN FTR = 'H' THEN 1 ELSE 0 END AS Win,
        CASE WHEN FTR = 'D' THEN 1 ELSE 0 END AS Draw,
        CASE WHEN FTR = 'A' THEN 1 ELSE 0 END AS Loss
    FROM 
        `mlops-479921.lab6_dataset.lab6_table`
    
    UNION ALL
    
    -- Step 2: Get Stats for Away Games
    SELECT 
        AwayTeam AS Team,
        FTAG AS GoalsFor,
        FTHG AS GoalsAgainst,
        CASE WHEN FTR = 'A' THEN 3 WHEN FTR = 'D' THEN 1 ELSE 0 END AS Points,
        CASE WHEN FTR = 'A' THEN 1 ELSE 0 END AS Win,
        CASE WHEN FTR = 'D' THEN 1 ELSE 0 END AS Draw,
        CASE WHEN FTR = 'H' THEN 1 ELSE 0 END AS Loss
    FROM 
        `mlops-479921.lab6_dataset.lab6_table`
)
-- Step 3: Aggregate and Order
SELECT 
    Team,
    COUNT(*) AS Played,
    SUM(Win) AS Won,
    SUM(Draw) AS Drawn,
    SUM(Loss) AS Lost,
    SUM(GoalsFor) AS GF,
    SUM(GoalsAgainst) AS GA,
    (SUM(GoalsFor) - SUM(GoalsAgainst)) AS GD,
    SUM(Points) AS Points
FROM 
    TeamMatches
GROUP BY 
    Team
ORDER BY 
    Points DESC, GD DESC, GF DESC;
```

Experiment with different functions and combinations of SQL clauses to gain insights from your data.

### Step 2: Visualize Data Using BigQuery UI

After running your queries, visualize the results directly within BigQuery:

In the BigQuery console, run your query. Click on **Explore Data** and choose a visualization type that best represents your data (e.g., bar chart, scatter plot, line chart). Customize your chart with appropriate axis labels, colors, and other formatting options.

**Task**: Create at least two different visualizations from your query results, such as:
- A bar chart comparing the total goals scored by different teams.
- A scatter plot showing the relationship between shots and goals to analyze shooting efficiency.

### Step 3: Export and Share Query Results

Once you've completed your analysis, export the query results for sharing with others or for further analysis:

**Export to Google Sheets**: Run the query, click on the **Save Results** button, and select **Google Sheets**. The data will be exported directly to a new Google Sheets file.

**Download as CSV**: Alternatively, download the query results as a CSV file for use in other tools like Excel or Looker Studio.

**Steps to Export to Google Sheets:**
1. Run your query.
2. Click the **Save Results** button at the bottom of the BigQuery interface.
3. Choose **Google Sheets** from the dropdown.
4. Your data will be exported directly to Google Sheets, ready for sharing or further processing.

## Looker Studio Visualization Sample

To enhance your data visualization skills further, import your dataset into Looker Studio and create professional visualizations. Here's a guide on setting up a few common charts using the football dataset:

**Import Dataset**: In Looker Studio, connect to your BigQuery dataset.

**Create Visualizations**:
- **Scorecard**: Display total home goals (203) as a key performance indicator.
- **Bar Chart**: Visualize average goals scored by each team, showing power rankings.
- **Scatter Plot**: Compare the relationship between shots and goals to identify shooting efficiency.
- **Filter Control**: Add a dropdown filter for HomeTeam to enable interactive exploration.

**Customize the Dashboard**: Adjust colors (using dark theme with orange/yellow accents), add filters, and modify the layout for a comprehensive dashboard.

## Changes from Original Lab

This football analytics lab differs from the original Iris dataset lab in several key ways:

### Data Complexity
- **Original**: Simple botanical measurements (150 rows, 5 columns)
- **This Lab**: Complex sports data with 139 matches and 131 columns including match results, statistics, and betting odds

### SQL Techniques
- **Original**: Basic aggregations with GROUP BY
- **This Lab**: 
  - Advanced Common Table Expressions (CTEs)
  - UNION ALL operations to combine home/away perspectives
  - Complex CASE statements for business logic (points calculation based on match results)
  - Multi-level sorting criteria (Points → Goal Difference → Goals For)

### Domain Context
- **Original**: Scientific classification problem with species comparison
- **This Lab**: Real-world sports analytics with industry-standard metrics like league tables, goal difference, shooting efficiency, and competitive rankings

### Visualization Focus
- **Original**: Species comparison with simple bar charts and scatter plots
- **This Lab**: 
  - Performance rankings with horizontal bar charts
  - Efficiency analysis (shots to goals conversion) using scatter plots
  - Season-level KPIs with scorecard metrics
  - Professional dark-themed dashboard suitable for sports analytics platforms


## Conclusion

This lab guides you through advanced querying techniques in BigQuery specifically for football/soccer analytics. You've learned to:
- Transform raw match data into meaningful league standings using CTEs and UNION operations
- Apply complex SQL patterns with business logic implementation
- Create industry-standard sports visualizations with professional styling
- Build dashboards in Looker Studio for executive-level reporting
- Export and share analytical insights with stakeholders

These skills are directly applicable to sports analytics roles, betting analysis, fantasy sports platforms, and sports journalism. The techniques demonstrated here form the foundation for more advanced predictive modeling and machine learning applications in sports.
